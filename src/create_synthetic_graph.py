import itertools
import logging
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from SPARQLWrapper import DIGEST, SPARQLWrapper

from config import RunConfig
from gen_triples import (
    RawTriple,
    apply_recursive_rule,
    apply_rule,
    from_binding_row,
    gen_extensional_graph,
)
from graph_metrics import GraphMetrics, PredicateProfile
from queries import (
    clear_named_graph,
    get_frequency,
    get_support,
    get_total_triples,
    insert_triples_gsp,
    insert_triples_sparql,
)
from rules import (
    HornRule,
    check_uninferrable_preds,
    get_predicate_mapping,
    get_ruleset_dependencies,
    parse_rule_set,
)
from utils import format_triple, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Update support and frequencies.
# ---------------------------------------------------------------------------
def get_closed_rules(
    sparql_client: SPARQLWrapper, graph_uri: str, rules: dict[str, HornRule]
) -> set[str]:
    """Queries the database to find which rules reached their target support.

    A rule is considered 'closed' when the count of distinct bindings
    satisfying both its body and head in the graph meets or exceeds
    its defined support threshold.
    """
    closed_rules: set[str] = set()

    # Check support
    for r_id, rule in rules.items():
        if support := get_support(sparql_client, rule, graph_uri):
            if support >= rule.support:
                closed_rules.add(r_id)

    return closed_rules


def get_closed_predicates(
    sparql_client: SPARQLWrapper, graph_uri: str, profiles: dict[str, PredicateProfile]
) -> set[str]:
    """Queries the database to find which predicates reached their target frequency.

    A predicate is considered 'closed' when the count of distinct triples containing the
    predicate is equal to the predicate's frequency.
    """

    closed_predicates: set[str] = set()

    for predicate, profile in profiles.items():
        if frequency := get_frequency(sparql_client, predicate, graph_uri):
            if frequency >= profile.frequency:
                closed_predicates.add(predicate)

    return closed_predicates


# ---------------------------------------------------------------------------
# Helper functions.
# ---------------------------------------------------------------------------
def _execute_stratification(
    sparql_client: SPARQLWrapper,
    graph_uri: str,
    rules: dict[str, HornRule],
    profiles: dict[str, PredicateProfile],
    extensional_predicates: set[str],
    intensional_predicates: set[str],
    term_mapping: dict[str, str],
    start_time: float,
    chunk_size: int = 50,
) -> None:
    """Handles the loop for the rule application and predicate closing."""
    rule_dependency = get_ruleset_dependencies(rules, intensional_predicates)
    predicate_to_rules = get_predicate_mapping(rules)

    closed_predicates = set(extensional_predicates)
    grounded_predicates = set(extensional_predicates)
    closed_rules: set[str] = set()

    def is_ready(rule_id: str) -> bool:
        """Evaluates if a rule should be included in the strata"""
        if rule_id in closed_rules:
            return False

        rule = rules[rule_id]
        if rule.head.predicate in closed_predicates:
            return False

        if any(r not in closed_rules for r in rule_dependency.get(rule_id, [])):
            return False

        body_preds = rule.get_body_predicates() - {rule.head.predicate}
        if not body_preds.issubset(grounded_predicates):
            return False

        return True

    strata = 0
    while True:
        strata += 1

        # Determine the ruleset for the next strata
        available_rules = [r_id for r_id in rules if is_ready(r_id)]

        direct_rules: dict[
            str, HornRule
        ] = {}  # TODO: Why uno es lista y el otro diccionario??
        direct_profiles: dict[str, PredicateProfile] = {}

        recursive_rules: list[HornRule] = []
        rescursive_profiles: dict[str, PredicateProfile] = {}

        for rule_id in available_rules:
            rule = rules[rule_id]
            predicate = rule.head.predicate

            if predicate in rule.get_body_predicates():
                recursive_rules.append(rule)
                rescursive_profiles[predicate] = profiles[predicate]

            else:
                direct_rules[rule_id] = rule
                direct_profiles[predicate] = profiles[predicate]

        logger.info("%d rules ready for Strata %d", len(available_rules), strata)

        if not (direct_rules or recursive_rules):
            logger.info(
                "No more rules to apply, execution finished after %.2f s.",
                time.time() - start_time,
            )
            break

        kwargs = {
            "graph_uri": graph_uri,
            "sparql_client": sparql_client,
        }
        # Apply direct rules
        if direct_rules:
            for rule in direct_rules.values():
                raw_triple_iterator = apply_rule(**kwargs, rule=rule)
                predicate = rule.head.predicate

                filtered_triples = filter_and_format_triples(
                    raw_triple_iterator=raw_triple_iterator,
                    predicate=predicate,
                    profile=profiles[predicate],
                    term_mapping=term_mapping,
                )

                _count = insert_triples_sparql(
                    **kwargs,
                    triple_stream=filtered_triples,
                    chunk_size=chunk_size,
                )
                grounded_predicates.add(predicate)
                r_id = rule.rule_id
                logger.info("Updated graph with %d triples from rule %s.", _count, r_id)

        # Apply recursive rules
        if recursive_rules:
            for rule in recursive_rules:
                raw_bindings = apply_recursive_rule(**kwargs, rule=rule)
                predicate = rule.head.predicate

                filtered_triple_iterator = filter_and_format_triples_recursive(
                    profile=profiles[predicate],
                    predicate=predicate,
                    bindings=raw_bindings,
                    rule=rule,
                    term_mapping=term_mapping,
                )

                _count = insert_triples_sparql(
                    **kwargs,
                    triple_stream=filtered_triple_iterator,
                    chunk_size=chunk_size,
                )
                grounded_predicates.add(predicate)
                r_id = rule.rule_id
                logger.info("Updated graph with %d triples from rule %s.", _count, r_id)

        # Update closed predicates
        closed_preds_old = closed_predicates
        closed_predicates = get_closed_predicates(**kwargs, profiles=profiles)

        if new_closed_preds := (closed_predicates - closed_preds_old):
            logger.info("Closed predicates in this strata: %s", new_closed_preds)
        else:
            logger.debug("No predicates were closed this strata.")

        if not (intensional_predicates - closed_predicates):
            logger.info(
                "All predicates closed. Execution finished after %.2f s.",
                time.time() - start_time,
            )
            break

        # Update closed rules
        closed_rules_old = set(closed_rules)

        affected_rules = {
            rule_id
            for pred in {rules[r].head.predicate for r in available_rules}
            for rule_id in predicate_to_rules[pred]
        } - closed_rules

        if rules_to_check := {r_id: rules[r_id] for r_id in affected_rules}:
            closed_rules.update(
                get_closed_rules(sparql_client, graph_uri, rules_to_check)
            )

        if new_closed_rules := (closed_rules - closed_rules_old):
            logger.info("Closed rules in this strata: %s", new_closed_rules)
        else:
            logger.debug("No rules were closed this strata.")

        logger.info("%d closed rules out of %d.", len(closed_rules), len(rules))

        if not (rules.keys() - closed_rules):
            logger.info(
                "All rules closed. Execution finished after %.2f s.",
                time.time() - start_time,
            )
            break


# ---------------------------------------------------------------------------
# Main funtion.
# ---------------------------------------------------------------------------
def create_synthetic_graph(
    sparql_client: SPARQLWrapper,
    rules_df: pd.DataFrame,
    profiles: dict[str, PredicateProfile],
    pca_threshold: float,
    ontology_file: Path,
    default_namespace: str,
) -> None:
    """Generates a new graph from a set of rules and metrics from the orginal graph.

    First, a new graph is created containing a random extensional database. Then, it
    uses the rules to generate the rest of the triples to complete the graph.

    Args:
        rules_df: DataFrame containing information about rules per row.
        graph_metrics: Predicates' frequency and distribution.
        output_path: Path to store the generated extensional graph and final synthetic
        database.
        namespace: Namespace URI.
        pca_threshold: Value to determine if a rule is positive or negative.
    """
    start_time = time.time()
    graph_uri = "http://SyntheticKG.org/"

    # Parse rules and setup
    rules, intensional_predicates, term_mapping = parse_rule_set(
        rules_df=rules_df,
        ontology_file=ontology_file,
        pca_threshold=pca_threshold,
        default_namespace=default_namespace,
    )

    extensional_predicates = profiles.keys() - intensional_predicates

    logger.info(
        "Found %d extensional predicates out of %d total predicates: %s",
        len(extensional_predicates),
        len(profiles),
        extensional_predicates,
    )
    if not extensional_predicates:
        raise ValueError("Cannot proceed with 0 extensional predicates.")

    check_uninferrable_preds(
        rules=rules,
        intensional_predicates=intensional_predicates,
        extensional_predicates=extensional_predicates,
    )

    # Clean the space where the new Synthetic graph is going to be stored
    clear_named_graph(graph_uri=graph_uri, sparql_client=sparql_client)
    logger.info(
        "Creating synthetic graph using %d rules and %d predicate profiles.",
        len(rules_df),
        len(profiles),
    )

    # Setp 1: Generate EDB
    gen_extensional_graph(
        sparql_client=sparql_client,
        predicate_profiles={k: profiles[k] for k in extensional_predicates},
        graph_uri=graph_uri,
        term_mapping=term_mapping,
    )

    # Step 2: Perform stratification and apply rules iteratively
    _execute_stratification(
        sparql_client=sparql_client,
        graph_uri=graph_uri,
        rules=rules,
        profiles=profiles,
        extensional_predicates=extensional_predicates,
        intensional_predicates=intensional_predicates,
        term_mapping=term_mapping,
        start_time=start_time,
    )

    count = get_total_triples(sparql_client, graph_uri)
    logger.info("Graph at %s has %d triples.", count, graph_uri)


def filter_and_format_triples(
    raw_triple_iterator: Iterator[RawTriple],
    profile: PredicateProfile,
    predicate: str,
    term_mapping: dict[str, str],
) -> Iterator[str]:
    # TODO: Change so it is aware of global profile of the triple
    """Filters raw triples against profile constraints and yields RDF strings."""

    # Pre-allocate state using dictionary comprehensions for speed
    for triple in raw_triple_iterator:
        # TODO: Any filtering of these triples must be done here with the profile
        yield format_triple(
            subject=triple.subject_val,
            predicate=triple.predicate,
            obj=triple.object_val,
            term_mapping=term_mapping,
        )

    logger.info("Successfully introduced triples for %s.", predicate)


def filter_and_format_triples_recursive(
    profile: PredicateProfile,
    predicate: str,
    bindings: list[dict[str, Any]],
    rule: HornRule,
    term_mapping: dict[str, str],
) -> Iterator[str]:
    """Filters raw triples against profile constraints and yields RDF strings."""

    if bindings:
        # Shuffle to avoid deterministic bias if multiple valid options exist
        shuffled_bindings = list(bindings)
        random.shuffle(shuffled_bindings)

        created = 0
        for binding_row in shuffled_bindings:
            if created >= profile.frequency:
                logger.debug("Closed %s, finishing generation.", predicate)
                break

            # Create that many triples
            triples: set[RawTriple] = set()
            for atom in rule.body:
                if atom.predicate == predicate:
                    subject_val, _ = from_binding_row(atom.subject, binding_row)
                    object_value, object_type = from_binding_row(atom.obj, binding_row)

                    triples.add(
                        RawTriple(
                            atom.predicate, subject_val, object_value, object_type
                        )
                    )

            # TODO: If we were to filter the triples, this must be done here
            for t in triples:
                # TODO: This does not take into account possible literals
                yield format_triple(
                    subject=t.subject_val,
                    predicate=t.predicate,
                    obj=t.object_val,
                    term_mapping=term_mapping,
                )
                # TODO: Update profile


def create_predicate_searchspace(
    predicate: str,
    profile: PredicateProfile,
    term_mapping: dict[str, str],
    searchspace_uri: str,
) -> None:
    """Generates and inserts a predicate search space into the database.

    Creates all possible triples for the given predicate using the cartesian
    product of the domain and range entities, and inserts them into a specific
    named graph using batching to ensure scalability.

    Args:
        database_endpoint: The URL of the SPARQL database endpoint.
        predicate: The URI of the predicate to link subjects and objects.
        profile: A profile object containing `domain` and `range` Counters.

    Returns:
        The URI of the generated search space named graph.

    Raises:
        Exception: If a SPARQL insertion batch fails.
    """

    count = len(profile.domain) * len(profile.range)

    logger.info("Creating searchspace for %s with %d triples", predicate, count)

    triple_generator: Iterator[str] = (
        format_triple(subj, predicate, obj, term_mapping)
        for subj, obj in itertools.product(profile.domain.keys(), profile.range.keys())
    )

    insert_triples_gsp(
        triples=triple_generator,
        graph_uri=searchspace_uri,
    )


if __name__ == "__main__":
    # Setup
    logging_level = logging.DEBUG
    setup_logging(level=logging_level)

    config_file = Path("configurations/gen_triples/french_royalty.json")
    config = RunConfig.from_json(config_file)

    rules_df = pd.read_csv(config.data.input_dir / config.rules.rules_file)

    sparql_client = SPARQLWrapper(
        endpoint=str(config.data.database_url / config.data.sparql_endpoint)
    )
    sparql_client.setHTTPAuth(DIGEST)
    sparql_client.setCredentials("dba", "dba")

    # Execute experiment
    create_synthetic_graph(
        sparql_client=sparql_client,
        rules_df=rules_df,
        ontology_file=config.data.input_dir / config.graph.ontology_file,
        profiles=GraphMetrics.from_uri(sparql_client, config.graph.uri).profiles,
        pca_threshold=0.0,
        default_namespace=config.graph.uri,
    )
