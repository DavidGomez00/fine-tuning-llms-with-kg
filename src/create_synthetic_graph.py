import itertools
import logging
import random
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from SPARQLWrapper import DIGEST, SPARQLWrapper
from yarl import URL

from config import RunConfig
from gen_triples import (
    RawTriple,
    apply_rule,
    from_binding_row,
    gen_graph,
)
from graph_metrics import GraphMetrics, PredicateProfile
from queries import (
    SparqlBindings,
    clear_graph_sparql,
    get_frequency,
    get_support,
    get_total_triples,
    insert_triples_sparql,
)
from rules import (
    HornRule,
    check_uninferrable_preds,
    get_predicate_mapping,
    get_ruleset_dependencies,
    get_term_mapping,
    parse_rule_set,
)
from utils import format_triple, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions.
# ---------------------------------------------------------------------------
class SyntheticGraphError(Exception):
    """Base exception for errors during graph generation."""

    pass


# ---------------------------------------------------------------------------
# Update support and frequencies.
# ---------------------------------------------------------------------------
def get_closed_rules(
    client: SPARQLWrapper, graph_uri: str, rules: dict[str, HornRule]
) -> set[str]:
    """Queries the database to find which rules reached their target support.

    A rule is considered 'closed' when the count of distinct bindings
    satisfying both its body and head in the graph meets or exceeds
    its defined support threshold.
    """
    closed_rules: set[str] = set()

    # Check support
    for r_id, rule in rules.items():
        support = get_support(client, rule, graph_uri)
        logger.debug("%s [%d/%d]", r_id, support, rule.support)
        if support:
            if support >= rule.support:
                closed_rules.add(r_id)

    return closed_rules


def get_closed_predicates(
    client: SPARQLWrapper, graph_uri: str, profiles: dict[str, PredicateProfile]
) -> set[str]:
    """Queries the database to find which predicates reached their target frequency.

    A predicate is considered 'closed' when the count of distinct triples containing the
    predicate is equal to the predicate's frequency.
    """

    closed_predicates: set[str] = set()

    for predicate, profile in profiles.items():
        frequency = get_frequency(client, predicate, graph_uri)
        logger.debug("%s [%d/%d]", predicate, frequency, profile.frequency)
        if frequency >= profile.frequency:
            closed_predicates.add(predicate)

    return closed_predicates


# ---------------------------------------------------------------------------
# Filter triples from results.
# ---------------------------------------------------------------------------
def filter_and_format_triples(
    raw_bindings: SparqlBindings,
    profile: PredicateProfile,
    rule: HornRule,
    term_mapping: dict[str, str],
    use_profile: bool = True,
) -> Iterator[str]:
    # TODO: Change so it is aware of global profile of the triple
    """Filters raw triples against profile constraints and yields RDF strings."""

    # Shuffle to avoid deterministic bias if multiple valid options exist
    shuffled_bindings = list(raw_bindings)
    random.shuffle(shuffled_bindings)

    created = 0
    for bindings_row in shuffled_bindings:
        if use_profile and created >= profile.frequency:
            logger.debug("%s is closed, can't add more triples.", rule.head.predicate)
            break

        triples: set[RawTriple] = set()

        for atom in rule.signature:
            if atom.predicate == rule.head.predicate:
                subject_val, _ = from_binding_row(atom.subject, bindings_row)
                object_value, object_type = from_binding_row(atom.obj, bindings_row)
                # TODO: If we were to filter the triples, this must be done here
                # if use_profile: <- Dont forget this if you implement filtering
                triples.add(
                    RawTriple(atom.predicate, subject_val, object_value, object_type)
                )

        for t in triples:
            # TODO: This does not take into account possible literals
            yield format_triple(
                subject=t.subject_val,
                predicate=t.predicate,
                obj=t.object_val,
                term_mapping=term_mapping,
            )
            # created += 1
            # TODO: Update profile


# ---------------------------------------------------------------------------
# Helper functions.
# ---------------------------------------------------------------------------
def _execute_stratification(
    client: SPARQLWrapper,
    graph_uri: str,
    rules: dict[str, HornRule],
    profiles: dict[str, PredicateProfile],
    term_mapping: dict[str, str],
    crud_endpoint: str,
    chunk_size: int = 50,
    use_searchspace: bool = True,
    use_head: bool = True,
) -> None:
    """Handles the loop for the rule application and closure."""

    intensional_preds = {rule.head.predicate for rule in rules.values()}
    extensional_preds = profiles.keys() - intensional_preds

    rule_dependency = get_ruleset_dependencies(rules, intensional_preds)
    predicate_to_rules = get_predicate_mapping(rules)

    closed_preds = set(extensional_preds)
    grounded_predicates = set(extensional_preds)
    closed_rule_ids: set[str] = set()

    def is_ready(rule_id: str) -> bool:
        """Evaluates if a rule should be included in the iteration. A rule is considered
        ready when its body is grounded, its support is not closed, and it does not
        depend on other rules."""

        rule = rules[rule_id]
        head_predicate = rule.head.predicate

        if rule_id in closed_rule_ids or head_predicate in closed_preds:
            return False

        body_predicates = rule.get_body_predicates() - {head_predicate}
        if not body_predicates.issubset(grounded_predicates):
            return False

        for r_id in rule_dependency.get(rule_id, []):
            if r_id not in closed_rule_ids:
                return False

        return True

    iter = 0
    while True:
        iter += 1

        available_rules = {r_id: r for r_id, r in rules.items() if is_ready(r_id)}
        if not available_rules:
            logger.info("No rules to apply.")
            break

        logger.info("Iter %d: Applying %d rules.", iter, len(available_rules))

        direct_rules: list[str] = []
        recursive_rules: list[str] = []

        for r_id, r in available_rules.items():
            if r.head.predicate in r.get_body_predicates():
                recursive_rules.append(r_id)
            else:
                direct_rules.append(r_id)

        # Apply direct rules first, then recursive rules
        for r_id in itertools.chain(direct_rules, recursive_rules):
            rule = rules[r_id]
            predicate = rule.head.predicate

            raw_bindings = apply_rule(
                client=client,
                graph_uri=graph_uri,
                rule=rule,
                profile=profiles[predicate],
                term_mapping=term_mapping,
                crud_endpoint=crud_endpoint,
                use_searchspace=use_searchspace,
                use_head=use_head,
            )

            if not raw_bindings:
                logger.warning("No results found for %s", r_id)
                continue

            logger.debug("Retrieved %d bindings for %s.", len(raw_bindings), r_id)

            filtered_triple_iterator = filter_and_format_triples(
                raw_bindings=raw_bindings,
                profile=profiles[predicate],
                rule=rule,
                term_mapping=term_mapping,
            )

            count = insert_triples_sparql(
                graph_uri=graph_uri,
                client=client,
                triple_stream=filtered_triple_iterator,
                chunk_size=chunk_size,
            )
            logger.debug("Counted filtered triples: %d", count)
            if count:
                grounded_predicates.add(predicate)
                logger.info(
                    "Rule %s yielded %d triples for %s.", r_id, count, predicate
                )

        updated_closure = False
        # Update predicates
        new_preds = get_closed_predicates(client, graph_uri, profiles) - closed_preds
        if new_preds:
            logger.debug("Closed predicates %s", new_preds)
            closed_preds.update(new_preds)
            updated_closure = True

        # Update rules
        affected_rule_ids: set[str] = set()
        for rule in available_rules.values():
            head_predicate = rule.head.predicate
            affected_rule_ids.update(predicate_to_rules[head_predicate])

        if pending_rule_ids := set(affected_rule_ids) - closed_rule_ids:
            rules_to_check = {r_id: rules[r_id] for r_id in pending_rule_ids}
            logger.debug("Rules to check: %s", rules_to_check.keys())

            if closed_r := (get_closed_rules(client, graph_uri, rules_to_check)):
                logger.debug("Closed rules in this strata: %s", closed_r)
                closed_rule_ids.update(closed_r)
                updated_closure = True

        # Log
        if updated_closure:
            logger.info(
                "Closed: Rules [%d/%d] | Predicates [%d/%d].",
                len(closed_rule_ids),
                len(rules),
                len(closed_preds),
                len(profiles),
            )

        if not (intensional_preds - closed_preds):
            logger.info("All predicates closed.")
            break

        if not (rules.keys() - closed_rule_ids):
            logger.info("All rules closed.")
            break


def complete_graph(
    client: SPARQLWrapper,
    graph_uri: str,
    rules: dict[str, HornRule],
    term_mapping: dict[str, str],
    crud_endpoint: str,
    chunk_size: int = 50,
    use_searchspace: bool = True,
    use_head: bool = True,
) -> None:
    """Completes a graph using only the given rules assuming they are all complete."""

    graph_metrics: GraphMetrics = GraphMetrics.from_uri(graph_uri)
    profiles = graph_metrics.profiles
    grounded_predicates = set(profiles.keys())
    rule_output: dict[str, int] = {r_id: 0 for r_id in rules.keys()}

    def is_ready(rule_id: str) -> bool:
        """Evaluates if a rule should be included in the iteration. A rule is considered
        ready when its body is grounded, its support is not closed, and it does not
        depend on other rules."""

        rule = rules[rule_id]
        head_predicate = rule.head.predicate

        body_predicates = rule.get_body_predicates() - {head_predicate}
        if not body_predicates.issubset(grounded_predicates):
            return False

        return True

    iter = 0
    prev_size = 0
    while True:
        iter += 1
        graph_size = get_total_triples(client, graph_uri)
        if graph_size > prev_size:
            prev_size = graph_size
        else:
            logger.info("No new triples generated.")
            break

        available_rules = {r_id: r for r_id, r in rules.items() if is_ready(r_id)}
        if not available_rules:
            logger.info("No rules to apply.")
            break

        logger.info("Iter %d: Applying %d rules.", iter, len(available_rules))

        direct_rules: list[str] = []
        recursive_rules: list[str] = []

        for r_id, r in available_rules.items():
            if r.head.predicate in r.get_body_predicates():
                recursive_rules.append(r_id)
            else:
                direct_rules.append(r_id)

        # Apply direct rules first, then recursive rules
        for r_id in itertools.chain(direct_rules, recursive_rules):
            rule = rules[r_id]
            predicate = rule.head.predicate

            raw_bindings = apply_rule(
                client=client,
                graph_uri=graph_uri,
                rule=rule,
                profile=profiles[predicate],
                term_mapping=term_mapping,
                crud_endpoint=crud_endpoint,
                use_searchspace=use_searchspace,
                use_head=use_head,
            )

            if not raw_bindings:
                logger.warning("No results found for %s", r_id)
                continue
            if len(raw_bindings) <= rule_output[r_id]:
                logger.debug("No now bindings from rule %s.", r_id)
            else:
                rule_output[r_id] = len(raw_bindings)
                logger.debug("Retrieved %d bindings for %s.", len(raw_bindings), r_id)

                filtered_triple_iterator = filter_and_format_triples(
                    raw_bindings=raw_bindings,
                    profile=profiles[predicate],
                    use_profile=False,
                    rule=rule,
                    term_mapping=term_mapping,
                )

                count = insert_triples_sparql(
                    graph_uri=graph_uri,
                    client=client,
                    triple_stream=filtered_triple_iterator,
                    chunk_size=chunk_size,
                )
                logger.debug("Counted filtered triples: %d", count)
                if count:
                    grounded_predicates.add(predicate)
                    logger.info(
                        "Rule %s yielded %d triples for %s.", r_id, count, predicate
                    )


# ---------------------------------------------------------------------------
# Main funtion.
# ---------------------------------------------------------------------------
def create_synthetic_graph(
    client: SPARQLWrapper,
    rules: dict[str, HornRule],
    term_mapping: dict[str, str],
    graph_metrics: GraphMetrics,
    graph_uri: str,
    crud_endpoint: str,
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
    intensional_preds = {rule.head.predicate for rule in rules.values()}
    extensional_preds = graph_metrics.profiles.keys() - intensional_preds
    if not extensional_preds:
        error_msg = "Cannot generate synthetic graph: Found 0 extensional predicates."
        logger.error(error_msg)
        raise SyntheticGraphError(error_msg)

    logger.info(
        "%d out of %d predicates are extensional.",
        len(extensional_preds),
        len(graph_metrics.profiles),
    )
    logger.debug(
        "\nIntensional preds.:\n\t%s\nExtensional preds.:\n\t%s\nProfiles:\n\t%s",
        "\n\t".join(intensional_preds),
        "\n\t".join(extensional_preds),
        "\n\t".join(graph_metrics.profiles.keys()),
    )

    # Clean the space where the new Synthetic graph is going to be stored
    clear_graph_sparql(graph_uri=graph_uri, client=client)

    # Generate the graph initially as the extensional database (EDB)
    edb_profiles = {k: graph_metrics.profiles[k] for k in extensional_preds}
    gen_graph(
        client=client,
        predicate_profiles=edb_profiles,
        graph_uri=graph_uri,
        term_mapping=term_mapping,
    )

    # Stratify and apply rules iteratively
    _execute_stratification(
        client=client,
        graph_uri=graph_uri,
        rules=rules,
        profiles=graph_metrics.profiles,
        term_mapping=term_mapping,
        crud_endpoint=crud_endpoint,
    )
    logger.info("Execution finished after %.2f s.", time.time() - start_time)

    original_count = graph_metrics.triple_count
    count = get_total_triples(client, graph_uri)

    logger.info("Original graph has %d triples.", original_count)
    logger.info("Synthetic graph at <%s> has %d triples.", graph_uri, count)


def synthetic_graph_experiment() -> None:
    ## -------------------------------- Setup ------------------------------------------
    config_file = Path("configurations/gen_triples/french_royalty.json")
    # config_file = Path("configurations/gen_triples/simpsons.json")

    config = RunConfig.from_json(config_file)
    setup_logging(level=config.logging.level)
    logger.info("Confifuration correctly initialized.")

    client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    client.setHTTPAuth(DIGEST)
    client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    graph_metrics = GraphMetrics.from_uri(client, config.graph.uri)

    ## -------------- Previous evaluation of rules and Graph Metrics -------------------
    rule_dataframe = pd.read_csv(config.data.input_dir / config.rules.rules_file)

    term_mapping = get_term_mapping(config.data.input_dir / config.graph.ontology_file)

    rules = parse_rule_set(
        rule_dataframe=rule_dataframe,
        term_mapping=term_mapping,
        pca_threshold=config.rules.pca_threshold,
        default_namespace=config.graph.uri,
    )

    intensional_predicates = {rule.head.predicate for rule in rules.values()}
    extensional_predicates = graph_metrics.profiles.keys() - intensional_predicates

    if uninferrable_preds := check_uninferrable_preds(
        rules=rules,
        intensional_predicates=intensional_predicates,
        extensional_predicates=extensional_predicates,
    ):
        error_msg = (
            f"Rule set not inferrable under complete rule assumption. "
            f"The following predicates cannot be deduced: {uninferrable_preds}."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    ## ----------------------- Synthetic Graph Generation  -----------------------------
    try:
        create_synthetic_graph(
            client=client,
            rules=rules,
            term_mapping=term_mapping,
            graph_metrics=graph_metrics,
            crud_endpoint=URL(client.endpoint).with_name("sparql-graph-crud-auth"),
            graph_uri=f"http://Synthetic{config.graph.name}.org/",
        )
    except SyntheticGraphError as e:
        logger.info("Generation error: %s", e)


if __name__ == "__main__":
    synthetic_graph_experiment()
    # ## -------------------------------- Setup ------------------------------------------
    # config_file = Path("configurations/complete_graph/french_royalty.json")
    # # config_file = Path("configurations/gen_triples/simpsons.json")

    # config = RunConfig.from_json(config_file)
    # setup_logging(level=config.logging.level)
    # logger.info("Confifuration correctly initialized.")

    # client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    # client.setHTTPAuth(DIGEST)
    # client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    # graph_metrics = GraphMetrics.from_uri(client, config.graph.uri)

    # ## -------------- Previous evaluation of rules and Graph Metrics -------------------
    # rule_dataframe = pd.read_csv(config.data.input_dir / config.rules.rules_file)

    # term_mapping = get_term_mapping(config.data.input_dir / config.graph.ontology_file)

    # rules = parse_rule_set(
    #     rule_dataframe=rule_dataframe,
    #     term_mapping=term_mapping,
    #     pca_threshold=config.rules.pca_threshold,
    #     default_namespace=config.graph.uri,
    # )

    # intensional_predicates = {rule.head.predicate for rule in rules.values()}
    # extensional_predicates = graph_metrics.profiles.keys() - intensional_predicates

    # if uninferrable_preds := check_uninferrable_preds(
    #     rules=rules,
    #     intensional_predicates=intensional_predicates,
    #     extensional_predicates=extensional_predicates,
    # ):
    #     error_msg = (
    #         f"Rule set not inferrable under complete rule assumption. "
    #         f"The following predicates cannot be deduced: {uninferrable_preds}."
    #     )
    #     logger.error(error_msg)
    #     raise RuntimeError(error_msg)

    # try:
    #     graph_uri = "http://CompleteFR.org/"

    #     original_count = get_total_triples(client, graph_uri)
    #     complete_graph(
    #         client=client,
    #         term_mapping=term_mapping,
    #         rules=rules,
    #         profiles=graph_metrics.profiles,
    #         graph_uri=graph_uri,
    #         crud_endpoint=URL(client.endpoint).with_name("sparql-graph-crud-auth"),
    #         use_searchspace=False,
    #         use_head=False,
    #     )
    #     count = get_total_triples(client, graph_uri)

    #     logger.info("Original graph has %d triples.", original_count)
    #     logger.info("Complete graph at <%s> has %d triples.", graph_uri, count)

    # except SyntheticGraphError as e:
    #     logger.error("%s", e)
