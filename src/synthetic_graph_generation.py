import logging
import time
from pathlib import Path

from SPARQLWrapper import SPARQLWrapper

from graph_metrics import GraphMetrics, PredicateProfile
from queries import (
    clear_graph_sparql,
    count_triples,
    get_frequency,
    get_support,
    initialize_from_source,
)
from rules import (
    HornRule,
    get_dependencies_intensional,
    get_predicate_mapping,
)
from triple_generation import (
    apply_rules,
    create_extensional_graph,
)

logger = logging.getLogger(__name__)


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
        if support:
            if support >= rule.support:
                closed_rules.add(r_id)

    return closed_rules


def get_closed_preds(
    client: SPARQLWrapper, graph_uri: str, profiles: dict[str, PredicateProfile]
) -> set[str]:
    """Queries the database to find which predicates reached their target frequency.

    A predicate is considered 'closed' when the count of distinct triples containing the
    predicate is equal to the predicate's frequency.
    """

    closed_predicates: set[str] = set()

    for predicate, profile in profiles.items():
        frequency = get_frequency(client, predicate, graph_uri)
        if frequency >= profile.frequency:
            closed_predicates.add(predicate)

    return closed_predicates


def update_closure(
    client: SPARQLWrapper,
    graph_uri: str,
    profiles_to_check: dict[str, PredicateProfile],
    rules_to_check: dict[str, HornRule],
    closed_rule_ids: set[str],
    closed_preds: set[str],
) -> bool:
    """Updates the sets that track closed rules and predicates. Returns True if elements
    are added to any set.

    Args:
        client: Wrapper for SPARQL queries.
        graph_uri: URI where closure is measured.
        profiles_to_check: Predicates to check for closure.
        rules_to_check: Rules to check for closure.

    Returns:
        True if any rules or predicates are closed, False otherwise.
    """

    update = False

    if new_preds := get_closed_preds(client, graph_uri, profiles_to_check):
        logger.debug("Closed predicates %s", new_preds)
        closed_preds.update(new_preds)
        update = True

    if closed_rules := get_closed_rules(client, graph_uri, rules_to_check):
        logger.debug("Closed rules %s", closed_rules)
        closed_rule_ids.update(closed_rules)
        update = True

    return update


# ---------------------------------------------------------------------------
# Synthetic Graph Generation.
# ---------------------------------------------------------------------------
def create_synthetic_graph(
    client: SPARQLWrapper,
    rules: dict[str, HornRule],
    term_mapping: dict[str, str],
    profiles: dict[str, PredicateProfile],
    source: str,
    synthetic_graph_uri: str,
    chunk_size: int,
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

    initialize_from_source(
        source=source,
        new_graph_uri=synthetic_graph_uri,
        client=client,
        chunk_size=chunk_size,
    )

    intensional_preds = {rule.head.predicate for rule in rules.values()}
    extensional_preds = profiles.keys() - intensional_preds

    if not extensional_preds:
        error_msg = "Can't generate synthetic graph with 0 extensional predicates."
        raise ValueError(error_msg)

    logger.info(
        "%d out of %d predicates are extensional.",
        len(extensional_preds),
        len(profiles),
    )
    logger.debug(
        "\nIntensional preds.:\n\t%s\nExtensional preds.:\n\t%s\nProfiles:\n\t%s",
        "\n\t".join(intensional_preds),
        "\n\t".join(extensional_preds),
        "\n\t".join(profiles.keys()),
    )

    # Clean the space where the new Synthetic graph is going to be stored
    clear_graph_sparql(graph_uri=synthetic_graph_uri, client=client)

    # Generate the graph initially as the extensional database (EDB)
    edb_profiles = {k: profiles[k] for k in extensional_preds}
    start_time = time.time()
    create_extensional_graph(
        client=client,
        rules=rules,
        profiles=edb_profiles,
        edb_uri=synthetic_graph_uri,
        term_mapping=term_mapping,
        chunk_size=chunk_size,
    )
    logger.info(
        "Finished EDB generation after %f s at <%s> with %d triples",
        time.time() - start_time,
        synthetic_graph_uri,
        count_triples(client=client, graph_uri=synthetic_graph_uri),
    )

    intensional_dependencies = get_dependencies_intensional(rules=rules)

    closed_rule_ids: set[str] = set()
    closed_preds: set[str] = set()
    grounded_predicates = set(extensional_preds)

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

        for r_id in intensional_dependencies.get(rule_id, []):
            if r_id not in closed_rule_ids:
                return False

        return True

    # Stratify and apply rules iteratively
    predicate_to_rules = get_predicate_mapping(rules)

    logger.info("Generating synthetic graph...")
    iter = 0
    prev_size = count_triples(client, synthetic_graph_uri)
    while True:
        iter += 1
        available_rules = {r_id: r for r_id, r in rules.items() if is_ready(r_id)}
        if not available_rules:
            logger.info("No rules to apply.")
            break

        logger.info("--- Iter %d ---", iter)

        applied_rules = apply_rules(
            client=client,
            graph_uri=synthetic_graph_uri,
            rules=available_rules,
            use_head=True,
            term_mapping=term_mapping,
            chunk_size=chunk_size,
            profiles=profiles,
        )

        if not applied_rules:
            logger.info("No rules applied in this itreation.")
            break

        graph_size = count_triples(client, synthetic_graph_uri)

        if not (new_triples := graph_size - prev_size):
            logger.info("No triples produced in this iteration.")
            break

        logger.info("Added %d triples.", new_triples)
        prev_size = graph_size

        # Determine which rules should be checked for closure
        affected_rule_ids: set[str] = set()
        for predicate in {r.head.predicate for r in applied_rules.values()}:
            affected_rule_ids.update(predicate_to_rules[predicate])

        if pending_rule_ids := set(affected_rule_ids) - closed_rule_ids:
            rules_to_check = {r_id: rules[r_id] for r_id in pending_rule_ids}

        if update_closure(
            client=client,
            graph_uri=synthetic_graph_uri,
            profiles_to_check=profiles,
            rules_to_check=rules_to_check,
            closed_preds=closed_preds,
            closed_rule_ids=closed_rule_ids,
        ):
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


def run_synthetic_graph_experiment(config_file: Path) -> None:
    """Runs a Synthetic Graph generation experiment."""
    import time

    import pandas as pd
    from SPARQLWrapper import DIGEST

    from config import RunConfig
    from rules import check_uninferrable_preds, get_term_mapping, parse_rule_set
    from utils import setup_logging

    ## ------ Setup ------
    config = RunConfig.from_json(config_file)
    setup_logging(level=config.logging.level)
    logger.info("Confifuration correctly initialized.")

    input_dir = config.data.input_dir
    rules_file = input_dir / config.rules.rules_file

    client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    client.setHTTPAuth(DIGEST)
    client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    graph_metrics = GraphMetrics.from_uri(client, config.graph.base_graph_uri)

    ## ------ Previous evaluation of rules and Graph Metrics ------
    rule_dataframe = pd.read_csv(rules_file)

    term_mapping = get_term_mapping(
        ontology_file=input_dir / config.graph.ontology_file,
        default_namespace=config.graph.base_graph_uri,
    )

    rules = parse_rule_set(
        rule_dataframe=rule_dataframe,
        term_mapping=term_mapping,
        pca_threshold=config.rules.pca_threshold,
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

    ## ------ Synthetic Graph Generation  ------
    start_time = time.time()
    synthetic_graph_uri = config.graph.synthetic_graph_uri

    create_synthetic_graph(
        client=client,
        rules=rules,
        term_mapping=term_mapping,
        profiles=graph_metrics.profiles,
        source=config.graph.base_graph_uri,
        synthetic_graph_uri=synthetic_graph_uri,
        chunk_size=config.virtuoso.chunk_size,
    )

    original_count = graph_metrics.triple_count
    count = count_triples(client, synthetic_graph_uri)

    logger.info("Original graph has %d triples.", original_count)
    logger.info("Synthetic graph at <%s> has %d triples.", synthetic_graph_uri, count)
    logger.info("Execution finished after %d s.", time.time() - start_time)


if __name__ == "__main__":
    simpsons_config = Path("configurations/simpsons.json")
    fr_config = Path("configurations/french_royalty.json")
    run_synthetic_graph_experiment(fr_config)
