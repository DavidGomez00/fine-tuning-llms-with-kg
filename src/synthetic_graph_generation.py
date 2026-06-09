import logging

from SPARQLWrapper import SPARQLWrapper

from graph_metrics import GraphMetrics, PredicateProfile
from queries import (
    count_triples,
    get_frequency,
    get_support,
    initialize_from_source,
)
from rules import (
    HornRule,
    check_uninferrable_preds,
    get_dependencies_intensional,
    get_predicate_mapping,
)
from triple_generation import (
    apply_rules,
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


def generate_idb(
    client: SPARQLWrapper,
    rules: dict[str, HornRule],
    term_mapping: dict[str, str],
    source: str,
    synthetic_uri: str,
    chunk_size: int,
) -> None:
    """"""

    initialize_from_source(
        source=source,
        new_graph_uri=synthetic_uri,
        client=client,
        chunk_size=chunk_size,
    )

    graph_metrics = GraphMetrics.from_uri(client, synthetic_uri)
    profiles = graph_metrics.profiles

    intensional_preds = {rule.head.predicate for rule in rules.values()}
    extensional_preds = profiles.keys() - intensional_preds

    if uninferrable_preds := check_uninferrable_preds(
        rules=rules,
        intensional_predicates=intensional_preds,
        extensional_predicates=extensional_preds,
    ):
        error_msg = (
            f"Rule set not inferrable under complete rule assumption. "
            f"The following predicates cannot be deduced: {uninferrable_preds}."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

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
    prev_size = count_triples(client, synthetic_uri)
    while True:
        iter += 1
        available_rules = {r_id: r for r_id, r in rules.items() if is_ready(r_id)}
        if not available_rules:
            logger.info("No rules to apply.")
            break

        logger.info("--- Iter %d ---", iter)

        applied_rules = apply_rules(
            client=client,
            graph_uri=synthetic_uri,
            rules=available_rules,
            use_head=True,
            term_mapping=term_mapping,
            chunk_size=chunk_size,
            profiles=profiles,
        )

        if not applied_rules:
            logger.info("No rules applied in this itreation.")
            break

        graph_size = count_triples(client, synthetic_uri)

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
            graph_uri=synthetic_uri,
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
