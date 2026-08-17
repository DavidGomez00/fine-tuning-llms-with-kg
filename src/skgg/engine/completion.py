import logging

from SPARQLWrapper import SPARQLWrapper

from skgg.core.queries import initialize_graph
from skgg.core.rules import HornRule
from skgg.engine.generator import apply_rule
from skgg.engine.metrics import GraphMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph completion.
# ---------------------------------------------------------------------------
def complete_graph(
    client: SPARQLWrapper,
    rules: list[HornRule],
    term_mapping: dict[str, str],
    base_uri: str,
    complete_uri: str,
    chunk_size: int,
) -> None:
    """Completes a graph using only the given rules assuming they are all complete."""

    # Initialize complete graph from base URI
    initialize_graph(
        client=client,
        source=base_uri,
        new_graph_uri=complete_uri,
        chunk_size=chunk_size,
    )

    # Get the initial grounded preds
    graph_metrics = GraphMetrics.from_uri(client, complete_uri)
    grounded_preds = set(graph_metrics.profiles.keys())

    def is_ready(rule: HornRule) -> bool:
        """Returns True if the body from 'rule' is grounded."""
        body_preds = rule.get_body_predicates()
        return True if body_preds.issubset(grounded_preds) else False

    state = dict()
    for rule in rules:
        state[rule.rule_id] = 0

    step = 0
    while True:
        step += 1
        available_rules = [rule for rule in rules if is_ready(rule)]
        if not available_rules:
            logger.info("[Step %d]: No rules to apply. Completion ended.", step)
            break

        added = 0
        for rule in available_rules:
            count = apply_rule(
                client=client,
                graph_uri=complete_uri,
                rule=rule,
                term_mapping=term_mapping,
                chunk_size=chunk_size,
            )

            logger.debug("%s added %d triples.", rule.rule_id, count)
            state[rule.rule_id] += count
            added += count
            if count:
                grounded_preds.add(rule.head.predicate)

        if not added:
            logger.info(
                "[Step %d]: No more triples to add. Graph completion ended.", step
            )
            break

        state_msg = "\n".join(
            [
                f"\t{rule.rule_id}: {state[rule.rule_id]}"
                for rule in rules
                if state[rule.rule_id] > 0
            ]
        )
        logger.info("[Step %d]: Added triples\n%s", step, state_msg)
