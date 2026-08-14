import logging
from pathlib import Path

from rules import HornRule
from SPARQLWrapper import SPARQLWrapper

from kg_synth.core.queries import count_triples, initialize_graph
from kg_synth.engine.generator import apply_rules
from kg_synth.engine.metrics import GraphMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph completion.
# ---------------------------------------------------------------------------
def complete_graph(
    client: SPARQLWrapper,
    rules: dict[str, HornRule],
    term_mapping: dict[str, str],
    chunk_size: int,
    complete_graph_uri: str,
) -> None:
    """Completes a graph using only the given rules assuming they are all complete."""

    graph_metrics: GraphMetrics = GraphMetrics.from_uri(client, complete_graph_uri)
    grounded_preds = graph_metrics.profiles.keys()

    def is_ready(r_id: str) -> bool:
        """Returns True if a rule should be included in the iteration. A rule is
        included when its body is grounded."""
        rule = rules[r_id]
        body_preds = rule.get_body_predicates() - {rule.head.predicate}

        return True if body_preds.issubset(grounded_preds) else False

    iter = 0
    prev_size = count_triples(client, complete_graph_uri)
    while True:
        iter += 1
        available_rules = {r_id: r for r_id, r in rules.items() if is_ready(r_id)}
        if not available_rules:
            logger.info("No rules to apply.")
            break

        logger.info("--- Iter %d ---", iter)

        apply_rules(
            client=client,
            graph_uri=complete_graph_uri,
            rules=available_rules,
            use_head=False,
            term_mapping=term_mapping,
            chunk_size=chunk_size,
        )

        graph_size = count_triples(client, complete_graph_uri)

        if new_triples := graph_size - prev_size:
            logger.info("Added %d triples.", new_triples)
            prev_size = graph_size
        else:
            logger.info("No triples produced in this iteration.")
            break


def store_comlete_graph(
    client: SPARQLWrapper,
    rules: dict[str, HornRule],
    source: str,
    complete_graph_uri: str,
) -> None:
    """Runs a graph completion experiment"""
    if source == complete_graph_uri:
        raise ValueError("Source and output URI cannot be the same.")

    initialize_graph(
        client=client, source=source, new_graph_uri=complete_graph_uri, chunk_size=1000
    )
    source_count = count_triples(client, complete_graph_uri)

    start_time = time.time()

    complete_graph(
        client=client,
        rules=rules,
        term_mapping=term_mapping,
        chunk_size=config.virtuoso.chunk_size,
        complete_graph_uri=complete_graph_uri,
    )

    final_time = time.time() - start_time

    new_count = count_triples(client, complete_graph_uri)

    logger.info("Original graph has %d triples.", source_count)
    logger.info("Complete graph at <%s> has %d triples.", complete_graph_uri, new_count)
    logger.info("Execution finished in %d s.", final_time)


if __name__ == "__main__":
    import time

    simpsons_config = Path("configurations/simpsons.json")
    fr_config = Path("configurations/french_royalty.json")
    run_graph_completion_experimnent(
        config_file=simpsons_config,
        source=".data/Simpsons/simpsons.nt",
        complete_graph_uri="http://SimpsonFamily-Complete.org/",
    )
