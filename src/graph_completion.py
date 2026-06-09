import logging
from pathlib import Path

from SPARQLWrapper import SPARQLWrapper

from graph_metrics import GraphMetrics
from queries import count_triples, initialize_from_source
from rules import HornRule
from triple_generation import apply_rules

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


def run_graph_completion_experimnent(
    config_file: Path,
    source: str,
    complete_graph_uri: str,
) -> None:
    """Runs a graph completion experiment"""
    import time

    import pandas as pd
    from SPARQLWrapper import DIGEST

    from config import RunConfig
    from rules import get_term_mapping, parse_rule_set
    from utils import setup_logging

    ## ------ Setup ------
    config = RunConfig.from_json(config_file)
    setup_logging(level=config.logging.level)
    logger.info("Confifuration correctly initialized.")

    client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    client.setHTTPAuth(DIGEST)
    client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    input_dir = config.data.input_dir
    rules_file = config.rules.rules_file

    ## ------ Previous evaluation of rules and Graph Metrics ------
    initialize_from_source(
        client=client, source=source, new_graph_uri=complete_graph_uri, chunk_size=1000
    )
    source_count = count_triples(client, complete_graph_uri)

    rule_dataframe = pd.read_csv(input_dir / rules_file)

    term_mapping = get_term_mapping(
        ontology_file=input_dir / config.graph.ontology_file,
        default_namespace=complete_graph_uri,
    )

    rules = parse_rule_set(
        rule_dataframe=rule_dataframe,
        term_mapping=term_mapping,
        pca_threshold=config.rules.pca_threshold,
    )

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
    simpsons_config = Path("configurations/simpsons.json")
    fr_config = Path("configurations/french_royalty.json")
    run_graph_completion_experimnent(
        fr_config,
        source=".data/FrenchRoyalty/french_royalty.nt",
        complete_graph_uri="http://FrenchRoyalty.org/",
    )
