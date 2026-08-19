"""End-to-end experiment entry point: metrics → EDB → IDB → synthetic graph.

See `run_synthetic_graph_experiment` and AGENTS.md's "Running an experiment"
section for the full pipeline description.
"""

import logging
import time
from pathlib import Path

from skgg.config import RunConfig
from skgg.core.queries import get_triple_count
from skgg.core.rules import parse_rule_set
from skgg.engine.edb import generate_edb
from skgg.engine.idb import generate_idb
from skgg.engine.metrics import GraphMetrics
from skgg.utils import create_sparql_client, get_term_mapping, setup_logging

logger = logging.getLogger(__name__)


def run_synthetic_graph_experiment(
    config_file: Path,
    source: str | None = None,
) -> None:
    """Runs a Synthetic Graph generation experiment."""

    ## ------ Setup ------
    config = RunConfig.from_json(config_file)
    setup_logging(level=config.logging.level)
    logger.info("Confifuration correctly initialized.")

    input_dir = config.data.input_dir
    rules_file = input_dir / config.rules.rules_file

    # SPARQL client
    client = create_sparql_client(config)

    ## ------ Extraction of predicate profiles from original graph -------
    # Graph metrics
    graph_metrics = GraphMetrics.from_uri(client, config.graph.complete_uri)
    profiles = graph_metrics.profiles

    ## ------ Previous evaluation of rules ------
    term_mapping = get_term_mapping(
        ontology_file=input_dir / config.graph.ontology_file,
        default_namespace=config.graph.namespace,
    )

    rules = parse_rule_set(
        rules_file=rules_file,
        term_mapping=term_mapping,
        pca_threshold=config.rules.pca_threshold,
    )

    ## ------ EDB Generation  ------
    chunk_size = config.db_config.chunk_size
    edb_uri = config.graph.edb_uri
    synthetic_uri = config.graph.synthetic_uri
    if source is None:
        source = config.graph.complete_uri

    logger.info("Generating EDB...")
    start_time = time.time()

    generate_edb(
        client=client,
        term_mapping=term_mapping,
        rules=rules,
        edb_uri=edb_uri,
        chunk_size=chunk_size,
        profiles=graph_metrics.profiles,
    )

    edb_time = time.time() - start_time

    logger.info(
        "Finished EDB generation after %f s at <%s> with %d triples",
        edb_time,
        edb_uri,
        get_triple_count(client, edb_uri),
    )

    generate_idb(
        client=client,
        rules=rules,
        term_mapping=term_mapping,
        edb_uri=edb_uri,
        synthetic_uri=synthetic_uri,
        chunk_size=chunk_size,
        profiles=profiles,
    )

    original_count = get_triple_count(client, source)
    count = get_triple_count(client, synthetic_uri)

    logger.info("Original graph has %d triples.", original_count)
    logger.info("Synthetic graph at <%s> has %d triples.", synthetic_uri, count)
    logger.info("Execution finished after %d s.", time.time() - start_time)


if __name__ == "__main__":
    mario_config = Path("configurations/mario.json")
    fr_config = Path("configurations/french_royalty.json")
    run_synthetic_graph_experiment(mario_config)
