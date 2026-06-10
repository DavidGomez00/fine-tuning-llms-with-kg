import logging
import time
from pathlib import Path

from SPARQLWrapper import DIGEST, SPARQLWrapper

from config import RunConfig
from idb_generation import generate_edb, generate_idb
from queries import count_triples
from rules import get_term_mapping, parse_rule_set
from utils import setup_logging

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

    client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    client.setHTTPAuth(DIGEST)
    client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    ## ------ Previous evaluation of rules ------
    term_mapping = get_term_mapping(
        ontology_file=input_dir / config.graph.ontology_file,
        default_namespace=config.graph.base_graph_uri,
    )

    rules = parse_rule_set(
        rules_file=rules_file,
        term_mapping=term_mapping,
        pca_threshold=config.rules.pca_threshold,
    )

    ## ------ EDB Generation  ------
    chunk_size = config.virtuoso.chunk_size
    edb_uri = config.graph.edb_uri
    synthetic_uri = config.graph.synthetic_uri
    if source is None:
        source = config.graph.base_graph_uri

    logger.info("Generating EDB...")
    start_time = time.time()
    generate_edb(
        client=client,
        term_mapping=term_mapping,
        rules=rules,
        source=source,
        edb_uri=edb_uri,
        chunk_size=chunk_size,
    )

    edb_time = time.time() - start_time

    logger.info(
        "Finished EDB generation after %f s at <%s> with %d triples",
        edb_time,
        edb_uri,
        count_triples(client, edb_uri),
    )

    generate_idb(
        client=client,
        rules=rules,
        term_mapping=term_mapping,
        source=source,
        synthetic_uri=synthetic_uri,
        chunk_size=chunk_size,
    )

    original_count = count_triples(client, source)
    count = count_triples(client, synthetic_uri)

    logger.info("Original graph has %d triples.", original_count)
    logger.info("Synthetic graph at <%s> has %d triples.", synthetic_uri, count)
    logger.info("Execution finished after %d s.", time.time() - start_time)


if __name__ == "__main__":
    simpsons_config = Path("configurations/simpsons.json")
    fr_config = Path("configurations/french_royalty.json")
    run_synthetic_graph_experiment(fr_config, source="http://FrenchRoyaltyEDB.org/")
