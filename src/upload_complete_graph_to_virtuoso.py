"""Uploads a base graph from an .nt file. Then it creates a complete graph from the
set of rules."""

import logging
from pathlib import Path

from SPARQLWrapper import DIGEST, SPARQLWrapper

from config import RunConfig
from graph_completion import complete_graph
from queries import count_triples, initialize_graph
from rules import get_term_mapping, parse_rule_set
from utils import setup_logging

### EDIT THIS PATH   vvv
graph_config = Path("configurations/mario.json")
config = RunConfig.from_json(graph_config)

setup_logging(level=config.logging.level)
logger = logging.getLogger(__name__)

input_dir = config.data.input_dir
base_uri = config.graph.base_uri
complete_uri = config.graph.complete_uri

term_mapping = get_term_mapping(
    ontology_file=input_dir / config.graph.ontology_file,
    default_namespace=config.graph.namespace,
)

rules = parse_rule_set(
    rules_file=input_dir / config.rules.rules_file,
    term_mapping=term_mapping,
    pca_threshold=config.rules.pca_threshold,
)
# TODO: Fix rule parsing
rules = list(rules.values())
logger.debug("Parsed %d rules.", len(rules))


client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
client.setHTTPAuth(DIGEST)
client.setCredentials(config.virtuoso.user, config.virtuoso.password)

# Initialize base graph
initialize_graph(
    client=client,
    source=str(input_dir / config.graph.nt_file),
    new_graph_uri=base_uri,
    chunk_size=1000,
)

logger.info("Starting Graph Completion")

# Complete graph
complete_graph(
    client=client,
    rules=rules,
    term_mapping=term_mapping,
    base_uri=base_uri,
    complete_uri=complete_uri,
    chunk_size=config.virtuoso.chunk_size,
)

logger.info(
    "Complete graph in <%s> has %d triples.",
    complete_uri,
    count_triples(client, complete_uri),
)
