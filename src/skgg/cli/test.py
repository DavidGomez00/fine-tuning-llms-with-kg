"""Test executions"""

import logging
from pathlib import Path

from skgg.config import RunConfig
from skgg.core.queries import get_support
from skgg.core.rules import parse_rule_set
from skgg.utils import create_sparql_client, get_term_mapping, setup_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    mario_config = Path("configurations/mario.json")

    ## ------ Setup ------
    config = RunConfig.from_json(mario_config)
    setup_logging(level=config.logging.level)

    logger.info("Confifuration correctly initialized.")

    input_dir = config.data.input_dir
    rules_file = input_dir / config.rules.rules_file

    # SPARQL client
    client = create_sparql_client(config)

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

    for r_id, rule in rules.items():
        logger.info(
            "%s has support %d",
            r_id,
            get_support(client, rule, "http://SuperMario.org/graph/complete"),
        )
