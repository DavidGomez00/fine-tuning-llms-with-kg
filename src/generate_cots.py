"""This file defines methods to generate natural language Chain of Thoughts (CoTs) from
the instances of a rule present in a Knowledge Graph (KG)."""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import pandas as pd
from rdflib import Graph

from config import KGConfig, RunConfig
from knowledge_graphs import RuleRow, load_knowledge_graph
from rules import build_sparql_query, parse_horn_rule, query_result_to_natural_language

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configures the root logger to output to the console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Generate CoTs
# ---------------------------------------------------------------------------


def _create_summary(
    kg_config: KGConfig,
    output_dir: Path,
    graph_length: int,
    rules_df: pd.DataFrame,
    time: float,
) -> None:
    """Writes a summary report of the CoT generation process."""

    pca_series = rules_df["PCA_Confidence"]
    valid_pca = pca_series.dropna()
    neg = (valid_pca < kg_config.pca_threshold).sum()
    pos = (valid_pca >= kg_config.pca_threshold).sum()
    nan = pca_series.isna().sum()

    summary_content = (
        f"{kg_config.kg_name.upper()} CoT2 RULES — SUMMARY\n"
        f"\tTotal Rules        : {len(rules_df)}\n"
        f"\tOutput Directory   : {output_dir.absolute()}\n"
        f"\tKG Triples         : {graph_length}\n"
        f"\tPCA Threshold      : {kg_config.pca_threshold}\n"
        f"\tNamespace          : {kg_config.namespace}\n"
        f"\tNamespace Prefix   : {kg_config.namespace_prefix}\n\n"
        f"\tPositive (>= {kg_config.pca_threshold})  : {pos}\n"
        f"\tNegative (<  {kg_config.pca_threshold})  : {neg}\n"
        f"\tMissing PCA        : {nan}\n\n"
        f"\tTotal time: {time:.2f} s"
    )
    output_file = output_dir / "summary.txt"
    output_file.write_text(summary_content, encoding="utf-8")
    logger.debug(f"Summary saved to: {output_file.absolute()}")


def generate_cots_sparql(
    kg_config: KGConfig,
    graph: Graph,
    rules_df: pd.DataFrame,
    output_dir: Path,
    max_rules: int | None = None,
    max_groundings: int | None = None,
    save_summary: bool = False,
) -> None:
    """Generate natural language CoTs from a set of rules.

    A set of queries is generated to retreve the groundings of each rule from the graph.
    Then, the rules and groundings are used to generate natural language CoTs to answer
    a yes or no question.
    The answer of the question for each grounded body depends on the existance of the
    grounded head in the graph. If the head exists, the answer is yes, otherwise is no.

    Args:
        kg_config: Configuration object with information about the KG files used.
        graph: Graph from where the groundings are retrieved.
        rules_df: DataFrame containing the set of rules used to generate CoTs.
        output_dir: Output directory where CoT files are saved.
        max_rules: Limit of rules used to generate CoTs.
        max_groundings: Limit of groundings per rule used to generate CoTs.
        save_summary: Wether to create a summary file of the generation process.
    """

    # Make sure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cap the number of rules
    # TODO: possible problem when sorting rules in the DF!
    if max_rules is not None:
        rules_df = rules_df.head(max_rules)

    # Start cot generation
    _header = "\nNL-instances-CoT2  (SPARQL-based)"
    text = (
        f"{_header}\n"
        f"\tKG file          : {kg_config.kg_file}\n"
        f"\tRules CSV        : {kg_config.rules_csv}\n"
        f"\tNamespace        : {kg_config.namespace}\n"
        f"\tNamespace prefix : {kg_config.namespace_prefix}\n"
        f"\tPCA threshold    : {kg_config.pca_threshold}\n"
    )
    logger.info(text)

    # For each rule, generate a query
    query_mapping: defaultdict[str, str] = defaultdict()
    logger.info("Generating queries for %d rules.", len(rules_df))

    start_time = time.time()
    for rule_idx, row in enumerate(rules_df.itertuples(index=False), start=1):
        rule_id = str(rule_idx)
        typed_row = cast(RuleRow, row)
        rule = parse_horn_rule(
            row=typed_row, rule_id=rule_id, pca_threshold=kg_config.pca_threshold
        )

        query = build_sparql_query(
            rule_signature=rule.signature,
            ns_prefix=kg_config.namespace_prefix,
            namespace=kg_config.namespace,
        )
        query_mapping[rule_id] = query

    logger.info(
        "Executing %d queries and generating a text description of the results.",
        len(query_mapping),
    )

    for rule_id, query in query_mapping.items():
        # Query the graph
        result = graph.query(query)

        # Generate rule + groundings natural language description
        grounding_description = query_result_to_natural_language(
            kg_config=kg_config,
            result=result,
            rule=rule,
            max_groundings=max_groundings,
        )

        # Save the rule + groundings natural language description as a file
        rule_file_path = output_dir / f"rule_{rule_id}.txt"
        rule_file_path.write_text(grounding_description, encoding="utf-8")
        logger.debug("Saving description of %s in %s", rule_id, rule_file_path.name)

    total_time = time.time() - start_time

    logger.info(
        "Succesfully created %d query result descriptions in %f seconfs.",
        len(query_mapping),
        total_time,
    )

    if save_summary:
        _create_summary(
            kg_config=kg_config,
            output_dir=output_dir,
            graph_length=len(graph),
            rules_df=rules_df,
            time=total_time,
        )


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Set up logger
    setup_logging()

    # Load configuration
    configuration_json = Path("configurations/gen_fr_cots.json")
    config = RunConfig.from_json(configuration_json)

    # Load rules DataFrame
    rules_df = pd.read_csv(config.data.input_dir / config.kg.rules_csv)
    max_rules = len(rules_df)
    max_groundings = None
    if config.cot_generation is not None:
        max_rules = config.cot_generation.max_rules
        max_groundings = config.cot_generation.max_groundings

    # Load the graph
    from knowledge_graphs import load_knowledge_graph  # type: ignore

    graph = load_knowledge_graph(kg_file=config.data.input_dir / config.kg.kg_file)

    # Generate CoTs
    generate_cots_sparql(
        kg_config=config.kg,
        graph=graph,
        rules_df=rules_df,
        output_dir=config.data.output_dir,
        max_rules=max_rules,
        max_groundings=max_groundings,
    )
