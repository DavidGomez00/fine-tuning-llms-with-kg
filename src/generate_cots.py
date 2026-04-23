"""File to generate CoTs form a KG."""

# TODO: Where am I loading the KG? maybe plan better
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import CoTGenerationConfig, KGConfig, RunConfig
from KG import load_knowledge_graph
from rules import build_sparql_query, parse_horn_rule, query_result_to_natural_language

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configures the root logger to output to the console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _create_summary(
    kg_config: KGConfig,
    output_dir: Path,
    output_file: str,
    graph_length: int,
    rules_df: pd.DataFrame,
    time: float,
) -> None:
    """Writes a summary report of the rules transformed into natural language

    Args:
        kg_config: Configuration object containing KG and rules parameters.
        output_dir: Path to the output directory.
        output_file: Filename of the summary file.
        graph_length: The length of the knowledge graph used to create the CoTs.
        rules_df: DataFrame containing the parsed rules and their confidence scores.
        time: Time for the process to complete in seconds.
    """
    summary_path = output_dir / output_file

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
    summary_path.write_text(summary_content, encoding="utf-8")
    logger.info(f"Summary saved to: {summary_path}")


def generate_cots_sparql(
    kg_config: KGConfig,
    cot_config: CoTGenerationConfig,
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Generate natural language descriptions for each rule in the csv and each
    grounding of the rule in the KG."""

    # Files and paths
    kg_file_path = input_dir / kg_config.kg_file
    rules_csv_path = input_dir / kg_config.rules_csv

    if not kg_file_path.is_file():
        raise FileNotFoundError(f"File {kg_file_path} does not exist.")
    if not rules_csv_path.is_file():
        raise FileNotFoundError(f"File {rules_csv_path} does not exist.")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # 1. Load graph and rules
    graph = load_knowledge_graph(kg_file_path)
    rules_df = pd.read_csv(rules_csv_path, encoding="utf-8")
    logger.info("Loaded %s file with %d rules.", rules_csv_path.name, len(rules_df))
    rules_df = rules_df.head(cot_config.max_samples)

    # 2. For each rule, generate a query
    query_mapping: defaultdict[str, str] = defaultdict()
    logger.info("Generating queries for %d rules.", len(rules_df))

    start_time = time.time()
    for rule_idx, row in enumerate(rules_df.itertuples(index=False), start=1):
        rule_id = str(rule_idx)
        rule = parse_horn_rule(
            row=row, rule_id=rule_id, pca_threshold=kg_config.pca_threshold
        )

        query = build_sparql_query(
            rule_signature=rule.signature,
            ns_prefix=kg_config.namespace_prefix,
            namespace=kg_config.namespace,
        )
        query_mapping[rule_id] = query

    # 3. Query the KG
    logger.info(
        "Executing %d queries and generating a text description of the results.",
        len(query_mapping),
    )
    for rule_id, query in query_mapping.items():
        result = graph.query(query)

        # Generate rule + groundings natural language description
        grounding_description = query_result_to_natural_language(
            kg_config=kg_config, result=result, rule=rule
        )

        # 4. Save the rule + groundings natural language description as a file
        rule_file_path = output_dir / f"rule_{rule_id}.txt"
        rule_file_path.write_text(grounding_description, encoding="utf-8")
        logger.debug("Saving description of %s in %s", rule_id, rule_file_path.name)

    total_time = time.time() - start_time

    logger.info("Succesfully created %d query result descriptions.", len(query_mapping))
    if kg_config.rule_summary_file:
        _create_summary(
            kg_config=kg_config,
            output_dir=output_dir,
            output_file=kg_config.rule_summary_file,
            graph_length=len(graph),
            rules_df=rules_df,
            time=total_time,
        )


if __name__ == "__main__":
    # Set up logger
    setup_logging()

    # Load configuration
    configuration_json = Path("configurations/gen_fr_cots.json")
    config = RunConfig.from_json(configuration_json)

    # Generate CoTs
    generate_cots_sparql(
        kg_config=config.kg,
        cot_config=config.cot_generation,
        input_dir=config.data.input_dir,
        output_dir=config.data.output_dir,
    )
