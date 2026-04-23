"""Fichero con las funciones complejas del framework"""

# TODO: docstrings
# TODO (generate_data): method may be obsolete
# TODO (generate_data): Gestión de errores
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph

from config import CoTGenerationConfig, DirConfig, KGConfig, RunConfig
from KG import (
    create_relation_mapping,
    load_id2relation_mapping,
    load_knowledge_graph,
    parse_kg,
)
from model import (
    create_comparison_plot,
    create_results_table_png,
    evaluate_model,
    fine_tune,
    save_results,
)
from rules import (
    build_sparql_query,
    generate_cots,
    load_rules_from_path,
    parse_horn_rule,
    result_to_natural_language,
)

logger = logging.getLogger(__name__)


def _set_up(config: RunConfig) -> None:
    """Set up method to load and generate data before training."""

    # Parse KG file
    kg_input = config.data.input_dir / config.kg.raw_kg_file
    kg_file = config.data.input_dir / config.kg.kg_file
    parse_kg(kg_input, kg_file)

    # Create relation file
    relations_file = config.data.input_dir / config.kg.relation_file
    create_relation_mapping(kg_file, relations_file)


def _generate_data(
    data_config: DirConfig,
    kg_config: KGConfig,
    dataset_configs: dict[str, CoTGenerationConfig],
) -> None:
    """Generates necessary data to perform an experiment."""

    # Load data from files
    # TODO: error if kg is not parsed
    kg_file = data_config.input_dir / kg_config.kg_file
    graph, node_list = load_knowledge_graph(kg_file)

    relations_file = data_config.input_dir / kg_config.relation_file
    id2relation = load_id2relation_mapping(relations_file)

    rules_path = data_config.input_dir / data_config.rules_dir
    rules = load_rules_from_path(rules_path)

    datasets: dict[str, pd.DataFrame] = {}

    for split_name, data_config in dataset_configs.items():
        datasets[split_name] = generate_cots(
            graph=graph,
            node_list=node_list,
            id2relation=id2relation,
            rules=rules,
            config=data_config,
        )

    # Save generated data
    for split_name, dataset_df in datasets.items():
        output_path = data_config.output_dir / f"{split_name}.csv"
        dataset_df.to_csv(output_path, index=False)


def _load_datasets(config: RunConfig) -> dict[str, pd.DataFrame]:
    """Retrieves or generates the datasets necessary for the experiment.

    Args:
        config: Experiment configuration.

    Returns:
        A dictionary containing different datasets.
    """

    # TODO: This should be inside config
    dataset_configs = {
        "train_data_without_rules": CoTGenerationConfig(samples=2000, use_rules=False),
        "train_data_with_rules": CoTGenerationConfig(samples=2000),
        "test_data_with_rules": CoTGenerationConfig(samples=1000),
    }

    if config.run_settings.generate_datasets:
        _generate_data(
            data_config=config.data,
            kg_config=config.kg,
            dataset_configs=dataset_configs,
        )

    datasets: dict[str, pd.DataFrame] = {}

    # Load datasets from files
    for split_name, _ in dataset_configs.items():
        dataset_path = config.data.output_dir / f"{split_name}.csv"
        datasets[split_name] = pd.read_csv(
            dataset_path, encoding="utf-8"
        )  # TODO: comprobar este comando.

    return datasets


def fine_tuning_experiment() -> None:
    """Main method to execute an experiment."""

    config = RunConfig()
    experiment_results: dict[str, Any] = defaultdict(dict)

    # Maybe set up
    _set_up(config)

    datasets = _load_datasets(config)

    # Base model
    if config.run_settings.skip_base_eval:
        print("Skipping base model evaluation.")
    else:
        # Evaluation
        experiment_results["Base Model"] = evaluate_model(
            config=config,
            data=datasets["test_data_with_rules"],
        )

    # Baseline fine-tuning without rules (KG-LLM)
    if config.run_settings.skip_baseline:
        print("Skipping baseline evaluation.")
    else:
        # Training
        baseline_model, train_time = fine_tune(  # TODO: use training_time!!
            config=config,
            train_dataset=datasets["train_datasets_without_rules"],
        )

        # Evaluation
        experiment_results["Baseline Model"] = evaluate_model(
            config=config,
            model=baseline_model,
            data=datasets["test_data_with_rules"],
        )
        experiment_results["Baseline Model"].train_time = train_time

    # Final fine-tune with rules (NeSyKG-LLM)
    # Training
    final_model, train_time = fine_tune(
        config=config,
        train_dataset=datasets["train_datasets_with_rules"],
    )

    # Evaluation
    experiment_results["Final Model"] = evaluate_model(
        config=config,
        model=final_model,
        data=datasets["test_data_with_rules"],
    )
    experiment_results["Final Model"].train_time = train_time

    # Save all results
    save_results(config, experiment_results)

    # Generate result plots and summaries
    if config.run_settings.generate_table:
        experiment_table_path = config.data.output_dir / config.data.experiment_table
        create_results_table_png(experiment_results, experiment_table_path)

    if config.run_settings.generate_plots:
        create_comparison_plot(config, experiment_results)


def _create_summary(
    kg_config: KGConfig,
    data_config: DirConfig,
    graph: Graph,
    rules_df: pd.DataFrame,
    time: float,
) -> None:
    """Writes a summary report of the rules transformed into natural language

    Args:
        kg_config: Configuration object containing KG and rules parameters.
        data_config: Configuration object containing directory paths.
        graph: The loaded RDFLib knowledge graph.
        rules_df: DataFrame containing the parsed rules and their confidence scores.
    """
    summary_path = data_config.output_dir / kg_config.rule_summary_file

    pca_series = rules_df["PCA_Confidence"]
    valid_pca = pca_series.dropna()
    neg = (valid_pca < kg_config.pca_threshold).sum()
    pos = (valid_pca >= kg_config.pca_threshold).sum()
    nan = pca_series.isna().sum()

    summary_content = (
        f"{kg_config.kg_name.upper()} CoT2 RULES — SUMMARY\n"
        f"\tTotal Rules        : {len(rules_df)}\n"
        f"\tOutput Directory   : {data_config.output_dir.absolute()}\n"
        f"\tKG Triples         : {len(graph)}\n"
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


def generate_cots_sparql(kg_config: KGConfig, data_config: DirConfig) -> None:
    """Generate natural language descriptions for each rule in the csv and each
    grounding of the rule in the KG."""

    # Files and paths
    kg_file_path = data_config.input_dir / kg_config.kg_file
    rules_csv_path = data_config.input_dir / kg_config.rules_csv

    if not kg_file_path.is_file():
        raise FileNotFoundError(f"File {kg_file_path} does not exist.")
    if not rules_csv_path.is_file():
        raise FileNotFoundError(f"File {rules_csv_path} does not exist.")

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

    # 2. For each rule, generate a query
    results: defaultdict[str, Any] = defaultdict()
    logger.info("Generating queries for 5 rules.")
    start_time = time.time()
    for rule_idx, row in enumerate(rules_df.itertuples(index=False), start=1):
        if rule_idx == 6:
            break
        rule_id = str(rule_idx)
        rule = parse_horn_rule(
            row=row, rule_id=rule_id, pca_threshold=kg_config.pca_threshold
        )

        query = build_sparql_query(
            rule_signature=rule.signature,
            ns_prefix=kg_config.namespace_prefix,
            namespace=kg_config.namespace,
        )
        results[rule_id] = query

    # 3. Query the KG
    logger.info("Executing 5 queries and generating a text description of the results.")
    for rule_id, query_result in results.items():
        result = graph.query(query_result)

        # Generate rule + groundings natural language description
        grounding_description = result_to_natural_language(
            kg_config=kg_config, result=result, rule=rule
        )

        # 4. Save the rule + groundings natural language description as a file
        rule_file_path = data_config.output_dir / f"rule_{rule_id}.txt"
        rule_file_path.write_text(grounding_description, encoding="utf-8")
        logger.debug("Saving description of %s in %s", rule_id, rule_file_path.name)

    total_time = time.time() - start_time

    logger.info("Succesfully created 5 query result descriptions.")
    _create_summary(
        kg_config=kg_config,
        data_config=data_config,
        graph=graph,
        rules_df=rules_df,
        time=total_time,
    )


# -----
# Testing
# ----


def setup_logging() -> None:
    """Configures the root logger to output to the console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    # Set up logger
    setup_logging()

    # Load configuration
    configuration_json = Path("configurations/gen_fr_cots.json")
    config = RunConfig.from_json(configuration_json)

    # Generate CoTs
    generate_cots_sparql(kg_config=config.kg, data_config=config.data)
