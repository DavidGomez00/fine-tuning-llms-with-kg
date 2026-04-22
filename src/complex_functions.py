"""Fichero con las funciones complejas del framework"""

# TODO: docstrings
# TODO (generate_data): method may be obsolete
# TODO (generate_data): Gestión de errores

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

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
    convert_all_rules_to_natural_language,
    create_summary,
    generate_cots,
    load_rules_from_path,
)


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





if __name__ == "__main__":
    pass
