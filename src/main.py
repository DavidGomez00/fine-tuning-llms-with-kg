from typing import Dict

import pandas as pd

from configuration import CoTGenerationConfig, ExperimentConfig
from evaluation import EvaluationMetrics
from KG import (
    create_relation_mapping,
    load_id2relation_mapping,
    load_knowledge_graph,
    parse_kg,
)
from model import evaluate_model, load_model
from rules import (
    generate_CoTs,
    load_rules_from_path,
)


def set_up(config: ExperimentConfig) -> None:
    """Set up method to load and generate data before training."""

    # Parse KG file
    kg_input = config.data.data_dir / config.data.kg_file
    kg_processed = config.data.data_dir / config.data.kg_file_processed
    parse_kg(kg_input, kg_processed)

    # Create relation file
    relations_file = config.data.data_dir / config.data.relation_file
    create_relation_mapping(kg_processed, relations_file)


def generate_data(
    config: ExperimentConfig, dataset_configs: Dict[str, CoTGenerationConfig]
):
    """Generates necessary data to perform an experiment.

    # TODO: gestion de errores

    Args:
        config: Experiment configuratoin.
    """

    # Load data from files
    kg_processed = config.data.data_dir / config.data.kg_file_processed
    graph, node_list = load_knowledge_graph(kg_processed)

    relations_file = config.data.data_dir / config.data.relation_file
    id2relation = load_id2relation_mapping(relations_file)

    rules_path = config.data.data_dir / config.data.rules_dir
    rules = load_rules_from_path(rules_path)

    datasets: Dict[str, pd.DataFrame] = {}

    for split_name, data_config in dataset_configs.items():
        datasets[split_name] = generate_CoTs(
            graph=graph,
            node_list=node_list,
            id2relation=id2relation,
            rules=rules,
            config=data_config,
        )

    # Save generated data
    for split_name, dataset_df in datasets.items():
        output_path = config.data.output_dir / f"{split_name}.csv"
        dataset_df.to_csv(output_path, index=False)


def load_datasets(config: ExperimentConfig) -> Dict[str, pd.DataFrame]:
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
        generate_data(config, dataset_configs)

    datasets: Dict[str, pd.DataFrame] = {}

    # Load datasets from files
    for split_name, _ in dataset_configs.items():
        dataset_path = config.data.output_dir / f"{split_name}.csv"
        datasets[split_name] = pd.read_csv(
            dataset_path, encoding="utf-8"
        )  # TODO: comprobar este comando.

    return datasets


def main() -> None:
    """Main method to execute an experiment."""

    config = ExperimentConfig()

    # Maybe set up
    set_up(config)

    # Get datasets
    datasets = load_datasets(config)

    # Base model evaluation
    if config.run_settings.skip_base_eval:
        print("Skipping base model evaluation.")
    else:
        base_model, base_tokenizer = load_model(config)
        base_results = evaluate_model(
            config=config,
            model=base_model,
            tokenizer=base_tokenizer,
            data=datasets["test_data_with_rules"],
        )

    # Baseline model: Fine-tune without rules (KG-LLM)
    if config.run_settings.skip_baseline:
        print("Skipping baseline evaluation.")
    else:
        baseline_model, baseline_tokenizer = train_model()  # TODO
        baseline_results = evaluate_model(
            config=config,
            model=baseline_model,
            tokenizer=baseline_tokenizer,
            data=datasets["train_datasets_without_rules"],
        )

    # Final model: Fine-tune with rules (NeSyKG-LLM)
    final_model, final_tokenizer = train_model()  # TODO
    final_results = evaluate_model(
        config=config,
        model=baseline_model,
        tokenizer=baseline_tokenizer,
        data=datasets["train_datasets_with_rules"],
    )

    # Print results table
    print_results_table(results, training_times)

    # Save all results
    save_results(results, training_times)

    # Generate comparison plot
    if config.run_settings.generate_plots:
        create_comparison_plot(results)

    # Summary CSV
    summary_path = config.data.output_dir / config.data.summary_csv
    print(f"\nResults Summary ({summary_path}):")
    print("-" * 70)
    summary_df = pd.read_csv(summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
