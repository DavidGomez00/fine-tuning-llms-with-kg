"""Fichero con las funciones complejas del framework"""

# TODO: docstrings
# TODO (generate_data): method may be obsolete
# TODO (generate_data): Gestión de errores
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from config import CoTGenerationConfig, DirConfig, FineTuningConfig, KGConfig, RunConfig
from generate_cots import generate_cots_sparql
from KG import (
    load_knowledge_graph,
)
from model import (
    create_comparison_plot,
    create_results_table_png,
    evaluate_model,
    fine_tune,
    save_results,
)

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
# Fine-tuning
# ---------------------------------------------------------------------------


def _load_datasets(
    input_dir: Path, cot_config: CoTGenerationConfig
) -> dict[str, pd.DataFrame]:
    """Retrieves or generates the datasets necessary for the experiment.

    Args:
        config: Experiment configuration.

    Returns:
        A dictionary containing different datasets.
    """
    datasets: dict[str, pd.DataFrame] = {}

    # Load datasets from files
    for split_name in ["train_with_rules", "train_without_rules", "test"]:
        dataset_path = input_dir / f"{split_name}.csv"
        datasets[split_name] = pd.read_csv(dataset_path, encoding="utf-8")

    return datasets


def _generate_dataset(
    data_config: DirConfig, kg_config: KGConfig, ft_config: FineTuningConfig
) -> None:
    """Generates necessary datasets to perform an experiment."""

    # TODO: Meter esto en la configuración, ahora nismo sólo se están usando
    # los nombres de los elementos del dict.
    # TODO: Considerar que ya haya configuración puesta
    # TODO: Right now many configs about "use rules" or "use reasoning" are not being
    # used for generation.
    dataset_configs = {
        "train_with_rules": CoTGenerationConfig(
            max_samples=ft_config.train_with_rules_samples, use_rules=False
        ),
        "train_without_rules": CoTGenerationConfig(
            max_samples=ft_config.train_without_rules_samples, use_rules=True
        ),
        "test": CoTGenerationConfig(max_samples=ft_config.test_samples, use_rules=True),
    }

    datasets: defaultdict[str, pd.DataFrame] = defaultdict()
    for split, cot_config in dataset_configs.items():
        output_dir = config.data.output_dir / split
        datasets[split] = generate_cots_sparql(
            kg_config=kg_config,
            cot_config=cot_config,
            input_dir=data_config.input_dir,
            output_dir=output_dir,
        )


def fine_tuning_experiment(config: RunConfig) -> None:
    """Main method to execute an experiment."""

    experiment_results: dict[str, Any] = defaultdict(dict)

    # TODO: Implementar check por si el grafo no está parseado
    # kg_input = config.data.input_dir / config.kg.raw_kg_file
    kg_file = config.data.input_dir / config.kg.kg_file
    graph = load_knowledge_graph(kg_file=kg_file)

    if config.fine_tuning.generate_datasets:
        _generate_dataset(
            data_config=config.data, kg_config=config.kg, ft_config=config.fine_tuning
        )

    return None

    # TODO: dataset generation should generate .csv files for datasets
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


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Set up logger
    setup_logging()

    # Load configuration
    conf_json = Path("configurations/fine_tuning_fr.json")
    config = RunConfig.from_json(conf_json)

    # Fine-tune Llama3.1:8b with FR KG
    fine_tuning_experiment(config=config)
