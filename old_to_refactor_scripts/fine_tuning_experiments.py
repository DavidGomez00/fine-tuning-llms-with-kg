"""Fichero con las funciones complejas del framework"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from gen_cots import generate_cots_sparql
from matplotlib import pyplot as plt
from rdflib import Graph

from config import DirConfig, FineTuningConfig, KGConfig, RunConfig
from graphs import load_knowledge_graph
from old_to_refactor_scripts.model import (
    EvaluationMetrics,
    evaluate_model,
    fine_tune,
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
# Dataset loading and generation
# ---------------------------------------------------------------------------


def _generate_datasets(
    kg_config: KGConfig,
    output_dir: Path,
    rules_df: pd.DataFrame,
    graph: Graph,
) -> None:
    """Generates necessary datasets to perform an experiment."""

    # TODO: Meter esto en la configuración, ahora nismo sólo se están usando
    # los nombres de los elementos del dict.
    # TODO: Considerar que ya haya configuración puesta
    # TODO: Right now many configs about "use rules" or "use reasoning" are not being
    # used for generation.
    # TODO: Make the dataset creation with parameters

    splits = ["training_without_rules", "training_with_rules", "test"]

    for split in splits:
        dataset_output_dir = output_dir / split
        generate_cots_sparql(
            kg_config=kg_config,
            graph=graph,
            rules_df=rules_df,
            output_dir=dataset_output_dir,
            save_summary=True,
        )

    # Generate csv files with rules info for each split


def _load_datasets(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Loads datasets for the fine-tuning experiment."""
    datasets: dict[str, pd.DataFrame] = {}

    # Load datasets from files
    for split_name in ["train_with_rules", "train_without_rules", "test"]:
        dataset_path = input_dir / f"{split_name}.csv"
        datasets[split_name] = pd.read_csv(dataset_path, encoding="utf-8")

    return datasets


# ---------------------------------------------------------------------------
# Plot results and comparisons
# ---------------------------------------------------------------------------


def _save_summary_csv(results: dict[str, EvaluationMetrics], output_file: Path) -> None:
    """Helper method to generate and save the summary CSV dataframe."""
    rows = [
        {
            "Model": model_name,
            "Precision": metrics.precision,
            "Recall": metrics.recall,
            "F1": metrics.f1_score,
            "Size": metrics.eval_size,
            "Time (s)": metrics.eval_time,
            "Train Time (s)": metrics.train_time,
        }
        for model_name, metrics in results.items()
    ]
    df = pd.DataFrame(rows)
    df = df.round(
        {
            "Precision": 4,
            "Recall": 4,
            "F1": 4,
            "Time (s)": 2,
            "Train Time (s)": 2,
        }
    )
    df.to_csv(output_file, index=False)
    logger.debug("Summary CSV saved to %s", output_file)


def _create_comparison_plot(results: dict[str, Any], output_file: Path) -> None:
    """Creates a bar chart comparing model performances.

    Args:
        config: Dataclass with the experiment settings.
        results: Dictionary containing model names mapped to metric objects.

    Raises:
        NotADirectoryError: If the output dir specified in `config` does not exist.
    """

    models = list(results.keys())
    if not models:
        logger.warning("No results found to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_data = [
        (axes[0, 0], [results[m].accuracy for m in models], "Accuracy"),
        (axes[0, 1], [results[m].precision for m in models], "Precision"),
        (axes[1, 0], [results[m].recall for m in models], "Recall"),
        (axes[1, 1], [results[m].f1_score for m in models], "F1 Score"),
    ]

    for ax, data, title in metrics_data:
        bars = ax.bar(models, data, alpha=0.8)

        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f"{title} Comparison", fontsize=14, fontweight="bold")

        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Comparison plot saved to %s", output_file)


def _create_comparison_table(
    results: dict[str, EvaluationMetrics],
    output_path: Path,
) -> None:
    """Generate a formatted results table and save it as a .png file.

    Args:
        results: dict containing entries for different modes and their corresponding
        evaluation metrics objects.
        output_path: Path to save the generated table.
    """
    # Define columns to match your original console output
    columns = ["Model", "Precision", "Recall", "F1", "Size", "Time (s)", "Train (s)"]
    cell_text = []

    # Extract and format data
    for model_name, metrics in results.items():
        row = [
            model_name,
            f"{metrics.precision:.4f}",
            f"{metrics.recall:.4f}",
            f"{metrics.f1_score:.4f}",
            str(metrics.eval_size),
            f"{metrics.eval_time:.2f}",
            f"{metrics.train_time:.2f}",
        ]
        cell_text.append(row)

    # Matplotlib figure
    fig_width = 10
    fig_height = len(cell_text) * 0.5 + 1.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=cell_text, colLabels=columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    plt.title("EVALUATION RESULTS SUMMARY", weight="bold", size=14, pad=20)

    # Save figure to a file
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fine-tuning experiment
# ---------------------------------------------------------------------------


def fine_tuning_experiment(
    data_config: DirConfig, kg_config: KGConfig, ft_config: FineTuningConfig
) -> None:
    """Main method to execute the fine-tuning experiment.

    A base model is evaluated. Then, the same model is fine-tuned and evaluated with and
    without rules. Finally, tables and plots for performance comparison are generated.
    """

    experiment_results: dict[str, Any] = defaultdict(dict)

    if ft_config.generate_datasets:
        # Generate datasets
        graph = load_knowledge_graph(data_config.input_dir / kg_config.kg_file)
        rules_df = pd.read_csv(data_config.input_dir / kg_config.rules_csv)

        _generate_datasets(
            kg_config=kg_config,
            output_dir=data_config.output_dir,
            rules_df=rules_df,
            graph=graph,
        )

    datasets = _load_datasets(input_dir=data_config.input_dir)

    # Base model
    if ft_config.skip_base_eval:
        logger.info("Skipping base model evaluation.")
    else:
        # Evaluation
        experiment_results["Base Model"] = evaluate_model(
            config=config,
            data=datasets["test_data_with_rules"],
        )

    # Baseline fine-tuning without rules (KG-LLM)
    if ft_config.skip_baseline:
        logger.info("Skipping baseline evaluation.")
    else:
        # Training
        baseline_model, train_time = fine_tune(
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

    # Save results and plot summaries/comparisons
    _save_summary_csv(
        experiment_results,
        output_file=data_config.output_dir / ft_config.summary_csv_file,
    )
    _create_comparison_table(
        experiment_results, data_config.output_dir / ft_config.summary_table_file
    )
    _create_comparison_plot(
        experiment_results, data_config.output_dir / ft_config.summary_plot_file
    )


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Set up logger
    setup_logging()

    # Load configuration
    conf_json = Path("configurations/fine_tuning_fr.json")
    config = RunConfig.from_json(conf_json)
    if config.fine_tuning is None:
        raise ValueError("Config expected to have fine-tunning settings.")

    # Fine-tune Llama3.1:8b with FR KG
    fine_tuning_experiment(
        ft_config=config.fine_tuning, data_config=config.data, kg_config=config.kg
    )
