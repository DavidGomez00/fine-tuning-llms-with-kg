import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from configuration import ExperimentConfig


def _save_json_results(
    output_dir: Path,
    config: ExperimentConfig,
    results: Dict[str, Any],
    training_times: Dict[str, float],
) -> None:
    """Helper method to generate and save the JSON payload."""
    json_path = output_dir / config.data.results_file

    results_dict: Dict[str, Any] = {
        "experiment_info": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": config.model.name,
            "model_alias": config.model.alias,
            "train_samples": config.train.train_samples,
            "num_training_steps": config.train.training_steps,
        },
        "results": {},
    }

    for model_name, metrics in results.items():
        model_metrics_dict = metrics.to_dict()
        model_metrics_dict["training_time_seconds"] = training_times.get(
            model_name, 0.0
        )
        results_dict["results"][model_name] = model_metrics_dict

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    # TODO: implement logger
    print(f"JSON results saved to {json_path}")


def _save_summary_csv(
    output_dir: Path,
    config: ExperimentConfig,
    results: Dict[str, Any],
    training_times: Dict[str, float],
) -> None:
    """Helper method to generate and save the summary CSV dataframe."""
    csv_path = output_dir / config.data.summary_csv
    rows = [
        {
            "Model": model_name,
            "Precision": metrics.precision,
            "Recall": metrics.recall,
            "F1": metrics.f1_score,
            "Size": metrics.eval_size,
            "Time (s)": metrics.eval_time_seconds,
            "Train Time (s)": training_times.get(model_name, 0.0),
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

    df.to_csv(csv_path, index=False)
    # TODO: implement logging
    print(f"Summary CSV saved to {csv_path}")


def save_results(
    config: ExperimentConfig, results: Dict[str, Any], training_times: Dict[str, float]
) -> None:
    """Saves comprehensive results to multiple formats.

    Args:
        config: Dataclass with the experiment settings.
        results: Dictionary containing model names mapped to metric objects.
        training_times: Maps model names to training times in seconds.

    Raises:
        NotADirectoryError: If the output dir specified in `config` does not exist.
    """
    output_dir = Path(config.data.output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError(f"The directory does not exist: {output_dir}")

    _save_json_results(output_dir, config, results, training_times)
    _save_summary_csv(output_dir, config, results, training_times)


def create_comparison_plot(config: ExperimentConfig, results: Dict[str, Any]) -> None:
    """Creates a bar chart comparing model performances.

    Args:
        config: Dataclass with the experiment settings.
        results: Dictionary containing model names mapped to metric objects.

    Raises:
        NotADirectoryError: If the output dir specified in `config` does not exist.
    """
    output_dir = Path(config.data.output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError(f"The directory does not exist: {output_dir}")

    output_file = output_dir / "comparison_plot.png"  # TODO: Maybe move to config.data

    models = list(results.keys())
    if not models:
        # TODO: logging
        print("Warning: No results found to plot.")
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

    # TODO: Implement logging
    print(f"Comparison plot saved to {output_file}")
