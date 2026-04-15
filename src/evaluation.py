import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from configuration import ExperimentConfig


@dataclass
class EvaluationMetrics:
    """Container for comprehensive model evaluation metrics.

    Attributes:
        y_true: List of actual ground truth labels.
        y_pred: List of model-predicted labels.
        eval_time: Time elapsed during the evaluation phase in seconds.
        accuracy: Proportion of correct predictions over the total samples.
        precision: Proportion of true positives over the total predicted positives.
        recall: Proportion of true positives over the total actual positives.
        f1_score: Harmonic mean of precision and recall.
        eval_size: Total number of samples used during evaluation.
        train_time: Time elapsed during the training phase.

    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    eval_size: int

    eval_time: float = 0.0
    train_time: float = 0.0

    y_true: List[bool] = field(default_factory=list)
    y_pred: List[bool] = field(default_factory=list)

    @classmethod
    def from_predictions(
        cls, y_true: List[bool], y_pred: List[bool], **kwargs
    ) -> "EvaluationMetrics":
        """Factory method to calculate metrics directly from raw labels.

        Args:
            y_true: Ground truth labels.
            y_pred: Model predictions.
            eval_time: Time taken for evaluation. Defaults to 0.0.

        Returns:
            EvaluationMetrics: A new instance with all metrics calculated.
        """
        # Note: In a real script, you would use sklearn.metrics # TODO:
        tp = sum(t and p for t, p in zip(y_true, y_pred))
        fp = sum(not t and p for t, p in zip(y_true, y_pred))
        fn = sum(t and not p for t, p in zip(y_true, y_pred))
        tn = sum(not t and not p for t, p in zip(y_true, y_pred))

        total = len(y_true)
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return cls(
            y_true=y_true,
            y_pred=y_pred,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            eval_size=total,
            **kwargs,
        )

    def to_dict(self, include_predictions: bool = False) -> Dict[str, Any]:
        """Converts the stored metrics into a dictionary format.

        Args:
            include_predictions: If True, includes `y_true` and `y_pred` lists.

        Returns:
            A dictionary containing the metrics, rounded for readability.
        """
        result: Dict[str, Any] = {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "eval_size": self.eval_size,
            "eval_time": round(self.eval_time, 2),
            "train_time": round(self.train_time, 2),
        }

        if include_predictions:
            result["y_true"] = self.y_true
            result["y_pred"] = self.y_pred

        return result

    def __str__(self) -> str:
        """Returns a human-readable string representation of the main metrics."""
        return (
            f"Precision: {self.precision:.4f} | "
            f"Recall: {self.recall:.4f} | "
            f"F1: {self.f1_score:.4f} | "
            f"Size: {self.eval_size:.2f} | "
            f"Time: {self.eval_time:.2f} s"
        )


def save_results_table_png(
    results: Dict[str, EvaluationMetrics],
    output_path: Path,
) -> None:
    """
    Generate a formatted results table and save it as a .png file.

    Args:
        results: A dictionary containing EvaluationMetrics for each model in the experiment.
        filename: Name of the output file.
    """
    # 1. Define columns to match your original console output
    columns = ["Model", "Precision", "Recall", "F1", "Size", "Time (s)", "Train (s)"]
    cell_text = []

    # 2. Extract and format data
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


def _save_json_results(
    config: ExperimentConfig,
    results: Dict[str, EvaluationMetrics],
) -> None:
    """Helper method to generate and save the JSON payload.

    TODO: Document
    """

    json_path = config.data.output_dir / config.data.results_file

    results_dict: Dict[str, Any] = {
        "experiment_info": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": config.model.name,
            "model_alias": config.model.alias,
            "train_samples": config.train.train_samples,
            "training_steps": config.train.max_steps,
            "epochs": config.train.num_train_epochs,
        },
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    # TODO: implement logger
    print(f"JSON results saved to {json_path}")


def _save_summary_csv(
    config: ExperimentConfig,
    results: Dict[str, EvaluationMetrics],
) -> None:
    """Helper method to generate and save the summary CSV dataframe.

    Args:
        config: Experiment configuration.
        results: Evaluaiton metrics of each model in the experiment.
    """
    csv_path = config.data.output_dir / config.data.summary_csv
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

    df.to_csv(csv_path, index=False)

    # TODO: implement logging
    print(f"Summary CSV saved to {csv_path}")


def save_results(
    config: ExperimentConfig, results: Dict[str, EvaluationMetrics]
) -> None:
    """Saves comprehensive results to multiple formats in disk.

    Args:
        config: Dataclass with the experiment settings.
        results: Dictionary containing model names mapped to metric objects.

    Raises:
        NotADirectoryError: If the output dir specified in `config` does not exist.
    """
    if not config.data.output_dir.is_dir():
        raise NotADirectoryError(
            f"The directory does not exist: {config.data.output_dir}"
        )

    config.data.output_dir.mkdir(parents=True, exist_ok=True)

    _save_json_results(config, results)
    _save_summary_csv(config, results)


def create_comparison_plot(config: ExperimentConfig, results: Dict[str, Any]) -> None:
    """Creates a bar chart comparing model performances.

    Args:
        config: Dataclass with the experiment settings.
        results: Dictionary containing model names mapped to metric objects.

    Raises:
        NotADirectoryError: If the output dir specified in `config` does not exist.
    """
    if not config.data.output_dir.is_dir():
        raise NotADirectoryError(
            f"The directory does not exist: {config.data.output_dir}"
        )

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

    output_file = config.data.output_dir / config.data.experiment_plot

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.close(fig)

    # TODO: Implement logging
    print(f"Comparison plot saved to {output_file}")
