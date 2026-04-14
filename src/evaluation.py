from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
        training_time: Time elapsed during the training phase.

    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    eval_size: int

    y_true: List[bool] = field(default_factory=list)
    y_pred: List[bool] = field(default_factory=list)

    eval_time: Optional[float] = None
    training_time: Optional[float] = None

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
            "eval_time": (
                round(self.eval_time, 2) if self.eval_time is not None else None,
            ),
            "training_time": (
                round(self.training_time, 2) if self.training_time is not None else None
            ),
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
            f"Size: {self.eval_size if self.eval_size else None} | "
            f"Time: {self.eval_time:.2f if self.eval_time else None}s"
        )
