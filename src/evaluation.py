from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class EvaluationMetrics:
    """Container for comprehensive model evaluation metrics.

    Attributes:
        accuracy (float): Proportion of correct predictions over the total samples.
        precision (float): Proportion of true positives over the total predicted positives.
        recall (float): Proportion of true positives over the total actual positives.
        f1_score (float): Harmonic mean of precision and recall.
        eval_size (int): Total number of samples used during evaluation.
        eval_time_seconds (float): Time elapsed during the evaluation phase in seconds.
        training_time_seconds (float, optional): Time elapsed during the training phase. 
            Defaults to 0.0. #TODO ???
        y_true (List[bool], optional): List of actual ground truth labels. 
            Defaults to an empty list.
        y_pred (List[bool], optional): List of model-predicted labels. 
            Defaults to an empty list.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    eval_size: int
    eval_time_seconds: float
    training_time_seconds: Optional[float] = 0.0
    y_true: List[bool] = field(default_factory=list)
    y_pred: List[bool] = field(default_factory=list)

    @classmethod
    def from_predictions(cls, y_true: List[bool], y_pred: List[bool], 
                         eval_time: float = 0.0, **kwargs) -> 'EvaluationMetrics':
        """Factory method to calculate metrics directly from raw labels.

        Args:
            y_true (List[bool]): Ground truth labels.
            y_pred (List[bool]): Model predictions.
            eval_time (float, optional): Time taken for evaluation. Defaults to 0.0.
            **kwargs: Additional fields like training_time_seconds.

        Returns:
            EvaluationMetrics: A new instance with all metrics calculated.
        """
        # TODO: Evaluar si merece la pena implementar este método
        # Note: In a real script, you would use sklearn.metrics
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
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            eval_size=total,
            eval_time_seconds=eval_time,
            y_true=y_true,
            y_pred=y_pred,
            **kwargs
        )

    def to_dict(self, include_predictions: bool = False) -> Dict:
        """Converts the stored metrics into a dictionary format.

        Args:
            include_predictions (bool, optional): If True, includes `y_true` and 
                `y_pred` lists in the resulting dictionary. Defaults to False.

        Returns:
            Dict: A dictionary containing the metrics, rounded for readability.
        """
        result = {
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'eval_size': self.eval_size,
            'eval_time_seconds': round(self.eval_time_seconds, 2),
            'training_time_seconds': round(self.training_time_seconds, 2),
        }

        if include_predictions:
            result['y_true'] = self.y_true
            result['y_pred'] = self.y_pred

        return result

    def __str__(self) -> str:
        """Returns a human-readable string representation of the main metrics."""
        return (
            f"Precision: {self.precision:.4f} | "
            f"Recall: {self.recall:.4f} | "
            f"F1: {self.f1_score:.4f} | "
            f"Size: {self.eval_size} | "
            f"Time: {self.eval_time_seconds:.2f}s"
        )