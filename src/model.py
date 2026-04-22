"""File to implement the functionalities that deal with models, like model loading or
saving and training or evaluating the model performance."""

# TODO: Improve EvaluationMetrics class
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bitsandbytes as bnb
import matplotlib.pyplot as plt
import pandas as pd
import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from config import RunConfig


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

    y_true: list[bool] = field(default_factory=list)
    y_pred: list[bool] = field(default_factory=list)

    @classmethod
    def from_predictions(
        cls, y_true: list[bool], y_pred: list[bool], **kwargs: float | int
    ) -> "EvaluationMetrics":
        """Factory method to calculate metrics directly from raw labels.

        Args:
            y_true: Ground truth labels.
            y_pred: Model predictions.
            eval_time: Time taken for evaluation. Defaults to 0.0.

        Returns:
            EvaluationMetrics: A new instance with all metrics calculated.
        """
        # TODO: use sklearn.metrics
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

    def to_dict(self, include_predictions: bool = False) -> dict[str, Any]:
        """Converts the stored metrics into a dictionary format.

        Args:
            include_predictions: If True, includes `y_true` and `y_pred` lists.

        Returns:
            A dictionary containing the metrics, rounded for readability.
        """
        result: dict[str, Any] = {
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


def _load_model(
    config: RunConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads a causal language model and its tokenizer with 4-bit quantization.

    This function initializes a BitsAndBytes configuration to load the model in 4-bit
    precision, optimizing memory usage. It automatically maps the model across available
    GPUs based on the memory limits defined in the configuration. It also ensures the
    tokenizer uses the EOS token for padding.

    Args:
        config (ExperimentConfig): Experiment configuration dataclass.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizerBase]: A tuple containing:
            - model: The quantized language model loaded for causal LM.
            - tokenizer: The configured tokenizer.

    Raises:
        OSError: If the model name/path cannot be found on the Hugging Face Hub or
                 locally.
        ValueError: If the specified precision in the hardware config is invalid for
                    bitsandbytes.
    """

    # Create bits and bytes configuration # TODO: Pasar esto a configuración
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=config.hardware.precision,
    )

    # Determine maximum memmory per GPU
    n_gpus = config.hardware.n_gpus
    if n_gpus > 0:
        max_memory = {i: f"{config.hardware.max_memory_mb}MB" for i in range(n_gpus)}
    else:
        max_memory = None

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.fine_tuning.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.fine_tuning.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _get_linear_names(model: PreTrainedModel | torch.nn.Module) -> list[str]:
    """Find all linear layer names for LoRA targeting.

    Args:
        model: Model from which retrieve the linear names.

    Returns:
        The list of all the linear modules in the model.
    """
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def fine_tune(
    config: RunConfig,
    train_dataset: pd.DataFrame,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[PeftModel, float]:
    """
    Fine-tune a large language model using LoRA.

    Args:
        config: The experiment configuration object.
        train_dataset: Dataset over which the model will be fine-tuned.
        model: Model to be fine-tuned. If not specifyied, will use the one in config.
        tokenizer: Tokenizer for the model.

    Returns:
        Tuple containing the fine-tuned model and training time in seconds.
    """
    if model is None:
        model, tokenizer = _load_model(config)
    elif tokenizer is None:
        _, tokenizer = _load_model(config)  # TODO: custom models and tokenizers?

    # Prepare model for LoRA
    model.gradient_checkpointing_enable()
    peft_prepared_model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config.fine_tuning.lora.r,
        lora_alpha=config.fine_tuning.lora.lora_alpha,
        target_modules=_get_linear_names(peft_prepared_model),
        lora_dropout=config.fine_tuning.lora.lora_dropout,
        bias=config.fine_tuning.lora.bias,
        task_type=config.fine_tuning.lora.task_type,
    )

    peft_model = get_peft_model(peft_prepared_model, peft_config)
    assert isinstance(peft_model, PeftModel), "Failed to initialize PEFT model."
    peft_model.config.use_cache = False

    # Configure training
    training_args = TrainingArguments(
        output_dir=str(config.data.output_dir),
        per_device_train_batch_size=config.fine_tuning.per_device_batch_size,
        gradient_accumulation_steps=config.fine_tuning.gradient_accumulation_steps,
        max_steps=config.fine_tuning.max_steps,
        learning_rate=config.fine_tuning.learning_rate,
        warmup_steps=config.fine_tuning.warmup_steps,
        optim=config.fine_tuning.optim,
        logging_steps=config.fine_tuning.logging_steps,
        save_steps=config.fine_tuning.save_steps,
        fp16=(config.hardware.precision == torch.float16),
        bf16=(config.hardware.precision == torch.bfloat16),
    )

    trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Execute training
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Save and Return
    config.data.output_dir.mkdir(parents=True, exist_ok=True)

    if trainer.model is not None:
        assert isinstance(trainer.model, PeftModel), (
            "Model is not a PeftModel, this might cause a crash."
        )
        trainer.model.save_pretrained(str(config.data.output_dir))
    else:
        print("Model is None type, something went wrong during fine-tuning.")

    return peft_model, training_time


def evaluate_model(
    config: RunConfig,
    data: pd.DataFrame,
    model: PreTrainedModel | PeftModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> EvaluationMetrics:
    """
    Evaluate model with comprehensive metrics.

    Args:
        config: Experiment configuration
        data: Data over which evaluate the model.
        model: Base model. Defaults to whatever is defined in config

    Returns:
        EvaluationMetrics object with:
        - Precision, Recall, F1, Accuracy
        - Evaluation time in seconds

    """
    if model is None:
        model, tokenizer = _load_model(config)
    elif tokenizer is None:
        _, tokenizer = _load_model(config)  # TODO: custom models and tokenizers?

    device = config.hardware.device
    model.eval()

    y_true = []
    y_pred = []

    samples = min(len(data), config.fine_tuning.test_samples)
    data_subset = data.head(samples)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    start_time = time.time()

    for i in range(0, samples, config.fine_tuning.batch_size):
        batch_df = data_subset.iloc[i : i + config.fine_tuning.batch_size]

        # Inputs and ground truths for the whole batch
        input_texts = ["###Input:\n" + text for text in batch_df["input_text"]]
        expected_has_yes = ["yes" in text.lower() for text in batch_df["output_text"]]

        # Tokenize the batch (padding=True is required for batches)
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)

        # Get the length of the padded prompts to slice them out later
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(  # type: ignore
                **inputs,  # Passes both input_ids and attention_mask automatically
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        # Slice out the prompt (take only the newly generated tokens)
        generated_tokens = outputs[:, prompt_length:]

        # Decode the entire batch at once
        model_answers = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        # Extract metrics for the batch
        for expected, answer in zip(expected_has_yes, model_answers):
            y_true.append(expected)
            y_pred.append("yes" in answer.lower())

        # Progress tracking
        processed = min(i + config.fine_tuning.per_device_batch_size, samples)
        if (
            processed % (config.fine_tuning.per_device_batch_size * 5) == 0
            or processed == samples
        ):
            elapsed = time.time() - start_time
            print(
                f"Processed {processed}/{samples} samples... ({elapsed:.1f}s elapsed)"
            )

    # Restore the original padding side (in case you train/evaluate sequentially)
    tokenizer.padding_side = original_padding_side

    # End timing
    eval_time = time.time() - start_time

    return EvaluationMetrics.from_predictions(y_true, y_pred, eval_time=eval_time)


def create_results_table_png(
    results: dict[str, EvaluationMetrics],
    output_path: Path,
) -> None:
    """
    Generate a formatted results table and save it as a .png file.

    Args:
        results: A dictionary containing EvaluationMetrics for each model in the
                 experiment.
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
    config: RunConfig,
    results: dict[str, EvaluationMetrics],
) -> None:
    """Helper method to generate and save the JSON payload.

    TODO: Document
    """

    json_path = config.data.output_dir / config.fine_tuning.resuts_json_file

    results_dict: dict[str, Any] = {
        "experiment_info": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": config.fine_tuning.model_name,
            "model_alias": config.fine_tuning.model_alias,
            "train_samples": config.fine_tuning.train_samples,
            "training_steps": config.fine_tuning.max_steps,
            "epochs": config.fine_tuning.num_train_epochs,
        },
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    # TODO: implement logger
    print(f"JSON results saved to {json_path}")


def _save_summary_csv(
    config: RunConfig,
    results: dict[str, EvaluationMetrics],
) -> None:
    """Helper method to generate and save the summary CSV dataframe.

    Args:
        config: Experiment configuration.
        results: Evaluaiton metrics of each model in the experiment.
    """
    csv_path = config.data.output_dir / config.fine_tuning.summary_csv_file
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


def save_results(config: RunConfig, results: dict[str, EvaluationMetrics]) -> None:
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


def create_comparison_plot(config: RunConfig, results: dict[str, Any]) -> None:
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

    output_file = config.data.output_dir / config.fine_tuning.summary_plot_file

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.close(fig)

    # TODO: Implement logging
    print(f"Comparison plot saved to {output_file}")
