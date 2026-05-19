"""Implements functionalities for loading, fine-tuning and evaluating models, as well as
generating summaries and comparison figures from the results."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import bitsandbytes as bnb
import pandas as pd
import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from config import DirConfig, FineTuningConfig, HardwareConfig

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
# Evaluation
# ---------------------------------------------------------------------------


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


def evaluate_model(
    ft_config: FineTuningConfig,
    data: pd.DataFrame,
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> EvaluationMetrics:
    """
    Evaluate model with comprehensive metrics.

    Args:
        config: Experiment configuration
        data: Data over which evaluate the model.
        model: Model to be evaluated.

    Returns:
        EvaluationMetrics object with:
        - Precision, Recall, F1, Accuracy
        - Evaluation time in seconds

    """
    model.eval()

    y_true = []
    y_pred = []

    samples = min(len(data), ft_config.test_samples)
    data = data.head(samples)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    start_time = time.time()

    for i in range(0, samples, ft_config.per_device_batch_size):
        batch_df = data.iloc[i : i + ft_config.per_device_batch_size]

        # Inputs and ground truths for the whole batch
        # TODO: Modify to build the prompt from data
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

        # TODO: Modify for new training
        for expected, answer in zip(expected_has_yes, model_answers, strict=True):
            y_true.append(expected)
            y_pred.append("yes" in answer.lower())

    # Restore the original padding side (in case you train/evaluate sequentially)
    tokenizer.padding_side = original_padding_side

    # End timing
    eval_time = time.time() - start_time

    return EvaluationMetrics.from_predictions(y_true, y_pred, eval_time=eval_time)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    ft_config: FineTuningConfig, hw_config: HardwareConfig
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

    # Create bits and bytes configuration # TODO: Ver si hace falta configurar BnB
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=config.hardware.precision,
    # )

    # Determine maximum memmory per GPU
    if hw_config.n_gpus > 0:
        max_memory = {
            i: f"{hw_config.max_memory_mb}MB" for i in range(hw_config.n_gpus)
        }
    else:
        max_memory = None

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=ft_config.model_name,
        # quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )

    tokenizer = AutoTokenizer.from_pretrained(ft_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Model Fine-tuning
# ---------------------------------------------------------------------------


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
    ft_config: FineTuningConfig,
    data_config: DirConfig,
    hw_config: HardwareConfig,
    train_dataset: pd.DataFrame,
    model: PreTrainedModel = None,
    tokenizer: PreTrainedTokenizerBase = None,
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
    # Prepare model for LoRA
    model.gradient_checkpointing_enable()
    # peft_prepared_model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=ft_config.lora_r,
        lora_alpha=ft_config.lora_alpha,
        target_modules=_get_linear_names(model),
        lora_dropout=ft_config.lora_dropout,
        bias=ft_config.lora_bias,
        task_type=ft_config.lora_task_type,
    )

    peft_model = get_peft_model(model, peft_config)
    assert isinstance(peft_model, PeftModel), "Failed to initialize PEFT model."
    peft_model.config.use_cache = False

    # Configure training
    training_args = TrainingArguments(
        output_dir=str(data_config.output_dir),
        per_device_train_batch_size=ft_config.per_device_batch_size,
        gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
        max_steps=ft_config.max_steps,
        learning_rate=ft_config.learning_rate,
        warmup_steps=ft_config.warmup_steps,
        optim=ft_config.optim,
        logging_steps=ft_config.logging_steps,
        save_steps=ft_config.save_steps,
        fp16=(hw_config.precision == torch.float16),
        bf16=(hw_config.precision == torch.bfloat16),
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
    data_config.output_dir.mkdir(parents=True, exist_ok=True)

    if trainer.model is not None:
        assert isinstance(trainer.model, PeftModel), (
            "Model is not a PeftModel, this might cause a crash."
        )
        trainer.model.save_pretrained(str(data_config.output_dir))
    else:
        print("Model is None type, something went wrong during fine-tuning.")

    return peft_model, training_time
