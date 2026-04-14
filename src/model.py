import time
from typing import Tuple

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from configuration import ExperimentConfig


def load_model(
    config: ExperimentConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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
        OSError: If the model name/path cannot be found on the Hugging Face Hub or locally.
        ValueError: If the specified precision in the hardware config is invalid for bitsandbytes.
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
        pretrained_model_name_or_path=config.model.name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def evaluate_model(
    config: ExperimentConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    data: pd.DataFrame,
    max_samples: int = 200,  # TODO: pasar a conf
):
    """
    Evaluate model with comprehensive metrics.

    Args:
        model: Pretrained model.
        tokenizer: Tokenizer for the model.
        data: Data over which evaluate the model.
        max_samples: Maximum samples to evaluate.
        device: Device in which perform computations

    Returns:
        EvaluationMetrics object with:
        - Precision, Recall, F1, Accuracy
        - Evaluation time in seconds
    """
    device = config.hardware.device
    model.eval()

    y_true = []
    y_pred = []

    samples = min(len(data), max_samples)

    # Start timing
    start_time = time.time()

    for idx, row in data.head(samples).iterrows():
        input_text = "###Input:\n" + row["input_text"]
        expected_has_yes = "yes" in row["output_text"].lower()

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(  # type: ignore
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sam+ple=False,
            )

        model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_has_yes = "yes" in model_answer.lower()

        y_true.append(expected_has_yes)
        y_pred.append(model_has_yes)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Processed {idx + 1}/{num_samples} samples... ({elapsed:.1f}s elapsed)"
            )

    # End timing
    eval_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0)

    metrics = EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        eval_size=num_samples,
        eval_time_seconds=eval_time,
        y_true=y_true,
        y_pred=y_pred,
    )

    return metrics


print("Enhanced evaluation function defined!")
