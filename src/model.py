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
from evaluation import EvaluationMetrics


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
) -> EvaluationMetrics:
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

    samples = min(len(data), config.test.samples)
    data_subset = data.head(samples)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    start_time = time.time()

    for i in range(0, samples, config.test.batch_size):
        batch_df = data_subset.iloc[i : i + config.test.batch_size]

        # Inputs and ground truths for the whole batch
        input_texts = ["###Input:\n" + text for text in batch_df["input_text"]]
        expected_has_yes = ["yes" in text.lower() for text in batch_df["output_text"]]
        # TODO: Asegurarme de que esta búsqueda de "yes" no tiene en cuenta el prompt.

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
        processed = min(i + config.test.batch_size, samples)
        if processed % (config.test.batch_size * 5) == 0 or processed == samples:
            elapsed = time.time() - start_time
            print(
                f"Processed {processed}/{samples} samples... ({elapsed:.1f}s elapsed)"
            )

    # Restore the original padding side (in case you train/evaluate sequentially)
    tokenizer.padding_side = original_padding_side

    # End timing
    eval_time = time.time() - start_time

    return EvaluationMetrics.from_predictions(y_true, y_pred, eval_time=eval_time)
