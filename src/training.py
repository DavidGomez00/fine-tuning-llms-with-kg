import time

import bitsandbytes as bnb
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

from evaluation import EvaluationMetrics
from finetuning_config import ExperimentConfig


def _load_model(
    config: ExperimentConfig,
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
        pretrained_model_name_or_path=config.model.name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
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
    config: ExperimentConfig,
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
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=_get_linear_names(peft_prepared_model),
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )

    peft_model = get_peft_model(peft_prepared_model, peft_config)
    assert isinstance(peft_model, PeftModel), "Failed to initialize PEFT model."
    peft_model.config.use_cache = False

    # Configure training
    training_args = TrainingArguments(
        output_dir=str(config.data.output_dir),
        per_device_train_batch_size=config.train.per_device_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        max_steps=config.train.max_steps,
        learning_rate=config.train.learning_rate,
        warmup_steps=config.train.warmup_steps,
        optim=config.train.optim,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
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
    config: ExperimentConfig,
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
