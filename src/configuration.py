import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class ModelConfig:
    """Configuration of the base model for fine-tuning.

    Attributes:
        name: Name of the model to be used (as huggingface path).
        alias: Alias for the model.
        context_window: LLM's context window.
    """

    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    alias: str = "LLaMA-3.2-1B"
    context_window: int = 512


@dataclass
class TrainingConfig:
    """Configuration for the training loop and hyperparameters.

    Attributes:
        train_samples: Number of samples for training.
        eval_samples: Number of samples for evaluation.
        test_samples: Number of samples for testing.
        training_steps: Number of training steps.
        batch_size: Batch size.
        gradient_accumulation_steps: Steps to accumulate gradients before updating.
        learning_rate: Peak learning rate for the optimizer.
        warmup_steps: Steps to increment the learning rate from 0 to learning_rate.
        logging_steps: Frequency of logging training metrics.
        save_steps: Frequency of saving model checkpoints.
        max_path_length: Maximum length of knowledge graph paths.
        pca_threshold: Confidence threshold for PCA (PCA-Confidence).
    """

    train_samples: int = 2000
    eval_samples: int = 2000
    test_samples: int = 1000
    training_steps: int = 500
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-3
    warmup_steps: int = 10
    logging_steps: int = 50
    save_steps: int = 100

    # TODO: Asegurarse de que va aquí
    pca_threshold: float = 0.3


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (LoRA) fine-tuning.

    Attributes:
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
    """

    lora_r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1


@dataclass
class GPUConfig:
    """GPU Settings to use in a LLM KG-based fine-tuning experiment.

    Attributes:
        n_gpus: Number of CUDA ready available GPUs.
        device: Which device to perform computations on.
        precision: Precision used by PyTorch tensors.
        max_memory_mb: Maximum memory allocation in megabytes.
    """

    n_gpus: int = torch.cuda.device_count()
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    precision: torch.dtype = torch.float16
    max_memory_mb: int = 40960


@dataclass
class DataConfig:
    """Configuration for dataset and output file paths.

    # TODO: Change paths and filenames.

    Attributes:
        data_dir: Base directory containing the knowledge graph data.
        rules_dir: Directory containing the logical rules.
        output_dir: Directory where outputs and checkpoints will be saved.
        kg_file: Filename of the knowledge graph data.
        kg_file_processed: Filename to store the processed knowledge graph file.
        relation_file: Filename of the mapping for relation IDs to relation names.
        results_file: Filename for the main JSON results.
        summary_csv: Filename for the CSV summary table.
        plot_file: Filename of the plotting comparison (if generated).
    """

    data_dir: Path = Path("/content/drive/MyDrive/KG-LLM/KG/LDM")
    rules_dir: Path = Path("rules")
    output_dir: Path = Path("/content/drive/MyDrive/KG-LLM/LDM/newoutput")
    kg_file: str = "train2id.txt"
    kg_file_processed: str = "train2id_processed.txt"
    relation_file: str = "/content/drive/MyDrive/KG-LLM/LDM/newoutput"
    results_file: str = "results.json"
    summary_csv: str = "results_summary.csv"
    plot_file: str = "plotting_result.png"
    max_path_length: int = 10


@dataclass
class RunConfig:
    """Toggable settings for the specific run configuration.

    Attributes:
        skip_base_eval: If True, skips evaluating the base model before fine-tuning.
        skip_baseline: If True, skips the standard baseline comparisons.
        save_predictions: Whether to save model predictions to disk.
        generate_plots: Whether to generate and save evaluation plots.
    """

    skip_base_eval: bool = False
    skip_baseline: bool = False
    save_predictions: bool = True
    generate_plots: bool = True


@dataclass
class ExperimentConfig:
    """Master configuration encompassing all experiment settings.

    Attributes:
        data: Configuration for paths and files.
        model: Base model configuration.
        training: Training loop parameters.
        lora: LoRA fine-tuning parameters.
        hardware: Compute and memory settings.
        run_settings: Execution configuration toggles.
    """

    # Using default factory ensuers new instances of sub-configs are created
    # for each ExperimentConfig, preventig shared state bugs.
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    hardware: GPUConfig = field(default_factory=GPUConfig)
    run_settings: RunConfig = field(default_factory=RunConfig)

    def setup_experiment(self) -> None:
        """Initializes system requirements based on the configuration.

        Creates necessary directories and prints initialization info.
        """
        os.makedirs(self.data.output_dir, exist_ok=True)

        ## TODO: Logging


@dataclass
class CoTGenerationConfig:
    """Configuration settings for generating CoT from KGs for training.

    Attributes:
        samples: Number of CoT samples to generate.
        max_path_length: Maximmum length of the paths in the KG.
        include_reasoning: Whether to include the reasoning process in natural language.
        use_rules: Whether to include the rules descriptions in the context in natural language.
        max_rules_in_context: Maximmum number of rules to be added to the prompt.
        max_attempts_multiplier: Multiplies by the number of samples to define the maximum number of attemps (prevents infinite loops).
    """

    samples: int = 1000
    max_path_length: int = 10
    include_reasoning: bool = True
    use_rules: bool = True
    max_rules_in_context: int = 3
    max_attempts_multiplier: int = 10
