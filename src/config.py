"""Configuration classes for the experiment and preprocessing settings."""

# TODO: Arreglar documentación

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from typing_extensions import Self


@dataclass
class FineTuningConfig:
    """Configuration for the training loop and hyperparameters."""

    # Generated outputs
    resuts_json_file: str = "results.json"
    summary_csv_file: str = "summary.csv"
    summary_plot_file: str = "experiment_plot.png"
    summary_table_file: str = "experiment_table.png"

    # Training loop parameters
    per_device_batch_size: int = 1
    num_train_epochs: float = 1.0
    max_steps: int = 500
    learning_rate: float = 2e-3
    warmup_steps: int = 10
    optim: str = "paged_adamw_8bit"
    gradient_accumulation_steps: int = 4
    logging_steps: int = 50
    save_steps: int = 100

    # Base model configuration
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    model_alias: str = "LLaMA-3.2-1B"
    context_window: int = 512

    # Generate datasets for fine-tuning
    generate_datasets: bool = False
    train_samples: int = 2000
    eval_samples: int = 2000
    test_samples: int = 1000

    # Fine-tuning experiment behavior
    skip_base_eval: bool = False
    skip_baseline: bool = False
    save_predictions: bool = True
    generate_plots: bool = True
    generate_table: bool = True

    # LoRA Config
    lora_r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_bias: Literal["none", "lora_only", "all"] = "none"
    lora_task_type: str = "CAUSAL_LM"


@dataclass
class CoTGenerationConfig:
    """Configuration settings for generating CoT from KGs for training."""

    # Behavior for generating CoTs
    max_samples: int = 100
    max_path_length: int = 10
    include_reasoning: bool = True
    use_rules: bool = True
    max_rules_in_context: int = 3
    attempts_multiplier: int = 10


@dataclass
class DirConfig:
    """Configuration for base directories and output files."""

    input_dir: Path = Path("/.data/")
    output_dir: Path = Path("/.experiments/")
    rules_dir: Path = Path("rules")  # TODO: determine if this is relevant


@dataclass
class HardwareConfig:
    """Hardware settings to use in a LLM KG-based fine-tuning experiment.

    Attributes:
        n_gpus: Number of CUDA ready available GPUs.
        device: Which device to perform computations on.
        precision: Precision used by PyTorch tensors.
        max_memory_mb: Maximum memory allocation in megabytes.
    """

    n_gpus: int = field(default_factory=lambda: torch.cuda.device_count())
    device: Literal["gpu", "cpu"] = field(
        default_factory=lambda: "gpu" if torch.cuda.is_available() else "cpu"
    )
    precision: torch.dtype = torch.float16
    max_memory_mb: int = 40960


@dataclass
class KGConfig:
    """Configuration for KG related files and paths."""

    kg_name: str = "KG"
    raw_kg_file: str = "train2id.txt"
    kg_file: str = "train2id_processed.txt"
    relation_file: str = "relation2id"
    namespace: str = "example.org"
    namespace_prefix: str = "ex"


@dataclass
class RulesConfig:
    """Configuration for rule files and paths."""

    rules_csv: Path = Path("rules.csv")
    pca_threshold: float = 0.5
    max_rules: int | None = None


@dataclass
class RunConfig:
    """Master configuration object for the experiment run."""

    fine_tuning: FineTuningConfig | None
    cot_generation: CoTGenerationConfig | None
    data: DirConfig = field(default_factory=DirConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    kg: KGConfig = field(default_factory=KGConfig)

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        """Loads a RunConfig from a JSON file."""
        # Read JSON contents
        path_obj = Path(json_path)
        try:
            with open(path_obj, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {path_obj.absolute()}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format in {path_obj.name}: {e}")
            raise

        _precision_mapping: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        data_cfg = data.get("data", {})
        hw_cfg = data.get("hardware", {})
        rules_cfg = data.get("rules", {})

        # --- Non-optional attributes ---
        # TODO: error if missing!
        kg_config = KGConfig(**data["kg"])

        dir_config = DirConfig(
            input_dir=Path(data_cfg.get("input_dir", "/.data/")),
            output_dir=Path(data_cfg.get("output_dir", "/.experiments/")),
            rules_dir=Path(data_cfg.get("rules_dir", "rules")),
        )

        hardware_config = HardwareConfig(
            n_gpus=hw_cfg.get("n_gpus", torch.cuda.device_count()),
            device=hw_cfg.get("device", "gpu" if torch.cuda.is_available() else "cpu"),
            precision=_precision_mapping.get(
                hw_cfg.get("precision", ""), torch.float16
            ),
            max_memory_mb=hw_cfg.get("max_memory_mb", 40960),
        )

        rules_config = RulesConfig(
            rules_csv=Path(rules_cfg.get("rules_csv", "rules.csv")),
            pca_threshold=rules_cfg.get("pca_threshold", 0.5),
            max_rules=rules_cfg.get("max_rules"),
        )

        # --- Optional attributes ---
        cot_config = (
            CoTGenerationConfig(**data["cot_generation"])
            if "cot_generation" in data
            else None
        )
        fine_tuning = (
            FineTuningConfig(**data["fine_tuning"]) if "fine_tuning" in data else None
        )

        return cls(
            fine_tuning=fine_tuning,
            cot_generation=cot_config,
            data=dir_config,
            hardware=hardware_config,
            rules=rules_config,
            kg=kg_config,
        )
