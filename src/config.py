"""Configuration classes for the experiment and preprocessing settings."""

# TODO: Arreglar documentación
# TODO: Is DirConfig.rules_dir used?
# TODO: Mix rules and KG?

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for the training loop and hyperparameters."""

    # Fine-tuning experiment behavior
    skip_base_eval: bool = False
    skip_baseline: bool = False
    save_predictions: bool = True
    generate_plots: bool = True
    generate_table: bool = True

    # Generated outputs
    resuts_json_file: str = "results.json"
    summary_csv_file: str = "summary.csv"
    summary_plot_file: str = "experiment_plot.png"
    summary_table_file: str = "experiment_table.png"

    # Dataset generation for fine-tuning
    generate_datasets: bool = False
    train_samples: int = 2000
    eval_samples: int = 2000
    test_samples: int = 1000

    # Base model configuration
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    model_alias: str = "LLaMA-3.2-1B"
    context_window: int = 512

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

    input_dir: Path = Path(".data/")
    output_dir: Path = Path(".experiments/")
    rules_dir: Path = input_dir / "rules"

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.rules_dir = Path(self.rules_dir)

        self._validate_path(self.input_dir, "input_dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_path(path: Path, field_name: str) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration Error: The {field_name} does not exist at {path}"
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"Configuration Error: The {field_name} at {path} is not a directory."
            )


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
    precision: str = "float16"
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
    rules_csv: Path = Path("rules.csv")
    pca_threshold: float = 0.5
    max_rules: int | None = None
    rule_summary_file: str = "rules_summary.txt"


@dataclass
class RunConfig:
    """Master configuration object for the experiment run."""

    fine_tuning: FineTuningConfig | None
    cot_generation: CoTGenerationConfig | None
    data: DirConfig = field(default_factory=DirConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
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

        def _get_config_section(key: str) -> dict[str, Any]:
            """Fetches a section from JSON or returns an empty dict with logging."""
            if key not in data:
                logger.debug(
                    "Configuration section '%s' not found; using defaults.", key
                )
                return {}

            section = data[key]
            if not isinstance(section, dict):
                raise ValueError(
                    f"Expected '{key}' to be a mapping, got {type(section).__name__}"
                )
            return section

        kg_config = KGConfig(**_get_config_section("kg"))
        dir_config = DirConfig(**_get_config_section("data"))
        hardware_config = HardwareConfig(**_get_config_section("hardware"))

        # --- Optional attributes ---
        if "cot_generation" in data:
            cot_config = CoTGenerationConfig(**data["cot_generation"])
        else:
            logger.debug("Optional section 'cot_generation' omitted.")
            cot_config = None

        if "fine_tuning" in data:
            fine_tuning = FineTuningConfig(**data["fine_tuning"])
        else:
            logger.debug("Optional section 'fine_tuning' omitted.")
            fine_tuning = None

        return cls(
            fine_tuning=fine_tuning,
            cot_generation=cot_config,
            data=dir_config,
            hardware=hardware_config,
            kg=kg_config,
        )

    def __post_init__(self) -> None:
        logger.debug("Confifuration correctly instantiated.")
