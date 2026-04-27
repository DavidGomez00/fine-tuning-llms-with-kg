"""Configuration classes for the experiment and preprocessing settings."""

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

    # Generated outputs
    resuts_json_file: str = "results.json"
    summary_csv_file: str = "summary.csv"
    summary_plot_file: str = "experiment_plot.png"
    summary_table_file: str = "experiment_table.png"

    # Dataset generation for fine-tuning
    generate_datasets: bool = False
    train_with_rules_samples: int = 10
    train_without_rules_samples: int = 10
    test_samples: int = 5

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
    max_rules: int = 100
    max_groundings: int = 100
    rule_summary_file: str = "rules_summary.txt"


@dataclass
class DirConfig:
    """Configuration for base input and output directories."""

    input_dir: Path = Path(".data/")
    output_dir: Path = Path(".experiments/new_experiment")

    def __post_init__(self) -> None:
        """Validate input and create output directories."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

        self._validate_path(self.input_dir, "input_dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_path(path: Path, field_name: str) -> None:
        """Check wether a path is valid."""
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
    """Hardware settings.

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
    """Knowledge Graph settings.

    Attributes:
        kg_name: Not formated name of the knowledge graph.
        namespace: Namespace for the knowledge graph entities.
        namespace_prefix: Prefix for the knowledge graph entities' namespace.
        kg_file: Name of the knowledge graph formated file.
        relation_file: Name of the relation mapping file.
        rules_csv: Name of the CSV containing rules.
    """

    kg_name: str
    namespace: str
    namespace_prefix: str
    pca_threshold: float
    kg_file: str
    relation_file: str
    rules_csv: str


@dataclass
class RunConfig:
    """Master configuration object for the experiment run."""

    kg: KGConfig
    fine_tuning: FineTuningConfig | None
    cot_generation: CoTGenerationConfig | None
    data: DirConfig = field(default_factory=DirConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        """Loads a RunConfig from a JSON file."""

        def _get_config_section(key: str, required: bool = False) -> dict[str, Any]:
            """Fetches a section from JSON.

            Args:
                key: The JSON key to fetch.
                required: If True, raises a KeyError if the section is missing.

            Raises:
                KeyError: If a required section is missing from the JSON.
                ValueError: If the section exists but is not a JSON object (mapping).
            """
            if key not in data:
                if required:
                    raise KeyError(
                        f"Configuratoin Error: Missing mandatory section {key}"
                    )
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

        # Read JSON contents
        json_path = Path(json_path)
        try:
            with open(json_path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {json_path.absolute()}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {json_path.name}: {e}")
            raise

        kg_config = KGConfig(**_get_config_section("kg", required=True))
        dir_config = DirConfig(**_get_config_section("data"))
        hardware_config = HardwareConfig(**_get_config_section("hardware"))

        # --- Optional attributes ---
        if "cot_generation" in data:
            cot_config = CoTGenerationConfig(**data["cot_generation"])
        else:
            logger.debug("Optional section 'cot_generation' not defined. Skipping.")
            cot_config = None

        if "fine_tuning" in data:
            fine_tuning = FineTuningConfig(**data["fine_tuning"])
        else:
            logger.debug("Optional section 'fine_tuning' not defined. Skipping.")
            fine_tuning = None

        return cls(
            fine_tuning=fine_tuning,
            cot_generation=cot_config,
            data=dir_config,
            hardware=hardware_config,
            kg=kg_config,
        )

    def __post_init__(self) -> None:
        """Validate paths for input and output files."""

        logger.debug("Confifuration correctly initialized.")


if __name__ == "__main__":
    # Tets many possible errors in the configuration
    complete_config = Path("configurations/tests/complete.json")
    missing_cot = Path("configurations/tests/missing_cot.json")
    missing_data = Path("configurations/tests/missing_data.json")
    missing_ft = Path("configurations/tests/missing_ft.json")
    missing_hw = Path("configurations/tests/missing_hw.json")
    missing_kg = Path("configurations/tests/missing_kg.json")
