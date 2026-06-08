"""Configuration classes for the experiment and preprocessing settings."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from typing_extensions import Self
from yarl import URL


@dataclass
class FineTuningConfig:
    """Configuration for the training loop and hyperparameters."""

    # Fine-tuning experiment behavior
    skip_base_eval: bool = False
    skip_baseline: bool = False

    # Generated outputs
    results_json_file: str = "results.json"
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
class DataConfig:
    """Configuration for input and output directories."""

    input_dir: Path = Path(".data/")
    output_dir: Path = Path(".experiments/new_experiment")
    database_url: URL = URL("http://localhost:8890/")
    sparql_endpoint: str = "sparql-auth"
    crud_endpoint: str = "sparql-graph-crud-auth"

    def __post_init__(self) -> None:
        """Validate input and create output directories."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.database_url = URL(self.database_url)

        self._validate_path(self.input_dir, "input_dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_path(path: Path, field_name: str) -> None:
        """Check whether a path is valid."""
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration Error: The {field_name} does not exist at {path}"
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"Configuration Error: {field_name} at {path} is not a directory."
            )


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class GraphConfig:
    """Knowledge Graph settings.

    TODO: Docstrings
    """

    name: str
    ontology_file: str
    nt_file: str
    base_graph_uri: str
    synthetic_graph_uri: str


@dataclass(frozen=True)
class VirtuosoConfig:
    user: str = "dba"
    password: str = "dba"
    chunk_size: int = 5000


@dataclass(frozen=True)
class RulesConfig:
    """
    TODO: Docstrings
    """

    rules_file: str
    pca_threshold: float


@dataclass(frozen=True)
class LoggingConfig:
    """TODO: Docs"""

    level: int | str = logging.INFO


@dataclass
class RunConfig:
    """Master configuration object for the experiment run."""

    graph: GraphConfig
    rules: RulesConfig
    fine_tuning: FineTuningConfig | None
    cot_generation: CoTGenerationConfig | None
    virtuoso: VirtuosoConfig = field(default_factory=VirtuosoConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        """Loads a RunConfig from a JSON file."""

        def get_section(key: str, required: bool = False) -> dict[str, Any]:
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
                        f"Configuration Error: Missing mandatory section {key}"
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

        with open(json_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        virtuoso_config = VirtuosoConfig(**get_section("virtuoso", required=True))
        graph_config = GraphConfig(**get_section("graph", required=True))
        rules_config = RulesConfig(**get_section("rules", required=True))
        data_config = DataConfig(**get_section("data", required=True))
        hardware_config = HardwareConfig(**get_section("hardware"))
        logging_config = LoggingConfig(**get_section("logging"))

        # --- Optional attributes ---
        if "cot_generation" in data:
            cot_config = CoTGenerationConfig(**data["cot_generation"])
        else:
            # TODO: print
            cot_config = None

        if "fine_tuning" in data:
            fine_tuning = FineTuningConfig(**data["fine_tuning"])
        else:
            # TODO: print
            fine_tuning = None

        return cls(
            data=data_config,
            graph=graph_config,
            rules=rules_config,
            hardware=hardware_config,
            fine_tuning=fine_tuning,
            cot_generation=cot_config,
            logging=logging_config,
            virtuoso=virtuoso_config,
        )

    def __post_init__(self) -> None:
        """Validate config."""
        pass
