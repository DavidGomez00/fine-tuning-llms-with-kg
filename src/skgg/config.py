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

    # Base URL of the database instance
    database_url: URL = URL("http://localhost:8890/")

    # Path/suffix for SPARQL endpoint
    sparql_endpoint: str = "sparql"
    crud_endpoint: str = "sparql-graph-crud-auth"

    def get_full_sparql_url(self) -> str:
        """Returns the full SPARQL endpoint for SPARQLWrapper."""
        return str(self.database_url / self.sparql_endpoint)

    def __post_init__(self) -> None:
        """Validate input and create output directories."""
        self.input_dir = Path(self.input_dir)
        self.database_url = URL(self.database_url)

        self._validate_path(self.input_dir, "input_dir")

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
class DatabaseAuthConfig:
    """Database authentication and batching settings."""

    user: str | None = "dba"
    password: str | None = "dba"

    # Supported: "DIGEST" (Virtuoso default), "BASIC" (GraphDB default), or "NONE"
    auth_type: Literal["DIGEST", "BASIC", "NONE"] = "DIGEST"
    chunk_size: int = 5000


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
    """Knowledge Graph settings: file locations and the named-graph URIs used to
    key each stage of the pipeline (see AGENTS.md's "Architecture" section for how
    base/complete/EDB/synthetic relate).

    Attributes:
        name: Human-readable name for the graph/experiment.
        ontology_file: Filename (relative to `data.input_dir`) of the ontology
            (.ttl) used to build the term-to-namespace mapping.
        nt_file: Filename (relative to `data.input_dir`) of the base graph in
            N-Triples format, consumed by `cli/upload.py`.
        namespace: Default namespace URI used to resolve unprefixed terms.
        base_uri: Named-graph URI for the raw, uploaded base graph.
        complete_uri: Named-graph URI for the base graph after rule-based
            completion (`engine/completion.py`) — this is the source graph that
            metrics are extracted from.
        edb_uri: Named-graph URI for the generated Extensional Database.
        synthetic_uri: Named-graph URI for the final synthetic graph (EDB + IDB).
    """

    name: str
    ontology_file: str
    nt_file: str
    namespace: str
    base_uri: str
    complete_uri: str
    edb_uri: str
    synthetic_uri: str


@dataclass(frozen=True)
class RulesConfig:
    """Settings for loading and filtering the Horn rule set.

    Attributes:
        rules_file: Filename (relative to `data.input_dir`) of the rules CSV,
            parsed by `core.rules.parse_rule_set`.
        pca_threshold: Minimum PCA confidence a rule must have to be classified
            "POSITIVE" (kept); rules below it are classified "NEGATIVE" and
            excluded from generation. `None` skips filtering.
    """

    rules_file: str
    pca_threshold: float


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings for an experiment run.

    Attributes:
        level: Root logger level, as an int (e.g. `logging.DEBUG`) or a level
            name string (e.g. `"DEBUG"`). Passed to `utils.setup_logging`.
    """

    level: int | str = logging.INFO


@dataclass
class RunConfig:
    """Master configuration object for the experiment run."""

    graph: GraphConfig
    rules: RulesConfig
    fine_tuning: FineTuningConfig | None
    cot_generation: CoTGenerationConfig | None
    db_config: DatabaseAuthConfig = field(default_factory=DatabaseAuthConfig)
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

        db_config = DatabaseAuthConfig(**get_section("db_config", required=True))
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
            db_config=db_config,
        )

    def __post_init__(self) -> None:
        """Validate config."""
        pass
