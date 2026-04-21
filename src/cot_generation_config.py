from dataclasses import dataclass
from pathlib import Path


@dataclass
class RuleToNLForCoTGenerationConfig:
    """TODO: docs"""

    rules_csv: Path = Path("rules.csv")
    kg_file: Path = Path("FrenchRoyalty.nt")
    kg_name: str = "KG"
    output_dir: Path = Path("outputs")
    namespace: str = "example.org"
    namespace_prefix: str = "ex"
    pca_threshold: float = 0.5
    max_rules: int | None = None
