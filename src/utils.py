import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configures the root logger to output to the console.

    Args:
        level: The logging level to set. Accepts standard logging integers
               (e.g., logging.DEBUG) or strings (e.g., "INFO", "DEBUG").
    """
    if isinstance(level, str):
        level = level.upper()

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-8s | %(levelname)-6s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Force urllib3 and its connectionpool child to be quiet
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# File parsing.
# ---------------------------------------------------------------------------
def filter_file(
    input_file: Path,
    target_string: str,
    output_filename: str | None,
) -> None:
    """
    Writes lines from input_file that don't contain target_string to the output file.
    """
    output_file = input_file
    if output_filename is not None:
        output_file = input_file.with_name(output_filename)

    with (
        open(input_file, encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        count_removed = 0
        for line in infile:
            if target_string in line:
                count_removed += 1
            else:
                outfile.write(line)


def filter_rules(
    rules_file: Path | str, pca_threshold: float = 0.0, std_threshold: float = 0.0
) -> None:

    rules_file = Path(rules_file)
    rules_dataframe = pd.read_csv(rules_file)

    if std_threshold:
        metric = "Std"
        threshold = std_threshold
    else:
        metric = "PCA"
        threshold = pca_threshold

    filtered_df = rules_dataframe[rules_dataframe[f"{metric}_Confidence"] >= threshold]
    filtered_df.to_csv(
        rules_file.with_name(f"rules_{metric}_{threshold}.csv"), index=False
    )


if __name__ == "__main__":
    filter_rules(Path(".data/FrenchRoyalty/french_royalty.csv"), std_threshold=1)
