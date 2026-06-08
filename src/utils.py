import csv
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class CustomLogicFormatter(logging.Formatter):
    """A formatter that applies custom logic to the final log string."""

    def format(self, record: logging.LogRecord) -> str:
        result = super().format(record)
        custom_output = result.replace("69", "69(<- nice)")
        return custom_output


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
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # custom_formatter = CustomLogicFormatter(
    #     fmt="%(asctime)s | %(name)-8s | %(levelname)-6s | %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    # root_logger = logging.getLogger()
    # for handler in root_logger.handlers:
    #     handler.setFormatter(custom_formatter)

    # Force urllib3 and its connectionpool child to be quiet
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Triple parsing
# ---------------------------------------------------------------------------
def format_term(
    term: str,
    term_mapping: dict[str, str] | None = None,
) -> str:
    """Ensures a term is wrapped in one set of brackets with the correct namespace."""
    if (term.startswith("<") and term.endswith(">")) or term.startswith("?"):
        return term

    if term.startswith("http"):
        return f"<{term}>"

    if term_mapping is not None:
        namespace = term_mapping.get(term, term_mapping.get("default"))
        if namespace is not None:
            return f"<{namespace}{term}>"

    message = "Default namespace not defined, aborting."
    raise ValueError(f"Error parsing term {term}: {message}")


def format_triple(
    subject: str,
    predicate: str,
    obj: str,
    term_mapping: dict[str, str],
) -> str:
    """Returns triple is in SPARQL format with the correct namespace and a final '.'."""
    subject_str = format_term(subject, term_mapping)
    predicate_str = format_term(predicate, term_mapping)
    object_str = format_term(obj, term_mapping)

    return f"{subject_str} {predicate_str} {object_str} ."


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


def tsv_to_nt(tsv_file: Path, nt_file: Path, term_mapping: dict[str, str]):
    """Parses a tsv file into a .nt file."""
    with tsv_file.open(encoding="utf-8") as tsv_f:
        with nt_file.open("w", encoding="utf-8") as nt_f:
            rd = csv.reader(tsv_f, delimiter="\t")
            for line in rd:
                if not line:
                    continue
                if len(line) == 3:
                    triple = format_triple(
                        line[0], line[1], line[2], term_mapping
                    ).strip()
                    nt_f.write(f"{triple}\n")

                else:
                    logger.error("Parsed line: %s", line)
                    raise ValueError("Error: Found != 3 elements in a non empty row.")

    logger.debug(".nt file saved at %s", nt_file)


if __name__ == "__main__":
    filter_rules(Path(".data/FrenchRoyalty/french_royalty.csv"), std_threshold=1)
