import logging
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Force urllib3 and its connectionpool child to be quiet
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Triple parsing
# ---------------------------------------------------------------------------

uri_pattern = re.compile(r"[#\/:]")


def get_local_name(term: str) -> str:
    """Returns the term without the URI."""
    return uri_pattern.split(term)[-1]


def format_term(
    term: str,
    term_mapping: dict[str, str],
    default_ns: str = "http://ExplicitFormat.org",
) -> str:
    """Ensures a term is wrapped in one set of brackets with the correct namespace."""
    if (term.startswith("<") and term.endswith(">")) or term.startswith("?"):
        return term

    if term.startswith("http"):
        return f"<{term}>"

    namespace = term_mapping.get(term)

    if namespace is None:
        namespace = default_ns
        logger.warning("Prefix not found for term '%s'. Using default namespace", term)

    # Assume it's a local name and add namespace + brackets
    return f"<{namespace}{term}>"


def format_triple(
    subject: str,
    predicate: str,
    obj: str,
    term_mapping: dict[str, str],
    default_ns: str = "http://ExplicitFormat.org",
) -> str:
    """Returns triple is in SPARQL format with the correct namespace and a final '.'."""
    subject_str = format_term(subject, term_mapping, default_ns)
    predicate_str = format_term(predicate, term_mapping, default_ns)
    object_str = format_term(obj, term_mapping, default_ns)

    return f"{subject_str} {predicate_str} {object_str} ."


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


def filter_rules(rules_file: Path, pca_threshold: float) -> None:
    rules_df = pd.read_csv(rules_file)
    filtered_df = rules_df[rules_df["PCA_Confidence"] >= 0.9]
    filtered_df.to_csv(
        rules_file.with_name(f"rules_PCA_{pca_threshold}.csv"), index=False
    )
