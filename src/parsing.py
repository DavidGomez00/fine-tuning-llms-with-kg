import csv
import logging
from pathlib import Path

from rules import format_triple

logger = logging.getLogger(__name__)


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
