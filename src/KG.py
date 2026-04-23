"""File for managing KG related functions."""
# TODO: fix docstrings

import logging
from collections import defaultdict
from pathlib import Path

from rdflib import Graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load KG from files
# ---------------------------------------------------------------------------


def deprecated_load_knowledge_graph(
    file_path: str | Path,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Loads a Knowledge Graph from a specifically formatted text file.

    The expected file format starts with a line indicating the number of edges,
    followed by lines with space-separated triples: `node1 node2 relation_id`.

    Args:
        file_path (Union[str, Path]): Path to file containing KG triples.

    Returns:
        Tuple[Dict[str, Dict[str, int]], List[str]]: A tuple containing:
            - graph: An adjacency dictionary where graph[node1][node2] = relation_id.
            - nodes: A list of all unique nodes found in the graph.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Knowledge Graph file not found: {file_path}")

    graph: dict[str, dict[str, int]] = defaultdict(dict)
    nodes: set[str] = set()

    with open(file_path, encoding="utf-8") as f:
        try:
            _ = int(f.readline().strip())
        except ValueError:
            logger.warning(
                "First line of '%s' is not a valid integer. Proceeding anyway.",
                file_path.name,
            )

        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split()

            # Skip empty lines
            if not parts:
                continue

            if len(parts) == 3:
                node1, node2, relation_str = parts
                nodes.update([node1, node2])

                try:
                    graph[node1][node2] = int(relation_str)
                except ValueError:
                    logger.warning(
                        "Invalid ID in line %d: '%s' is not an integer. Skipping."
                    )
            else:
                logger.warning(
                    "Malformed triple on line %d. Expected 3 parts, got %d.",
                    line_num,
                    len(parts),
                )

    # Convert the defaultdict back to a standard dict before returning
    # to prevent accidental empty key creations later.
    return dict(graph), list(nodes)


def load_knowledge_graph(kg_file: Path) -> Graph:
    """Loads a knowledge graph from file."""
    format = "nt" if kg_file.name.endswith(".nt") else "turtle"
    graph = Graph().parse(kg_file, format=format)
    logger.debug("Loaded %s knowledge grapth file.", kg_file.name)
    return graph


def parse_kg(input_file: Path, output_file: Path) -> int:
    """Preprocess Knowledge Graph data.

    The input file in tab-separated format is converted to space-separated format. The
    result is saved in `output_file`.

    Args:
        input_file: Path to the document to parse.
        output_file: Path to the file where the document is saved.

    Returns:
        int: The number of valid triples successfully processed and written.

    Raises:
        FileNotFoundError: If the input_file does not exist.
        IOError: If there is an issue reading or writting the files.
    """
    if not input_file.is_file():
        raise FileNotFoundError(f"The input file '{input_file}' does not exist.")

    with (
        open(input_file, encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        # Count valid lines without loading the whole file into memory
        valid_count = sum(1 for line in f_in if len(line.strip().split("\t")) == 3)

        # Write the header
        f_out.write(f"{valid_count}\n")

        # Reset the reading pointer back to the beggining of the input file
        f_in.seek(0)

        # Write the valid data
        for line_num, line in enumerate(f_in, 1):
            parts = line.strip("\t")
            if len(parts) == 3:
                f_out.write(
                    f"{parts[0]} {parts[1]} {parts[2]}\n"
                )  # TODO: igual mejor con str.join()
            else:
                # TODO: logging of incorrect lines??
                pass

    return valid_count


def create_relation_mapping(kg_file_path: Path, relations_file_path: Path) -> None:
    """Creates the relation mapping from a file defined in config.

    Extracts unique relations from `kg_file_path` and writes them into
    `relations_file_path` along with their numerical IDs.

    Args:
        kg_file_path: File containing the knowledge graph triples.
        relations_file_path: Destination file for the relation mapping.

    Raises:
        FileNotFoundError: If `kg_file_path` does not exist.
        ValueError: If the input file is empty or the first line isn't an integer.
    """
    if not kg_file_path.is_file():
        raise FileNotFoundError(f"The input file does not exist: {kg_file_path}")

    relations = set()
    expected_triples = 0

    with open(kg_file_path, encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"The file {kg_file_path} is completely empty.")

        try:
            expected_triples = int(first_line)
        except ValueError:
            logger.error(
                "The first line of %s must be the integer count of triples.",
                kg_file_path.name,
            )
            raise

        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split()
            if len(parts) == 3:
                relations.add(parts[2])
            elif parts:
                logger.warning(
                    "Malformed triple at line %d in %s: %s",
                    line_num,
                    kg_file_path,
                    line.strip(),
                )

    # Save the triples
    relations_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(relations_file_path, "w", encoding="utf-8") as f:
        f.write(f"{len(relations)}\n")
        for _, rel in enumerate(sorted(relations)):
            f.write(f"relation_{rel}\t{rel}\n")

    logger.info("Processed %d expected triples.", expected_triples)
    logger.info(
        "Found and saved %d unique relations to %s", len(relations), relations_file_path
    )


def load_id2relation_mapping(file_path: str | Path) -> dict[int, str]:
    """Loads a mapping of relation IDs to their natural language name from a file.

    The expected file format is a header line with the number of relations,
    followed by lines with tab-separated 'relation_name' and 'relation_id'.

    Args:
        file_path (Union[str, Path]): Path to dictionary file.

    Returns:
        Dict[int, str]: TAhe dictionary where keys are integer IDs and values
            are the natural language relation names.

    Raises:
        FileNotFoundError: If the dictionary file does not exist.
        ValueError: If there is an issue parsing the IDs into integers.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {file_path}")

    id2relation: dict[int, str] = {}

    with open(file_path, encoding="utf-8") as f:
        # Read and discard the first line (number of relations)
        _ = f.readline()

        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                relation, relation_id = parts
                try:
                    id2relation[int(relation_id)] = relation
                except ValueError:
                    logger.warning(
                        "Line %d: Could not parse ID '%s' as integer. Skipping.",
                        line_num,
                        relation_id,
                    )
            else:
                logger.warning(
                    "Line %d: Invalid format, expected 2 separated parts. Skipping.",
                    line_num,
                )

    return id2relation
