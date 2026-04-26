"""File for managing KG related functions."""

# TODO: Finish PredicateProfile and GraphMetrics
# TODO: load KG from CSV files?
# TODO: Do I need to throw errors if relation2id does not exists in this script?
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import RDF, Graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge Graph Metrics
# ---------------------------------------------------------------------------


@dataclass
class PredicateProfile:
    """Tracks subject and object frequency distributions for a specific predicate."""

    subjects: Counter[str] = field(default_factory=Counter)
    objects: Counter[str] = field(default_factory=Counter)


@dataclass
class GraphMetrics:
    """A structured container for RDF graph metrics and properties."""

    profiles: dict[str, PredicateProfile]
    total_triples: int
    unique_subjects: int
    unique_predicates: int
    unique_objects: int
    predicate_frequencies: dict[str, int]
    class_frecuencies: dict[str, int]


def get_kg_metrics(graph: Graph) -> GraphMetrics:
    """Calculates frequency and cardinality metrics for a graph.

    Args:
        kg_file: Path to file with KG triples.

    Returns:
        A GraphMetrics dataclass containing cardinalities and frequency distributions.
    """

    # Sets for distinct cardinality counting
    subjects: set[str] = set()
    objects: set[str] = set()

    # Counters and mappings
    pred_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    profiles: defaultdict[str, PredicateProfile] = defaultdict(PredicateProfile)

    total_triples = 0

    # Single pass through the graph
    for s, p, o in graph:
        s_str, p_str, o_str = str(s), str(p), str(o)

        total_triples += 1
        pred_counter[p_str] += 1

        subjects.add(s_str)
        objects.add(o_str)

        # Track instances of rdf:type for class distribution
        # NOTE: We check the original 'p' against the RDF.type URIRef for speed,
        # but store the string representation of 'o'
        if p == RDF.type:
            class_counter[o_str] += 1

        profiles[p_str].subjects[s_str] += 1
        profiles[p_str].objects[o_str] += 1

    return GraphMetrics(
        profiles=dict(profiles),
        total_triples=total_triples,
        unique_subjects=len(subjects),
        unique_predicates=len(pred_counter),
        unique_objects=len(objects),
        predicate_frequencies=dict(pred_counter),
        class_frecuencies=dict(class_counter),
    )


# ---------------------------------------------------------------------------
# Knowledge Graph Loading
# ---------------------------------------------------------------------------


def load_knowledge_graph(kg_file: Path) -> Graph:
    """Loads a knowledge graph to a rdflib.Graph."""
    format = "nt" if kg_file.name.endswith(".nt") else "turtle"
    graph = Graph().parse(kg_file, format=format)
    logger.debug("Loaded %s knowledge graph with %d triples", kg_file.name, len(graph))
    return graph


def load_id2relation_mapping(file_path: Path) -> dict[int, str]:
    """Loads a mapping of relation IDs to their natural language name from a file.

    Raises:
        FileNotFoundError: If the dictionary file does not exist.
        ValueError: If there is an issue parsing the IDs into integers.
    """
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
