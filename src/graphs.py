"""File for managing KG related functions."""

import logging
import re
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

    domain: Counter[str] = field(default_factory=Counter)
    range: Counter[str] = field(default_factory=Counter)
    frequency: int = 0
    reflexive: int = 0


@dataclass
class GraphMetrics:
    """A structured container for RDF graph metrics and properties."""

    predicates: dict[str, PredicateProfile]
    total_triples: int
    class_frequencies: dict[str, int]


uri_pattern = re.compile(r"[#\/:]")


def get_kg_metrics(graph: Graph) -> GraphMetrics:
    """Calculates frequency and cardinality metrics for a graph.

    Args:
        kg_file: Path to file with KG triples.

    Returns:
        A GraphMetrics dataclass containing cardinalities and frequency distributions.
    """

    def _get_local_name(term: str) -> str:
        """Helper function to resolve URIs."""
        return uri_pattern.split(term)[-1]

    # Counters and mappings
    class_counter: Counter[str] = Counter()
    predicates: defaultdict[str, PredicateProfile] = defaultdict(PredicateProfile)
    total_triples = 0

    # Single pass through the graph
    for s, p, o in graph:
        s_str = _get_local_name(str(s))
        p_str = _get_local_name(str(p))
        o_str = _get_local_name(str(o))

        total_triples += 1
        predicates[p_str].frequency += 1
        predicates[p_str].domain[s_str] += 1
        predicates[p_str].range[o_str] += 1

        if s_str == o_str:
            logger.debug(
                "Getting the Graph metrics I found that %s is refelxive:\n%s\n%s",
                p_str,
                s_str,
                o_str,
            )
            predicates[p_str].reflexive += 1

        # Track instances of rdf:type for class distribution
        # NOTE: We check the original 'p' against the RDF.type URIRef for speed,
        # but store the string representation of 'o'
        if p == RDF.type:
            class_counter[o_str] += 1

    metrics = GraphMetrics(
        predicates=dict(predicates),
        total_triples=total_triples,
        class_frequencies=dict(class_counter),
    )

    reflexive_preds = 0
    for _, profile in predicates.items():
        if profile.reflexive > 0:
            reflexive_preds += 1

    logger.debug(
        "Loaded graph metrics for %d predicates, from which %d are reflexive.",
        len(predicates),
        reflexive_preds,
    )
    return metrics


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
