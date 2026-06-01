"""File for managing KG related functions."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper

from queries import (
    get_domain,
    get_preds_and_freqs,
    get_range,
    get_reflexivity,
    get_total_triples,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge Graph Metrics
# ---------------------------------------------------------------------------
@dataclass
class PredicateProfile:
    """Tracks subject and object frequency distributions for a specific predicate."""

    domain: dict[str, int] = field(default_factory=dict)
    range: dict[str, int] = field(default_factory=dict)
    frequency: int = 0
    reflexivity: int = 0


@dataclass
class GraphMetrics:
    """A structured container for RDF graph metrics and properties."""

    profiles: dict[str, PredicateProfile]
    triple_count: int

    @classmethod
    def from_uri(cls, client: SPARQLWrapper, graph_uri: str) -> "GraphMetrics":
        """Instantiates GraphMetrics by delegating aggregation to the SPARQL endpoint.

        Scales efficiently by querying distributions per-predicate, avoiding massive
        data transfers and database ResultSetMaxRows limits.
        """

        triple_count = get_total_triples(client, graph_uri)
        logger.debug(
            "Retrieving metrics from %s with %d triples.", graph_uri, triple_count
        )
        profiles: dict[str, PredicateProfile] = {}

        predicates = get_preds_and_freqs(client, graph_uri) or {}

        for predicate, frequency in predicates.items():
            if not predicate.startswith("<"):
                predicate = f"<{predicate}>"

            reflexivity = get_reflexivity(client, graph_uri, predicate)
            domain = get_domain(client, graph_uri, predicate)
            p_range = get_range(client, graph_uri, predicate)

            profiles[predicate] = PredicateProfile(
                frequency=frequency,
                domain=domain,
                range=p_range,
                reflexivity=reflexivity,
            )

        logger.debug("Loaded DB metrics for %d predicates.", len(profiles))

        for predicate, profile in profiles.items():
            if "?f" in profile.domain.keys():
                raise ValueError(f"Error ?f en {predicate} domain.")
        #     logger.debug(
        #         "Predicate %s (freq %d | reflx %d)",
        #         predicate,
        #         profile.frequency,
        #         profile.reflexivity,
        #     )
        #     logger.debug("Domain:")
        #     for subject, freq in profile.domain.items():
        #         logger.debug("%s | %d", subject, freq)
        #     logger.debug("Range:")
        #     for obj, freq in profile.range.items():
        #         logger.debug("%s | %d", obj, freq)

        return cls(profiles=profiles, triple_count=triple_count)

    @classmethod
    def from_rdflib(cls, graph: Graph) -> "GraphMetrics":
        """Calculates frequency and cardinality metrics for a graph.

        Args:
            kg_file: Path to file with KG triples.

        Returns:
            GraphMetrics dataclass containing cardinalities and frequency distributions.
        """

        # Counters and mappings
        profiles: dict[str, PredicateProfile] = defaultdict(PredicateProfile)
        triple_count = 0

        # Single pass through the graph
        for s, p, o in graph:
            s_str = str(s)
            p_str = f"<{str(p)}>"
            o_str = str(o)

            triple_count += 1
            profiles[p_str].frequency += 1
            profiles[p_str].domain[s_str] += 1
            profiles[p_str].range[o_str] += 1

            if s_str == o_str:
                profiles[p_str].reflexivity += 1

        metrics = GraphMetrics(
            profiles=dict(profiles),
            triple_count=triple_count,
        )

        reflexive_preds = 0
        for _, profile in profiles.items():
            if profile.reflexivity > 0:
                reflexive_preds += 1

        logger.debug("Loaded graph metrics for %d predicates.", len(profiles))
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
