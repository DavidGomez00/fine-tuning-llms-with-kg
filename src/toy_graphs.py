"""Generates a toy dataset to test other functionalities."""

import csv
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from rdflib import Graph, URIRef

logger = logging.getLogger(__name__)

# Extracted as an immutable constant
DEFAULT_MOVIE_TRIPLES: frozenset[tuple[str, str, str]] = frozenset(
    [
        ("Keanu_Reeves", "acted_in", "The_Matrix"),
        ("Carrie_Anne_Moss", "acted_in", "The_Matrix"),
        ("Laurence_Fishburne", "acted_in", "The_Matrix"),
        ("Lana_Wachowski", "directed", "The_Matrix"),
        ("The_Matrix", "has_genre", "SciFi"),
        ("The_Matrix", "has_genre", "Action"),
        ("Keanu_Reeves", "acted_in", "John_Wick"),
        ("Chad_Stahelski", "directed", "John_Wick"),
        ("John_Wick", "has_genre", "Action"),
        ("John_Wick", "has_genre", "Thriller"),
        ("Carrie_Anne_Moss", "acted_in", "Memento"),
        ("Christopher_Nolan", "directed", "Memento"),
        ("Memento", "has_genre", "Thriller"),
        ("Memento", "has_genre", "Mystery"),
        ("Keanu_Reeves", "knows", "Carrie_Anne_Moss"),
        ("Carrie_Anne_Moss", "knows", "Keanu_Reeves"),
        ("Keanu_Reeves", "knows", "Laurence_Fishburne"),
    ]
)


TOY_RULES: tuple[dict[str, str | int | float], ...] = (
    {
        "Body": "?actorA acted_in ?movie ?actorB acted_in ?movie",
        "Head": "?actorA worked_with ?actorB",
        "Head_Coverage": 1.0,
        "Std_Confidence": 1.0,
        "PCA_Confidence": 1.0,
        "Positive_Examples": 5,
        "Body_size": 2,
        "PCA_Body_size": 2,
        "Functional_variable": "?actorB",
    },
    {
        "Body": "?director directed ?movie ?actor acted_in ?movie",
        "Head": "?director directed_actor ?actor",
        "Head_Coverage": 1.0,
        "Std_Confidence": 1.0,
        "PCA_Confidence": 1.0,
        "Positive_Examples": 4,
        "Body_size": 2,
        "PCA_Body_size": 2,
        "Functional_variable": "?actor",
    },
    {
        "Body": "?actor acted_in ?movie ?movie has_genre ?genre",
        "Head": "?actor acts_in_genre ?genre",
        "Head_Coverage": 1.0,
        "Std_Confidence": 1.0,
        "PCA_Confidence": 1.0,
        "Positive_Examples": 6,
        "Body_size": 2,
        "PCA_Body_size": 2,
        "Functional_variable": "?genre",
    },
    {
        "Body": "?director directed ?movie ?movie has_genre ?genre",
        "Head": "?director directs_genre ?genre",
        "Head_Coverage": 1.0,
        "Std_Confidence": 1.0,
        "PCA_Confidence": 1.0,
        "Positive_Examples": 4,
        "Body_size": 2,
        "PCA_Body_size": 2,
        "Functional_variable": "?genre",
    },
    {
        "Body": "?personA knows ?personB ?personB knows ?personC",
        "Head": "?personA has_mutual_connection ?personC",
        "Head_Coverage": 1.0,
        "Std_Confidence": 1.0,
        "PCA_Confidence": 1.0,
        "Positive_Examples": 2,
        "Body_size": 2,
        "PCA_Body_size": 2,
        "Functional_variable": "?personC",
    },
    {
        "Body": "?director directed_actor ?co_star ?co_star worked_with ?actor",
        "Head": "?director warm_lead_for ?actor",
        "Head_Coverage": 1.0,
        "Std_Confidence": 1.0,
        "PCA_Confidence": 1.0,
        "Positive_Examples": 2,
        "Body_size": 2,
        "PCA_Body_size": 2,
        "Functional_variable": "?actor",
    },
)


def _infer_toy_triples(
    base_triples: Iterable[tuple[str, str, str]],
) -> set[tuple[str, str, str]]:
    """Infers new triples based on predefined domain rules using index lookups."""
    knowledge_graph = set(base_triples)
    new_triples_added = True

    while new_triples_added:
        initial_size = len(knowledge_graph)

        idx: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
        for subj, pred, obj in knowledge_graph:
            idx[pred].add((subj, obj))

        # 1. Co-Star Rule
        for a1, m1 in idx["acted_in"]:
            for a2, m2 in idx["acted_in"]:
                if m1 == m2 and a1 != a2:
                    knowledge_graph.add((a1, "worked_with", a2))

        # 2. Director-Actor Link
        for d, m1 in idx["directed"]:
            for a, m2 in idx["acted_in"]:
                if m1 == m2:
                    knowledge_graph.add((d, "directed_actor", a))

        # 3. Actor Genre Profiler
        for a, m1 in idx["acted_in"]:
            for m2, g in idx["has_genre"]:
                if m1 == m2:
                    knowledge_graph.add((a, "acts_in_genre", g))

        # 4. Director Genre Profiler
        for d, m1 in idx["directed"]:
            for m2, g in idx["has_genre"]:
                if m1 == m2:
                    knowledge_graph.add((d, "directs_genre", g))

        # 5. Network Expansion Rule
        for p1, p2 in idx["knows"]:
            for p3, p4 in idx["knows"]:
                if p2 == p3 and p1 != p4:
                    knowledge_graph.add((p1, "has_mutual_connection", p4))

        # 6. Casting Recommendation Rule
        for d, a1 in idx["directed_actor"]:
            for a2, a3 in idx["worked_with"]:
                if a1 == a2 and d != a3:
                    knowledge_graph.add((d, "warm_lead_for", a3))

        if len(knowledge_graph) == initial_size:
            new_triples_added = False

    return knowledge_graph


def create_example_graph(
    base_triples: Iterable[tuple[str, str, str]] | None = None,
    namespace: str = "http://example.org/",
) -> Graph:
    """Creates a fully materialized, simple graph example to test functionalities.

    Args:
        base_triples: Initial set of (subject, predicate, object) tuples. If None,
            a default movie-based dataset is used.
        namespace: The URI namespace to apply to all node strings.

    Returns:
        An rdflib.Graph instance containing both base and inferred triples.
    """
    if base_triples is None:
        base_triples = DEFAULT_MOVIE_TRIPLES

    inferred_triples = _infer_toy_triples(base_triples)
    graph = Graph()
    for subj, pred, obj in inferred_triples:
        graph.add(
            (
                URIRef(f"{namespace}{subj}"),
                URIRef(f"{namespace}{pred}"),
                URIRef(f"{namespace}{obj}"),
            )
        )

    logger.debug(
        "Generated %d new inferred triples. Total KG size: %d",
        len(inferred_triples) - len(list(base_triples)),
        len(graph),
    )
    return graph


def export_toy_dataset(output_dir: Path | str) -> None:
    """Exports both the materialized toy graph and its semantic rules.

    Args:
        output_dir: Path to the directory where the .nt and .csv files will be saved.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Export Graph
    graph_file = out_path / "complete_movies_KG.nt"
    toy_graph = create_example_graph()
    toy_graph.serialize(destination=str(graph_file), format="nt")
    logger.info("Exported toy graph to %s", graph_file)

    # 2. Export Rules CSV
    rules_file = out_path / "movies_rules.csv"
    headers = [
        "Body",
        "Head",
        "Head_Coverage",
        "Std_Confidence",
        "PCA_Confidence",
        "Positive_Examples",
        "Body_size",
        "PCA_Body_size",
        "Functional_variable",
    ]

    with open(rules_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(TOY_RULES)

    logger.info("Exported toy rules to %s", rules_file)


if __name__ == "__main__":
    # Example Usage:
    toy_graph = create_example_graph()
    toy_graph.serialize(destination="movies_KG.nt", format="nt", encoding="utf-8")
