"""This file is dedicated to triple generation. Triple generation must generate a set of
triples following a set of rules, using cardinality values."""

import itertools
import logging
import random
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from config import DirConfig, KGConfig, RunConfig
from knowledge_graphs import GraphMetrics, get_kg_metrics, load_knowledge_graph
from rules import parse_body, parse_head

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configures the root logger to output to the console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def gen_triples(
    kg_config: KGConfig, data_config: DirConfig, kg_metrics: GraphMetrics
) -> Iterator[tuple[str, str, str]]:
    """Generates triples for a given set of rules and real KG metrics."""
    # Define the set of rules
    # TODO: Unify paths
    rules_csv_path = data_config.input_dir / kg_config.rules_csv
    rules_df = pd.read_csv(rules_csv_path)

    # Take 1 as an example
    # TODO: delete or config
    rules_df = rules_df.head(1)

    # Retrieve the set of predicates in the rule set to reduce scope
    body_predicates: set[str] = set()
    for body in rules_df["Body"]:
        for atom in sorted(parse_body(body)):
            body_predicates.add(atom.predicate)
    head_predicates: set[str] = set()
    for head in rules_df["Head"]:
        head_predicates.add(parse_head(head).predicate)

    predicates_in_rules: set[str] = body_predicates | head_predicates

    # 1. Pre-compute predicate population and weights
    pred_population: list[str] = list(predicates_in_rules)
    pred_weights: list[int] = list(
        kg_metrics.predicate_frequencies[pred] for pred in pred_population
    )

    # 2. Pre-compute localized subject/object distributions per predicate
    # Structure: {predicate_str: ((subj_pop, subj_weights), (obj_pop, obj_weights))}
    distributions: dict[
        str, tuple[tuple[list[str], list[int]], tuple[list[str], list[int]]]
    ] = {}
    for p in pred_population:
        subjects: list[str] = kg_metrics.profiles[p].subjects.keys()
        subject_values: list[int] = kg_metrics.profiles[p].subjects.values()
        objects: list[str] = kg_metrics.profiles[p].objects.keys()
        object_values: list[int] = kg_metrics.profiles[p].objects.values()
        distributions[p] = ((subjects, subject_values), (objects, object_values))

    # 3. Stream the generation
    for _ in range(kg_metrics.total_triples):
        # It handles weighted probabilities automatically via the standard C library
        p = random.choices(pred_population, weights=pred_weights, k=1)[0]

        # Retrieve the pre-computed subject and object distributions for this predicate
        (s_pop, s_weights), (o_pop, o_weights) = distributions[p]

        s = random.choices(s_pop, weights=s_weights, k=1)[0]
        o = random.choices(o_pop, weights=o_weights, k=1)[0]

        yield (s, p, o)


def generate_intentional_database(
    graph_metrics: GraphMetrics,
    output_file: Path,
    rules_df: pd.DataFrame,
    namespace: str = "http://example.org/",
    ns_prefix: str = "ex",
) -> None:
    """Generates the intentional database from the real graph metrics.

    Args:
        graph_metrics: GraphMetrics object with the characteristics of the real KG.

    Returns:
        A dict mapping each predicate to a set of generated triples (s, p, o).
    """

    # Select only those predicates that are not present in any rule head
    all_head_predicates: set[str] = set()

    # For the moment consider all predicates in the graph that are not in the rule
    # heads. TODO: Should we include predicates not present in the rules?

    for row in rules_df.itertuples(index=False):
        all_head_predicates.add(parse_head(str(row.Head)).predicate)

    intentional_predicates = graph_metrics.predicates.keys() - all_head_predicates

    logger.debug(
        "Loaded %d unique predicates, from which %d are intentional.",
        len(graph_metrics.predicates.keys() | all_head_predicates),
        len(intentional_predicates),
    )

    # Generate intentional graph with the intentional predicates
    intentional_graph: list[str] = []
    for predicate in intentional_predicates:
        key = f"{predicate}"
        p_domain = graph_metrics.predicates[key].domain
        p_range = graph_metrics.predicates[key].range

        # Possible pairs in p
        possible_pairs = list(itertools.product(p_domain, p_range))
        pairs = random.sample(
            possible_pairs,
            graph_metrics.predicates[key].frecuency,
        )

        # Generate triples
        triples = "\n".join(
            f"<{namespace}{subject}> <{ns_prefix}:{predicate}> <{namespace}{object}> ."
            for subject, object in pairs
        )
        intentional_graph.append(triples)
    intentional_graph_text = "\n".join(intentional_graph)
    output_file.write_text(intentional_graph_text, encoding="utf-8")


if __name__ == "__main__":
    # Set up logger
    setup_logging()

    # Load config
    config_file = Path("configurations/generate_triples_fr.json")
    config = RunConfig.from_json(config_file)

    # Load files
    kg_file_path = config.data.input_dir / config.kg.kg_file
    graph = load_knowledge_graph(kg_file_path)
    kg_metrics = get_kg_metrics(graph)
    rules_csv_path = config.data.input_dir / config.kg.rules_csv
    rules_df = pd.read_csv(rules_csv_path)

    # Test gen_triples
    intentional_graph = config.data.output_dir / "intentional_graph.nt"
    generate_intentional_database(kg_metrics, intentional_graph, rules_df)
