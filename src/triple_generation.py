"""Generates synthetic Knowledge Graphs using extensional data and Horn rules.

This module provides the core pipeline for triple generation, applying a set of graph
metrics and logical rules to produce a complete, synthetic N-Triples dataset.
"""

import logging
import random
import shutil
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib.query import Result, ResultRow

from config import RunConfig
from graphs import (
    GraphMetrics,
    PredicateProfile,
    get_kg_metrics,
    load_knowledge_graph,
)
from rules import (
    Atom,
    HornRule,
    build_ruleset_query,
    get_uninferrable_predicates,
    parse_rule_set,
)

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
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _gen_triple_instances(
    predicate: str,
    profile: PredicateProfile,
    namespace: str = "http://example.org/",
) -> Iterator[str]:
    """Generates a sequence of random instances for a predicate.

    Args:
        predicate: The predicate string.
        profile: Profile with informaiton about the predicate in the original Graph.
        namespace: The URI namespace.

    Yields:
        Formatted triple strings in N-Triples/Turtle style format.
    """
    subjects = list(profile.domain.elements())
    objects = list(profile.range.elements())

    random.shuffle(objects)

    for subject, obj in zip(subjects, objects, strict=True):
        yield f"<{namespace}{subject}> <{namespace}{predicate}> <{namespace}{obj}> ."


def _gen_triple_from_results(
    rules: dict[str, HornRule], results: Result, namespace: str
) -> Iterator[str]:
    """Generates a sequence of triple instances from a quuery Result.

    Args:
        results: Query Result object.
        namespace: Namespace URI.

    Yields:
        Formatted triple strings in N-Triples/Turtle style format.
    """
    for result_row in results:
        assert isinstance(result_row, ResultRow)

        row_dict: dict[str, Any] = {
            str(var_name): result_row[var_name]
            for var_name in result_row.labels
            if result_row[var_name] is not None
        }

        # Identify the rule
        rule_id = str(row_dict.pop("rule_id", "Unknown_rule"))
        # Safety guard: Prevent KeyError if the query didn't bind the rule_id properly
        if rule_id not in rules:
            logger.warning("Found unknown rule_id: %s. Skipping row.", rule_id)
            continue
        rule_head = rules[rule_id].head

        # Get the subject of the new triple
        if rule_head.subject.startswith("?"):
            raw_subj = row_dict.get(rule_head.subject.removeprefix("?"))
            if raw_subj is None:
                continue
            triple_subject = Atom.to_sparql(raw_subj, namespace)
        else:
            triple_subject = Atom.to_sparql(rule_head.subject, namespace)

        # Get the object of the new triple
        if rule_head.obj.startswith("?"):
            raw_obj = row_dict.get(rule_head.obj.removeprefix("?"))
            if raw_obj is None:
                continue
            triple_object = Atom.to_sparql(raw_obj, namespace)
        else:
            triple_object = Atom.to_sparql(rule_head.obj, namespace)

        # NOTE: We are not contemplating vars in predicates
        triple_predicate = Atom.to_sparql(rule_head.predicate, namespace)

        yield f"{triple_subject} {triple_predicate} {triple_object} ."


def _generate_graph_file(
    predicate_profiles: dict[str, PredicateProfile],
    output_file: Path,
    namespace: str = "http://example.org/",
) -> None:
    """Generates a graph file directly to disk to minimize memory footprint.

    Args:
        predicate_profiles: Mapping of predicates to their profile objects.
        output_file: Pathlib object defining the target file.
        namespace: The URI namespace.
    """
    count = 0
    # Stream directly to the file to save memory
    with output_file.open("w", encoding="utf-8") as f:
        for predicate, profile in predicate_profiles.items():
            triple_generator = _gen_triple_instances(
                predicate=predicate,
                profile=profile,
                namespace=namespace,
            )

            for triple in triple_generator:
                count += 1
                f.write(f"{triple}\n")

    logger.info("Created %s with %d triples.", output_file, count)


def _check_uninferrable_preds(
    rules: dict[str, HornRule],
    intensional_predicates: set[str],
    extensional_predicates: set[str],
) -> None:
    """Calls the method to see if there is any uninferrable intensional predicate in the
    rule set.

    Args:
        rules: Dict containing all rules in the set.
        intensional_predicates: Set of all predicates that must be inferrable.
        extensional_predicates: Set of extensional predicates assumed to be inferrable.

    Raises:
        ValueError: If there are any non-inferrable predicates.
    """

    rule_mapping: dict[str, list[set[str]]] = defaultdict(list)
    for _, rule in rules.items():
        head_predicate = rule.head.predicate
        if head_predicate in intensional_predicates:
            body_intensional = (
                rule.signature.get_body_predicates() - extensional_predicates
            )
            rule_mapping[head_predicate].append(body_intensional)

    uninferrable_preds = get_uninferrable_predicates(
        rule_mapping=rule_mapping,
        intensional_predicates=intensional_predicates,
    )

    if uninferrable_preds:
        raise ValueError(
            f"Rule set not inferrable under complete rule assumption. "
            f"The following predicates cannot be deduced: {uninferrable_preds}"
        )
    else:
        logger.debug("All intensional predicates can be successfully deduced.")


def create_synthetic_graph(
    rules_df: pd.DataFrame,
    graph_metrics: GraphMetrics,
    output_path: Path,
    namespace: str,
    pca_threshold: float,
) -> None:
    """Generates a synthetic graph from a set of rules and metrics.

    First, a new graph is created containing a random extensional database. Then, it
    uses the rules to generate the rest of the triples to complete the graph.

    Args:
        rules_df: DataFrame containing information about rules per row.
        graph_metrics: Predicates' frequency and distribution.
        output_path: Path to store the generated extensional graph and final synthetic
        database.
        namespace: Namespace URI.
        pca_threshold: Value to determine if a rule is positive or negative.
    """
    logger.info(
        "Creating synthetic graph using %d rules and %d predicate profiles.",
        len(rules_df),
        len(graph_metrics.predicates),
    )

    # Parse rules to Horn Rules and determine extensional predicates
    rules, intensional_predicates = parse_rule_set(rules_df, pca_threshold)
    extensional_predicates = graph_metrics.predicates.keys() - intensional_predicates
    logger.info(
        "Found %d extensional predicates out of %d total predicates.",
        len(extensional_predicates),
        len(graph_metrics.predicates),
    )
    if len(extensional_predicates) == 0:
        raise ValueError("Cannot proceed with 0 extensional predicates.")

    # Check that the predicates can be deduced under complete rule assumption
    _check_uninferrable_preds(
        rules=rules,
        intensional_predicates=intensional_predicates,
        extensional_predicates=extensional_predicates,
    )

    extensional_file = output_path / "extensional_graph.nt"
    synthetic_file = output_path / "syntheticKG.nt"

    # Create the extensional database
    _generate_graph_file(
        predicate_profiles={
            k: graph_metrics.predicates[k] for k in extensional_predicates
        },
        namespace=namespace,
        output_file=extensional_file,
    )

    # Initialize synthtic graph with the extensional DB
    shutil.copy(str(extensional_file), str(synthetic_file))
    logger.debug("Copied extensional graph to initialize the synthetic graph.")

    # Deduce the intensional triples
    extensional_graph = load_knowledge_graph(extensional_file)
    seen_triples: set[str] = set()
    with open(str(synthetic_file), encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped:
                seen_triples.add(line_stripped)

    with open(str(synthetic_file), "a", encoding="utf-8") as of:
        iteration = 1

        while True:
            logger.debug("--- Starting deduction iteration %d ---", iteration)

            # Get results from rule queries
            ruleset_query = build_ruleset_query(rules, namespace)
            results = extensional_graph.query(ruleset_query)

            # Parse results into triples
            deduced_triples: set[str] = set(
                _gen_triple_from_results(rules, results, namespace)
            )

            # If there is nothing new to add, break the loop
            new_triples = deduced_triples - seen_triples
            logger.debug("Found %d new triples to add to the graph.", len(new_triples))
            if not new_triples:
                logger.info(
                    "Generation complete. Reached fixed point at %d total triples.",
                    len(seen_triples),
                )
                break

            # Write new triples to file and uodate the graph
            chunk_to_parse = ""
            for triple_str in new_triples:
                of.write(f"{triple_str}\n")
                seen_triples.add(triple_str)
                chunk_to_parse += f"{triple_str}\n"

            # Flush the buffer to write the triples in the .nt file
            of.flush()

            # Graph's parse method handles duplicates and allows to query next iter
            # without reloading the graph.
            extensional_graph.parse(data=chunk_to_parse, format="nt")

            logger.debug("Added %d new unique triples.", len(new_triples))
            iteration += 1


if __name__ == "__main__":
    setup_logging()

    # Load config
    config_file = Path("configurations/gen_triples/simpleKG_config.json")
    config = RunConfig.from_json(config_file)

    # Load files
    kg_file_path = config.data.input_dir / config.kg.kg_file
    graph = load_knowledge_graph(kg_file_path)
    graph_metrics = get_kg_metrics(graph)
    rules_csv_path = config.data.input_dir / config.kg.rules_csv
    rules_df = pd.read_csv(rules_csv_path)

    # Run experiment
    create_synthetic_graph(
        rules_df=rules_df,
        graph_metrics=graph_metrics,
        output_path=config.data.output_dir,
        namespace=config.kg.namespace,
        pca_threshold=config.kg.pca_threshold,
    )
