"""Generates synthetic Knowledge Graphs using extensional data and Horn rules.

This module provides the core pipeline for triple generation, applying a set of graph
metrics and logical rules to produce a complete, synthetic N-Triples dataset.
"""

import itertools
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from SPARQLWrapper import DIGEST, JSON, POST, SPARQLWrapper

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
# Extensional database generation
# ---------------------------------------------------------------------------


def _build_triple(subject: str, predicate: str, obj: str, namespace: str) -> str:
    """Module-level private helper for formatting triples."""
    return f"<{namespace}{subject}> <{namespace}{predicate}> <{namespace}{obj}> ."


def _decrement_frequency(counts: dict[str, int], term: str) -> None:
    """Module-level private helper for managing frequency state."""
    if term in counts:
        counts[term] -= 1
        if counts[term] == 0:
            del counts[term]


def _gen_extensional_triples(
    predicate: str,
    profile: PredicateProfile,
    namespace: str = "http://example.org/",
) -> Iterator[str]:
    """Generates a random set of unique triples for a predicate, preserving the domain
    and range frequency distributions and the rate of reflexive triple instances.

    To ensure these metrics are maintained, the reflexive triples are generated first.
    Then, any subject which frequency matches the unique number of objects can be
    resolved directly by assigning it to each unique object in distinct triples.
    The same is true for objects.

    If none of the above conditions is met, we select a random subject and create as
    many unique triples as its frequency, then check again the remaining entities.

    Args:
        predicate: The predicate.
        profile: Profile containing the distribution of domain and range entities as
            Counter objects, as well as the reflexive triple count and predicate name.
        namespace: URI namespace applied to all subjects, predicates, and objects.

    Yields:
        Formatted triple strings in N-Triples format, e.g.:
        "<http://example.org/s> <http://example.org/p> <http://example.org/o> ."

    Raises:
        ValueError: If duplicate-free assignment is impossible due to frequency limits.
    """

    max_sub_freq = max(profile.domain.values(), default=0)
    if max_sub_freq > len(profile.range):
        raise ValueError(
            f"Can't create unique triples for '{predicate}': a subject appears "
            f"{max_sub_freq} times but only {len(profile.range)} unique objects exist"
        )

    max_obj_freq = max(profile.range.values(), default=0)
    if max_obj_freq > len(profile.domain):
        raise ValueError(
            f"Can't create unique triples for '{predicate}': an object appears "
            f"{max_obj_freq} times but only {len(profile.domain)} unique subjects exist"
        )

    # Avoid mutating the profiles
    p_domain = profile.domain.copy()
    p_range = profile.range.copy()
    reflexive_count = profile.reflexive

    logger.debug("Generating extensional triples for %s", predicate)

    # Ensure refelxivity first
    for subject in list(p_domain.keys()):
        if reflexive_count <= 0:
            break
        if subject in p_range:
            yield _build_triple(subject, predicate, subject, namespace)
            _decrement_frequency(p_domain, subject)
            _decrement_frequency(p_range, subject)
            reflexive_count -= 1

    while p_domain:
        progress_made = False

        # Check for direct matches on domain
        for subject, frequency in list(p_domain.items()):
            obj_choices = list(p_range.keys() - {subject})
            if frequency == len(obj_choices):
                for obj in obj_choices:
                    yield _build_triple(subject, predicate, obj, namespace)
                    _decrement_frequency(p_range, obj)
                del p_domain[subject]
                progress_made = True

        # Check for direct matches on range
        for obj, frequency in list(p_range.items()):
            subj_choices = list(p_domain.keys() - {obj})
            if frequency == len(subj_choices):
                for subject in subj_choices:
                    yield _build_triple(subject, predicate, obj, namespace)
                    _decrement_frequency(p_domain, subject)
                del p_range[obj]
                progress_made = True

        # Re-evaluate direct matches before falling back to random assignment
        if progress_made:
            continue

        # Random assignment fallback
        if p_domain:
            subject = random.choice(list(p_domain.keys()))
            required_count = p_domain[subject]
            available_objects = list(p_range.keys() - {subject})

            if len(available_objects) < required_count:
                raise ValueError(f"Error: Ran out of objects for '{predicate}'")

            logger.debug("No direct match, assigning objects to %s", subject)
            chosen_objects = random.sample(available_objects, required_count)

            for obj in chosen_objects:
                yield _build_triple(subject, predicate, obj, namespace)
                _decrement_frequency(p_range, obj)

            del p_domain[subject]


def gen_extensional_graph(
    predicate_profiles: dict[str, PredicateProfile],
    database_endpoint: str,
    namespace: str = "http://example.org/",
    graph_uri: str = "http://example.org/synthetic_graph",
    chunk_size: int = 5000,
) -> None:
    """Generates an extensional graph and inserts it directly into a SPARQL endpoint.

    Args:
        endpoint_url: The URL of the SPARQL endpoint (e.g., "http://localhost:8890/sparql").
        predicate_profiles: Mapping of predicates to their profile objects.
        namespace: The URI namespace.
        chunk_size: The maximum number of triples to insert in a single HTTP request.
    """
    sparql = SPARQLWrapper(database_endpoint)
    sparql.setHTTPAuth(DIGEST)
    sparql.setCredentials("dba", "dba")

    sparql.setMethod(POST)

    count = 0

    triple_stream = itertools.chain.from_iterable(
        _gen_extensional_triples(
            predicate=predicate,
            profile=profile,
            namespace=namespace,
        )
        for predicate, profile in predicate_profiles.items()
    )

    for chunk in _chunk_iterable(triple_stream, chunk_size):
        insert_query = f"INSERT DATA {{\n  GRAPH <{graph_uri}> {{\n"
        insert_query += "\n".join(chunk)
        insert_query += "\n  }\n}"

        sparql.setQuery(insert_query)
        sparql.query()

        count += len(chunk)
        logger.debug(
            "Inserted chunk of %d triples. Total so far: %d", len(chunk), count
        )

    logger.info("Extensional generation complete. Inserted %d total triples.", count)


# ---------------------------------------------------------------------------
# Intensional data generation
# ---------------------------------------------------------------------------
def _resolve_term(row_dict: dict[str, Any], term: str, namespace: str) -> str:
    if term.startswith("?"):
        val = row_dict.get(term.removeprefix("?"))
        return Atom._to_sparql(val, namespace) if val else ""
    else:
        return str(Atom._to_sparql(term, namespace))


def _gen_triple_from_results(
    rules: dict[str, HornRule],
    results: dict[str, Any],
    namespace: str,
) -> Iterator[str]:
    """Generates a sequence of triple instances from a quuery Result."""

    # Safely navigate the W3C SPARQL JSON structure
    bindings = results.get("results", {}).get("bindings", [])

    for result_row in bindings:
        row_dict: dict[str, str] = {
            k: v.get("value")
            for k, v in result_row.items()
            if v.get("value") is not None
        }

        rule_id = row_dict.pop("rule_id", "Unknown_rule")
        if rule_id not in rules:
            logger.warning("Found unknown rule_id: %s. Skipping row.", rule_id)
            continue

        rule_head = rules[rule_id].head
        triple_subject = _resolve_term(row_dict, rule_head.subject, namespace)
        triple_object = _resolve_term(row_dict, rule_head.obj, namespace)
        triple_predicate = Atom._to_sparql(rule_head.predicate, namespace)

        if not (triple_subject and triple_object):
            continue

        yield f"{triple_subject} {triple_predicate} {triple_object} ."


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


def _chunk_iterable(iterable: Iterable[str], size: int) -> Iterable[tuple[str, ...]]:
    """Yields successive chunks of a given size from an iterable."""
    iterator = iter(iterable)
    while chunk := tuple(itertools.islice(iterator, size)):
        yield chunk


def gen_intensional_graph(
    database_endpoint: str,
    rules: dict[str, HornRule],
    namespace: str,
    chunk_size: int = 5000,
    graph_uri: str = "http://example.org/synthetic_graph",
) -> None:
    """Generates an intensional database from an extensional database."""

    sparql = SPARQLWrapper(database_endpoint)
    sparql.setHTTPAuth(DIGEST)
    sparql.setCredentials("dba", "dba")

    ruleset_query = build_ruleset_query(rules, namespace)

    seen_triples: set[str] = set()
    iteration = 1

    while True:
        logger.debug("--- Starting deduction iteration %d ---", iteration)

        # Get the result from the rule set query
        sparql.setQuery(ruleset_query)
        sparql.setReturnFormat(JSON)
        sparql.setMethod(POST)  # Standard for potentially large queries

        results: dict[str, Any] = sparql.queryAndConvert()

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
        sparql.setMethod(POST)
        for chunk in _chunk_iterable(new_triples, chunk_size):
            insert_query = f"INSERT DATA {{\n  GRAPH <{graph_uri}> {{\n"
            insert_query += "\n".join(chunk)
            insert_query += "\n  }\n}"

            sparql.setQuery(insert_query)
            sparql.query()  # Execute the insert

        seen_triples.update(new_triples)
        logger.debug("Added %d new unique triples.", len(new_triples))

        iteration += 1


def create_synthetic_graph(
    database_endpoint: str,
    rules_df: pd.DataFrame,
    graph_metrics: GraphMetrics,
    namespace: str,
    pca_threshold: float,
) -> None:
    """Generates a new graph from a set of rules and metrics from the orginal graph.

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

    # Create the extensional database
    gen_extensional_graph(
        predicate_profiles={
            k: graph_metrics.predicates[k] for k in extensional_predicates
        },
        namespace=namespace,
        database_endpoint=database_endpoint,
    )

    # Create the intensional database
    gen_intensional_graph(
        database_endpoint=database_endpoint, rules=rules, namespace=namespace
    )


def test_create_synthetic_graph(
    config_file: Path,
    logging_level: int | str = logging.INFO,
) -> None:
    setup_logging(logging_level)

    # Load config
    config = RunConfig.from_json(config_file)

    # Load files
    kg_file_path = config.data.input_dir / config.kg.kg_file
    graph = load_knowledge_graph(kg_file_path)
    graph_metrics = get_kg_metrics(graph)
    rules_csv_path = config.data.input_dir / config.kg.rules_csv
    rules_df = pd.read_csv(rules_csv_path)

    # Run experiment
    create_synthetic_graph(
        database_endpoint="http://localhost:8890/sparql-auth",
        rules_df=rules_df,
        graph_metrics=graph_metrics,
        namespace=config.kg.namespace,
        pca_threshold=config.kg.pca_threshold,
    )


def test_gen_extensional_db(
    config_file: Path,
    logging_level: int | str = logging.INFO,
    extensional_graph_name: str = "extensional_graph.nt",
) -> None:
    setup_logging(logging_level)

    # Load config
    config = RunConfig.from_json(config_file)

    # Load files
    kg_file_path = config.data.input_dir / config.kg.kg_file
    graph = load_knowledge_graph(kg_file_path)
    graph_metrics = get_kg_metrics(graph)
    rules_csv_path = config.data.input_dir / config.kg.rules_csv
    rules_df = pd.read_csv(rules_csv_path)

    # Parse rules to Horn Rules and determine extensional predicates
    rules, intensional_predicates = parse_rule_set(rules_df, config.kg.pca_threshold)
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
    extensional_file = config.data.output_dir / extensional_graph_name

    # Create the extensional database
    gen_extensional_graph(
        predicate_profiles={
            k: graph_metrics.predicates[k] for k in extensional_predicates
        },
        namespace=config.kg.namespace,
        database_endpoint="http://localhost:8890/sparql-auth",
    )

    extensional_graph = load_knowledge_graph(extensional_file)
    logger.info("Extensional graph with %d triples.", len(extensional_graph))


if __name__ == "__main__":
    config_file = Path("configurations/gen_triples/moviesKG_config.json")
    logging_level = logging.INFO

    test_create_synthetic_graph(
        config_file=config_file,
        logging_level=logging_level,
    )
