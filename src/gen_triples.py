"""Generates synthetic Knowledge Graphs using extensional data and Horn rules.

This module provides the core pipeline for triple generation, applying a set of graph
metrics and logical rules to produce a complete, synthetic N-Triples dataset.
"""

import itertools
import logging
import random
from collections.abc import Iterator
from typing import Any, NamedTuple

from SPARQLWrapper import SPARQLWrapper

from graph_metrics import PredicateProfile
from queries import (
    SparqlBindings,
    build_rule_query,
    get_select_results,
    insert_triples_gsp,
    insert_triples_sparql,
)
from rules import HornRule
from utils import format_triple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic triple generation
# ---------------------------------------------------------------------------
class RawTriple(NamedTuple):
    """Represents an extracted, unformatted triple."""

    predicate: str
    subject_val: str
    object_val: str
    object_type: str


def from_binding_row(term: str, bindings_row: dict[str, Any]) -> tuple[str, str]:
    """Safely extracts a term from a single binding row."""
    if term.startswith("?"):
        var_name = term.lstrip("?")
        val = bindings_row.get(var_name, {}).get("value", var_name)
        v_type = bindings_row.get(var_name, {}).get("type", "uri")
        return val, v_type
    return term, "uri"


def iter_raw_triples(
    rules: dict[str, HornRule], bindings: list[dict[str, Any]]
) -> Iterator[RawTriple]:
    """Extracts raw triples from SPARQL bindings based on rule definitions."""

    shuffled_bindings = list(bindings)
    random.shuffle(shuffled_bindings)

    for bindings_row in shuffled_bindings:
        rule_id = bindings_row.get("rule_id", {}).get("value")
        if not rule_id or rule_id not in rules:
            logger.warning("Results are expected to have a rule_id binding. Skipping.")
            continue

        rule = rules[rule_id]

        s_val, _ = from_binding_row(rule.head.subject, bindings_row)
        o_val, o_type = from_binding_row(rule.head.obj, bindings_row)

        if not s_val or not o_val:
            continue

        yield RawTriple(
            predicate=rule.head.predicate,
            subject_val=s_val,
            object_val=o_val,
            object_type=o_type,
        )


def apply_rule(
    client: SPARQLWrapper,
    rule: HornRule,
    graph_uri: str,
    term_mapping: dict[str, str],
    profile: PredicateProfile,
    crud_endpoint: str,
) -> SparqlBindings:
    """Applies a rule to a graph and retuns a raw triple Iterator that generates the
    resulted triples from the graph."""

    searchspace_uri = "http://SearchSpace.org/"

    if rule.head.predicate in rule.get_body_predicates():
        create_predicate_searchspace(
            predicate=rule.head.predicate,
            profile=profile,
            searchspace_uri=searchspace_uri,
            term_mapping=term_mapping,
            client=client,
            crud_endpoint=crud_endpoint,
        )

    query = build_rule_query(rule.signature, graph_uri, searchspace_uri)
    return get_select_results(client, query)


# ---------------------------------------------------------------------------
# Extensional Database (EDB) creation
# ---------------------------------------------------------------------------
def triple_iterator(
    predicate: str,
    profile: PredicateProfile,
    term_mapping: dict[str, str],
) -> Iterator[str]:
    """Generates a random set of unique triples for a predicate, preserving the domain
    and range frequency distributions and the rate of reflexive triple instances.

    To ensure these metrics are maintained, the reflexive triples are generated first.
    Then, any subject which frequency matches the unique number of objects can be
    resolved directly by assigning it to each unique object in distinct triples.
    The same is true for objects.

    If none of the above conditions is met, a random subject is selected and generated
    until it's frequency is closed and the loop starts again untill no more triples can
    be generated.

    Args:
        predicate: The predicate value.
        profile: Profile containing the properties of the predicate.
        term_mapping: Mapping from a term to its namespace.

    Yields:
        Formatted triple strings in N-Triples format, e.g.:
        "<http://example.org/s> <http://example.org/p> <http://example.org/o> ."

    Raises:
        ValueError: If duplicate-free assignment is impossible due to frequency limits.
    """

    def _decrement_frequency(counts: dict[str, int], term: str) -> None:
        """Module-level private helper for managing frequency state."""
        if term in counts:
            counts[term] -= 1
            if counts[term] == 0:
                del counts[term]

    # Fail fast if it is not possible to retrieve unique triples
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
    reflexive_count = profile.reflexivity

    # Ensure refelxivity first
    for subject in list(p_domain.keys()):
        if reflexive_count <= 0:
            break
        if subject in p_range:
            yield format_triple(subject, predicate, subject, term_mapping)
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
                    yield format_triple(subject, predicate, obj, term_mapping)
                    _decrement_frequency(p_range, obj)
                del p_domain[subject]
                progress_made = True

        # Check for direct matches on range
        for obj, frequency in list(p_range.items()):
            subj_choices = list(p_domain.keys() - {obj})
            if frequency == len(subj_choices):
                for subject in subj_choices:
                    yield format_triple(subject, predicate, obj, term_mapping)
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

            chosen_objects = random.sample(available_objects, required_count)

            for obj in chosen_objects:
                yield format_triple(subject, predicate, obj, term_mapping)
                _decrement_frequency(p_range, obj)

            del p_domain[subject]


def gen_graph(
    client: SPARQLWrapper,
    predicate_profiles: dict[str, PredicateProfile],
    graph_uri: str,
    term_mapping: dict[str, str],
    chunk_size: int = 1000,
) -> None:
    """Generates a graph following the predicate distributions and inserts it into a DB
    through SPARQL.

    Args:
        client: SPARQL client.
        predicate_profiles: Mapping from the predicate to its distributions.
        graph_uri: Graph URI.
        term_mapping: Mapping from terms to their prefix.
        chunk_size: Maximum number of triples to insert per SPARQL query.
    """

    # Use a chain of iterators since we are passing an iterator for each predicate
    triple_stream = itertools.chain.from_iterable(
        triple_iterator(predicate, profile, term_mapping)
        for predicate, profile in predicate_profiles.items()
    )

    count = insert_triples_sparql(client, graph_uri, triple_stream, chunk_size)

    logger.info(
        "Graph generated at <%s> with %d triples for %d predicates.",
        graph_uri,
        count,
        len(predicate_profiles),
    )


def create_predicate_searchspace(
    client: SPARQLWrapper,
    predicate: str,
    profile: PredicateProfile,
    term_mapping: dict[str, str],
    searchspace_uri: str,
    crud_endpoint: str,
) -> None:
    """Generates and inserts a predicate search space into the database.

    Creates all possible triples for the given predicate using the cartesian
    product of the domain and range entities, and inserts them into a specific
    named graph using batching to ensure scalability.

    Args:
        database_endpoint: The URL of the SPARQL database endpoint.
        predicate: The URI of the predicate to link subjects and objects.
        profile: A profile object containing `domain` and `range` Counters.

    Returns:
        The URI of the generated search space named graph.

    Raises:
        Exception: If a SPARQL insertion batch fails.
    """

    count = len(profile.domain) * len(profile.range)

    logger.info("Creating searchspace for %s with %d triples", predicate, count)

    triple_generator: Iterator[str] = (
        format_triple(subj, predicate, obj, term_mapping)
        for subj, obj in itertools.product(profile.domain.keys(), profile.range.keys())
    )

    insert_triples_gsp(
        graph_uri=searchspace_uri,
        triples=triple_generator,
        client=client,
        crud_endpoint=crud_endpoint,
    )
