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
    build_federated_query,
    build_rule_query,
    get_select_results,
    insert_triples_sparql,
)
from rules import HornRule
from utils import format_triple

# ---------------------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------


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
        val = bindings_row.get(var_name, {}).get("value", term)
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
    sparql_client: SPARQLWrapper,
    rule: HornRule,
    graph_uri: str,
) -> Iterator[RawTriple]:
    """Applies a rule to a graph and retuns a raw triple Iterator that generates the
    resulted triples from the graph."""
    query = build_rule_query(rule.signature, graph_uri)
    raw_bindings = get_select_results(sparql_client, query)
    logger.debug("Retrieved %d results from rule %s.", len(raw_bindings), rule.rule_id)
    return iter_raw_triples({rule.rule_id: rule}, raw_bindings)


def apply_recursive_rule(
    sparql_client: SPARQLWrapper,
    rule: HornRule,
    graph_uri: str,
) -> SparqlBindings:
    """Applias a recursive rule to a graph to extend it."""
    query = build_federated_query(rule.signature, graph_uri, "http://SearchSpace.org/")
    raw_bindings = get_select_results(sparql_client, query)
    logger.debug("Retrieved %d results from rule %s.", len(raw_bindings), rule.rule_id)
    return raw_bindings


# ---------------------------------------------------------------------------
# Extensional Database (EDB) creation
# ---------------------------------------------------------------------------
def get_extensional_triples(
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


def gen_extensional_graph(
    sparql_client: SPARQLWrapper,
    predicate_profiles: dict[str, PredicateProfile],
    graph_uri: str,
    term_mapping: dict[str, str],
    chunk_size: int = 1000,
) -> None:
    """Generates an extensional graph and inserts it directly into a SPARQL endpoint.

    Args:
        predicate_profiles: Mapping of predicates to their profile objects.
        namespace: The URI namespace.
        chunk_size: The maximum number of triples to insert in a single HTTP request.
    """

    triple_stream = itertools.chain.from_iterable(
        get_extensional_triples(predicate, profile, term_mapping)
        for predicate, profile in predicate_profiles.items()
    )

    insert_triples_sparql(sparql_client, graph_uri, triple_stream, chunk_size)

    logger.info("EDB generated at <%s>.", graph_uri)
