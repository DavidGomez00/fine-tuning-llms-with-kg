"""Generates synthetic Knowledge Graphs using extensional data and Horn rules.

This module provides the core pipeline for triple generation, applying a set of graph
metrics and logical rules to produce a complete, synthetic N-Triples dataset.
"""

import itertools
import logging
from collections.abc import Iterator
from typing import TypedDict

from SPARQLWrapper import SPARQLWrapper

from skgg.core.queries import (
    SparqlBinding,
    build_rule_query,
    clear_graph_sparql,
    from_binding_row,
    get_existing_triples,
    insert_triples_gsp,
    insert_triples_sparql,
    run_select_query,
)
from skgg.core.rules import Atom, HornRule
from skgg.engine.metrics import PredicateProfile
from skgg.utils import format_triple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile update.
# ---------------------------------------------------------------------------
def decrement_counts(counts: dict[str, int], term: str) -> None:
    """Module-level private helper for managing frequency state."""
    if term in counts:
        counts[term] -= 1
        if counts[term] == 0:
            del counts[term]
    else:
        logger.warning(
            "Decrementing the count of %s when it does not exist in counter.", term
        )


def update_closed_preds(
    edb_profiles: dict[str, PredicateProfile],
    closed_preds: set[str],
) -> bool:
    """Module-level private helpter to update the state of the predicates. Returns True
    if new predicates are closed."""
    new = False
    for predicate, profile in edb_profiles.items():
        if profile.frequency <= 0 and predicate not in closed_preds:
            closed_preds.add(predicate)
            new = True

    return new


def is_assignment_solvable(profile: PredicateProfile, subject: str, obj: str) -> bool:
    """Checks if assigning a (subject, object) pair maintains graph solvability.

    Simulates the assigment to evaluate the Gale-Ryser / Havel-Hakimi conditions without
    mutating or copying the profile object.

    Args:
        profile: Predicate profile tracking available domains and ranges.
        subject: Subject term to be assigned.
        obj: Object term to be assigned.

    Returns:
        True if the assignment leaves the graph in a solvable state, False otherwise.
    """
    if subject not in profile.domain or obj not in profile.range:
        logger.warning("Bad triple: (%s, %s).\n%s", subject, obj, profile)
        return False

    s_domain_len = len(profile.domain) - (1 if profile.domain.get(subject) == 1 else 0)
    s_range_len = len(profile.range) - (1 if profile.range.get(obj) == 1 else 0)

    max_domain_freq = max(
        (count - 1 if k == subject else count for k, count in profile.domain.items()),
        default=0,
    )
    max_range_freq = max(
        (count - 1 if k == obj else count for k, count in profile.range.items()),
        default=0,
    )

    return max_domain_freq <= s_range_len and max_range_freq <= s_domain_len


# ---------------------------------------------------------------------------
# Create searchspace.
# ---------------------------------------------------------------------------
def create_searchspace(
    client: SPARQLWrapper,
    profiles: dict[str, PredicateProfile],
    term_mapping: dict[str, str],
    searchspace_uri: str,
) -> None:
    """Generates and inserts a searchspace into the DB using one or more predicates.

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
    logger.debug("Creating searchspace for %s", list(profiles.keys()))

    for predicate, profile in profiles.items():
        triple_generator: Iterator[str] = (
            format_triple(subj, predicate, obj, term_mapping)
            for subj, obj in itertools.product(
                profile.domain.keys(), profile.range.keys()
            )
        )
        insert_triples_gsp(
            graph_uri=searchspace_uri,
            triples=triple_generator,
            client=client,
        )


# ---------------------------------------------------------------------------
# Apply rules.
# ---------------------------------------------------------------------------
class GraphSources(TypedDict):
    target: str
    others: list[str]


def triples_from_bindings(
    bindings: list[SparqlBinding], atoms: list[Atom], term_mapping: dict[str, str]
) -> Iterator[str]:
    """Maps bindings to RDF formatted triples using the patterns in 'atoms'.

    Args:
        bindings: A list of SPARQL binding rows to evaluate.
        atoms: A list of body atoms providing the triple patterns.
        term_mapping: Mapping of terms to their string representations.

    Returns:
        A set of all possible formatted triple strings.
    """
    return (
        format_triple(
            subject=from_binding_row(atom.subject, binding_row)[0],
            predicate=atom.predicate,
            obj=from_binding_row(atom.obj, binding_row)[0],
            term_mapping=term_mapping,
        )
        for atom in atoms
        for binding_row in bindings
    )


def apply_rule(
    client: SPARQLWrapper,
    graph_uri: str,
    rule: HornRule,
    term_mapping: dict[str, str],
    chunk_size: int,
    profile: PredicateProfile | None = None,
) -> int:
    """Inserts novel triples generated from the rule to 'graph_uri'. If a profile is
    provided, restricts triple generation to profile constraints.

    Args:
        client: SPARQLWrapper client.
        graph_uri: URI of the graph where data is queried and inserted.
        rule: Rule represented as a Horn Rule.
        term_mapping: Mapping from a term to its corresponding prefix.
        chunk_size: Maximum number of triples to insert per SPARQL query.
        profile: Contains the constraints of the head predicate.

    Returns:
        Number of novel triples inserted to the graph.
    """

    # Retrieve bindings. If using profile, generates a searchspace.
    try:
        graph_sources: GraphSources = {
            "target": graph_uri,
            "others": [],
        }
        searchspace_uri = "http://Searchspace.org/"

        use_profile = profile is not None

        if use_profile and rule.head.predicate in rule.get_body_predicates():
            # Create a searchspace
            create_searchspace(
                client=client,
                profiles={rule.head.predicate: profile},
                term_mapping=term_mapping,
                searchspace_uri=searchspace_uri,
            )
            # Add the searchspace as a source
            graph_sources.update({"others": [searchspace_uri]})

        # Query the graph
        query = build_rule_query(rule=rule.signature, sources=graph_sources)
        if not (raw_bindings := run_select_query(client, query)):
            return 0

    # Clear the searchspace
    finally:
        clear_graph_sparql(client, searchspace_uri)

    # Get the candidate triples that already exist in the graph
    existing_triples = get_existing_triples(
        client=client,
        graph_uri=graph_uri,
        candidate_triples=triples_from_bindings(
            bindings=raw_bindings,
            atoms=[rule.head],
            term_mapping=term_mapping,
        ),
        term_mapping=term_mapping,
        chunk_size=chunk_size,
    )

    def filter_triples() -> Iterator[str]:
        """Helper generator. Yields novel and constraint-valid triples."""
        for triple in triples_from_bindings(
            bindings=raw_bindings,
            atoms=[rule.head],
            term_mapping=term_mapping,
        ):
            if triple in existing_triples:
                # logger.debug("%s already exists in the graph.", triple)
                continue

            if use_profile:
                subject, predicate, obj = triple.strip(" .").split(sep=" ")
                # logger.debug(
                #    "Subject: %s | Predicate: %s | Object: %s", subject, predicate, obj
                # )
                if (
                    profile.frequency <= 0
                    or profile.domain.get(subject, 0) <= 0
                    or profile.range.get(subject, 0) <= 0
                    or not is_assignment_solvable(profile, subject, obj)
                ):
                    # logger.debug("%s violates profile constraints.", triple)
                    continue

            yield triple

    # Yield triples that do not exist already in the graph
    return insert_triples_sparql(
        graph_uri=graph_uri,
        client=client,
        triple_stream=filter_triples(),
        chunk_size=chunk_size,
    )
