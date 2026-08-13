import copy
import logging
import random
import uuid
from collections.abc import Iterator

from SPARQLWrapper import SPARQLWrapper

from kg_synth.core.queries import (
    SparqlBinding,
    build_rule_query,
    clear_graph_sparql,
    from_binding_row,
    get_existing_triples,
    initialize_graph,
    insert_triples_sparql,
    run_select_query,
)
from kg_synth.core.rules import (
    Atom,
    HornRule,
    RuleSignature,
    format_triple,
    get_extensional_dependencies,
)
from kg_synth.engine.generator import (
    GraphSources,
    create_searchspace,
    decrement_counts,
    is_assignment_solvable,
    triples_from_bindings,
    update_closed_preds,
)
from kg_synth.engine.metrics import GraphMetrics, PredicateProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Direct matching of triples.
# ---------------------------------------------------------------------------
def check_direct_matches(
    client: SPARQLWrapper,
    edb_uri: str,
    edb_profiles: dict[str, PredicateProfile],
    term_mapping: dict[str, str],
    chunk_size: int,
    closed_preds: set[str],
) -> int:
    """Retrieves the triples that must be added to the EDB from a set of rules and
    profiles and inserts them into the EDB.

    Returns the number of triples inserted to the EDB."""

    def direct_triples() -> Iterator[str]:
        """Yields triples that must be added to the EDB."""
        while True:
            progress_made = False

            for predicate, profile in edb_profiles.items():
                if predicate in closed_preds:
                    continue
                # Check for direct matches on domain
                for subject, frequency in list(profile.domain.items()):
                    obj_choices = list(profile.range.keys() - {subject})
                    if frequency == len(obj_choices):
                        for obj in obj_choices:
                            yield format_triple(subject, predicate, obj, term_mapping)
                            decrement_counts(profile.range, obj)
                            profile.frequency -= 1
                        del profile.domain[subject]
                        progress_made = True

                # Check for direct matches on range
                for obj, frequency in list(profile.range.items()):
                    subj_choices = list(profile.domain.keys() - {obj})
                    if frequency == len(subj_choices):
                        for subject in subj_choices:
                            yield format_triple(subject, predicate, obj, term_mapping)
                            decrement_counts(profile.domain, subject)
                            profile.frequency -= 1
                        del profile.range[obj]
                        progress_made = True

            if not progress_made:
                break

    triples = direct_triples()
    return insert_triples_sparql(
        client=client,
        graph_uri=edb_uri,
        triple_stream=triples,
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Step 2: Extract ext. predicates from rule bodies
# ---------------------------------------------------------------------------
def _filter_bindings(
    client: SPARQLWrapper,
    edb_uri: str,
    intensional_preds: set[str],
    raw_bindings: list[SparqlBinding],
    searchspace_profiles: dict[str, PredicateProfile],
    rule: HornRule,
    term_mapping: dict[str, str],
    closed_preds: set[str],
) -> Iterator[str]:
    """Yields triples for EDB generation while checking if the triple is allowed by
    predicate metrics."""

    excluded_preds = intensional_preds | closed_preds
    only_extensional = rule.get_body_predicates().isdisjoint(excluded_preds)

    if only_extensional and len(raw_bindings) < rule.support:
        raise ValueError(
            f"{rule.rule_id}'s support is {rule.support} but only "
            f"{len(raw_bindings)} bindings found."
        )

    # NOTE: If we implement the MRV (Minimum Remaining Values) heuristic later, we would
    # replace random.shuffle with a sort function here.
    candidate_bindings = list(raw_bindings)
    random.shuffle(candidate_bindings)

    selected_bindings = _select_valid_bindings(
        client=client,
        edb_uri=edb_uri,
        bindings=candidate_bindings,
        rule=rule,
        searchspace_profiles=searchspace_profiles,
        term_mapping=term_mapping,
    )

    if len(selected_bindings) < rule.support and only_extensional:
        logger.warning(
            "%s's support is %d, but %d bindings were retrieved.",
            rule.rule_id,
            rule.support,
            len(selected_bindings),
        )

    # Produce the new triples from selected bindings
    for idx in selected_bindings:
        binding_row = candidate_bindings[idx]
        for atom in (a for a in rule.body if a.predicate in searchspace_profiles):
            predicate = atom.predicate
            subject = from_binding_row(atom.subject, binding_row)[0]
            obj = from_binding_row(atom.obj, binding_row)[0]

            triple = format_triple(
                subject=subject,
                predicate=predicate,
                obj=obj,
                term_mapping=term_mapping,
            )

            decrement_counts(searchspace_profiles[predicate].domain, subject)
            decrement_counts(searchspace_profiles[predicate].range, obj)
            searchspace_profiles[predicate].frequency -= 1

            yield triple


def _select_valid_bindings(
    client: SPARQLWrapper,
    edb_uri: str,
    bindings: list[SparqlBinding],
    rule: HornRule,
    searchspace_profiles: dict[str, PredicateProfile],
    term_mapping: dict[str, str],
    max_backtracks: int = 10000,
    chunk_size: int = 1000,
) -> list[int]:
    """Selects a set of valid bindings that satisfy rule support and profile metrics.

    Args:
        client: Wrapper for SPARQL queries.
        edb_uri: URI of the Extensional Database.
        bindings: A list of SPARQL binding rows to evaluate.
        rule: The HornRule containing the body atoms to check.
        searchspace_profiles: Profiles tracking remaining allowed frequencies.
        term_mapping: Mapping of terms to their string representations.

    Returns:
        A list of indices corresponding to the accepted bindings.
    """
    logger.debug(
        "Searching through %d bindings (%d needed).", len(bindings), rule.support
    )
    body_atoms = [a for a in rule.body if a.predicate in searchspace_profiles]

    all_potential_triples = triples_from_bindings(bindings, body_atoms, term_mapping)
    existing_triples = get_existing_triples(
        client=client,
        graph_uri=edb_uri,
        candidate_triples=all_potential_triples,
        term_mapping=term_mapping,
        chunk_size=chunk_size,
    )

    added_bindings: list[int] = []
    backtrack_counter = [0]

    def backtrack(
        current_idx: int,
        current_profiles: dict[str, PredicateProfile],
        current_triples: set[str],
    ) -> bool:
        """Recursive DFS CSP solver."""
        if len(added_bindings) >= rule.support:
            return True  # Success state

        if current_idx >= len(bindings):
            return False  # Failure by invalid state

        if backtrack_counter[0] > max_backtracks:
            # Failure by timeout
            raise TimeoutError(
                "CSP Backtrack budget exceeded. Graph may be unsatisfable."
            )

        binding_row = bindings[current_idx]
        is_valid = True

        branch_profiles = copy.deepcopy(current_profiles)
        branch_triples = set(current_triples)

        for atom in body_atoms:
            predicate = atom.predicate
            profile = branch_profiles[predicate]
            logger.debug("Branch_profile state: %s", profile)

            subject = from_binding_row(atom.subject, binding_row)[0]
            obj = from_binding_row(atom.obj, binding_row)[0]

            logger.debug("Subject: %s | Object: %s", subject, obj)

            triple = format_triple(subject, predicate, obj, term_mapping)

            logger.debug("Trying triple: %s", triple)

            if triple in existing_triples or triple in branch_triples:
                logger.debug("Triple already exists in selected triples or EDB.")
                continue

            if (
                not profile.frequency
                or subject not in profile.domain
                or obj not in profile.range
                or not is_assignment_solvable(profile, subject, obj)
            ):
                logger.debug("Triple violates profiles.")
                is_valid = False
                break

            # Apply mutations to the current branch state
            branch_triples.add(triple)
            decrement_counts(profile.domain, subject)
            decrement_counts(profile.range, obj)
            profile.frequency -= 1

        if is_valid:
            added_bindings.append(current_idx)
            logger.debug(
                "Binding %d valid, let's check %d.", current_idx, current_idx + 1
            )

            if backtrack(current_idx + 1, branch_profiles, branch_triples):
                return True  # Bubble up successful state

            added_bindings.pop()
            backtrack_counter[0] += 1

        return backtrack(current_idx + 1, current_profiles, current_triples)

    success = backtrack(0, searchspace_profiles, current_triples=set())

    if not success:
        logger.warning("Could not find a valid combination to satisfy rule support.")

    return added_bindings


def check_triples_from_rule(
    rule: HornRule,
    intensional_preds: set[str],
    closed_preds: set[str],
    edb_profiles: dict[str, PredicateProfile],
    client: SPARQLWrapper,
    term_mapping: dict[str, str],
    edb_uri: str,
    chunk_size: int,
) -> int:
    """Retrieves triples satisfying a rule's body and inserts them into the EDB.

    Args:
        rule: The HornRule being evaluated.
        intensional_preds: Set of intensional predicate strings.
        closed_preds: Set of already closed predicate strings.
        edb_profiles: Global predicate profiles tracking domains and ranges.
        client: Wrapper for SPARQL queries.
        term_mapping: Mapping of terms to their string representations.
        edb_uri: The URI of the target EDB.
        chunk_size: Maximum number of triples to insert per SPARQL query.

    Returns:
        The number of successfully inserted triples.
    """

    excluded_preds = intensional_preds | closed_preds
    new_body = {atom for atom in rule.body if atom.predicate not in excluded_preds}
    if not new_body:
        return 0

    logger.debug("Generating predicates from %s: %s", rule.rule_id, list(new_body))

    new_rule = HornRule(
        signature=RuleSignature(
            rule_id=rule.rule_id,
            body=frozenset(new_body),
            head=Atom("", "", ""),  # Dummy head for query builder
        ),
        support=rule.support,
        head_coverage=rule.head_coverage,
        std_confidence=rule.std_confidence,
        pca_confidence=rule.pca_confidence,
        classification=rule.classification,
    )

    # Create the searchspace
    target_preds = new_rule.get_body_predicates()
    searchspace_profiles = {
        pred: edb_profiles[pred] for pred in target_preds if pred in edb_profiles
    }

    # Concurrency-safe unique URI
    searchspace_uri = f"http://SearchSpace.org/{uuid.uuid4().hex}"

    try:
        create_searchspace(
            client=client,
            profiles=searchspace_profiles,
            term_mapping=term_mapping,
            searchspace_uri=searchspace_uri,
        )

        sources = GraphSources(target=searchspace_uri, others=[])
        query = build_rule_query(rule=new_rule.signature, sources=sources)
        # logger.debug("Query: %s", query)
        bindings = run_select_query(client, query)

    finally:
        # Guarantee cleanup even if the query engine timeouts or filtering fails
        clear_graph_sparql(client=client, graph_uri=searchspace_uri)

    # Filter the retrieved bindings
    triple_stream = _filter_bindings(
        client=client,
        edb_uri=edb_uri,
        rule=rule,
        raw_bindings=bindings,
        term_mapping=term_mapping,
        searchspace_profiles=searchspace_profiles,
        intensional_preds=intensional_preds,
        closed_preds=closed_preds,
    )

    return insert_triples_sparql(
        client=client,
        graph_uri=edb_uri,
        triple_stream=triple_stream,
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Step 3: Produce triples from a random subject.
# ---------------------------------------------------------------------------
def insert_random_triples(
    client: SPARQLWrapper,
    edb_uri: str,
    profile: PredicateProfile,
    predicate: str,
    term_mapping: dict[str, str],
    chunk_size: int,
):
    """Generates random triples from a single predicate's profile and inserts them.

    Args:
        client: Wrapper for SPARQL queries.
        edb_uri: The URI where the EDB triples will be inserted.
        profile: The predicate profile tracking available domains and ranges.
        predicate: The predicate string for the triples.
        term_mapping: Mapping of terms to their string representations.
        chunk_size: Maximum number of triples to insert per SPARQL query.

    Returns:
        The number of inserted triples.

    Raises:
        ValueError: If there are not enough available objects to satisfy the domain.
    """

    available_subjects = list(profile.domain.keys())
    if not available_subjects:
        logger.warning("Empty domain for predicate %s.", predicate)
        return 0

    subject = random.choice(available_subjects)
    required_count = profile.domain[subject]

    # TODO: I am not excluding the subject from the profile, ponder this.
    available_objects = list(profile.range.keys())
    if len(available_objects) < required_count:
        raise ValueError(
            f"Error assigning triples for {predicate}. "
            f"{subject} must be in {required_count} triples. "
            f"There are only {len(available_objects)} different objects available."
        )

    # Pre-calculate simulated domain metrics (remains constant for this choice)
    sim_domain_len = len(profile.domain) - 1
    max_domain = max((v for k, v in profile.domain.items() if k != subject), default=0)
    chosen_objects: list[str] = []

    while True:
        chosen_objects = random.sample(available_objects, required_count)
        chosen_set = set(chosen_objects)

        # Simulate the range length: Count the objects that won't drop to 0
        sim_range_len = sum(
            1
            for obj, count in profile.range.items()
            if not (count == 1 and obj in chosen_set)
        )

        max_range = max(
            (
                count - 1 if obj in chosen_set else count
                for obj, count in profile.range.items()
            ),
            default=0,
        )

        if max_domain <= sim_range_len and max_range <= sim_domain_len:
            break

        logger.debug("Selected random assignments invalid, trying again.")

    def random_matches() -> Iterator[str]:
        """Yield random triples generated for a predicate using its profile."""

        profile.frequency -= required_count
        del profile.domain[subject]

        for obj in chosen_objects:
            decrement_counts(profile.range, obj)
            yield format_triple(subject, predicate, obj, term_mapping)

    return insert_triples_sparql(
        client=client,
        graph_uri=edb_uri,
        triple_stream=random_matches(),
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Generate EDB.
# ---------------------------------------------------------------------------
def generate_edb(
    client: SPARQLWrapper,
    rules: dict[str, HornRule],
    term_mapping: dict[str, str],
    edb_uri: str,
    chunk_size: int,
    profiles: dict[str, PredicateProfile],
) -> None:
    """Generates an EDB from a set of rules and predicate profiles.

    Args:
        client: Wrapper for SPARQL queries.
        rules: Mapping of rule IDs to HornRule objects.
        term_mapping: Mapping of terms to their string representations.
        edb_uri: The URI where the Extensional Database (EDB) will be generated.
        chunk_size: Maximum number of triples to insert per SPARQL query.
        profiles: The metrics for each predicate in the original graph.
    """
    # Instantiate a new graph
    initialize_graph(
        client=client,
        source=None,
        new_graph_uri=edb_uri,
        chunk_size=chunk_size,
    )

    intensional_preds = {r.head.predicate for r in rules.values()}
    extensional_preds = profiles.keys() - intensional_preds

    if not extensional_preds:
        logger.warning("Retrieved 0 extensional predicates, EDB will be empty.")
        return

    logger.debug(
        "\nIntensional preds.:\n\t%s\nExtensional preds.:\n\t%s\nProfiles:\n\t%s",
        "\n\t".join(sorted(intensional_preds)),
        "\n\t".join(sorted(extensional_preds)),
        "\n\t".join(sorted(profiles)),
    )

    logger.info("Creating EDB from %d extensional predicates.", len(extensional_preds))

    # Filter rules to use only those containing extensional predicates
    relevant_rules = {
        r_id: r
        for r_id, r in rules.items()
        if any(pred not in intensional_preds for pred in r.get_predicates())
    }

    edb_profiles = {pred: profiles[pred] for pred in extensional_preds}
    rule_dependency = get_extensional_dependencies(rules)

    checked_rules: set[str] = set()
    closed_preds: set[str] = set()

    total_profiles = len(edb_profiles)
    check_rules = True
    step = 0

    def _evaluate_closure() -> bool:
        """Updates closed predicates and logs progress. Returns True if complete."""
        if update_closed_preds(edb_profiles, closed_preds):
            closed = len(closed_preds)
            logger.info(
                "[Step %d]: Closed ext. predicates [%d/%d].",
                step,
                closed,
                total_profiles,
            )

            if closed == total_profiles:
                logger.info("All ext. predicates closed.")
                return True
        return False

    while len(closed_preds) < total_profiles:
        step += 1
        progress = False

        # Step 1: Check direct matches
        if d_count := check_direct_matches(
            client=client,
            edb_uri=edb_uri,
            edb_profiles=edb_profiles,
            term_mapping=term_mapping,
            chunk_size=chunk_size,
            closed_preds=closed_preds,
        ):
            progress = True
            logger.debug("[Step %d]: Added %d triples directly.", step, d_count)

            if _evaluate_closure():
                break

        # Step 2: Check rule bodies
        if check_rules:
            for r_id, r in relevant_rules.items():
                if r_id in checked_rules or (rule_dependency[r_id] - checked_rules):
                    continue

                excluded_preds = intensional_preds | closed_preds
                if len(r.get_predicates() - excluded_preds) > 1:
                    r_count = check_triples_from_rule(
                        rule=r,
                        intensional_preds=intensional_preds,
                        closed_preds=closed_preds,
                        edb_profiles=edb_profiles,
                        client=client,
                        term_mapping=term_mapping,
                        edb_uri=edb_uri,
                        chunk_size=chunk_size,
                    )
                    checked_rules.add(r_id)

                    if len(checked_rules) == len(relevant_rules):
                        check_rules = False

                    logger.debug(
                        "[Step %d]: Checked rules [%d/%d]",
                        step,
                        len(checked_rules),
                        len(relevant_rules),
                    )
                    progress = True

                    if r_count:
                        logger.debug(
                            "[Step %d]: Added %d triples from %s",
                            step,
                            r_count,
                            r.rule_id,
                        )
                        if _evaluate_closure():
                            break
                    break

        if len(closed_preds) == total_profiles:
            break

        # Step 3: Assign randomly
        if not progress:
            open_preds = list(edb_profiles.keys() - closed_preds)
            predicate = random.choice(open_preds)
            profile = edb_profiles[predicate]

            ran_count = insert_random_triples(
                client=client,
                edb_uri=edb_uri,
                profile=profile,
                predicate=predicate,
                term_mapping=term_mapping,
                chunk_size=chunk_size,
            )
            logger.debug(
                "[Step %d]: Added %d triples by random assignment.", step, ran_count
            )

            if ran_count and _evaluate_closure():
                break


if __name__ == "__main__":
    import time
    from pathlib import Path

    from SPARQLWrapper import DIGEST

    from kg_synth.config import RunConfig
    from kg_synth.core.queries import count_triples
    from kg_synth.core.rules import get_term_mapping, parse_rule_set
    from kg_synth.utils import setup_logging

    # Config setup
    simpson_config = Path("configurations/simpsons.json")
    french_config = Path("configurations/french_royalty.json")

    ##### TO RUN ANOTHER GRAPH EDIT THIS vvvv ##
    config = RunConfig.from_json(simpson_config)

    # Logging
    setup_logging(level=config.logging.level)
    logger.info("Confifuration correctly initialized.")

    # Graph settings
    graph_uri = config.graph.complete_uri
    edb_uri = config.graph.edb_uri

    # Input files
    input_dir = config.data.input_dir
    ontology_file = input_dir / config.graph.ontology_file
    term_mapping = get_term_mapping(ontology_file, default_namespace=graph_uri)
    rules_file = input_dir / config.rules.rules_file
    rules = parse_rule_set(rules_file, term_mapping=term_mapping, pca_threshold=1)

    # SPARQLWrapper client
    client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    client.setHTTPAuth(DIGEST)
    client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    # Graph metrics
    graph_metrics = GraphMetrics.from_uri(client, graph_uri)

    # Generate EDB
    logger.info("Starting EDB generation from <%s>...", graph_uri)
    start_time = time.time()
    generate_edb(
        client=client,
        rules=rules,
        term_mapping=term_mapping,
        edb_uri=edb_uri,
        chunk_size=config.virtuoso.chunk_size,
        profiles=graph_metrics.profiles,
    )
    logger.info("Finished execution at %d s.", time.time() - start_time)
    logger.info("EDB at <%s> with %d triples.", edb_uri, count_triples(client, edb_uri))
