import itertools
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import requests
from requests.auth import HTTPDigestAuth
from SPARQLWrapper import GET, JSON, URLENCODED, SPARQLWrapper
from yarl import URL

from kg_synth.core.rules import HornRule, RuleSignature, format_term, format_triple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions.
# ---------------------------------------------------------------------------
def _chunk_iter(iterable: Iterable[str], size: int) -> Iterable[tuple[str, ...]]:
    """Yields successive chunks of a given size from an iterable."""
    iterator = iter(iterable)
    while chunk := tuple(itertools.islice(iterator, size)):
        yield chunk


def _get_update_client(client: SPARQLWrapper) -> SPARQLWrapper:
    """Returns a SPARQLWrapper pointing to the write endpoint for updates."""

    if "/repositories/" in client.endpoint and not client.endpoint.endswith(
        "/statements"
    ):
        update_client = SPARQLWrapper(f"{client.endpoint.rstrip('/')}/statements")
        update_client.http_auth = client.http_auth
        update_client.user = getattr(client, "user", None)
        update_client.passwd = getattr(client, "passwd", None)
        return update_client

    return client


# ---------------------------------------------------------------------------
# SPARQL query generation.
# ---------------------------------------------------------------------------
def build_rule_query(rule: RuleSignature, sources: dict[str, str | list[str]]) -> str:
    """Creates a query for the rule signature."""

    # Get the variables from the atomns with extensional predicates
    variables = rule.get_body_variables()
    proj = " ".join(sorted(list(variables)))

    patterns_str = "\n      ".join([f"{atom} ." for atom in sorted(rule.body)])

    unique_values_str = ""
    if len(rule.get_variables()) > 1:
        expressions = [
            f"{v1} != {v2}"
            for v1, v2 in itertools.combinations(sorted(set(rule.get_variables())), 2)
        ]
        unique_values_str = f"FILTER ({' && '.join(expressions)})"

    # Define the Graph sources for the query
    t_source = sources.get("target", None)
    if t_source is None:
        raise ValueError("Target source not specified.")
    sources = [t_source] + [s for s in sources.get("others", [])]
    source_str = "\n    ".join(f"FROM <{g}>" for g in sources)

    query = f"""
    SELECT DISTINCT ?rule_id {proj}
    {source_str}
    WHERE {{
      BIND ("{rule.rule_id}" AS ?rule_id)
      {patterns_str}
      {unique_values_str}
    }}
    """
    return query


# ---------------------------------------------------------------------------
# Insert to database.
# ---------------------------------------------------------------------------
# TODO: Unify the insert functions so it does work in 1 function with any source.
def insert_triples_sparql(
    client: SPARQLWrapper,
    graph_uri: str,
    triple_stream: Iterable[str],
    chunk_size: int,
) -> int:
    """Inserts triples into Virtuoso using SPARQL in batches.

    Args:
        client: An instantiated and configured SPARQLWrapper client.
        graph_uri: The URI of the target named graph.
        triple_stream: An iterable yielding individual SPARQL triple strings.
        chunk_size: Maximum number of triples to insert per SPARQL query.

    Returns:
        The number of new triples added to the graph. Relies on the upstream generator
        to strictly yield novel triples.
    """
    total_inserted = 0

    for chunk in _chunk_iter(triple_stream, chunk_size):
        if unique_chunk := set(chunk):
            triples_payload = "\n".join(unique_chunk)

            query = f"""
            INSERT DATA {{
            GRAPH <{graph_uri}> {{
                {triples_payload}
            }}
            }}"""

            execute_insert_query(client, query)
            total_inserted += len(unique_chunk)

    return total_inserted


# TODO: Create searchspace depends on this functions, but it doesn't work with graphDB.
def insert_triples_gsp(
    client: SPARQLWrapper,
    graph_uri: str,
    triples: Iterator[str],
    chunk_size: int = 10000,
) -> None:
    """Inserts triples into Virtuoso using the Graph Store HTTP Protocol.

    Sends raw N-Triples data directly to the REST API, preventing SQL translation buffer
    overflows and drastically speeding up ingestion.

    Args:
        graph_uri: Target named graph URI.
        triples: Iterator yielding N-Triple formatted strings.
        chunk_size: Number of triples to send per HTTP POST request.
        auth: A (username, password) tuple for basic authentication.

    Raises:
        requests.HTTPError: If the Virtuoso server rejects the payload.
    """
    params = {"graph-uri": graph_uri}
    headers = {"Content-Type": "application/n-triples"}

    total_inserted = 0

    with requests.Session() as session:
        session.auth = HTTPDigestAuth(*(client.user, client.passwd))

        while True:
            batch = list(itertools.islice(triples, chunk_size))
            if not batch:
                break

            payload = "\n".join(batch) + "\n"
            response = session.post(
                url=URL(client.endpoint).with_name("sparql-graph-crud-auth"),
                params=params,
                headers=headers,
                data=payload,
            )
            response.raise_for_status()
            total_inserted += len(batch)


def insert_graph_sparql(
    client: SPARQLWrapper,
    graph_uri: str,
    chunk_size: int,
    nt_file: Path | str,
) -> None:
    """Overwrites a graph with contents from an .nt file or a URI.

    Args:
        client: The SPARQL wrapper client used to execute queries.
        graph_uri: The URI of the named graph to overwrite.
        chunk_size: The number of triples to insert per batch.
        nt_file: The local file path to the .nt file.
        from_uri: The remote URI pointing to an .nt file.

    Raises:
        ValueError: If neither `nt_file` nor `from_uri` is provided.
    """

    nt_file = Path(nt_file)
    if not nt_file.is_file():
        raise ValueError("Invalid input file: %s", nt_file)

    clear_graph_sparql(client, graph_uri)

    def _parse_line(text: str) -> str:
        """Parses a line from a .nt file to a valid triple."""
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            return ""
        return stripped

    def _triple_stream(file_path: Path) -> Iterator[str]:
        """Streams the triples locally from an .nt file."""
        with file_path.open(encoding="utf-8") as f:
            for line in f:
                if triple := _parse_line(line):
                    yield triple

    iterator = _triple_stream(nt_file)

    n = insert_triples_sparql(client, graph_uri, iterator, chunk_size)
    logger.debug("Inserted %d triples to <%s> from '%s'.", n, graph_uri, nt_file.name)


def clear_graph_sparql(client: SPARQLWrapper, graph_uri: str) -> None:
    """Removes all triples from a specified named graph.

    Args:
        database_endpoint: The URL of the SPARQL database endpoint.
        graph_uri: The URI of the named graph to clear.

    Raises:
        Exception: If the SPARQL CLEAR operation fails.
    """
    update_client = _get_update_client(client)
    update_client.setMethod("POST")
    query = f"CLEAR SILENT GRAPH <{graph_uri}>"
    update_client.setQuery(query)

    if hasattr(update_client, "parameters"):
        if "query" in update_client.parameters:
            del update_client.parameters["query"]
        update_client.parameters["update"] = query

    try:
        update_client.query()
    except Exception:
        logger.exception("Failed to clear graph <%s>.", graph_uri)
        raise


# ---------------------------------------------------------------------------
# Download from database.
# ---------------------------------------------------------------------------
# TODO: Tiene esto que estar aquí??
def download_graph_raw(
    client: SPARQLWrapper,
    graph_uri: str,
    output_path: Path,
    file_name: str,
    limit: int = 10000,
) -> None:
    """Directly stores graph contents to a disk file."""
    endpoint = URL(client.endpoint).with_name("sparql")

    offset = 0
    total_triples = 0

    logger.info("Starting extraction from <%s>...", graph_uri)

    output_path.mkdir(exist_ok=True, parents=True)
    output_file = output_path / file_name

    with output_file.open("a", encoding="utf-8") as f:
        while True:
            query = f"""
            CONSTRUCT {{ ?s ?p ?o }}
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
            LIMIT {limit} OFFSET {offset}
            """

            response = requests.get(
                endpoint,
                params={"query": query},
                headers={"Accept": "application/n-triples"},
            )

            if response.status_code == 200:
                triples = response.text.strip()

                if not triples:
                    break

                f.write(triples + "\n")

                chunk_size = len([line for line in triples.split("\n") if line.strip()])
                total_triples += chunk_size

                logger.debug("Downloaded %d triples so far...", total_triples)

                if chunk_size < limit:
                    break

                offset += limit
            else:
                error_msg = (
                    f"Failed to query endpoint (code {response.status_code})\n"
                    f"{response.text}"
                )
                logger.error(error_msg)
                raise Exception(error_msg)

    logger.info(f"Successfully saved {total_triples} triples to {output_file}.")


# ---------------------------------------------------------------------------
# initialize graph in database.
# ---------------------------------------------------------------------------
def initialize_graph(
    client: SPARQLWrapper, source: str | None, new_graph_uri: str, chunk_size: int
) -> None:
    """Overwrites the new graph URI's content with the source's content.

    Args:
        client: Wrapper for SPARQL queries.
        source: A .nt file path or a Graph URI. If None, the graph is only cleared.
        new_graph_uri: URI where the source content will be written.
        chunk_size: Maximum number of triples to insert per SPARQL query.

    Raises:
        ValueError: If the source format is not valid (neither a .nt file nor a URI).
    """

    if source is None:
        clear_graph_sparql(client, new_graph_uri)
        logger.debug("Initializated empty graph at <%s>.", new_graph_uri)
        return

    clean_source = str(source).strip("<>")
    clean_target = new_graph_uri.strip("<>")
    is_nt_file = clean_source.endswith(".nt")
    is_uri = clean_source.startswith(("http:", "https:"))

    if not (is_nt_file or is_uri):
        raise ValueError(f"Invalid source '{source}'. Expected a .nt file or a URI.")

    if is_uri and clean_source == clean_target:
        return

    clear_graph_sparql(client, new_graph_uri)

    if is_nt_file:
        insert_graph_sparql(
            client=client,
            graph_uri=new_graph_uri,
            nt_file=source,
            chunk_size=chunk_size,
        )

    else:
        copy_graph_sparql(
            client=client,
            source_graph_uri=clean_source,
            target_graph_uri=new_graph_uri,
        )


# ---------------------------------------------------------------------------
# Handle SPARQL query responses.
# ---------------------------------------------------------------------------
SparqlBinding = dict[str, dict[str, str]]


def run_select_query(client: SPARQLWrapper, query: str) -> list[SparqlBinding]:
    """Executes a SELECT query and returns the bindings."""
    # TODO (optim): Maybe we want this as an iterator
    client.setMethod(GET)
    client.setReturnFormat(JSON)
    client.setQuery(query)

    try:
        response = client.queryAndConvert()

        if isinstance(response, dict) and "results" in response:
            raw_bindings = response["results"].get("bindings", [])
            return cast(list[SparqlBinding], raw_bindings)

        raise ValueError("Failed to retrieve bindings from query results.")

    except Exception:
        logger.error("SPARQL execution failed for query:\n%s", query)
        raise


def execute_insert_query(client: SPARQLWrapper, query: str) -> None:
    """Execute an INSERT query."""
    update_client = _get_update_client(client)

    # Configure for SPARQL UPDATE
    update_client.setMethod("POST")
    update_client.setRequestMethod(URLENCODED)

    update_client.setQuery(query)

    if hasattr(update_client, "parameters"):
        if "query" in update_client.parameters:
            del update_client.parameters["query"]
        update_client.parameters["update"] = query

    try:
        update_client.query()
    except Exception:
        logger.exception("Failed to insert chunk! Query:\n%s", query)
        raise


def copy_graph_sparql(
    client: SPARQLWrapper, source_graph_uri: str, target_graph_uri: str
) -> None:
    """Copy all contents from a graph to another."""
    update_client = _get_update_client(client)

    update_client.setMethod("POST")
    update_client.setRequestMethod(URLENCODED)

    query = f"""
    COPY GRAPH <{source_graph_uri}> TO GRAPH <{target_graph_uri}>
    """
    update_client.setQuery(query)

    if hasattr(update_client, "parameters"):
        if "query" in update_client.parameters:
            del update_client.parameters["query"]
        update_client.parameters["update"] = query

    try:
        update_client.query()
        logger.debug(
            "Successfully copied <%s> to <%s>.", source_graph_uri, target_graph_uri
        )
    except Exception:
        logger.exception(
            "Failed to copy <%s> to <%s>.", source_graph_uri, target_graph_uri
        )
        raise


def from_binding_row(term: str, binding_row: SparqlBinding) -> tuple[str, str]:
    """Safely extracts a term from a single binding row."""
    if term.startswith("?"):
        var_name = term.lstrip("?")
        val = format_term(binding_row.get(var_name, {}).get("value", var_name))
        v_type = binding_row.get(var_name, {}).get("type", "uri")
        return val, v_type
    return term, "uri"


# ---------------------------------------------------------------------------
# Query metrics.
# ---------------------------------------------------------------------------
def get_predicate_frequencies(client: SPARQLWrapper, graph_uri: str) -> dict[str, int]:
    """Retrieves all unique predicates in the graph and the frequency of each one."""

    predicate_frequencies: dict[str, int] = {}
    query = f"""
        SELECT ?predicate (COUNT(*) AS ?frequency)
        WHERE {{ 
          GRAPH <{graph_uri}> {{
            ?s ?predicate ?o .
          }} 
        }}
        GROUP BY ?predicate
        """

    results = run_select_query(client, query)
    if not results:
        return predicate_frequencies

    for row in results:
        predicate = row["predicate"]["value"]
        frequency = int(row["frequency"]["value"])
        predicate_frequencies[predicate] = frequency

    return predicate_frequencies


def get_domain(client: SPARQLWrapper, graph_uri: str, predicate: str) -> dict[str, int]:
    """Retrieves the distribution of subjects for a predicate in a graph."""

    domain: dict[str, int] = {}

    query = f"""
    SELECT ?subject (COUNT(*) AS ?count) 
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?subject {predicate} ?o 
      }}
    }} 
    GROUP BY ?subject
    """

    if results := run_select_query(client, query):
        for row in results:
            subject = f"<{row['subject']['value']}>"
            frequency = int(row["count"]["value"])
            domain[subject] = frequency

        return domain

    logger.warning("Retrieved None for the domain of predicate %s.", predicate)
    return domain


def get_range(client: SPARQLWrapper, graph_uri: str, predicate: str) -> dict[str, int]:
    """Retrieves the distribution of objects for a predicate in a graph."""

    p_range: dict[str, int] = {}

    query = f"""
    SELECT ?obj (COUNT(*) AS ?count) 
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s {predicate} ?obj 
      }}
    }} 
    GROUP BY ?obj
    """

    if results := run_select_query(client, query):
        for row in results:
            obj = f"<{row['obj']['value']}>"
            frequency = int(row["count"]["value"])
            p_range[obj] = frequency
        return p_range

    logger.warning("Retrieved None for the domain of predicate %s.", predicate)
    return p_range


def get_reflexivity(client: SPARQLWrapper, graph_uri: str, predicate: str) -> int:
    """Retrieves how many triples with this predicate are reflexive (obj == subj)."""

    query = f"""
    SELECT (COUNT(*) AS ?c)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s {predicate} ?s 
      }} 
    }}"""

    if results := run_select_query(client, query):
        return int(results[0]["c"]["value"])
    return 0


def get_support(client: SPARQLWrapper, rule: HornRule, graph_uri: str) -> int:
    """Returns the support for the rule in the graph."""

    patterns = "\n        ".join(
        [f"{atom} ." for atom in rule.body] + [f"{rule.head} ."]
    )
    proj = " ".join(rule.get_variables())

    query = f"""
    SELECT (COUNT(*) AS ?supp)
    WHERE {{
      SELECT DISTINCT {proj}
      WHERE {{
        GRAPH <{graph_uri}> {{
        {patterns}
      }}  
      }}
    }}"""

    if results := run_select_query(client, query):
        return int(results[0]["supp"]["value"])

    logger.warning("Retrieved None for %s support in %s.", rule.rule_id, graph_uri)
    return 0


def get_frequency(client: SPARQLWrapper, predicate: str, graph_uri: str) -> int:
    """Returns the number of times a predicate appears in the graph."""

    query = f"""
    SELECT (COUNT(*) AS ?frequency)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s {predicate} ?o .
      }}
    }}"""

    if results := run_select_query(client, query):
        return int(results[0]["frequency"]["value"])

    logger.warning("Retrieved None for %s frequency in %s.", predicate, graph_uri)
    return 0


def count_triples(client: SPARQLWrapper, graph_uri: str) -> int:
    """Returns the total triples in a graph."""

    query = f"""
    SELECT (COUNT(*) AS ?total)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s ?p ?o .
      }}
    }}"""

    if results := run_select_query(client, query):
        total_triples = int(results[0]["total"]["value"])
        if total_triples == 0:
            logger.warning("Retrieved 0 triples from %s.", graph_uri)

        return total_triples
    logger.warning("Retrieved None for total triples in %s.", graph_uri)
    return 0


# ---------------------------------------------------------------------------
# Helpers ofr IDB/EDB generation.
# ---------------------------------------------------------------------------
def get_existing_triples(
    client: SPARQLWrapper,
    graph_uri: str,
    candidate_triples: Iterable[str],
    term_mapping: dict[str, str],
    chunk_size: int,
) -> set[str]:
    """Return triples from 'candidate_triples' that already exist in 'edb_uri'."""
    existing_triples = set()

    for chunk in _chunk_iter(candidate_triples, chunk_size):
        formatted_values = (f"({triple.strip(' .')})" for triple in chunk)
        values_clause = " ".join(formatted_values)

        query = f"""
            SELECT ?s ?p ?o
            WHERE {{
                GRAPH <{graph_uri}> {{
                    ?s ?p ?o .
                    VALUES (?s ?p ?o) {{ {values_clause} }}
                }}
            }}
        """

        bindings = run_select_query(client, query)

        existing_triples.update(
            format_triple(
                subject=from_binding_row("?s", binding_row)[0],
                predicate=from_binding_row("?p", binding_row)[0],
                obj=from_binding_row("?o", binding_row)[0],
                term_mapping=term_mapping,
            )
            for binding_row in bindings
        )

    return existing_triples
