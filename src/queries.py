import itertools
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import requests
from requests.auth import HTTPDigestAuth
from SPARQLWrapper import GET, JSON, POST, SPARQLWrapper

from rules import HornRule, RuleSignature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPARQL Query generation.
# ---------------------------------------------------------------------------
def build_rule_query(rule: RuleSignature, graph_uri: str) -> str:
    """Builds a SPARQL query for a rule."""
    var_set = set(rule.get_head_variables())
    proj = " ".join(var_set)

    patterns = [f"{atom} ." for atom in rule.body] + [f"{rule.head} ."]

    where_patterns = "\n      ".join(patterns)
    return f"""
    SELECT DISTINCT ?rule_id {proj}
    WHERE {{
      BIND ("{rule.rule_id}" AS ?rule_id)
      GRAPH <{graph_uri}> {{
        {where_patterns}
      }} 
    }}
    """


def build_ruleset_query(rules: dict[str, HornRule], include_head: bool = False) -> str:
    """Builds a single SPARQL query to retrieve variables satisfying each rule.

    Args:
        rules: A list of HornRule objects containing body and head definitions.
        namespace: The ontology or graph namespace used for SPARQL translation.
        include_head: If True, includes the head triple pattern in the WHERE clause.

    Returns:
        A formatted SPARQL query string containing UNIONed subqueries.
    """
    subqueries: list[str] = []
    global_vars: set[str] = {"?rule_id"}

    for rule_id, rule in rules.items():
        var_set = set(rule.get_head_variables())
        global_vars.update(var_set)
        proj = " ".join(var_set)

        patterns = [f"{atom} ." for atom in rule.body]
        if include_head:
            patterns.append(f"{rule.head} .")

        where_patterns = "\n      ".join(patterns)
        subqueries.append(
            f"  {{\n"
            f'    SELECT ("{rule_id}" AS ?rule_id) {proj}\n'
            f"    WHERE {{\n"
            f"      {where_patterns}\n"
            f"    }}\n"
            f"  }}"
        )

    outer_proj = " ".join(global_vars)

    subqueries_joined = "\n  UNION\n".join(subqueries)
    query = f"SELECT {outer_proj}\nWHERE {{\n{subqueries_joined}\n}}"

    logger.debug("Created ruleset query for %d rules:\n%s", len(rules), query)
    return query


def build_federated_query(
    rule: RuleSignature,
    main_graph: str,
    searchspace_graph: str,
) -> str:
    """Builds a SPARQL query that routes patterns to specific named graphs.

    Iterates through a rule's body and routes triple patterns containing the
    target predicate to the search space graph, while directing all other
    patterns to the main graph.

    Args:
        rule: The HornRule object containing the body patterns.
        main_graph: The URI of the primary named graph.
        searchspace_graph: The URI of the search space named graph.

    Returns:
        A formatted SPARQL SELECT query string.
    """

    head_pred = rule.head.predicate
    proj = " ".join(set(rule.get_head_variables()))
    patterns: list[str] = []

    for atom in rule.body:
        graph_str = searchspace_graph if atom.predicate == head_pred else main_graph

        graph_block = f"""
        GRAPH <{graph_str}> {{
          {atom} .
        }}"""

        patterns.append(graph_block)

    where_clause = " .\n".join(patterns)

    # Construct the final query string
    query = f"""
    SELECT DISTINCT {proj}
    WHERE {{
      {where_clause}
    }}"""

    logger.debug("Generated federated query for rule %s:\n%s", rule.rule_id, query)
    return query


# ---------------------------------------------------------------------------
# Insert to database.
# ---------------------------------------------------------------------------
def insert_triples_sparql(
    sparql_client: SPARQLWrapper,
    graph_uri: str,
    triple_stream: Iterable[str],
    chunk_size: int,
) -> int:
    """Inserts triples into Virtuoso using SPARQL in batches.

    Args:
        sparql_client: An instantiated and configured SPARQLWrapper client.
        graph_uri: The URI of the target named graph.
        triple_stream: An iterable yielding individual SPARQL triple strings.
        chunk_size: The maximum number of triples to insert per SPARQL query.

    Returns:
        The total number of triples successfully processed.
    """
    count = 0

    def chunk_iter(iterable: Iterable[str], size: int) -> Iterable[tuple[str, ...]]:
        """Yields successive chunks of a given size from an iterable."""
        iterator = iter(iterable)
        while chunk := tuple(itertools.islice(iterator, size)):
            yield chunk

    for chunk in chunk_iter(triple_stream, chunk_size):
        triples_payload = "\n".join(chunk)

        insert_query = f"""
        INSERT DATA {{
          GRAPH <{graph_uri}> {{
            {triples_payload}
          }}
        }}"""

        sparql_client.setQuery(insert_query)
        sparql_client.query()

        count += len(chunk)

    return count


def insert_triples_gsp(
    graph_uri: str,
    triples: Iterator[str],
    crud_endpoint: str,
    chunk_size: int = 50000,
    auth: tuple[str, str] = ("dba", "dba"),
) -> None:
    """Inserts triples into Virtuoso using the Graph Store HTTP Protocol.

    Sends raw N-Triples data directly to the REST API, preventing SQL translation buffer
    overflows and drastically speeding up ingestion.

    Args:
        crud_endpoint: Virtuoso CRUD URL ("http://.../sparql-graph-crud").
        graph_uri: Target named graph URI.
        triples: Iterator yielding N-Triple formatted strings.
        chunk_size: Number of triples to send per HTTP POST request.
        auth: A (username, password) tuple for basic authentication.

    Raises:
        requests.HTTPError: If the Virtuoso server rejects the payload.
    """
    # TODO: Fix parameters

    params = {"graph-uri": graph_uri}
    headers = {"Content-Type": "application/n-triples"}

    total_inserted = 0
    logger.debug("Inserting triples to %s via GSP.", graph_uri)
    with requests.Session() as session:
        session.auth = HTTPDigestAuth(*auth)

        while True:
            batch = list(itertools.islice(triples, chunk_size))
            if not batch:
                break
            payload = "\n".join(batch)
            response = session.post(
                url=crud_endpoint,
                params=params,
                headers=headers,
                data=payload,
            )
            response.raise_for_status()
            total_inserted += len(batch)

    logger.debug(
        "Successfully uploaded %d triples to <%s> via GSP.",
        total_inserted,
        graph_uri,
    )


# ------------------------------- Fix ---------------------------------------


def insert_graph_sparql(
    sparql_client: SPARQLWrapper,
    graph_uri: str,
    auth: HTTPDigestAuth,
    nt_file: Path,
    database_endpoint: str,
) -> None:
    """Overwrites a graph from an .nt file to the database management system."""

    clear_named_graph(graph_uri=graph_uri, sparql_client=sparql_client)

    headers = {"Content-Type": "application/n-triples"}

    # Target the specific Virtuoso Graph Store CRUD endpoint
    crud_endpoint = f"{database_endpoint.rstrip('/')}/sparql-graph-crud-auth"

    with nt_file.open("rb") as f:
        response = requests.put(
            crud_endpoint,
            params={"graph-uri": graph_uri},
            data=f,
            headers=headers,
            auth=auth,
        )

    if response.status_code in (200, 201):
        logger.debug("Successfully inserted the graph!")
    else:
        logger.error("Failed with status %s\n%s", response.status_code, response.text)


def clear_named_graph(sparql_client: SPARQLWrapper, graph_uri: str) -> None:
    """Removes all triples from a specified named graph.

    Args:
        database_endpoint: The URL of the SPARQL database endpoint.
        graph_uri: The URI of the named graph to clear.

    Raises:
        Exception: If the SPARQL CLEAR operation fails.
    """
    sparql_client.setMethod(POST)

    # CLEAR SILENT empties the graph safely even if it doesn't exist yet
    query = f"CLEAR SILENT GRAPH <{graph_uri}>"
    sparql_client.setQuery(query)

    try:
        sparql_client.query()
        logger.debug("Successfully cleared graph <%s>.", graph_uri)
    except Exception as e:
        logger.error("Failed to clear graph <%s>: %s", graph_uri, e)
        raise


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SPARQL query response handling.
# ---------------------------------------------------------------------------
SparqlBindings = list[dict[str, dict[str, str]]]


def get_select_results(sparql_client: SPARQLWrapper, query: str) -> SparqlBindings:
    """Handles the response for SELECT queries using SPARQLWrapper."""
    sparql_client.setMethod(GET)
    sparql_client.setReturnFormat(JSON)
    sparql_client.setQuery(query)

    try:
        response = sparql_client.queryAndConvert()

        if isinstance(response, dict) and "results" in response:
            raw_bindings = response["results"].get("bindings", [])
            return cast(SparqlBindings, raw_bindings)

        raise ValueError("Failed to retrieve bindings from query results.")

    except Exception:
        logger.error("SPARQL execution failed for query:\n%s", query)
        raise


# ---------------------------------------------------------------------------
# Query metrics.
# ---------------------------------------------------------------------------
def get_preds_and_freqs(sparql_client: SPARQLWrapper, graph_uri: str) -> dict[str, int]:
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

    results = get_select_results(sparql_client, query)
    if not results:
        return predicate_frequencies

    for row in results:
        predicate = row["predicate"]["value"]
        frequency = int(row["?frequency"]["value"])
        predicate_frequencies[predicate] = frequency

    return predicate_frequencies


def get_domain(
    sparql_client: SPARQLWrapper, graph_uri: str, predicate: str
) -> dict[str, int]:
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

    if results := get_select_results(sparql_client, query):
        for row in results:
            subject = row["?subject"]["value"]
            frequency = int(row["?count"]["value"])
            domain[subject] = frequency

        return domain

    logger.warning("Retrieved None for the domain of predicate %s.", predicate)
    return domain


def get_range(
    sparql_client: SPARQLWrapper, graph_uri: str, predicate: str
) -> dict[str, int]:
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

    if results := get_select_results(sparql_client, query):
        for row in results:
            obj = row["?obj"]["value"]
            frequency = int(row["?count"]["value"])
            p_range[obj] = frequency
        return p_range

    logger.warning("Retrieved None for the domain of predicate %s.", predicate)
    return p_range


def get_reflexivity(
    sparql_client: SPARQLWrapper, graph_uri: str, predicate: str
) -> int:
    """Retrieves how many triples with this predicate are reflexive (obj == subj)."""

    query = f"""
    SELECT (COUNT(*) AS ?c)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s {predicate} ?s 
      }} 
    }}"""

    if results := get_select_results(sparql_client, query):
        return int(results[0]["c"]["value"])
    return 0


def get_support(sparql_client: SPARQLWrapper, rule: HornRule, graph_uri: str) -> int:
    """Returns the support for the rule in the graph."""

    patterns = "\n".join([f"{atom} ." for atom in rule.body] + [f"{rule.head}"])

    query = f"""
    SELECT (COUNT(DISTINCT ?target) AS ?supp)
    WHERE {{
      GRAPH <{graph_uri}> {{
        {patterns}
      }}
    }}"""

    if results := get_select_results(sparql_client, query):
        return int(results[0]["supp"]["value"])

    logger.warning("Retrieved None for %s support in %s.", rule.rule_id, graph_uri)
    return 0


def get_frequency(sparql_client: SPARQLWrapper, predicate: str, graph_uri: str) -> int:
    """Returns the number of times a predicate appears in the graph."""

    query = f"""
    SELECT (COUNT(*) AS ?frequency)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s {predicate} ?o .
      }}
    }}"""

    if results := get_select_results(sparql_client, query):
        return int(results[0]["frequency"]["value"])

    logger.warning("Retrieved None for %s frequency in %s.", predicate, graph_uri)
    return 0


def get_total_triples(sparql_client: SPARQLWrapper, graph_uri: str) -> int:
    """Returns the total triples in a graph."""

    query = f"""
    SELECT (COUNT(*) AS ?total)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s ?p ?o .
      }}
    }}"""

    if results := get_select_results(sparql_client, query):
        return int(results[0]["total"]["value"])

    logger.warning("Retrieved None for total triples in %s.", graph_uri)
    return 0


if __name__ == "__main__":
    from SPARQLWrapper import DIGEST

    # Clean the french royalty graph
    client = SPARQLWrapper(endpoint="http://localhost:8890/sparql-auth")
    client.setHTTPAuth(DIGEST)
    client.setCredentials("dba", "dba")
    clear_named_graph(client, "http://FrenchRoyalty.org/")
