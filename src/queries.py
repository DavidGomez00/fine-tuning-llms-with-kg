import itertools
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import requests
from requests.auth import HTTPDigestAuth
from SPARQLWrapper import GET, JSON, SPARQLWrapper
from yarl import URL

from rules import HornRule, RuleSignature
from utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPARQL Query generation.
# ---------------------------------------------------------------------------
def build_rule_query(rule: RuleSignature, ebd_uri: str, searchspace_uri: str) -> str:
    """Builds a SPARQL query for a rule."""
    t_vars = set(rule.get_head_variables())
    t_predicate = rule.head.predicate

    if t_predicate in rule.get_body_predicates():
        for atom in (a for a in rule.body if a.predicate == t_predicate):
            s_var, o_var = atom.get_variables()
            if s_var is not None:
                t_vars.add(s_var)
            if o_var is not None:
                t_vars.add(o_var)

    proj = " ".join(t_vars)

    intensional_patterns: list[str] = []
    extensional_patterns: list[str] = []

    for atom in rule.body:
        if atom.predicate != t_predicate:
            extensional_patterns.append(f"{atom} .")
        else:
            intensional_patterns.append(f"{atom} .")

    blocks = []
    if extensional_patterns:
        extensional_join = "\n            ".join(extensional_patterns)
        extensional_block = f"""
          GRAPH <{ebd_uri}> {{
            {extensional_join}
          }}"""
        blocks.append(extensional_block)
    if intensional_patterns:
        intensional_patterns.append(f"{rule.head} .")
        intensional_join = "\n            ".join(intensional_patterns)
        intensional_block = f"""
          GRAPH <{searchspace_uri}> {{
            {intensional_join}
          }}"""
        blocks.append(intensional_block)

    where_clause = ".\n    ".join(blocks)
    query = f"""
    SELECT DISTINCT ?rule_id {proj}
    WHERE {{
      BIND ("{rule.rule_id}" AS ?rule_id)      
        {where_clause}
    }}
    """

    logger.debug("Built query for rule %s (%s):", rule.rule_id, rule.head)
    return query


# ---------------------------------------------------------------------------
# Insert to database.
# ---------------------------------------------------------------------------
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
        chunk_size: The maximum number of triples to insert per SPARQL query.

    Returns:
        The total number of triples successfully processed.
    """
    client.setMethod("POST")
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

        client.setQuery(insert_query)
        client.query()

        count += len(chunk)

    return count


def insert_triples_gsp(
    graph_uri: str,
    triples: Iterator[str],
    client: SPARQLWrapper,
    crud_endpoint: str,
    chunk_size: int = 50000,
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
    params = {"graph-uri": graph_uri}
    headers = {"Content-Type": "application/n-triples"}

    total_inserted = 0
    logger.debug("Inserting triples to %s via GSP.", graph_uri)

    with requests.Session() as session:
        session.auth = HTTPDigestAuth(*(client.user, client.passwd))

        while True:
            batch = list(itertools.islice(triples, chunk_size))
            if not batch:
                break

            payload = "\n".join(batch) + "\n"
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


def insert_graph_sparql(
    client: SPARQLWrapper,
    graph_uri: str,
    nt_file: Path,
    chunk_size: int,
) -> None:
    """Overwrites a graph from an .nt file to the database management system."""

    clear_graph_sparql(client, graph_uri)

    def _parse_line(text: str) -> str:
        """Parses a line from a .nt file to a valid triple."""
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            return ""
        return stripped

    def _triple_stream(file_path: Path) -> Iterator[str]:
        """Streams the triples from an nt file."""
        with file_path.open(encoding="utf-8") as f:
            for line in f:
                if triple := _parse_line(line):
                    yield triple

    iterator = _triple_stream(nt_file)
    insert_triples_sparql(client, graph_uri, iterator, chunk_size)


def clear_graph_sparql(client: SPARQLWrapper, graph_uri: str) -> None:
    """Removes all triples from a specified named graph.

    Args:
        database_endpoint: The URL of the SPARQL database endpoint.
        graph_uri: The URI of the named graph to clear.

    Raises:
        Exception: If the SPARQL CLEAR operation fails.
    """
    client.setMethod("POST")

    # CLEAR SILENT empties the graph safely even if it doesn't exist yet
    query = f"CLEAR SILENT GRAPH <{graph_uri}>"
    client.setQuery(query)

    try:
        client.query()
        logger.debug("Successfully cleared graph <%s>.", graph_uri)
    except Exception:
        logger.exception("Failed to clear graph <%s>: %s", graph_uri)
        raise


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
# SPARQL query response handling.
# ---------------------------------------------------------------------------
SparqlBindings = list[dict[str, dict[str, str]]]


def get_select_results(client: SPARQLWrapper, query: str) -> SparqlBindings:
    """Handles the response for SELECT queries using SPARQLWrapper."""
    client.setMethod(GET)
    client.setReturnFormat(JSON)
    client.setQuery(query)

    try:
        response = client.queryAndConvert()

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
def get_preds_and_freqs(client: SPARQLWrapper, graph_uri: str) -> dict[str, int]:
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

    results = get_select_results(client, query)
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

    if results := get_select_results(client, query):
        for row in results:
            subject = row["subject"]["value"]
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

    if results := get_select_results(client, query):
        for row in results:
            obj = row["obj"]["value"]
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

    if results := get_select_results(client, query):
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

    if results := get_select_results(client, query):
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

    if results := get_select_results(client, query):
        return int(results[0]["frequency"]["value"])

    logger.warning("Retrieved None for %s frequency in %s.", predicate, graph_uri)
    return 0


def get_total_triples(client: SPARQLWrapper, graph_uri: str) -> int:
    """Returns the total triples in a graph."""

    query = f"""
    SELECT (COUNT(*) AS ?total)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s ?p ?o .
      }}
    }}"""

    if results := get_select_results(client, query):
        total_triples = int(results[0]["total"]["value"])
        if total_triples == 0:
            logger.warning("Retrieved 0 triples from %s.", graph_uri)

        return total_triples
    logger.warning("Retrieved None for total triples in %s.", graph_uri)
    return 0


if __name__ == "__main__":
    from SPARQLWrapper import DIGEST

    from config import RunConfig

    config_file = Path("configurations/gen_triples/simpsons.json")
    config = RunConfig.from_json(config_file)

    setup_logging(level=config.logging.level)

    client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    client.setHTTPAuth(DIGEST)
    client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    # --- Execution ---
