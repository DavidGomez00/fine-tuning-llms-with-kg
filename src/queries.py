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
def build_rule_query(rule: RuleSignature, sources: dict[str, str | list[str]]) -> str:
    """Creates a query for the rule signature."""

    # Get the variables from the atomns with extensional predicates
    variables = rule.get_variables()
    proj = " ".join(sorted(list(variables)))

    if not (source := sources.get("target")):
        raise ValueError("Invalid sources to form query.")

    patterns_str = "\n      ".join([f"{atom} ." for atom in sorted(rule.body)])
    source_str = f"FROM <{source}>"

    query = f"""
    SELECT DISTINCT ?rule_id {proj}
    {source_str}
    WHERE {{
      BIND ("{rule.rule_id}" AS ?rule_id)
      {patterns_str}
    }}
    """
    return query


def build_filtered_query(
    rule: RuleSignature,
    sources: dict[str, str | list[str]],
    use_head: bool,
) -> tuple[str, str]:
    """Builds SPARQL SELECT queries to find instantiations of a rule.

    Args:
        rule: The rule signature containing the head and body atoms.
        graph_uri: The primary graph URI (extensional data) to query.
        searchspace_uri: The secondary graph URI for recursive/intensional search.
        use_head: If True, includes the rule's head in the query body when recursive.
        use_searchspace: If True, includes searchspace_uri in the FROM datasets.

    Returns:
        TODO
    """

    # Source Graphs
    target = sources.get("target", "")
    sources_block = "\n    ".join(
        f"FROM <{g}>" for g in sorted([target] + [s for s in sources.get("others", [])])
    )

    if not target:
        raise ValueError("Expected a value for target graph.")

    # Patterns and variables
    body_patterns = [f"{atom} ." for atom in sorted(rule.body)]
    proj_variables = set(rule.get_head_variables())

    diff_values_filter = ""
    if len(rule.get_variables()) > 1:
        expressions = [
            f"{v1} != {v2}"
            for v1, v2 in itertools.combinations(sorted(set(rule.get_variables())), 2)
        ]
        diff_values_filter = f"FILTER ({' && '.join(expressions)})"

    h_predicate = rule.head.predicate

    # In case of recursive rules
    if h_predicate in rule.get_body_predicates():
        proj_variables.update(
            var
            for atom in rule.body
            if atom.predicate == h_predicate
            for var in atom.get_variables()
            if var is not None
        )

        if use_head:
            body_patterns.append(f"{rule.head} .")

    # Filtered setup
    flags = []
    bind_statements = []
    not_exists_statements = []

    for i, atom in enumerate(sorted([a for a in rule if a.predicate == h_predicate])):
        flag_variable = f"?is_new_{i}"
        flags.append(flag_variable)

        # Bind statement so Python can read the boolean flag
        bind_str = (
            f"BIND (\n"
            f"        NOT EXISTS {{ GRAPH <{target}> {{ {atom} . }} }}\n"
            f"        AS {flag_variable}\n"
            f"      )"
        )
        bind_statements.append(bind_str)

        # Used to strictly compel Virtuoso to drop the rows natively
        not_exists_statements.append(
            f"NOT EXISTS {{ GRAPH <{target}> {{ {atom} . }} }}"
        )

    proj = " ".join(sorted(proj_variables) + flags)

    # Filtered query
    pattern_block = "\n      ".join(body_patterns)
    filter_block = ""
    if not_exists_statements:
        pure_or_conditions = " || ".join(not_exists_statements)
        filter_block = f"FILTER (\n        {pure_or_conditions}\n      )"

    bind_block = "\n      ".join(bind_statements)
    filtered_query = (
        f"    SELECT DISTINCT ?rule_id {proj}\n"
        f"    {sources_block}\n"
        f"    WHERE {{\n"
        f"      {pattern_block}\n"
        f'      BIND ("{rule.rule_id}" AS ?rule_id)\n'
        f"      {bind_block}\n"
        f"      {filter_block}\n"
        f"      {diff_values_filter}\n"
        f"    }}"
    )

    # Ask query
    ask_query = (
        f"  ASK\n"
        f"  {sources_block}\n"
        f"  WHERE {{\n"
        f"    SELECT (1 AS ?_force)\n"
        f"    WHERE {{\n"
        f"      {pattern_block}\n\n"
        f"      {filter_block}\n"
        f"      {diff_values_filter}\n"
        f"    }}\n"
        f"  }}"
    )

    return filtered_query, ask_query


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
        chunk_size: Maximum number of triples to insert per SPARQL query.

    Returns:
        The number of new triples added to the graph. Relies on the upstream generator
        to strictly yield novel triples.
    """
    total_inserted = 0

    def chunk_iter(iterable: Iterable[str], size: int) -> Iterable[tuple[str, ...]]:
        """Yields successive chunks of a given size from an iterable."""
        iterator = iter(iterable)
        while chunk := tuple(itertools.islice(iterator, size)):
            yield chunk

    for chunk in chunk_iter(triple_stream, chunk_size):
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


def insert_graph_from_nt_sparql(
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
    client.setMethod("POST")

    # CLEAR SILENT empties the graph safely even if it doesn't exist yet
    query = f"CLEAR SILENT GRAPH <{graph_uri}>"
    client.setQuery(query)

    try:
        client.query()
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


def initialize_from_source(
    source: str, new_graph_uri: str, client: SPARQLWrapper, chunk_size: int
) -> None:
    """Overwrites the new graph URI's content with the source's content.

    Args:
        source: .nt file or URI.
        new_graph_uri: URI where the source content will be written.
        client: Wrapper for SPARQL queries.
        chunk_size: Maximum number of triples to insert per SPARQL query.

    Raises:
        ValueError: If the source format is not valid.
    """
    if source.endswith(".nt"):
        clear_graph_sparql(client, new_graph_uri)
        insert_graph_from_nt_sparql(
            client=client,
            graph_uri=new_graph_uri,
            nt_file=source,
            chunk_size=chunk_size,
        )

    else:
        uri = source.removeprefix("<").removesuffix(">")
        if uri.startswith("http://"):
            clear_graph_sparql(client, new_graph_uri)
            copy_graph_sparql(
                client=client,
                source_graph_uri=uri,
                target_graph_uri=new_graph_uri,
            )
        else:
            raise ValueError("Invalid source, expected a .nt file or a Graph URI.")


# ---------------------------------------------------------------------------
# SPARQL query response handling.
# ---------------------------------------------------------------------------
SparqlBinding = dict[str, dict[str, str]]


def execute_select_query(client: SPARQLWrapper, query: str) -> list[SparqlBinding]:
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


def execute_ask_query(client: SPARQLWrapper, query: str) -> bool:
    """Executes a SPARQL ASK query and returns the boolean result.

    Returns:
        True if the graph pattern is satisfied, False otherwise.

    Raises:
        ValueError: If the query results do not contain a boolean response.
        Exception: Reraises any exceptions encountered during query execution.
    """
    client.setMethod(GET)
    client.setReturnFormat(JSON)
    client.setQuery(query)

    try:
        response = client.queryAndConvert()

        # Type guard for the expected JSON dictionary structure of an ASK query
        if isinstance(response, dict) and "boolean" in response:
            return bool(response["boolean"])

        raise ValueError("Failed to retrieve boolean value from ASK query results.")

    except Exception:
        logger.exception("SPARQL execution failed for ASK query:\n%s", query)
        raise


def execute_insert_query(client: SPARQLWrapper, query: str) -> None:
    """Execute an INSERT query."""
    client.setMethod("POST")
    client.setQuery(query)

    try:
        client.query()

    except Exception:
        logger.exception("Failed to insert chunk! Query:\n%s", query)
        raise


def copy_graph_sparql(
    client: SPARQLWrapper, source_graph_uri: str, target_graph_uri: str
) -> None:
    """Copy all contents from a graph to another."""

    query = f"""
    COPY GRAPH <{source_graph_uri}> TO GRAPH <{target_graph_uri}>
    """

    client.setQuery(query)
    client.setMethod("POST")

    try:
        client.query()
        logger.debug(
            "Successfully copied <%s> to <%s>.", source_graph_uri, target_graph_uri
        )
    except Exception:
        logger.exception(
            "Failed to copy <%s> to <%s>.", source_graph_uri, target_graph_uri
        )
        raise


def from_binding_row(term: str, bindings_row: SparqlBinding) -> tuple[str, str]:
    """Safely extracts a term from a single binding row."""
    if term.startswith("?"):
        var_name = term.lstrip("?")
        val = bindings_row.get(var_name, {}).get("value", var_name)
        v_type = bindings_row.get(var_name, {}).get("type", "uri")
        return val, v_type
    return term, "uri"


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

    results = execute_select_query(client, query)
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

    if results := execute_select_query(client, query):
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

    if results := execute_select_query(client, query):
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

    if results := execute_select_query(client, query):
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

    if results := execute_select_query(client, query):
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

    if results := execute_select_query(client, query):
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

    if results := execute_select_query(client, query):
        total_triples = int(results[0]["total"]["value"])
        if total_triples == 0:
            logger.warning("Retrieved 0 triples from %s.", graph_uri)

        return total_triples
    logger.warning("Retrieved None for total triples in %s.", graph_uri)
    return 0


if __name__ == "__main__":
    import pandas as pd
    from SPARQLWrapper import DIGEST

    from config import RunConfig

    config_file = Path("configurations/complete_graph/french_royalty.json")
    config = RunConfig.from_json(config_file)

    input_dir = config.data.input_dir
    graph_uri = config.graph.base_graph_uri
    ontology_file = input_dir / config.graph.ontology_file
    rule_file = input_dir / config.rules.rules_file

    setup_logging(level=config.logging.level)

    client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    client.setHTTPAuth(DIGEST)
    client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    rule_dataframe = pd.read_csv(rule_file)

    initialize_from_source(
        client=client,
        source=str(input_dir / config.graph.nt_file),
        new_graph_uri=graph_uri,
        chunk_size=1000,
    )

    # term_mapping = get_term_mapping(
    #     ontology_file=ontology_file, default_namespace=graph_uri
    # )
    # rules = parse_rule_set(
    #     rule_dataframe=rule_dataframe, term_mapping=term_mapping, pca_threshold=1
    # )

    # for rule_id, rule in rules.items():
    #     logger.info("%s", rule_id)
    #     logger.info(
    #         "Rule support: %d",
    #         get_support(client=client, rule=rule, graph_uri=graph_uri),
    #     )
