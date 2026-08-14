from pathlib import Path

import requests
from SPARQLWrapper import SPARQLWrapper

from utils import setup_logging


def download_graph_raw(
    client: SPARQLWrapper,
    graph_uri: str,
    output_path: Path,
    output_file: str,
    limit: int = 10000,
) -> None:
    """Directly stores graph contents to a disk file."""
    endpoint = client.endpoint

    offset = 0
    total_triples = 0

    print("Starting extraction from <%s>...", graph_uri)

    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_file, "a", encoding="utf-8") as f:
        while True:
            # 2. Paginate the CONSTRUCT query
            query = f"""
            CONSTRUCT {{ ?s ?p ?o }}
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
            LIMIT {limit} OFFSET {offset}
            """

            # 3. Request N-Triples format directly from Virtuoso
            response = requests.get(
                endpoint,
                params={"query": query},
                headers={"Accept": "application/n-triples"},
            )

            if response.status_code == 200:
                triples = response.text.strip()

                # If the chunk is empty, we have extracted everything
                if not triples:
                    break

                f.write(triples + "\n")

                # Count how many lines/triples we got in this chunk
                chunk_size = len([line for line in triples.split("\n") if line.strip()])
                total_triples += chunk_size
                print(f"Downloaded {total_triples} triples so far...")

                # If the chunk is smaller than our limit, it was the final chunk
                if chunk_size < limit:
                    break

                offset += limit
            else:
                print(f"Failed to query endpoint. Status Code: {response.status_code}")
                print(response.text)
                break

    print(f"Done! Successfully saved {total_triples} triples to {output_file}.")


if __name__ == "__main__":
    from SPARQLWrapper import DIGEST

    from config import RunConfig

    config_file = Path("configurations/gen_triples/french_royalty.json")
    config = RunConfig.from_json(config_file)

    setup_logging(level=config.logging.level)

    client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
    client.setHTTPAuth(DIGEST)
    client.setCredentials(config.virtuoso.user, config.virtuoso.password)

    output_path = Path(
        "/home/master/GitHub/fine-tuning-llms-with-kg/.data/Synthetic/FrenchRoyalty/synthetic_graph.nt"
    )
    download_graph_raw(
        client=client,
        graph_uri="http://SyntheticKG.org/",
        output_path=output_path,
        file_name="synthetic_graph.nt",
    )
