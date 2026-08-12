from pathlib import Path

from SPARQLWrapper import DIGEST, SPARQLWrapper

from config import RunConfig
from queries import initialize_graph
from utils import setup_logging

### EDIT THIS PATH   vvv
graph_config = Path("configurations/mario.json")
config = RunConfig.from_json(graph_config)

input_dir = config.data.input_dir
graph_uri = config.graph.base_uri  # EDIT THIS URI

setup_logging(level=config.logging.level)

client = SPARQLWrapper(str(config.data.database_url / config.data.sparql_endpoint))
client.setHTTPAuth(DIGEST)
client.setCredentials(config.virtuoso.user, config.virtuoso.password)

initialize_graph(
    client=client,
    source=str(input_dir / config.graph.nt_file),
    new_graph_uri=graph_uri,
    chunk_size=1000,
)
