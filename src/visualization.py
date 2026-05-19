import networkx as nx
from pyvis.network import Network
from SPARQLWrapper import JSON, SPARQLWrapper


def visualize_graph(endpoint_url: str, limit: int = 100) -> None:
    """Queries the graph DB and generates an interactive HTML visualization."""

    sparql = SPARQLWrapper(endpoint_url)

    # Query a subset of triples (visualizing millions of nodes at once will crash the browser)
    query = f"""
    SELECT ?s ?p ?o
    WHERE {{
        GRAPH <http://example.org/synthetic_graph> {{
            ?s ?p ?o .
        }}
    }}
    LIMIT {limit}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.queryAndConvert()

    # Initialize a directed graph
    G = nx.DiGraph()

    # Parse triples into the NetworkX graph
    bindings = results.get("results", {}).get("bindings", [])
    for row in bindings:
        # Strip the namespace for cleaner labels in the UI
        subject = row["s"]["value"].split("/")[-1]
        predicate = row["p"]["value"].split("/")[-1]
        obj = row["o"]["value"].split("/")[-1]

        G.add_node(subject, title=subject)
        G.add_node(obj, title=obj)
        G.add_edge(subject, obj, label=predicate)

    # Generate the interactive PyVis network
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    # Add physics buttons for user interaction
    net.show_buttons(filter_=["physics"])
    net.show("knowledge_graph.html", notebook=False)


if __name__ == "__main__":
    # Run the visualizer
    visualize_graph("http://localhost:8890/sparql")
