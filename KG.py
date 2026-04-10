import os


def preprocess_kg_file(input_file, output_file):
    """ Preprocess KG file from tab-separated to space-separated format.
    Input: 
        input_file: document to process.
        output_file: file to save processed document.
    Returns:
        len(lines): Number of lines in input_file.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(output_file, 'w') as f:
        f.write(f"{len(lines)}\n")
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                f.write(f"{parts[0]} {parts[1]} {parts[2]}\n")
    print(f"Preprocessed {len(lines)} triples")
    return len(lines)


def load_knowledge_graph(file_path):
    """ Load knowledge graph from file_path.
    Input:
        - file_path: Path to file containing KG triples.
    Return:
        - graph: dictionary?
        - list(nodes): set of nodes in the graph as list.
    """
    # TODO: Este bucle dentro del "with open" seguro que se puede implementar mejor
    graph = {}
    nodes = set()
    with open(file_path, 'r') as f:
        num_lines = int(f.readline())
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                node1, node2, relation = parts
                nodes.add(node1)
                nodes.add(node2)
                if node1 not in graph:
                    graph[node1] = {}
                graph[node1][node2] = int(relation)
    return graph, list(nodes)


def load_relation_mapping(file_path):
    """ Load relation ID to name mapping.
    Input:
        - file_path: Path to file containing mapping.
    Return:
        - relation2id: Dictionary.
    """
    relation2id = {}
    with open(file_path, 'r') as f:
        num_relations = int(f.readline())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, relation_id = parts
                relation2id[int(relation_id)] = relation
    return relation2id

# TODO: Mover esto al main
# Preprocess the knowledge graph
kg_input = os.path.join(CONFIG['data_dir'], CONFIG['kg_file'])
kg_processed = os.path.join(CONFIG['data_dir'], 'train2id_processed.txt')
preprocess_kg_file(kg_input, kg_processed)
print("Knowledge graph preprocessed!")