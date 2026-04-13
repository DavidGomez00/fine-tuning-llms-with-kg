import os

from typing import Union, List, Dict, Tuple
from pathlib import Path
from collections import defaultdict

def preprocess_kg_file(input_file: Union[str, Path], output_file: Union[str, Path]) -> int:
    """ Preprocess KG file from tab-separated to space-separated format.
    
    Args: 
        input_file (Union[str, Path]): Absoulute or relative path to the document to parse.
        output_file (Union[str, Path]): Absolute or relative path to the file where the 
            parsed document is saved.
    
    Returns:
        int: The number of valid triples successfully processed and written.

    Raises: 
        FileNotFoundError: If the input_file does not exist.
        IOError: If there is an issue reading or writting the files.
    """
    in_path = Path(input_file)
    if not in_path.is_file():
        raise FileNotFoundError(f"The input file '{in_path}' does not exist.")
    
    with open(in_path, 'r', encoding='utf-8') as f_in, \
        open(output_file, 'w', encoding='utf-8') as f_out:
        
        # Count valid lines without loading the whole file into memory
        valid_cont = sum(1 for line in f_in if len(line.strip().split('\t')) == 3)

        # Write the header
        f_out.write(f"{valid_count}\n")

        # Reset the reading pointer back to the beggining of the input file
        f_in.seek(0)

        # Write the valid data
        for line_num, line in enumerate(f_in, 1):
            parts = line.strip('\t')
            if len(parts) == 3:
                f_out.write(f"{parts[0]} {parts[1]} {parts[2]}\n") # TODO: igual mejor con str.join()
            else:
                #TODO: logging of incorrect lines??
                pass 

    return valid_cont


def load_knowledge_graph(file_path: Union[str, Path]) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """ Loads a Knowledge Graph from a specifically formatted text file.

    The expected file format starts with a line indicating the number of edges,
    followed by lines with space-separated triples: `node1 node2 relation_id`.
    
    Args:
        file_path (Union[str, Path]): Path to file containing KG triples.

    Returns:
        Tuple[Dict[str, Dict[str, int]], List[str]]: A tuple containing:
            - graph: An adjacency dictionary where graph[node1][node2] = relation_id.
            - nodes: A list of all unique nodes found in the graph.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Knowledge Graph file not found: {file_path}")

    graph: Dict[str, Dict[str, int]] = defaultdict(dict)
    nodes: set[str] = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            _ = int(f.readline().strip())
        except ValueError:
            # TODO log warning
            print(f"Warning: First line of {file_path} is not a valid integer. Proceeding anywat.")

        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split()
            
            # Skip empty lines
            if not parts:
                continue

            if len(parts) == 3:
                node1, node2, relation_srt = parts
                nodes.update([node1, node2])

                try:
                    graph[node1][node2] = int(relation_str)
                except ValueError:
                    # TODO: Logger implementation.
                    print(f"Error: Invalid relation ID in line {line_num}: '{relation_str}' is not an integer.")
            else:
                # TODO: impl logger
                print(f"Warning: Malformed triple on line {line_num}. Expected 3 parts, got {len(parts)}.")
    # Convert the defaultdict back to a standard dict before returning
    # to prevent accidental empty key creations later.
    return dict(graph), list(nodes)


def load_relation_mapping(file_path: Union[str, Path]) -> Dict[int, str]:
    """ Loads a mapping of relation IDs to their natural language name from a file.

    The expected file format is a header line with the number of relations,
    followed by lines with tab-separated 'relation_name' and 'relation_id'.
    
    Args:
        file_path (Union[str, Path]): Path to dictionary file.
    
    Returns:
        Dict[int, str]: TAhe dictionary where keys are integer IDs and values
            are the natural language relation names.
    
    Raises: 
        FileNotFoundError: If the dictionary file does not exist.
        ValueError: If there is an issue parsing the IDs into integers.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {file_path}")

    id2relation: Dict[int, str] = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        # Read and discard the first line (number of relations)
        _ = f.readline()

        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, relation_id = parts
                try:
                    id2relation[int(relation_id)] = relation
                except ValueError:
                    # TODO: logger
                    print(f"Warning: Line {line_num}: Could not parse ID '{relation_id}' as integer. Skipping.")
            else:
                #TODO: Logger
                print(f"Warning: Line {line_num}: Invalid formal, expected 2 separated parts. Skipping.")
    
    return id2relation

# TODO: Mover esto al main
# Preprocess the knowledge graph
kg_input = os.path.join(CONFIG['data_dir'], CONFIG['kg_file'])
kg_processed = os.path.join(CONFIG['data_dir'], 'train2id_processed.txt')
preprocess_kg_file(kg_input, kg_processed)
print("Knowledge graph preprocessed!")