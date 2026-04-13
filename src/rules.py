import re
import pandas as pd

from typing import Dict, Any, Union, List, optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class RuleDefinition:
    """ Represents a parsed logic rule form a knowledge graph.

    Attributes:
        rule_id (str | None): The extracted numerical ID of the rule.
        rule_text (str | None): The natural language representation of the rule.
        head (str | None): The head predicate.
        body (str | None): The body predicate.
        instances (List[str]): A list of real instances found in the KG.
        pca_confidence (float | None): The PCA confidence score.
        classification (str | None): The rule's classification type.
    """
    rule_id: Optional[str] = None
    rule_text: Optional[str] = None
    head: Optional[str] = None
    body: Optional[str] = None
    instances: List[str] = field(default_factory=list)
    pca_confidence: Optional[float] = None
    classification: Optional[str] = None

@dataclass
class CoTGenerationConfig:
    """ Configuration settings for generating CoT from KGs for training.
    
    Attributes:
        total_samples (int, optional): Number of CoT samples to generate. Defaults to 1000.
        max_path_length (int, optional): Maximmum length of the paths in the KG. Defaults to 10.
        include_reasoning (bool, optional): Whether to include the reasoning process in 
            natural language. Defaults to True.
        use_rules (bool, optional): Whether to include the rules descriptions in the 
            context in natural language. Defaults to True.
        max_rules_in_context (int, optional): Maximmum number of rules to be added to the
            prompt. Defaults to 3.
        max_attempts_multiplier (int, optional): Multiplies by the number of samples to
            define the maximum number of attemps (prevents infinite loops). Defaults to 10.
    """
    total_samples: Optional[int] = 1000
    max_path_length: Optional[int] = 10
    include_reasoning: Optional[bool] = True
    use_rules: Optional[bool] = True
    max_rules_in_context: Optional[int] = 3
    max_attempts_multiplier: Optional[int] = 10

def parse_rule_file(file_path: Union[str, Path]) -> RuleDefinition:
    """ Parses a rule text file to extract rule definitions and metrics.

    This function reads a specifically formatted text file containing logic rules
    and uses regular expressions to extract metadata such as the rule ID, logical 
    head and body, real instances from a Knowledge Graph, and confidence metrics.

    Args:
        file_path (Union[str, Path]): The absolute or relative path to the rule file.

    Returns:
        RuleDefinition: A datacalss object containing the parsed rule metadata.
           
    Raises:
        FileNotFoundError: If the file specified in `file_path` does not exist.
        IOError: If there is an issue reading the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Initialize the dataclass object
    rule_info = RuleDefinition()

    # Extract rule ID and text
    rule_match = re.search(r'Rule (\d+):\s*(.+?)(?=\n\nFormal Rule:)', content, re.DOTALL)
    if rule_match:
        rule_info.rule_id = rule_match.group(1)
        rule_info.rule_text = rule_match.group(2).strip()

    # Extract head and body predicates
    head_match = re.search(r'Head:\s*(.+)', content)
    body_match = re.search(r'Body:\s*(.+)', content)
    if head_match:
        rule_info.head = head_match.group(1).strip()
    if body_match:
        rule_info.body = body_match.group(1).strip()

    # Extract instances
    instances_section = re.search(r'Real Instances from Knowledge Graph.*?:\n\n(.+?)(?=\n\nRule Statistics:)', content, re.DOTALL)
    if instances_section:
        for line in instances_section.group(1).strip().split('\n'):
            if line.strip():
                rule_info.instances.append(line.strip())

    # Extract PCA confidence and classification
    pca_match = re.search(r'PCA Confidence:\s*([\d.]+)', content)
    classification_match = re.search(r'Rule Classification:\s*(\w+)', content)
    if pca_match:
        rule_info.pca_confidence = float(pca_match.group(1))
    if classification_match:
        rule_info.classification = classification_match.group(1)

    return rule_info


def load_all_rules(rules_directory: Union[str, Path]) -> List[RuleDefinition]:
    """ Load all parsed rules from a specified directory.
    
    Args: 
        rules_directory (Union[str, Path]): Absolute or relative path to the rules dir.
    
    Returns:
        list[RuleDefinition]: A list containing RuleDefinitions for each rule.

    Raises: 
        NotADirectoryError: If the provided rules_directory does not exist or is not a
                            valid directory.
    """
    dir_path = Path(rules_directory)

    # Check directory
    if not dir_path.is_dir():
        raise NotADirectoryError(f"The directory '{rules_directory}' wa not found.")
    
    rules: List[RuleDefinition] = []
    rule_files = sorted(dir_path.glob('rule_*.txt'))

    for file_path in rule_files:
        try:
            rule_info = parse_rule_file(file_path)
            rules.append(rule_info)

        except Exception as e:
            print(f"Error loading {file_path}: {e}") # TODO: implement a logger
    return rules


def create_rule_context(rules: List[RuleDefinition], max_rules: int = 3) -> str:
    """ Create a natural language string describing symbolic rules for prompt injection.
    
    Args:
        rules (List[RuleDefinition]): List containing each of the rules as a RuleDefinition.
        max_rules (int): Maximum number of rule descriptions to be added to context.
                         Set to 3 by default.
    
    Returns:
        string: Natural language string describing the rules added to context.
    """
    if not rules:
        return ""
    
    context = ["###Symbolic Rules (for reference):"]

    for i, rule in enumerate(rules[:max_rules], 1):
        if rule.rule_text:
            context.append(f"{i}. {rule.rule_text}")
            if rule.pca_confidence is not None:
                context.append(f"  (Confidence: {rule.pca_confidence:.3f})")
    context.append("")

    return "\n".join(context)


def generate_training_data_with_rules(
    graph: Dict[Any, Dict[Any, Any]],
    node_list: List[Any],
    relation2id: Dict[Any, str],
    rules: List[RuleDefinition],
    config: CoTGenerationConfig = CoTGenerationConfig()
    ) -> pd.DataFrame:
    """ Generates training data with chain-of-thought reasoning from a Knowledge Graph.

    Args:
        graph (Dict): A dictionary representing the Knowledge Graph adjacency list.
        node_list (List): A flat list of all available nodes in the graph.
        relation2id (Dict): Mapping from relation IDs to natural language names.
        rules (List[RuleDefinition]): List of parsed symbolic rules.
        config (CoTGenerationConfig, optional): Settings for generation behaviour.

    Returns:
        pd.DataFrame: A DataFrame containing the generated training examples.
    """
    data = []
    unique_paths: Set[str] = set()
    pos_count = 0
    neg_count = 0

    # Initialize Context
    rule_context = ""
    if config.use_rules and rules:
        rule_context = create_rule_context(rules, max_rules=config.max_rules_in_context)

    has_active_rules = bool(rule_context) # See if there are rules in context

    max_attempts = config.total_samples * config.max_attempts_multiplier
    attempts = 0
    half_samples = config.total_samples // 2

    # CoT generation loop
    while len(data) < config.total_samples and attempts < max_attempts:
        attempts += 1

        path_length = random.randint(2, config.max_path_length)
        first_node = random.choice(node_list)
        visited = {first_node}

        path_text = ""
        reasoning_text = ""
        previous_node = first_node

        # Build the path
        for step in range(path_length - 1):
            if previous_node not in graph or not graph[previous_node]:
                # Disconnected node logic
                node = random.choice(node_list)
                safety = 0
                while node in visited and safety < 100:
                    node = random.choice(node_list)
                    safety += 1
                
                path_text += f'node_{previous_node} not connected with node_{node}. '
                if config.include_reasoning:
                    reasoning_text += f'node_{previous_node} not connected with node_{node} means there is no relationship. '

                visited.add(node)
                previous_node = node

            else:
                # Connected node logic
                next_node = random.choice(list(graph[previous_node].keys()))
                safety = 0
                while next_node in visited and safety < 100:
                    next_node = random.choice(list(graph[previous_node].keys()))
                    safety += 1

                relation = graph[previous_node][next_node]
                rel_name = relation2id.get(relation, f"relation_{relation}")

                path_text += f'node_{previous_node} has {rel_name} with node_{next_node}. '
                if config.include_reasoning:
                    reasoning_text += f'node_{previous_node} has {rel_name} with node_{next_node}. '
                
                visited.add(next_node)
                previous_node = next_node

        last_node = previous_node

        # Filter duplicates
        if path_text in unique_paths:
            continue
        unique_paths.add(path_text)

        # Check connectivity and balance dataset
        question = f'Is node_{first_node} connected with node_{last_node}?'
        is_connected = (first_node in graph and last_node in graph[first_node]) or \
                       (last_node in graph and first_node in graph[last_node])

        if is_connected and pos_count >= half_samples:
            continue
        if not is_connected and neg_count >= half_samples:
            continue

        # Construct answer logic
        answer = reasoning_text if config.include_reasoning else ""
        if has_active_rules and config.include_reasoning:
            context_phrase = "Applying symbolic rules and path analysis together: "
            answer += context_phrase
        answer += 'The answer is yes.' if is_connected else 'The answer is no.'

        if is_connected:
            pos_count += 1
        else:
            neg_count += 1

        # Construct prompt
        instruction = ""
        if config.include_reasoning:
            instruction = "###Instruction:\nAnswer the following yes/no question by reasoning step-by-step."
            if has_active_rules:
                instruction += " Use the symbolic rules as additional context along with the path information."
            instruction += "\n"
        
        prompt_parts = filter(None, [
            instruction,
            rule_context if has_active_rules else "",
            f"###Input:\n{path_text}{question}",
            f"###Response:\n{answer}"
        ])
        prompt = "\n".join(prompt_parts)

        # Append data
        data.append({
            'Prompt': prompt,
            'input_text': path_text + question,
            'output_text': answer,
            'has_rule_context': has_active_rules,
            'is_connected': is_connected
        })

    print(f"Generated {len(data)} samples (positive: {pos_count}, negative: {neg_count})") #TODO: add logger
    return pd.DataFrame(data)


