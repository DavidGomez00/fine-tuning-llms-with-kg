import re

def parse_rule_file(file_path):
    """ Generate rule definition in natural language from a rule file.
    INPUT: 
        - file_path
    RETURN:
        - dict
    """
    with open(file_path, 'r') as f:
        content = f.read()

    rule_info = {
        'rule_id': None,
        'rule_text': None,
        'head': None,
        'body': None,
        'instances': [],
        'pca_confidence': None,
        'classification': None
    }

    # Extract rule ID and text
    rule_match = re.search(r'Rule (\d+):\s*(.+?)(?=\n\nFormal Rule:)', content, re.DOTALL)
    if rule_match:
        rule_info['rule_id'] = rule_match.group(1)
        rule_info['rule_text'] = rule_match.group(2).strip()

    # Extract head and body predicates
    head_match = re.search(r'Head:\s*(.+)', content)
    body_match = re.search(r'Body:\s*(.+)', content)
    if head_match:
        rule_info['head'] = head_match.group(1).strip()
    if body_match:
        rule_info['body'] = body_match.group(1).strip()

    # Extract instances
    instances_section = re.search(r'Real Instances from Knowledge Graph.*?:\n\n(.+?)(?=\n\nRule Statistics:)', content, re.DOTALL)
    if instances_section:
        for line in instances_section.group(1).strip().split('\n'):
            if line.strip():
                rule_info['instances'].append(line.strip())

    # Extract PCA confidence and classification
    pca_match = re.search(r'PCA Confidence:\s*([\d.]+)', content)
    classification_match = re.search(r'Rule Classification:\s*(\w+)', content)
    if pca_match:
        rule_info['pca_confidence'] = float(pca_match.group(1))
    if classification_match:
        rule_info['classification'] = classification_match.group(1)

    return rule_info


def load_all_rules(rules_directory):
    """ Load all rules from a directory.
    INPUT: 
        - rules_directory
    RETURNS:
        - list[dict]
    """
    rules = []
    rule_files = sorted(Path(rules_directory).glob('rule_*.txt'))
    for file_path in rule_files:
        try:
            rule_info = parse_rule_file(file_path)
            rules.append(rule_info)
            print(f"Loaded {file_path.name}: {rule_info['rule_text'][:60] if rule_info['rule_text'] else 'N/A'}...")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return rules


def create_rule_context(rules, max_rules=3):
    """ Create a natural language string describing symbolic rules for prompt injection.
    INPUT:
        - rules: list of rules.
    RETURN:
        - str
    """
    if not rules:
        return ""
    rule_context = "\n###Symbolic Rules (for reference):\n"
    for i, rule in enumerate(rules[:max_rules], 1):
        if rule.get('rule_text'):
            rule_context += f"{i}. {rule['rule_text']}\n"
            if rule.get('pca_confidence') is not None:
                rule_context += f"   (Confidence: {rule['pca_confidence']:.3f})\n"
    rule_context += "\n"
    return rule_context


def generate_training_data_with_rules(
    graph,
    node_list,
    relation2id,
    rules,
    total_samples=1000,
    max_path_length=10,
    include_reasoning=True,
    use_rules=True
):
    """ Generate training data with chain-of-thought reasoning.

    Args:
        graph: Knowledge graph dictionary
        node_list: List of nodes
        relation2id: Relation ID to name mapping
        rules: List of symbolic rules
        total_samples: Number of samples to generate
        max_path_length: Maximum path length
        include_reasoning: Whether to include reasoning traces
        use_rules: Whether to include symbolic rule context

    Returns:
        DataFrame with training examples.
    """
    # TODO: Implementar patrón de Objetos de Configuración en los argumentos de la función
    data = []
    unique_paths = set()
    pos_count = 0
    neg_count = 0

    rule_context = ""
    if use_rules and rules:
        rule_context = create_rule_context(rules, max_rules=CONFIG['max_rules_in_context'])

    max_attempts = total_samples * 10
    attempts = 0

    while len(data) < total_samples and attempts < max_attempts:
        attempts += 1

        path_length = random.randint(2, max_path_length)
        first_node = random.choice(node_list)
        visited = {first_node}
        path_text = ""
        reasoning_text = ""
        previous_node = first_node

        for step in range(path_length - 1):
            if previous_node not in graph or not graph[previous_node]:
                node = random.choice(node_list)
                safety = 0
                while node in visited and safety < 100:
                    node = random.choice(node_list)
                    safety += 1
                path_text += f'node_{previous_node} not connected with node_{node}. '
                if include_reasoning:
                    reasoning_text += f'node_{previous_node} not connected with node_{node} means there is no relationship. '
                visited.add(node)
                previous_node = node
            else:
                next_node = random.choice(list(graph[previous_node].keys()))
                safety = 0
                while next_node in visited and safety < 100:
                    next_node = random.choice(list(graph[previous_node].keys()))
                    safety += 1

                relation = graph[previous_node][next_node]
                rel_name = relation2id.get(relation, f"relation_{relation}")

                path_text += f'node_{previous_node} has {rel_name} with node_{next_node}. '
                if include_reasoning:
                    reasoning_text += f'node_{previous_node} has {rel_name} with node_{next_node}. '
                visited.add(next_node)
                previous_node = next_node

        last_node = previous_node

        if path_text in unique_paths:
            continue
        unique_paths.add(path_text)

        question = f'Is node_{first_node} connected with node_{last_node}?'
        is_connected = (first_node in graph and last_node in graph[first_node]) or \
                       (last_node in graph and first_node in graph[last_node])

        if is_connected and pos_count >= total_samples // 2:
            continue
        if not is_connected and neg_count >= total_samples // 2:
            continue

        if is_connected:
            if include_reasoning:
                answer = reasoning_text
                if use_rules and rule_context:
                    answer += "Applying symbolic rules and path analysis together: "
                answer += 'The answer is yes.'
            else:
                answer = 'The answer is yes.'
            pos_count += 1
        else:
            if include_reasoning:
                answer = reasoning_text
                if use_rules and rule_context:
                    answer += "Based on symbolic rules and path analysis: "
                answer += 'The answer is no.'
            else:
                answer = 'The answer is no.'
            neg_count += 1

        # Construct prompt
        if include_reasoning:
            if use_rules and rule_context:
                prompt = f"###Instruction:\nAnswer the following yes/no question by reasoning step-by-step. Use the symbolic rules as additional context along with the path information.\n{rule_context}###Input:\n{path_text}{question}\n\n###Response:\n{answer}"
            else:
                prompt = f"###Instruction:\nAnswer the following yes/no question by reasoning step-by-step.\n\n###Input:\n{path_text}{question}\n\n###Response:\n{answer}"
        else:
            if use_rules and rule_context:
                prompt = f"{rule_context}###Input:\n{path_text}{question}\n\n###Response:\n{answer}"
            else:
                prompt = f"###Input:\n{path_text}{question}\n\n###Response:\n{answer}"

        data.append({
            'Prompt': prompt,
            'input_text': path_text + question,
            'output_text': answer,
            'has_rule_context': use_rules and bool(rule_context),
            'is_connected': is_connected
        })

    print(f"Generated {len(data)} samples (positive: {pos_count}, negative: {neg_count})")
    return pd.DataFrame(data)