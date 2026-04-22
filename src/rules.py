"""Rules - Rule file that defines rule representations and logic."""
# TODO: Can rules have empty heads or empty bodies?
# TODO: Are subject, predicate, or object components of a rule/body always a variable?
# TODO: Are there any expected formats or standards for the rules in the .csv files?
# TODO: Docstrings

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import rdflib

from config import RunConfig, KGConfig, DirConfig
from sparql import build_sparql_query

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Atom:
    """Represents a triple composed of subject predicate object.

    The values of the each of these attributes are strings that represent either a
    variable (e.g., ?a), a resource (e.g., ex:Patient), or a literal.

    Attributes:
        subject: Subject of the triple.
        predicate: Represents the relation between subject and object.
        obj: Object of the triple.
    """

    subject: str
    predicate: str
    obj: str

    def __str__(self) -> str:
        """Returns a string of the atom as '(subject predicate object)'."""
        return f"({self.subject} {self.predicate} {self.obj})"

    def to_natural_language(self) -> str:
        """Returns a string with a natural language description of the atom."""
        nl_pred = self.predicate_to_nl()
        nl_obj = self.entity_to_nl()
        return f"{nl_pred} {nl_obj}"

    def predicate_to_nl(self) -> str:
        """Return the camelCase predicate to spaced lower-case natural language."""
        nl_pred = re.sub(r"([A-Z])", r" \1", self.predicate).strip().lower()
        if nl_pred.startswith("has "):
            nl_pred = nl_pred[4:]
        return nl_pred

    def entity_to_nl(self) -> str:
        """Return "-" spaced entity to " " spaced natural language."""
        return self.obj.replace("_", " ")


@dataclass(frozen=True, slots=True)
class RuleSignature:
    """Signature of a rule.

    Represents the rule as a body -> head.

    Attributes:
        body: The body of a rule is a set of atoms.
        head: The head of a rule is a single atom.
    """

    body: frozenset[Atom]
    head: Atom

    def __str__(self) -> str:
        """Returns the rule signature as a string '(atom), ... ,(atom) -> head'."""
        body_str = " AND ".join(f"{atom}" for atom in self.body)
        return f"{body_str} -> {str(self.head)}"

    def to_natural_language(self) -> str:
        """Returns a string with a natural language description of the rule."""
        nl_body = " AND ".join(b.to_natural_language() for b in self.body)
        nl_head = self.head.to_natural_language()
        rule_desc = (
            f"If {nl_body}, then {nl_head}.\n\n"
            f"Formal Rule:\nHead: {self.head}\nBody: {self.body}"
        )
        return rule_desc


@dataclass(slots=True)
class HornRule:
    """Represents a Horn Rule.

    Attributes:
        rule_id: String representing the ID of the rule.
        signature: The representation of the rule, containing rule's head and body.
        pca_confidence: The PCA confidence score of this rule.
        support: The support of this rule.
        head_coverage: The head coverage of the rule
    """

    rule_id: str
    signature: RuleSignature
    std_confidence: float = 0.0
    pca_confidence: float = 0.0
    support: int = 0
    head_coverage: float = 0.0
    classification: str = "UNKNOWN"

    @property
    def head(self) -> Atom:
        """Exposes the head of the rule directly for convenience."""
        return self.signature.head

    @property
    def body(self) -> frozenset[Atom]:
        """Exposes the body of the rule directly for convenience."""
        return self.signature.body

    def __str__(self) -> str:
        """Returns a formatted string representation of the rule and its stats."""
        return (
            f"{self.rule_id}: {self.signature} | "
            f"PCA conf.: {self.pca_confidence}, "
            f"supp.: {self.support}, "
            f"hc: {self.head_coverage}"
        )


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

# Compile the regex once at the module level.
ATOM_PATTERN = re.compile(r"(\?\w+)\s+(\S+)\s+(\S+)")


def parse_body(body_str: str) -> frozenset[Atom]:
    """Parse body string containing one or more atoms.

    # TODO: Define how body strings are expected.

    Args:
        body_str: String representing one or more atoms separated by whitespace.

    Returns:
        A frozenset of parsed Atom objects. Returns an empty frozenset
        if the input string is empty or falsy.
    """
    if not body_str:
        return frozenset()

    return frozenset(
        Atom(m.group(1), m.group(2), m.group(3))
        for m in ATOM_PATTERN.finditer(body_str)
    )


def parse_head(head_str: str) -> Atom:
    """Parse head string containing one atom.

    # TODO: Define how body strings are expected.

    Args:
        head_str: String representing the head atom.

    Returns:
        The parsed Atom object, or None if the string does not contain
        enough valid components.

    Raises:
        ValueError: If the string format is not valid.
    """
    if not head_str:
        raise ValueError("Head string format is not valid: Empty string.")

    parts = head_str.strip().split()

    # TODO: Consider if rules should have strictly 3 components.
    if len(parts) < 3:
        raise ValueError("Head string format is not valid: Too few components.")

    return Atom(*parts[:3])


def parse_rule_file(file: Path) -> RuleSignature:
    """To implement"""
    raise NotImplementedError()


def parse_rule_csv(rule_csv: Path) -> pd.DataFrame:
    """Parse a CSV containing a set of Horn Rules.

    Args:
        rule_csv: Path to the CSV file.

    Returns:
        A pandas DataFrame with the parsed and cleaned rules.

    Raises:
        ValueError: If the CSV does not contain 'Body' and 'Head' columns.
        FileNotFoundError: If the CSV path does not exist.
    """
    rules_df = pd.read_csv(rule_csv)

    # Ensure expected schema
    required_cols = {"Body", "Head"}
    if not required_cols.issubset(rules_df.columns):
        raise ValueError(
            f"CSV is missing required columns. "
            f"Expected {required_cols}, got {set(rules_df.columns)}"
        )

    # Clean the DF
    mask_body = rules_df["Body"].str.strip().str.startswith("?", na=False)
    mask_head = rules_df["Head"].str.strip().str.startswith("?", na=False)
    mask = mask_body & mask_head

    clean_df = rules_df[mask].reset_index(drop=True)
    removed = len(rules_df) - len(clean_df)

    if removed:
        # TODO: Implement logger
        print(f"Removed {removed} invalid rows from rules dataframe")

    return clean_df


def _get_local_name(uri: str) -> str:
    """Extract the name of a resource from a URI string.

    rpartition searches from the right end of the string, stops at the first match, and
    always returns a 3-tuple: (before, separator, after). If a match is not found, the
    whole string is returned in "after".
    """
    if "#" in uri:
        return uri.rpartition("#")[-1]
    return uri.rpartition("/")[-1]


def results_to_natural_language(
    kg_config: KGConfig,
    results: rdflib.query.Result,
    rule: HornRule,
) -> str:
    """Generate a description in natural language for a rule and all of its groundings.

    Args:
        rules_df: Pandas dataframe with the rule info.
        rule_idx: Index of the rule.

    Returns:
        A string ?
    """

    if rule.classification == "POSITIVE":
        operator = ">="
    elif rule.classification == "NEGATIVE":
        operator = "<"
    else:
        operator = "?"

    rule_metrics_text = (
        f"\nRule Statistics:\n- PCA Confidence: {rule.pca_confidence:.4f}\n"
        f"- Rule Classification: {rule.classification}"
        f"(PCA Confidence {rule.pca_confidence:.4f} {operator})"
        f"threshold {kg_config.pca_threshold})\n"
        f"- Standard Confidence: {rule.std_confidence:.4f}\n"
        f"- Positive Examples: {rule.support:.4f}\n"
        f"- Head Coverage: {rule.head_coverage:.4f}\n"
    )

    # Header: Rule information
    rule_text = (
        f"{rule.signature.to_natural_language()}\n\n"
        f"Real Instances from Knowledge Graph ({len(results)} found):\n\n"
    )

    # Body: Rule groundings information
    if len(results) == 0:
        groundings_text = "No matching instances found in the Knowledge Graph.\n"
    else:
        pca_text = (
            f"The path is classified as {rule.classification} "
            f"(PCA Confidence {rule.pca_confidence:.4f} {operator} "
            f"threshold {kg_config.pca_threshold})"
        )
        instances: list[str] = []
        # Check if the head exists in the answers
        for row in results:
            row_dict = row.asdict()

            # Get the answer
            answer = "yes" if row_dict.get("_head_exists") == "true" else "no"

            # Head: Show full head if exists, else show just the variable
            head_string = (
                f"{row_dict.get(rule.signature.head.subject)} has "
                f"{rule.signature.head.predicate_to_nl()}"
                f"{rule.signature.head.entity_to_nl()}"
                if answer == "yes"
                else f"{row_dict.get(rule.signature.head.subject)}"
            )

            # Body: rendered with their correct bound subject
            body_facts_str: list[str] = []
            for atom in rule.signature.body:
                subject_str = (
                    row_dict.get(atom.subject.lstrip("?"), "")
                    if atom.subject.startswith("?")
                    else atom.subject
                )
                obj_str = (
                    row_dict.get(atom.obj.lstrip("?"), "")
                    if atom.obj.startswith("?")
                    else atom.obj
                )
                body_facts_str.append(f"{subject_str} has {atom.predicate} {obj_str}")

            facts_str = "\n".join(fact_str for fact_str in body_facts_str)
            row_text = f"{head_string}\n{facts_str} {pca_text} Answer: {answer}"
            instances.append(row_text)

        groundings_text = "\n".join(instance_str for instance_str in instances)

    final_text = f"{rule_text}{groundings_text}{rule_metrics_text}"

    return final_text


def convert_all_rules_to_natural_language(
    kg_config: KGConfig, 
    data_config: DirConfig, 
    graph: rdflib.Graph, 
    rules_df: pd.DataFrame
) -> None:
    """Creates a natural language description of all rules and groundings for each rule
    in the rules_csv file and stores them in separate files.

    TODO: Complete docs.
    TODO: Este bucle for sobra seguro.
    """
    for rule_idx, row in enumerate(rules_df.itertuples(index=False)):
        # Build a HornRule object
        rule_id = str(rule_idx + 1)

        rule_signature = RuleSignature(
            head=parse_head(str(row.Head)),
            body=parse_body(str(row.Body)),
        )

        std_confidence = float(row.Std_Confidence)

        pca_confidence = row.PCA_Confidence
        if pca_confidence is None:
            pca_confidence = "Not available"
            classification = "UNKNOWN"
        elif float(pca_confidence) >= kg_config.pca_threshold:
            classification = "POSITIVE"
        else:
            classification = "NEGATIVE"

        support = int(row.Positive_Examples)

        head_coverage = float(row.Head_Coverage)

        rule = HornRule(
            rule_id=rule_id,
            signature=rule_signature,
            std_confidence=std_confidence,
            pca_confidence=float(pca_confidence),
            support=support,
            head_coverage=head_coverage,
            classification=classification,
        )

        # Retrieve groundings of the HornRule from the KG
        query_str = build_sparql_query(
            rule, kg_config.namespace_prefix, kg_config.namespace
        )
        results = graph.query(query_str)

        # Generate a rule + groundings natural language description
        natural_language_results: str = results_to_natural_language(
            kg_config=kg_config, results=results, rule=rule
        )

        # Save the rule + groundings natural language description as a file
        rule_file_path = data_config.output_dir / f"rule_{rule_id}.txt"
        rule_file_path.write_text(natural_language_results, encoding="utf-8")


def load_rules_from_path(rules_directory: Path) -> list[RuleSignature]:
    """TODO: docs"""
    if not rules_directory.is_dir():
        raise NotADirectoryError(f"The directory '{rules_directory}' was not found.")

    rules: list[RuleSignature] = []

    rule_files = list(rules_directory.glob("rule_*.txt"))

    def extract_rule_number(file_path: Path) -> int:
        match = re.search(r"\d+", file_path.name)
        return int(match.group()) if match else 0

    rule_files.sort(key=extract_rule_number)

    for file_path in rule_files:
        try:
            rule_info = parse_rule_file(file_path)
            rules.append(rule_info)

        except ValueError as e:
            # TODO: implement logging
            print(f"Warning: Skipping malformed file {file_path.name}: {e}")

        except FileNotFoundError as e:
            # TODO: Implement logging
            print(f"Cannot find file {file_path.name}: {e}")

    # TODO: Logger maybe
    return rules


def create_rule_context(rules: list[RuleSignature], max_rules: int = 3) -> str:
    """TODO: docs and logic"""
    raise NotImplementedError


# TODO: Move to main
def generate_cots(
    config: RunConfig,
    graph: dict[Any, dict[Any, Any]],
    node_list: list[Any],
    id2relation: dict[Any, str],
    rules: list[RuleSignature],
) -> pd.DataFrame:
    """TODO: docs and logic or DELETE"""
    data: list[dict[str, Any]] = []
    unique_paths: set[str] = set()
    pos_count = 0
    neg_count = 0

    # Initialize Context
    rule_context = ""
    if config.cot_generation.use_rules and rules:
        rule_context = create_rule_context(
            rules, max_rules=config.cot_generation.max_rules_in_context
        )

    has_active_rules = bool(rule_context)  # See if there are rules in context

    max_attempts = (
        config.cot_generation.max_samples * config.cot_generation.attempts_multiplier
    )
    attempts = 0
    half_samples = config.cot_generation.max_samples // 2

    # CoT generation loop
    while len(data) < config.cot_generation.max_samples and attempts < max_attempts:
        attempts += 1

        path_length = random.randint(2, config.cot_generation.max_path_length)
        first_node = random.choice(node_list)
        visited = {first_node}

        path_text = ""
        reasoning_text = ""
        previous_node = first_node

        # Build the path
        for _ in range(path_length - 1):
            if previous_node not in graph or not graph[previous_node]:
                # Disconnected node logic
                node = random.choice(node_list)
                safety = 0
                while node in visited and safety < 100:
                    node = random.choice(node_list)
                    safety += 1

                path_text += f"node_{previous_node} not connected with node_{node}. "
                if config.cot_generation.include_reasoning:
                    reasoning_text += f"node_{previous_node} not connected with node_{node} means there is no relationship. "

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
                rel_name = id2relation.get(relation, f"relation_{relation}")

                path_text += (
                    f"node_{previous_node} has {rel_name} with node_{next_node}. "
                )
                if config.cot_generation.include_reasoning:
                    reasoning_text += (
                        f"node_{previous_node} has {rel_name} with node_{next_node}. "
                    )

                visited.add(next_node)
                previous_node = next_node

        last_node = previous_node

        # Filter duplicates
        if path_text in unique_paths:
            continue
        unique_paths.add(path_text)

        # Check connectivity and balance dataset
        question = f"Is node_{first_node} connected with node_{last_node}?"
        is_connected = (first_node in graph and last_node in graph[first_node]) or (
            last_node in graph and first_node in graph[last_node]
        )

        if is_connected and pos_count >= half_samples:
            continue
        if not is_connected and neg_count >= half_samples:
            continue

        # Construct answer logic
        answer = reasoning_text if config.cot_generation.include_reasoning else ""
        if has_active_rules and config.cot_generation.include_reasoning:
            context_phrase = "Applying symbolic rules and path analysis together: "
            answer += context_phrase
        answer += "The answer is yes." if is_connected else "The answer is no."

        if is_connected:
            pos_count += 1
        else:
            neg_count += 1

        # Construct prompt
        instruction = ""
        if config.cot_generation.include_reasoning:
            instruction = "###Instruction:\nAnswer the following yes/no question by reasoning step-by-step."
            if has_active_rules:
                instruction += " Use the symbolic rules as additional context along with the path information."
            instruction += "\n"

        prompt_parts = filter(
            None,
            [
                instruction,
                rule_context if has_active_rules else "",
                f"###Input:\n{path_text}{question}",
                f"###Response:\n{answer}",
            ],
        )
        prompt = "\n".join(prompt_parts)

        # Append data
        data.append(
            {
                "Prompt": prompt,
                "input_text": path_text + question,
                "output_text": answer,
                "has_rule_context": has_active_rules,
                "is_connected": is_connected,
            }
        )

    print(
        f"Generated {len(data)} samples: {neg_count} negative and {pos_count} positive."
    )  # TODO: add logger
    return pd.DataFrame(data)
