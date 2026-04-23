"""Rules - Rule file that defines rule representations and logic."""
# TODO: Can rules have empty heads or empty bodies?
# TODO: Are subject, predicate, or object components of a rule/body always a variable?
# TODO: Are there any expected formats or standards for the rules in the .csv files?
# TODO: Docstrings

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import rdflib

from config import KGConfig, RunConfig

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
        """Returns a string of the atom as 'subject predicate object' string."""
        return f"{self.subject} {self.predicate} {self.obj}"

    def __lt__(self, other: "Atom") -> bool:
        """Enables native sorting of Atoms by their string representation."""
        if not isinstance(other, Atom):
            return NotImplemented
        return str(self) < str(other)

    def get_predicate_desc(self) -> str:
        """Return the camelCase predicate to spaced lower-case natural language."""
        nl_pred = re.sub(r"([A-Z])", r" \1", self.predicate).strip().lower()
        if nl_pred.startswith("has "):
            nl_pred = nl_pred[4:]
        return nl_pred

    def get_obj_desc(self) -> str:
        """Return "-" spaced entity to " " spaced natural language."""
        return self.obj.replace("_", " ")

    def get_description(self) -> str:
        """Returns a natural language description of the atom."""
        nl_pred = self.get_predicate_desc()
        nl_obj = self.get_obj_desc()
        return f"{nl_pred} {nl_obj}"


@dataclass(frozen=True, slots=True)
class RuleSignature:
    """Signature of a Horn Rule."""

    body: frozenset[Atom]
    head: Atom

    def __str__(self) -> str:
        """Returns the formal representation of the rule as 'atom AND ... -> head'."""
        body_desc = " AND ".join(f"{atom}" for atom in sorted(self.body))
        return f"{body_desc} -> {self.head}"

    def get_description(self) -> str:
        """Returns a natural language description of the rule."""
        sorted_body = sorted(self.body)

        body_desc = " AND ".join(atom.get_description() for atom in sorted_body)
        head_desc = self.head.get_description()

        body_formal = " AND ".join(f"{atom}" for atom in sorted_body)

        return (
            f"If {body_desc}, then {head_desc}.\n\n"
            f"Formal Rule:\n\tHead: {self.head}\n\tBody: {body_formal}"
        )

    def get_variables(self) -> str:
        """Return all the variables present in the rule as 'var ... var'."""
        all_atoms = {self.head} | self.body

        return " ".join(
            term
            for atom in sorted(all_atoms)
            for term in (atom.subject, atom.obj)
            if term.startswith("?")
        )


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
    support: int = 0
    head_coverage: float = 0.0
    std_confidence: float = 0.0
    pca_confidence: float | None = None
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
    """Parse body string containing one or more atoms."""
    if not body_str:
        return frozenset()

    return frozenset(
        Atom(m.group(1), m.group(2), m.group(3))
        for m in ATOM_PATTERN.finditer(body_str)
    )


def parse_head(head_str: str) -> Atom:
    """Parse a head string into an Atom based on the first three whitespace-separated
    components."""
    if not head_str:
        raise ValueError("Head string format is not valid: Empty string.")

    parts = head_str.split()
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


# ---------------------------------------------------------------------------
# Build SPARQL query from rules
# ---------------------------------------------------------------------------


def build_sparql_query(
    rule_signature: RuleSignature, ns_prefix: str, namespace: str
) -> str:
    """Builds a SPARQL SELECT query from a rule signature.

    Body atoms map to required triple patterns.
    The head atom maps to an OPTIONAL triple pattern to determine a boolean answer.

    Args:
        rule: The signature containing the body and head atoms.
        ns_prefix: The short prefix alias for the RDF namespace (e.g., 'ex').
        namespace: The full URI string of the RDF namespace.

    Returns:
        A formatted SPARQL SELECT query string.
    """

    def format_term(term: str) -> str:
        """Format a term: variables stay as-is, constants get the namespace prefix"""
        return term if term.startswith("?") else f"{ns_prefix}:{term}"

    def format_atom(atom: Atom) -> str:
        """Format an entire atom into a SPARQL triple pattern."""
        # NOTE: This assumes predicates NEVER contain variables
        subject = format_term(atom.subject)
        obj = format_term(atom.obj)
        return f"\t{subject} {ns_prefix}:{atom.predicate} {obj} ."

    select_vars = rule_signature.get_variables()

    def term(t: str) -> str:
        """Format a term for SPARQL: variable stays as-is, constants get prefix."""
        return t if t.startswith("?") else f"{ns_prefix}:{t}"

    body_lines = "\n".join(format_atom(atom) for atom in rule_signature.body)
    head_line = f"{format_atom(rule_signature.head)}"

    return (
        f"PREFIX {ns_prefix}: <{namespace}>\n"
        f"SELECT DISTINCT {select_vars} ?_head_exists WHERE {{\n"
        f"\t{body_lines}\n"
        f"\tOPTIONAL {{\n"
        f"\t{head_line}\n"
        f"\t\tBIND(true AS ?_head_exists)\n"
        f"\t}}\n"
        f"}}"
    )


# ---------------------------------------------------------------------------
# Transform rules to natural language
# ---------------------------------------------------------------------------


def _get_local_name(uri: str) -> str:
    """Extract the name of a resource from a URI string.

    rpartition searches from the right end of the string, stops at the first match, and
    always returns a 3-tuple: (before, separator, after). If a match is not found, the
    whole string is returned in "after".
    """
    if "#" in uri:
        return uri.rpartition("#")[-1]
    return uri.rpartition("/")[-1].replace("_", " ")


def _build_grounding_text(
    result: rdflib.query.Result, rule: HornRule, pca_threshold: str, operator: str
) -> str:
    """Helper function to generate natural language descriptions for each grounding from
    a retrieved query result."""

    pca_text = (
        f"The path is classified as {rule.classification} "
        f"(PCA conf. {rule.pca_confidence:.4f} {operator} {pca_threshold})"
    )

    instances: list[str] = []
    for row in result:
        row_dict = row.asdict()

        # Varaible ?_head_exists only has values bound if a head is found, else is None
        answer = "yes" if row_dict.get("_head_exists") is not None else "no"

        # Head: Show full head if exists, else show just the subject value
        subject_key = rule.signature.head.subject.removeprefix("?")
        subject = _get_local_name(row_dict.get(subject_key))
        head_text = f"{subject}"
        if answer == "yes":
            obj_key = rule.signature.head.obj.removeprefix("?")
            obj = _get_local_name(
                row_dict.get(obj_key, rule.signature.head.get_obj_desc())
            )
            predicate = _get_local_name(rule.signature.head.get_predicate_desc())
            head_text = f"{subject} has {predicate} {obj}"

        # Body: rendered with their correct bound subject
        body_facts_str: list[str] = []
        for atom in rule.signature.body:
            subject_key = atom.subject.removeprefix("?")
            obj_key = atom.obj.removeprefix("?")
            subject = _get_local_name(row_dict.get(subject_key, subject_key))
            obj = _get_local_name(row_dict.get(obj_key, obj_key))
            body_facts_str.append(f"{subject} has {atom.predicate} {obj}")

        facts_str = "\n".join(fact_str for fact_str in body_facts_str)
        row_text = f"{head_text}, {facts_str}\n{pca_text}\nAnswer: {answer}\n"
        instances.append(row_text)

    return "\n".join(instance_str for instance_str in instances)


def query_result_to_natural_language(
    kg_config: KGConfig,
    result: rdflib.query.Result,
    rule: HornRule,
) -> str:
    """Generate natural language description for a rule and its groundings from a query
    result."""

    operator_mapping = {"POSITIVE": ">=", "NEGATIVE": "<", "UNKNOWN": "?"}
    operator = operator_mapping.get(rule.classification, "?")

    # Description header (rule information)
    desc_header = (
        f"{rule.signature.get_description()}\n\n"
        f"Real Instances from Knowledge Graph ({len(result)} found):"
    )

    # Description body (rule groundings)
    if len(result) == 0:
        desc_body = "No matching instances found in the Knowledge Graph."
    else:
        desc_body = _build_grounding_text(
            result=result,
            rule=rule,
            pca_threshold=kg_config.pca_threshold,
            operator=operator,
        )

    # Desciption footer
    _footer_title = "Rule Metrics:"
    _rule_classification_text = (
        f"Rule Classification: {rule.classification} "
        f"(PCA conf. {rule.pca_confidence:.4f} {operator} {kg_config.pca_threshold})"
    )
    desc_footer = (
        f"{_footer_title}\n"
        f"\tPCA Confidence: {rule.pca_confidence:.4f}\n"
        f"\tStandard Confidence: {rule.std_confidence:.4f}\n"
        f"\t{_rule_classification_text}\n"
        f"\tPositive Examples: {rule.support:.4f}\n"
        f"\tHead Coverage: {rule.head_coverage:.4f}"
    )

    final_text = f"{desc_header}\n\n{desc_body}\n\n{desc_footer}"

    return final_text


class RuleRow(Protocol):
    """Protocol defining the expected structure of a rules DataFrame row."""

    Head: Any
    Body: Any
    Std_Confidence: Any
    Positive_Examples: Any
    Head_Coverage: Any


def parse_horn_rule(row: RuleRow, rule_id: str, pca_threshold: float) -> HornRule:
    """Extracts a HornRule object from a pandas DataFrame row.

    Args:
        row: A named tuple representing a row form the rules DataFrame.
        rule_id: The assigned string identifier for the rule.
        pca_threshold: The threshold for clarifying a rule as POSITIVE.

    Returns:
        A populated HornRule instance.
    """

    pca_raw = getattr(row, "PCA_Confidence", None)
    pca_conf: float | None = None if pd.isna(pca_raw) else float(pca_raw)

    if pca_conf is None:
        classification = "UNKNOWN"
    else:
        classification = "POSITIVE" if pca_conf >= pca_threshold else "NEGATIVE"

    return HornRule(
        rule_id=rule_id,
        signature=RuleSignature(
            head=parse_head(str(row.Head)),
            body=parse_body(str(row.Body)),
        ),
        std_confidence=float(row.Std_Confidence),
        pca_confidence=pca_conf if pca_conf is not None else None,
        support=int(row.Positive_Examples),
        head_coverage=float(row.Head_Coverage),
        classification=classification,
    )


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
            logger.warning("Warning: Skipping malformed file %s: %s", file_path.name, e)

        except FileNotFoundError as e:
            logger.error("Cannot find file %s: %s", file_path.name, e)

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

    logger.info(
        "Generated %d samples: %d negative and %d positive.",
        len(data),
        neg_count,
        pos_count,
    )

    return pd.DataFrame(data)
