"""Rules - Rule file that defines rule representations and logic."""
# TODO: Can rules have empty heads or empty bodies?
# TODO: Are subject, predicate, or object components of a rule/body always a variable?
# TODO: Are there any expected formats or standards for the rules in the .csv files?
# TODO: Docstrings

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import rdflib

from config import KGConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configures the root logger to output to the console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Atom:
    """Represents a triple composed of subject predicate object.

    The values of the each of these attributes are strings that represent a variable
    (e.g., ?a), a resource (e.g., ex:Patient), or a literal (e.g., 10.0).

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
        variables = {
            term
            for atom in sorted(all_atoms)
            for term in (atom.subject, atom.obj)
            if term.startswith("?")
        }

        return " ".join(var for var in variables)

    def get_predicates(self) -> set[str]:
        """Return the set of unique predicates in the rule."""
        all_atoms = {self.head} | self.body
        return {atom.predicate for atom in sorted(all_atoms)}


@dataclass(slots=True)
class HornRule:
    """Represents a Horn Rule.

    Attributes:
        rule_id: String representing the ID of the rule.
        signature: Representation of the rule that contains head and body.
        pca_confidence: PCA confidence score of this rule.
        support: Support of this rule.
        head_coverage: Head coverage of the rule.
    """

    rule_id: str
    signature: RuleSignature
    support: int | float | None = None
    head_coverage: float | None = None
    std_confidence: float | None = None
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


class RuleRow(Protocol):
    """Definines the expected structure of a rule DataFrame row."""

    Head: Any
    Body: Any
    Std_Confidence: Any
    Positive_Examples: Any
    Head_Coverage: Any


def parse_body(body_str: str) -> frozenset[Atom]:
    """Parses a body string containing one or more atoms into a frozen set of atoms."""
    if not body_str:
        return frozenset()

    return frozenset(
        Atom(m.group(1), m.group(2), m.group(3))
        for m in ATOM_PATTERN.finditer(body_str)
    )


def parse_head(head_str: str) -> Atom:
    """Parses a head string into an Atom."""
    if not head_str:
        raise ValueError("Head string format is not valid: Empty string.")

    parts = head_str.split()
    if len(parts) < 3:
        raise ValueError("Head string format is not valid: Too few components.")

    return Atom(*parts[:3])


def parse_horn_rule(
    row: RuleRow,
    rule_id: str,
    pca_threshold: float | None = None,
) -> HornRule:
    """Extracts a HornRule object from a pandas DataFrame row.

    Args:
        row: A named tuple representing a row form the rules DataFrame.
        rule_id: Assigned string identifier for the rule.
        pca_threshold: Threshold for classificating a rule as POSITIVE.

    Returns:
        A populated HornRule instance.
    """

    def _fetch_attr(row: RuleRow, key: str) -> Any:
        """Returns an attribute from a RuleRow or None if it's nan."""
        value_raw = getattr(row, key, None)
        return None if pd.isna(value_raw) else float(value_raw)

    support = _fetch_attr(row, "Positive_Examples")
    head_coverage = _fetch_attr(row, "Head_Coverage")
    std_conf = _fetch_attr(row, "Std_Confidence")
    pca_conf = _fetch_attr(row, "PCA_Confidence")

    if pca_conf is None or pca_threshold is None:
        classification = "UNKNOWN"
    else:
        classification = "POSITIVE" if pca_conf >= pca_threshold else "NEGATIVE"

    return HornRule(
        rule_id=rule_id,
        signature=RuleSignature(
            head=parse_head(str(row.Head)),
            body=parse_body(str(row.Body)),
        ),
        support=support,
        head_coverage=head_coverage,
        std_confidence=std_conf,
        pca_confidence=pca_conf,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Build SPARQL query from rules
# ---------------------------------------------------------------------------


def build_sparql_query(
    rule_signature: RuleSignature, ns_prefix: str, namespace: str
) -> str:
    """Builds a SPARQL SELECT query from a rule signature.

    Body atoms map to required triple patterns. The head atom maps to an OPTIONAL triple
    pattern to determine a boolean answer. This method assumes that predicates never
    contain variables.

    Args:
        rule_signature: Signature containing the body and head atoms.
        ns_prefix: Short prefix alias for the RDF namespace (e.g., 'ex').
        namespace: Full URI string of the RDF namespace.

    Returns:
        A formatted SPARQL SELECT query string.
    """

    def format_term(term: str) -> str:
        """Formats terms: variables stay as-is, constants get the namespace prefix"""
        return term if term.startswith("?") else f"{ns_prefix}:{term}"

    def format_atom(atom: Atom) -> str:
        """Formats an entire atom into a SPARQL triple pattern."""
        # NOTE: This assumes predicates NEVER contain variables
        subject = format_term(atom.subject)
        obj = format_term(atom.obj)
        return f"{subject} {ns_prefix}:{atom.predicate} {obj} ."

    select_vars = rule_signature.get_variables()
    body_lines = "\n\t".join(format_atom(atom) for atom in rule_signature.body)
    head_line = f"{format_atom(rule_signature.head)}"

    return (
        f"PREFIX {ns_prefix}: <{namespace}>\n"
        f"SELECT DISTINCT {select_vars} ?_head_exists WHERE {{\n\t"
        f"{body_lines}\n"
        f"\tOPTIONAL {{\n"
        f"\t\t{head_line}\n"
        f"\t\tBIND(true AS ?_head_exists)\n"
        f"\t}}\n}}"
    )


# ---------------------------------------------------------------------------
# Transform rules to natural language
# ---------------------------------------------------------------------------


def _get_local_name(uri: str) -> str:
    """Extracts the name of a resource from a URI string."""
    if "#" in uri:
        return uri.rpartition("#")[-1]
    return uri.rpartition("/")[-1].replace("_", " ")


def _build_grounding_text(
    result: rdflib.query.Result,
    rule: HornRule,
    pca_threshold: float | None,
    operator: str,
    max_groundings: int | None,
) -> str:
    """Helper function to generate natural language descriptions of a query result."""

    pca_text = (
        f"The path is classified as {rule.classification} "
        f"(PCA conf. {rule.pca_confidence:.4f} {operator} {pca_threshold})"
    )

    instances: list[str] = []
    for row_idx, row in enumerate(result):
        # TODO: Implement something more efficient
        if max_groundings is not None and row_idx >= max_groundings:
            break

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
            subject = _get_local_name(row_dict.get(subject_key, subject_key))
            obj_key = atom.obj.removeprefix("?")
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
    max_groundings: int | None,
) -> str:
    """Generates natural language description for a rule and its query results."""

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
            max_groundings=max_groundings,
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


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Test SAPRQL generation
    rule = HornRule(
        rule_id="Test_rule",
        signature=RuleSignature(
            head=parse_head("?a hasSuccesor ?b"),
            body=parse_body("?a hasChild ?b"),
        ),
        std_confidence=0.9,
        pca_confidence=0.8,
        support=345,
        head_coverage=0.7,
        classification="POSITIVE",
    )
    query = build_sparql_query(
        rule_signature=rule.signature,
        ns_prefix="ex",
        namespace="http://www.example.org/",
    )
    print(query)
