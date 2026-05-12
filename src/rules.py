"""Defines data structures and logic for Horn Rule-based systems.

Provides the HornRule dataclass, Pandas CSV parsing, and core logic operations
(like forward chaining) to verify and evaluate inferrable predicates within a rule set.
"""

import itertools
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


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configures the root logger to output to the console.

    Args:
        level: The logging level to set. Accepts standard logging integers
               (e.g., logging.DEBUG) or strings (e.g., "INFO", "DEBUG").
    """
    if isinstance(level, str):
        level = level.upper()

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

CAMEL_CASE_PATTERN = re.compile(r"(?<=[a-z])([A-Z])")


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

    def __contains__(self, item: str) -> bool:
        """Returns True if the term is in the atom, False otherwise."""
        return item in (self.subject, self.predicate, self.obj)

    @staticmethod
    def _clean_term(term: str) -> str:
        """Cleans a single term by resolving URIs, variables and formatting."""
        # TODO: Maybe change name ot "to_natural_language" ??
        # If it's a variable just return without "?"
        if term.startswith("?"):
            return term.removeprefix("?")

        # Extract local name (splits by #, /, or : and takes the last element)
        term = re.split(r"[#\/:]", term)[-1]

        term = CAMEL_CASE_PATTERN.sub(r" \1", term)
        term = term.replace("_", " ").strip().lower()
        return term

    @staticmethod
    def _to_sparql(term: str, namespace: str) -> str:
        """Returns a term in sparql compatible format."""
        if term.startswith("?") or term.startswith("<"):
            return term
        else:
            # Remove URI
            term = re.split(r"[#\/:]", term)[-1]
            return f"<{namespace}{term}>"

    def get_local_names(self) -> tuple[str, str, str]:
        """Extracts the name of a resource from a URI string."""
        return (
            self._clean_term(self.subject),
            self._clean_term(self.predicate),
            self._clean_term(self.obj),
        )

    def get_description(self) -> str:
        """Returns a natural language description of the atom."""
        return " ".join(term for term in self.get_local_names())

    def to_sparql(self, namespace: str) -> str:
        """Returns the atom in sparql format."""
        return (
            f"{self._to_sparql(self.subject, namespace)} "
            f"{self._to_sparql(self.predicate, namespace)} "
            f"{self._to_sparql(self.obj, namespace)} ."
        )


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

    def get_variables(self, sort: bool = True) -> list[str]:
        """Return unique variables starting with '?' in this rule.

        Args:
            sort: Whether to return variables in alphabetical order to ensure stable
            SPARQL SELECT clauses.
        """
        vars_set = {
            term
            for atom in (self.head, *self.body)
            for term in (atom.subject, atom.obj)
            if term.startswith("?")
        }
        return sorted(vars_set) if sort else list(vars_set)

    def get_head_variables(self, sort: bool = True) -> list[str]:
        """Return unique variables starting with '?' in this rule's head.

        Args:
            sort: Whether to return variables in alphabetical order to ensure stable
            SPARQL SELECT clauses.
        """
        vars_set = {
            term for term in (self.head.subject, self.head.obj) if term.startswith("?")
        }
        return sorted(vars_set) if sort else list(vars_set)

    def get_predicates(self) -> set[str]:
        """Return the set of unique predicates in the rule."""
        return {atom.predicate for atom in (self.body | {self.head})}

    def get_body_predicates(self) -> set[str]:
        """Return the set of unique predicates in the body atoms."""
        return {atom.predicate for atom in self.body}

    def head_with_ns(self, namespace: str) -> str:
        return self.head.to_sparql(namespace)

    def body_with_ns(self, namespace: str) -> set[str]:
        return set(atom.to_sparql(namespace) for atom in self.body)

    def rule_query_body(self, var_mappings: dict[str, str], namespace: str) -> set[str]:
        return {
            (
                f"{var_mappings.get(atom.subject, atom.subject)} "
                f"<{namespace}{atom.predicate}> "
                f"{var_mappings.get(atom.obj, atom.obj)}"
            )
            for atom in self.body
        }


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
            f"| Id: {self.rule_id} | Signature: {self.signature} | "
            f"PCA conf.: {self.pca_confidence} | "
            f"supp.: {self.support} | "
            f"hc: {self.head_coverage} |"
        )


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

# Compile the regex once at the module level.
ATOM_PATTERN = re.compile(r"(\?\w+)\s+(\S+)\s+(\S+)")


class RuleRow(Protocol):
    """Definines the expected structure of a rule DataFrame row."""

    Head: str
    Body: str
    Std_Confidence: float
    Positive_Examples: float
    Head_Coverage: float
    PCA_Confidence: float
    Classification: str


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


def _parse_metric(value: float | None) -> float | None:
    """Returns float or Python None for Pandas/Numpy NaNs securely."""
    if pd.isna(value) or value is None:
        return None
    else:
        return float(value)


def parse_horn_rule(
    row: RuleRow,
    rule_id: str,
) -> HornRule:
    """Extracts a HornRule object from a pandas DataFrame row.

    Args:
        row: A named tuple representing a row form the rules DataFrame.
        rule_id: Assigned string identifier for the rule.

    Returns:
        A populated HornRule instance.
    """

    rule = HornRule(
        rule_id=rule_id,
        signature=RuleSignature(
            head=parse_head(str(row.Head)),
            body=parse_body(str(row.Body)),
        ),
        support=_parse_metric(row.Positive_Examples),
        head_coverage=_parse_metric(row.Head_Coverage),
        std_confidence=_parse_metric(row.Std_Confidence),
        pca_confidence=_parse_metric(row.PCA_Confidence),
        classification=row.Classification,
    )

    logger.debug("Parsed rule: %s", rule)
    return rule


# --
# Rule set handling
# --
def parse_rule_set(
    rules_df: pd.DataFrame, pca_threshold: float | None
) -> tuple[dict[str, HornRule], set[str]]:
    """Parse a DataFrame into a dict of HornRules identified by rule_id.

    Args:
        rules_df: DataFrame containing information for each rule in each row.

    Returns:
        A tuple containing
            - A dict of HornRules identified by rule_id.
            - A set of strings representing the predicates in the rules' head.
    """
    # Doing this here is more optimal than comparing for each row
    if pca_threshold is not None:
        rules_df["Classification"] = "NEGATIVE"
        rules_df.loc[rules_df["PCA_Confidence"] >= pca_threshold, "Classification"] = (
            "POSITIVE"
        )
        rules_df.loc[rules_df["PCA_Confidence"].isna(), "Classification"] = "UNKNOWN"
    else:
        rules_df["Classification"] = "UNKNOWN"

    rules: dict[str, HornRule] = {}
    intensional_predicates: set[str] = set()

    for row_id, row in enumerate(rules_df.itertuples(index=False), start=1):
        rule_id = f"rule_{row_id}"
        rule = parse_horn_rule(row, rule_id)

        rules[rule.rule_id] = rule
        intensional_predicates.add(rule.head.predicate)

    return rules, intensional_predicates


def get_uninferrable_predicates(
    rule_mapping: dict[str, list[set[str]]],
    intensional_predicates: set[str],
) -> set[str]:
    """Identifies intensional predicates that cannot be inferred from the ruleset.

    Uses bottom-up forward chaining to avoid deep recursion and safely handle cyclic
    dependencies.

    Args:
        rule_mapping: Dictionary mapping head predicates to lists of their rule bodies.
        intensional_predicates: Set of all intensional predicates to be inferred.

    Returns:
        A set of strings containing the predicates that cannot be deduced.
        An empty set indicates all intensional predicates are inferrable.
    """
    deducible: set[str] = set()

    # Iteratively expand the set of deducible predicates
    while True:
        added_new = False
        for head, bodies in rule_mapping.items():
            if head in deducible:
                continue
            if any(body.issubset(deducible) for body in bodies):
                deducible.add(head)
                added_new = True

        if not added_new:
            break

    uninferrable = intensional_predicates - deducible
    return uninferrable


# ---------------------------------------------------------------------------
# Build SPARQL query from rules
# ---------------------------------------------------------------------------


def build_rule_query(
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

    select_vars = " ".join(rule_signature.get_variables())
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


def build_ruleset_query(rule_set: dict[str, HornRule], namespace: str) -> str:
    """Builds a single query to retrieve variables that satisfy each rule in the set."""
    subqueries: list[str] = []
    global_vars: set[str] = {"?rule_id"}

    for rule_id, rule in rule_set.items():
        var_list = rule.signature.get_head_variables(sort=True)
        global_vars.update(var_list)

        proj = " ".join(var_list)

        body_sparql = " . \n      ".join(
            [atom.to_sparql(namespace) for atom in rule.body]
        )
        subqueries.append(
            f"  {{\n"
            f'    SELECT ("{rule_id}" AS ?rule_id) {proj}\n'
            f"    WHERE {{\n"
            f"      {body_sparql}\n"
            f"    }}\n"
            f"  }}"
        )

    outer_proj = " ".join(sorted(global_vars))
    query = f"SELECT {outer_proj}\nWHERE {{\n" + "\n  UNION\n".join(subqueries) + "\n}"

    logger.debug("Created ruleset query for %d rules:\n%s", len(rule_set), query)
    return query


# ---------------------------------------------------------------------------
# Transform rules to natural language
# ---------------------------------------------------------------------------


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

    bounded_result = (
        itertools.islice(result, max_groundings) if max_groundings else result
    )

    for row in bounded_result:
        row_data: dict[str, Any] = {str(var): row[var] for var in row.labels}
        answer = "yes" if row_data.get("_head_exists") is not None else "no"

        # Unpack head
        sub_key, pred_key, obj_key = rule.head.get_local_names()

        subject = Atom._clean_term(str(row_data.get(sub_key, sub_key)))

        if answer == "yes":
            obj = Atom._clean_term(str(row_data.get(obj_key, obj_key)))
            predicate = Atom._clean_term(str(row_data.get(pred_key, pred_key)))
            head_text = f"{subject} has {predicate} {obj}"
        else:
            head_text = str(subject)

        # Unpack body
        body_facts: list[str] = []
        for atom in rule.signature.body:
            sub_key, _, obj_key = atom.get_local_names()

            subject = Atom._clean_term(str(row_data.get(sub_key, sub_key)))
            obj = Atom._clean_term(str(row_data.get(obj_key, obj_key)))

            body_facts.append(f"{subject} has {atom.predicate} {obj}")

        facts_str = "\n".join(body_facts)
        row_text = f"{head_text}, {facts_str}\n{pca_text}\nAnswer: {answer}\n"
        instances.append(row_text)

    return "\n".join(instances)


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

    # Description footer
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
    query = build_rule_query(
        rule_signature=rule.signature,
        ns_prefix="ex",
        namespace="http://www.example.org/",
    )
    print(query)
