"""Defines data structures and logic for Horn Rule-based systems.

Provides the HornRule dataclass, Pandas CSV parsing, and core logic operations
(like forward chaining) to verify and evaluate inferrable predicates within a rule set.
"""

import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from utils import format_term

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Term mappings.
# ---------------------------------------------------------------------------
DEFAULT_PREFIXES: dict[str, str] = {
    "type": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "Property": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "Class": "http://www.w3.org/2000/01/rdf-schema#",
    "subClassOf": "http://www.w3.org/2000/01/rdf-schema#",
    "sameAs": "http://www.w3.org/2002/07/owl#",
    "name": "http://xmlns.com/foaf/0.1/",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
CAMEL_CASE_PATTERN = re.compile(r"(?<=[a-z])([A-Z])")


@dataclass(frozen=True, slots=True)
class Atom:  # type: ignore
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

    def get_variables(self) -> tuple[str | None, str | None]:
        """Returns the variables in this atom, ordered as [subject_var, object_var]"""
        s_var = self.subject if self.subject.startswith("?") else None
        o_var = self.obj if self.obj.startswith("?") else None
        return (s_var, o_var)

    @staticmethod
    def _clean(term: str) -> str:
        """Cleans a single term by resolving URIs, variables and formatting."""
        # If it's a variable just return without "?"
        if term.startswith("?"):
            return term.removeprefix("?")

        # Extract local name (splits by #, /, or : and takes the last element)
        term = re.split(r"[#\/:]", term)[-1]

        term = CAMEL_CASE_PATTERN.sub(r" \1", term)
        term = term.replace("_", " ").strip().lower()
        return term

    def to_natural_language(self) -> tuple[str, str, str]:
        """Extracts the name of a resource from a URI string."""
        return (
            self._clean(self.subject),
            self._clean(self.predicate),
            self._clean(self.obj),
        )


@dataclass(frozen=True, slots=True)
class RuleSignature:  # type: ignore
    """Signature of a Horn Rule."""

    rule_id: str
    body: frozenset[Atom]
    head: Atom

    def __str__(self) -> str:
        """Returns the formal representation of the rule as 'atom AND ... -> head'."""
        body_desc = " AND ".join(f"{atom}" for atom in sorted(self.body))
        return f"{body_desc} -> {self.head}"

    def __iter__(self) -> Iterator[Atom]:
        """Iterates over all atoms in the rule (body and head).

        Yields:
            Atom: Each atom composing the rule's body, followed by the head atom.
        """
        yield from self.body
        yield self.head

    def to_natural_language(self) -> str:
        """Returns a natural language description of the rule."""
        sorted_body = sorted(self.body)

        body_desc = " AND ".join(
            " ".join(atom.to_natural_language()) for atom in sorted_body
        )
        head_desc = " ".join(self.head.to_natural_language())

        body_formal = " AND ".join(f"{atom}" for atom in sorted_body)

        return (
            f"If {body_desc}, then {head_desc}.\n\n"
            f"Formal Rule:\n\tHead: {self.head}\n\tBody: {body_formal}"
        )

    def get_variables(self) -> set[str]:
        """Return unique variables starting with '?' in this rule."""
        return {
            term
            for atom in (self.head, *self.body)
            for term in (atom.subject, atom.obj)
            if term.startswith("?")
        }

    def get_head_variables(self) -> list[str]:
        """Return the head variables in a list [subject, object] or [subject]"""
        return [
            term for term in (self.head.subject, self.head.obj) if term.startswith("?")
        ]

    def get_predicates(self) -> set[str]:
        """Return the set of unique predicates in the rule."""
        return {atom.predicate for atom in (self.body | {self.head})}

    def get_body_predicates(self) -> set[str]:
        """Return the set of unique predicates in the body atoms."""
        return {atom.predicate for atom in self.body}

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
class HornRule:  # type: ignore
    """Represents a Horn Rule.

    Attributes:
        signature: Representation of the rule that contains head and body.
        pca_confidence: PCA confidence score of this rule.
        support: Support of this rule.
        head_coverage: Head coverage of the rule.
    """

    signature: RuleSignature
    support: int | float
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

    @property
    def rule_id(self) -> str:
        """Exposes the signature's id for convenience."""
        return self.signature.rule_id

    def __str__(self) -> str:
        """Returns a formatted string representation of the rule and its stats."""
        return (
            f"| Id: {self.rule_id} | Signature: {self.signature} | "
            f"PCA conf.: {self.pca_confidence} | "
            f"supp.: {self.support} | "
            f"hc: {self.head_coverage} |"
        )

    def get_body_predicates(self) -> set[str]:
        """Return the set of unique predicates in the body atoms."""
        return self.signature.get_body_predicates()

    def get_head_variables(self) -> list[str]:
        """Returns a list of the head variables."""
        return self.signature.get_head_variables()

    def get_predicates(self) -> set[str]:
        """Returns a set containing all predicates present in the rule"""
        return self.signature.get_predicates()

    def get_variables(self) -> set[str]:
        """Returns a set with all the variables present in the rule."""
        return self.signature.get_variables()


# ---------------------------------------------------------------------------
# Rule custom errors.
# ---------------------------------------------------------------------------
class RulesError(Exception):
    """Base exception for errors during rule handling."""

    pass


# ---------------------------------------------------------------------------
# Rule parsing.
# ---------------------------------------------------------------------------
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


def parse_body(
    body_str: str, term_mapping: dict[str, str], default_ns: str
) -> frozenset[Atom]:
    """Parses a body string containing one or more atoms into a frozen set of atoms."""
    if not body_str:
        logger.warning("Parsing empty body. Is this supposed to happen?")
        return frozenset()

    return frozenset(
        Atom(
            format_term(m.group(1), term_mapping, default_ns),
            format_term(m.group(2), term_mapping, default_ns),
            format_term(m.group(3), term_mapping, default_ns),
        )
        for m in ATOM_PATTERN.finditer(body_str)
    )


def parse_head(head_str: str, term_mapping: dict[str, str], default_ns: str) -> Atom:
    """Parses a head string into an Atom."""
    if not head_str:
        raise ValueError("Head string format is not valid: Empty string.")

    parts = head_str.split()
    if len(parts) < 3:
        raise ValueError("Head string format is not valid: Too few components.")

    return Atom(
        format_term(parts[0], term_mapping, default_ns),
        format_term(parts[1], term_mapping, default_ns),
        format_term(parts[2], term_mapping, default_ns),
    )


def parse_horn_rule(
    row: RuleRow,
    rule_id: str,
    term_mapping: dict[str, str],
    default_namespace: str = "http:DefaultNamespace.org/",
) -> HornRule:
    """Extracts a HornRule object from a pandas DataFrame row.

    Args:
        row: A named tuple representing a row form the rules DataFrame.
        rule_id: Assigned string identifier for the rule.

    Returns:
        A populated HornRule instance.
    """

    def _parse_metric(value: float | None) -> float:
        """Returns float or Python None for Pandas/Numpy NaNs securely."""
        return 0.0 if (pd.isna(value) or value is None) else float(value)

    rule = HornRule(
        signature=RuleSignature(
            rule_id=rule_id,
            head=parse_head(str(row.Head), term_mapping, default_namespace),
            body=parse_body(str(row.Body), term_mapping, default_namespace),
        ),
        support=_parse_metric(row.Positive_Examples),
        head_coverage=_parse_metric(row.Head_Coverage),
        std_confidence=_parse_metric(row.Std_Confidence),
        pca_confidence=_parse_metric(row.PCA_Confidence),
        classification=row.Classification,
    )

    return rule


# -----------------------------------------------------------------------------
# Rule set handling
# ---------------------------------------------------------------------------
def parse_rule_set(
    rule_dataframe: pd.DataFrame,
    term_mapping: dict[str, str],
    pca_threshold: float | None,
    default_namespace: str,
) -> dict[str, HornRule]:
    """Parse a DataFrame into a dict of HornRules identified by rule_id.

    Args:
        rules_df: DataFrame containing information for each rule in each row.

    Returns:
        A tuple containing
            - A dict of HornRules identified by rule_id.
            - A set of strings representing the predicates in the rules' head.
    """

    if pca_threshold is not None:
        rule_dataframe["Classification"] = "NEGATIVE"
        rule_dataframe.loc[
            rule_dataframe["PCA_Confidence"] >= pca_threshold, "Classification"
        ] = "POSITIVE"
        rule_dataframe.loc[
            rule_dataframe["PCA_Confidence"].isna(), "Classification"
        ] = "UNKNOWN"
    else:
        rule_dataframe["Classification"] = "UNKNOWN"

    rules: dict[str, HornRule] = {}

    for row_id, row in enumerate(rule_dataframe.itertuples(index=False), start=1):
        rule_id = f"rule_{row_id}"
        rule = parse_horn_rule(
            row=row,
            rule_id=rule_id,
            term_mapping=term_mapping,
            default_namespace=default_namespace,
        )

        rules[rule.rule_id] = rule

    return rules


def check_uninferrable_preds(
    rules: dict[str, HornRule],
    intensional_predicates: set[str],
    extensional_predicates: set[str],
) -> set[str]:
    """Calls the method to see if there is any uninferrable intensional predicate in the
    rule set.

    Args:
        rules: Dict containing all rules in the set.
        intensional_predicates: Set of all predicates that must be inferrable.
        extensional_predicates: Set of extensional predicates assumed to be inferrable.

    Raises:
        ValueError: If there are any non-inferrable predicates.
    """

    # Create rule mappings
    rule_mapping: dict[str, list[set[str]]] = defaultdict(list)
    for _, rule in rules.items():
        head_predicate = rule.head.predicate
        if head_predicate in intensional_predicates:
            body_intensional = (
                rule.signature.get_body_predicates() - extensional_predicates
            )
            rule_mapping[head_predicate].append(body_intensional)

    deducible: set[str] = set()

    # Iteratively expand the set of deducible predicates
    while True:
        added_new = False
        for head, bodies in rule_mapping.items():
            if head in deducible:
                continue
            if any((body - {head}).issubset(deducible) for body in bodies):
                deducible.add(head)
                added_new = True

        if not added_new:
            break

    return intensional_predicates - deducible


def get_ruleset_dependencies(
    rules: dict[str, HornRule],
    intensional_predicates: set[str],
) -> dict[str, set[str]]:
    """Builds a dictionary that represents rule dependencies in a ruleset."""

    # TODO: Implement dependencies between recursive rules!
    # Group rules by head predicate
    by_head: dict[str, list[str]] = defaultdict(list)
    for rule_id, rule in rules.items():
        if rule.head.predicate in intensional_predicates:
            by_head[rule.head.predicate].append(rule_id)

    # Initialize dependencies
    rule_dependency: dict[str, set[str]] = {r_id: set() for r_id in rules}

    # Process subsumptions
    for head_predicate, rule_ids in by_head.items():
        if not rule_ids:
            logger.warning("No rules generate %s, skipping.", head_predicate)
            logger.warning("This should have not happened.")
            continue

        sorted_ids = sorted(rule_ids, key=lambda x: len(rules[x].get_body_predicates()))

        for i, current_id in enumerate(sorted_ids):
            current_body = rules[current_id].get_body_predicates()

            for next_id in sorted_ids[i + 1 :]:
                next_body = rules[next_id].get_body_predicates()
                if current_body.issubset(next_body):
                    rule_dependency[current_id].add(next_id)

    logger.info("Created dependency graph for %d rules.", len(rules))
    return rule_dependency


def get_term_mapping(ontology_file: Path) -> dict[str, str]:
    """Extracts term->namespace mappings from a Turtle file using line-by-line regex.

    Scales with O(1) memory footprint by avoiding in-memory graph construction.
    """
    term_mapping: dict[str, str] = DEFAULT_PREFIXES.copy()
    custom_mapping: dict[str, str] = {}
    prefixes: dict[str, str] = {}

    # Matches: @prefix fr: <http://FrenchRoyalty.org/> .
    prefix_pattern = re.compile(r"@prefix\s+([^:]+):\s*<([^>]+)>\s*\.")

    # Matches: fr:father a rdfs:Property (captures "fr" and "father")
    term_pattern = re.compile(r"^([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+)(?=\s)")

    with open(ontology_file, encoding="utf-8") as f:
        for line in f:
            line = line.lstrip()  # Keep right spaces, just clear indents

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # 1. Catch Prefix Declarations
            if line.startswith("@prefix"):
                match = prefix_pattern.search(line)
                if match:
                    prefix, uri = match.groups()
                    prefixes[prefix] = uri
                continue

            # 2. Catch Term Definitions
            match = term_pattern.search(line)
            if match:
                prefix, term = match.groups()
                if prefix in prefixes:
                    custom_mapping[term] = prefixes[prefix]
    logger.debug("Created term to prefix mapping.")
    term_mapping.update(custom_mapping)
    return term_mapping


def get_predicate_mapping(rules: dict[str, HornRule]) -> dict[str, set[str]]:
    """Returns a mapping from predicates to the ids of rules where they are present."""

    mapping: defaultdict[str, set[str]] = defaultdict(set)

    for r_id, rule in rules.items():
        for pred in rule.get_predicates():
            mapping[pred].add(r_id)

    return dict(mapping)
