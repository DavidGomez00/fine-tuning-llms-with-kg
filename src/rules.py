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

    def get_variables(self) -> tuple[str | None, str | None]:
        """Returns the variables in this atom, ordered as [subject_var, object_var]"""
        s_var = self.subject if self.subject.startswith("?") else None
        o_var = self.obj if self.obj.startswith("?") else None
        return (s_var, o_var)

    def to_natural_language(self) -> tuple[str, str, str]:
        """Extracts the name of a resource from a URI string."""

        def clean(term: str) -> str:
            """Resolve URIs, variables and formatting."""
            if term.startswith("?"):
                return term.removeprefix("?")

            term = re.split(r"[#\/:]", term)[-1]
            term = CAMEL_CASE_PATTERN.sub(r" \1", term)
            term = term.replace("_", " ").strip().lower()
            return term

        return (clean(self.subject), clean(self.predicate), clean(self.obj))


@dataclass(frozen=True, slots=True)
class RuleSignature:
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
        yield from sorted(self.body)
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

    def get_extensional_body(self, intesional_preds: set[str]) -> frozenset[Atom]:
        """Returns the rule's body excluding atoms with intensional predicates."""
        return frozenset(
            {atom for atom in self.body if atom.predicate not in intesional_preds}
        )

    def get_extensional_preds(self, intensional_preds: set[str]) -> set[str]:
        """Returns body predicates excluding the ones in 'intensional_preds'."""
        return {
            atom.predicate
            for atom in self.body
            if atom.predicate not in intensional_preds
        }


@dataclass(slots=True)
class HornRule:
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

    def get_extensional_body(self, intensional_preds: set[str]) -> frozenset[Atom]:
        """Returns rule's body excluding atoms that contain intensional predicates."""
        return self.signature.get_extensional_body(intensional_preds)

    def get_head_variables(self) -> list[str]:
        """Returns a list of the head variables."""
        return self.signature.get_head_variables()

    def get_predicates(self) -> set[str]:
        """Returns a set containing all predicates present in the rule"""
        return self.signature.get_predicates()

    def get_variables(self) -> set[str]:
        """Returns a set with all the variables present in the rule."""
        return self.signature.get_variables()

    def get_extensional_preds(self, intensional_preds: set[str]) -> set[str]:
        """Returns body predicates excluding the ones in 'intensional_preds'."""
        return self.signature.get_extensional_preds(intensional_preds)


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


def parse_body(body_str: str, term_mapping: dict[str, str]) -> frozenset[Atom]:
    """Parses a body string containing one or more atoms into a frozen set of atoms."""
    if not body_str:
        logger.warning("Parsing empty body. Is this supposed to happen?")
        return frozenset()

    return frozenset(
        Atom(
            format_term(m.group(1), term_mapping),
            format_term(m.group(2), term_mapping),
            format_term(m.group(3), term_mapping),
        )
        for m in ATOM_PATTERN.finditer(body_str)
    )


def parse_head(head_str: str, term_mapping: dict[str, str]) -> Atom:
    """Parses a head string into an Atom."""
    if not head_str:
        raise ValueError("Head string format is not valid: Empty string.")

    parts = head_str.split()
    if len(parts) < 3:
        raise ValueError("Head string format is not valid: Too few components.")

    return Atom(
        format_term(parts[0], term_mapping),
        format_term(parts[1], term_mapping),
        format_term(parts[2], term_mapping),
    )


def parse_horn_rule(
    row: RuleRow,
    rule_id: str,
    term_mapping: dict[str, str],
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
            head=parse_head(str(row.Head), term_mapping),
            body=parse_body(str(row.Body), term_mapping),
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
) -> dict[str, HornRule]:
    # TODO: Add csv reading for pandas here
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


def get_dependencies_intensional(rules: dict[str, HornRule]) -> dict[str, set[str]]:
    """Builds a dictionary that represents rule dependencies in a ruleset based on the
    head of the rule. A rule depends on other rules if they are more restrictive than it
    and produce the same head."""

    if any(rule.support is None for rule in rules.values()):
        raise ValueError("Can't determine rule dependencies for rules without support.")

    intensional_preds = {rule.head.predicate for rule in rules.values()}

    # Group rules by head predicate
    by_head: dict[str, list[str]] = defaultdict(list)
    for rule_id, rule in rules.items():
        if rule.head.predicate in intensional_preds:
            by_head[rule.head.predicate].append(rule_id)
    by_head = dict(by_head)  # Secure the dict type

    # Determine dependencies
    rule_dependency: dict[str, set[str]] = {r_id: set() for r_id in rules}

    for _, rule_ids in by_head.items():
        # Sort the rules that generate the same predicate by body length
        sorted_ids = sorted(
            rule_ids, key=lambda r_id: len(rules[r_id].get_body_predicates())
        )

        # Scan the rules from shortest to largest
        for i, current_id in enumerate(sorted_ids):
            current_rule = rules[current_id]
            current_body = current_rule.get_body_predicates()
            current_support = current_rule.support

            for next_id in sorted_ids[i + 1 :]:
                next_rule = rules[next_id]
                next_body = next_rule.get_body_predicates()
                next_support = next_rule.support

                # The rule only depends on others if they share the complete body of the
                # current rule
                if current_body.issubset(next_body):
                    # If they happen to have the same body, the lesser support is more
                    # restrictive
                    if (
                        len(next_body) == len(current_body)
                        and current_support >= next_support
                    ):
                        rule_dependency[current_id].add(next_id)
                    else:
                        rule_dependency[next_id].add(current_id)

    logger.info("Created dependency graph for %d rules.", len(rules))
    return rule_dependency


def get_extensional_dependencies(
    rules: dict[str, HornRule],
):
    """Builds a dictionary that represents extensional predicate dependencies. A rule is
    dependent of any other rule extensional-wise if they share an extensional predicate.

    From all the rules that share an extensional predicate, the larger rules are more
    restrictive.
    """

    intensional_preds = {rule.head.predicate for rule in rules.values()}

    rule_dependency: dict[str, set[str]] = {r_id: set() for r_id in rules}

    # Sort from smallest to largest body counting only extensional preds
    sorted_ids = sorted(
        (rule_id for rule_id in rules.keys()),
        key=lambda r_id: len(rules[r_id].get_extensional_body(intensional_preds)),
    )

    for i, current_id in enumerate(sorted_ids):
        current_rule = rules[current_id]
        current_ext_preds = current_rule.get_extensional_preds(intensional_preds)

        if not (current_ext_preds):
            continue

        for next_id in sorted_ids[i + 1 :]:
            next_rule = rules[next_id]
            next_ext_preds = next_rule.get_extensional_preds(intensional_preds)

            if any(pred in next_ext_preds for pred in current_ext_preds):
                c_length = len(current_rule.get_extensional_body(intensional_preds))
                n_length = len(next_rule.get_extensional_body(intensional_preds))

                if c_length == n_length and next_rule.support > current_rule.support:
                    rule_dependency[next_id].add(current_id)
                else:
                    rule_dependency[current_id].add(next_id)

    return rule_dependency


def get_term_mapping(ontology_file: Path, default_namespace: str) -> dict[str, str]:
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
    term_mapping.update({"default": default_namespace})
    return term_mapping


def get_predicate_mapping(rules: dict[str, HornRule]) -> dict[str, set[str]]:
    """Returns a mapping from predicates to the ids of rules where they are present."""

    mapping: defaultdict[str, set[str]] = defaultdict(set)

    for r_id, rule in rules.items():
        for pred in rule.get_predicates():
            mapping[pred].add(r_id)

    return dict(mapping)
