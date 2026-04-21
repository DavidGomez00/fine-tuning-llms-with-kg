from rules import RuleSignature


def build_sparql_query(rule: RuleSignature, ns_prefix: str, namespace: str) -> str:
    """
    Build a SPARQL SELECT query from a rule.

    # TODO: Fix docs.
    Body atoms  -> required triple patterns.
    Head atom   -> OPTIONAL triple pattern (to determine yes / no answer).
    All variables projected in SELECT DISTINCT.

    Args:
        rule: RuleSignature containing body and head of the rule.
        ns_prefix: No tengo idea.
        namespace: ok.

    Returns:
        The SPARQL query.
    """
    variables = set()

    for atom in rule.body:
        if str(atom.obj).startswith("?"):
            variables.add(str(atom.obj))
        if str(atom.subject).startswith("?"):
            variables.add(str(atom.subject))

    if str(rule.head.subject).startswith("?"):
        variables.add(str(rule.head.subject))
    if str(rule.head.obj).startswith("?"):
        variables.add(str(rule.head.obj))

    select_vars = " ".join(sorted(variables))

    def term(t: str) -> str:
        """Format a term for SPARQL: variable stays as-is, constants get prefix."""
        return t if t.startswith("?") else f"{ns_prefix}:{t}"

    body_lines = "\n".join(
        f"    {term(str(atom.subject))} {ns_prefix}:{str(atom.predicate)} {term(str(atom.object))} ."
        for atom in rule.body
    )
    head_line = f"    {term(rule.head.subject)} {ns_prefix}:{rule.head.predicate} {term(rule.head.obj)} ."

    return (
        f"PREFIX {ns_prefix}: <{namespace}>\n"
        f"SELECT DISTINCT {select_vars} ?_head_exists WHERE {{\n"
        f"{body_lines}\n"
        f"    OPTIONAL {{\n"
        f"{head_line}\n"
        f"        BIND(true AS ?_head_exists)\n"
        f"    }}\n"
        f"}}"
    )
