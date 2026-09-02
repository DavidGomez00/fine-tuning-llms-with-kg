# Backlog

Known issues and pending refactors, organized by module. See `AGENTS.md` for
the current architecture map. Resolved items are archived in
`BACKLOG_ARCHIVE.md` once checked off here, to keep this file scannable.

## `core/`

- [ ] **`rules.py`: `parse_rule_set` return type** *(low priority)* —
      `parse_rule_set` returns `dict[str, HornRule]` and still carries a stale
      `# TODO: Change the return value to a simple list of rules` comment.
      Changing it to `list[HornRule]` requires updating every dict-keyed
      consumer (`rules[rule_id]`, `.values()`, `.keys()` lookups in
      `engine/idb.py`, `engine/edb.py`, `engine/generator.py`), so it's
      bigger than a same-file cleanup — do as its own change.

## 'engine/'
- [ ] **`metrics.py`**: There has to be a way to pass the metrics without reading an actual graph. Pass the metrics through a JSON or smth.
  - [ ] **`main.py`**: We should read the original metrics from the JSON or data input, not through SPARQL queries.

- [ ] **`idb.py`**: Apply rule should not use the searchspace. It is happening that whenever a recursive rule is applied (e.g., a ^ p -> p), the searchspace is created and a set of triples are added to the final graph, instead of using the already grounded a and p first. We should use the searchspace when we have exhausted existing groundings for the triple. My initial guess is that we can try to retrieve bindings from the final graph and remove the already included triples. If 0 resulted triples are added, we can use searchspace.

## 'docs/'
- [ ] **concepts.md** *(low priority)*: Check all definitions are as intended (support, head coverage, etc).
- [ ] **getting_started.md** *(low priority)*: Define clearly all the neccessary inputs for the execution of the repo.