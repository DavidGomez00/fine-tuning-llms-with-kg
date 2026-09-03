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

## 'docs/'
- [ ] **concepts.md** *(low priority)*: Check remaining definitions are as
      intended (head coverage, std/PCA confidence). Support's definition was
      audited and fixed to match AMIE3 (see `get_support` in `core/queries.py`
      and its `get_head_variables()`-based projection).
- [ ] **getting_started.md** *(low priority)*: Define clearly all the neccessary inputs for the execution of the repo.

## Doubts
- [ ] Should the support be an upper bound as well? Depending on how the EDB is generated, the freqency of predicates can be reduced, thus reducing the support of some rules.