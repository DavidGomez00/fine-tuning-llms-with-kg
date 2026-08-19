# Backlog

Known issues and pending refactors, organized by module. See `AGENTS.md` for
the current architecture map.

## `cli/`

- [ ] **`main.py`** — No se ejecuta correctamente. `TODO: Debug main.py`.
- [ ] **`upload.py`** — Sin pendientes registrados.

## `config.py`

- [x] **Dead/unused config variables and classes.** `RunConfig` and its
      sub-configs carried several fields/classes with zero readers anywhere
      in `src/`.
  - **Deleted** `HardwareConfig` (`n_gpus`, `device`, `precision`,
    `max_memory_mb`) entirely — it was never read outside its own
    `default_factory` construction, and its only live effect was forcing
    the `torch` import in `config.py`. Removed the `RunConfig.hardware`
    field, its `from_json` section handling, and the `import torch`.
  - **Deleted** `DataConfig.crud_endpoint` — defined but never read
    anywhere. Also removed the matching (equally unused) `crud_endpoint`
    key from `configurations/french_royalty.json` and
    `configurations/simpsons.json`'s `data` sections — since `DataConfig`
    is built via `DataConfig(**get_section("data", ...))`, that stray key
    would otherwise now raise a `TypeError` on load.
  - **Deleted** `FineTuningConfig` / `CoTGenerationConfig` — initially kept
    as documented forward-looking scaffolding, but reconsidered as out of
    scope for now and removed along with `RunConfig.fine_tuning`/
    `cot_generation` and their `from_json` handling (no config JSON
    referenced either section). `AGENTS.md` updated to note the
    fine-tuning/CoT pipeline isn't implemented under `src/` without
    pointing at now-nonexistent config classes; reintroduce them once that
    pipeline is actually built.
  - **`classification`/`pca_threshold` filtering inconsistency** — resolved
    as docs-only: `RulesConfig.pca_threshold`'s docstring and `AGENTS.md`
    both claimed/implied `"NEGATIVE"`-classified rules are excluded from
    generation, but nothing downstream actually checks
    `rule.classification` before feeding rules into
    `generate_edb`/`generate_idb`, and no `utils.filter_rules` function
    exists. Corrected both docs to describe the actual (classify-only,
    no filtering) behavior instead of implementing the missing filter.
  - Verified via `mypy` (clean) and `RunConfig.from_json` against all three
    `configurations/*.json` — `mario.json`/`french_royalty.json` load as
    before, `simpsons.json` still fails only on its already-tracked missing
    `namespace` field.
- [x] **Smelly logic in `config.py`.**
  - **Deleted** `RunConfig.__post_init__` — it was a no-op (`pass`) despite a
    docstring claiming "Validate config."
  - **`RulesConfig.pca_threshold` is now a required `float`, never `None`** —
    its docstring already documented `None` as skipping classification, and
    it fed into `core.rules.parse_rule_set(pca_threshold: float | None)`,
    but no caller (`cli/main.py`, `cli/upload.py`) ever actually passed
    `None`. Tightened `parse_rule_set`'s signature to plain `float` and
    dropped its now-dead `None`-skips-classification branch to match.
  - **Unified error surfacing in `RunConfig.from_json`** — missing/malformed
    top-level sections used to raise clean `KeyError`/`ValueError`s via a
    `get_section` helper, but errors *inside* a section (missing/unexpected
    field) fell through to a raw dataclass `TypeError` with no
    "Configuration Error" context (e.g. the `simpsons.json` case below).
    Replaced `get_section` with `load_section`, which builds the section's
    dataclass directly and wraps any `TypeError` from its constructor into a
    `ValueError` prefixed the same way as the other config errors. Also
    added the missing `"Configuration Error: "` prefix to the
    not-a-mapping-section case, which lacked it.
  - Verified via `mypy` (clean) and by exercising all four error paths
    (missing section, non-mapping section, missing field, unexpected field)
    plus an end-to-end run against `configurations/mario.json`.

## `core/`

- [x] **`rules.py`** — Refactor dataclass functions to delete obsolete code;
      refactor and delete obsolete functions.
  - Deleted dead methods with zero callers anywhere in `src/`:
    `Atom.__contains__`, `Atom.get_variables`, `Atom.to_natural_language`
    (and `CAMEL_CASE_PATTERN`), `RuleSignature.__iter__`,
    `RuleSignature.to_natural_language`, `RuleSignature.__str__`,
    `RuleSignature.get_head_variables`/`HornRule.get_head_variables`,
    `HornRule.__str__`.
  - `get_dependencies_intensional` kept as-is (flagged dead-for-now by the
    author, needed once incomplete-rule support is built).
  - Renamed internal-only parsing helpers to signal they're private:
    `parse_body`→`_parse_body`, `parse_head`→`_parse_head`,
    `parse_horn_rule`→`_parse_horn_rule`, `RuleRow`→`_RuleRow`.
  - Fixed the `intesional_preds`→`intensional_preds` misspelling in
    `RuleSignature.get_extensional_body`, and rewrote `parse_rule_set`'s
    stale docstring (referenced a nonexistent `rules_df` param and a tuple
    return value).
  - Verified via `mypy` (no new errors) and an end-to-end run of
    `run_synthetic_graph_experiment` against `configurations/mario.json`.
  - Follow-ups spun out below: `parse_rule_set` return type, and the
    `classification`/`pca_threshold` filtering inconsistency.
- [ ] **`rules.py`: `parse_rule_set` return type** *(low priority)* —
      `parse_rule_set` returns `dict[str, HornRule]` and still carries a stale
      `# TODO: Change the return value to a simple list of rules` comment.
      Changing it to `list[HornRule]` requires updating every dict-keyed
      consumer (`rules[rule_id]`, `.values()`, `.keys()` lookups in
      `engine/idb.py`, `engine/edb.py`, `engine/generator.py`), so it's
      bigger than a same-file cleanup — do as its own change.
- [x] **`queries.py`: 3 funciones para escribir queries** — reducir o
      eliminar hasta sólo tener las que se usan.
  - Audited all 3 insert-triples functions
    (`insert_triples_sparql`/`insert_triples_bulk`/`insert_graph_sparql`,
    the last since renamed to `insert_graph` — see below): all 3 are
    actually used somewhere in `src/`, so this wasn't a dead-code deletion.
    `insert_triples_gsp()` was already renamed to `insert_triples_bulk()`
    in an earlier session and now supports GraphDB as well as Virtuoso
    (detects the backend from the endpoint, `/repositories/` ⇒ GraphDB, and
    POSTs raw N-Triples to each store's bulk-load REST endpoint —
    `.../statements?context=<grafo>` on GraphDB/RDF4J,
    `sparql-graph-crud-auth` on Virtuoso — instead of parsing `INSERT DATA`
    for hundreds of thousands of triples).
  - Found and fixed the one real gap: `insert_graph_sparql` (formerly
    listed here under its old name `insert_graph_from_nt_sparql`, already
    renamed in an earlier session but never updated here) — used by
    `cli/upload.py` to load the base graph from a `.nt` file, potentially
    the largest/most performance-sensitive insert path in the pipeline —
    was still going through the slow, chunked `insert_triples_sparql`
    (SPARQL `INSERT DATA`) instead of `insert_triples_bulk`'s fast REST
    path. Switched it over. `edb.py`/`generator.py`'s other
    `insert_triples_sparql` call sites are left as-is: they insert
    smaller, filtered/deduplicated runtime-generated streams with no
    documented at-scale problem pushing them onto the bulk path.
  - Since it no longer builds/executes SPARQL itself, renamed
    `insert_graph_sparql` → `insert_graph` (only caller: `initialize_graph`
    within `queries.py` itself).
  - `build_filtered_query`/`generate_triples_from_rule` — no longer exist
    anywhere in the codebase (already removed in an earlier session); the
    old sub-bullet about them was stale.
  - Verified via `mypy` (no new errors) and by running `cli/upload.py`
    end-to-end against `mario.nt` (16/16 non-comment lines loaded,
    matching exactly) followed by a full
    `run_synthetic_graph_experiment` run.
- [x] **`queries.py`: `download_graph_raw()` removed** — dropped the
      "produce a file with a graph in it" feature entirely for now. It had
      zero callers anywhere in `src/`/`notebooks/` (its only caller,
      `cli/download.py`, was already removed in an earlier session) and was
      the function responsible for the 17MB artifact previously cleaned out
      of the repo (`output_path.mkdir()` followed by `output_path /
      file_name` produces a nested directory when `output_path` already
      ends with the file name). Deleted the whole "Download from database"
      section, including its stale `# TODO: Tiene esto que estar aquí??`.
      `requests`/`URL`/`Path` imports are all still used elsewhere in the
      file, so no import cleanup was needed. As a side effect, this also
      removed one of the file's pre-existing `mypy` errors (a type mismatch
      inside the deleted function). Verified via `mypy` and end-to-end runs
      of `cli/upload.py` and `run_synthetic_graph_experiment`. Re-add a
      download/export feature from scratch if/when it's actually needed.
- [x] **`queries.py`: `copy_graph_sparql`** — reordenar, revisar uso y
      refactorizar.
  - **Uso**: confirmed used — its only caller is `initialize_graph` (for
    the "source is a graph URI, not a `.nt` file" branch), and the pipeline
    exercises it every run (e.g. copying `graph/base` → `graph/complete`).
  - **Refactor**: `clear_graph_sparql`, `execute_insert_query`, and
    `copy_graph_sparql` each duplicated the same ~15-line
    "configure the update client, work around `SPARQLWrapper`'s
    query/update param quirk, execute, log-and-raise on failure"
    boilerplate — and `clear_graph_sparql` was even missing the
    `setRequestMethod(URLENCODED)` call the other two had (harmless in
    practice, since that's already `SPARQLWrapper`'s own default, but
    still an inconsistency). Extracted the shared logic into a new
    `_execute_update_query()` helper (grouped with `_get_update_client` in
    the "Helper functions" section); all three now just build their query
    string and delegate to it. Also fixed `clear_graph_sparql`'s stale
    docstring (documented a nonexistent `database_endpoint` arg instead of
    `client`).
  - **Reorder**: moved `copy_graph_sparql`'s definition to sit right after
    `clear_graph_sparql` — both are graph-level SPARQL UPDATE operations
    feeding `initialize_graph`, previously separated by the unrelated
    "Handle SPARQL query responses" section.
  - As a side effect, de-duplicating the boilerplate also collapsed 2 of
    the file's pre-existing `mypy` errors (duplicate copies of the same
    type mismatch) down to 1 occurrence.
  - Verified via `mypy` and end-to-end runs of `cli/upload.py` (including
    its "Successfully copied ..." debug log) and
    `run_synthetic_graph_experiment`.
- [x] **`queries.py`: `chunk_iter`** — already resolved in an earlier
      session: it's `_chunk_iter()` and already grouped in the "Helper
      functions" section alongside `_get_update_client`/
      `_execute_update_query`. Stale sub-bullet, no action needed.
- [x] **`queries.py`: renombrar `run_select_query`/`execute_ask_query`/
      `execute_insert_query`** — `execute_ask_query` turned out to not
      exist in the file at all (confirmed via grep; likely never existed
      under that name, or was removed before this file's history was
      tracked here). Converged the "run a raw/pre-built query string
      against the store" layer on one verb, **"execute"** — matching
      `execute_insert_query` and the `_execute_update_query()` helper added
      in the previous pass, so only `run_select_query` needed to move
      (cheaper than moving the other two):
  - `run_select_query` → `execute_select_query`.
  - Also renamed for consistency, per user decision:
    - `clear_graph_sparql`/`copy_graph_sparql` → `clear_graph`/`copy_graph`
      — dropped the redundant `_sparql` suffix (the module talks SPARQL by
      default per its own docstring, so the suffix added no signal here).
      Left `insert_triples_sparql`/`insert_triples_bulk` alone: that pair's
      suffixes are load-bearing, disambiguating two real implementations
      of the same operation (SPARQL `INSERT DATA` vs. REST bulk-load).
    - `count_triples` → `get_triple_count` — the only metric-query function
      not prefixed `get_*` (siblings: `get_domain`, `get_range`,
      `get_reflexivity`, `get_support`, `get_frequency`,
      `get_predicate_frequencies`, `get_existing_triples`).
  - Updated every external caller: `engine/generator.py`, `engine/edb.py`
    (`execute_select_query`, `clear_graph`), `engine/metrics.py`,
    `cli/main.py`, `cli/upload.py` (`get_triple_count`) — plus re-sorted
    the import blocks touched, alphabetically (`ruff`'s `I` rules).
  - Verified via `mypy` (identical error set/line numbers to baseline —
    zero new issues) and end-to-end runs of `cli/upload.py` and
    `run_synthetic_graph_experiment`.
- [x] **`queries.py`: refactorizar el script en general** — ordenar
      funciones y organizar en secciones. Rewrote the file into 6 coherent
      sections instead of the previous scattered/mislabeled ones:
      `Helper functions` (private), `Query execution` (the public
      `execute_select_query`/`execute_insert_query` primitives, moved up
      from ~230 lines further down — `insert_triples_sparql` calls
      `execute_insert_query`, so this also fixes a forward-reference
      ordering issue), `SPARQL query generation` (`build_rule_query`),
      `Write to database` (`insert_triples_sparql`/`insert_triples_bulk`/
      `insert_graph`/`clear_graph`/`copy_graph`/`initialize_graph` — merges
      the old "Insert to database" and "initialize graph in database."
      sections, which mislabeled `clear_graph`/`copy_graph` as inserts and
      gave `initialize_graph` its own redundant section), `Query metrics`,
      and `Helpers for IDB/EDB generation` (fixed a `"ofr"` → `"for"`
      typo). Also moved the `SparqlBinding` type alias up next to `logger`
      since it's used throughout the file, not just by the "response
      handling" functions it used to sit next to. Pure reordering — no
      logic changes; verified the function set is byte-identical
      (`diff`'d sorted `def` names before/after) and behavior is unchanged
      via `mypy` (identical error set, just shifted line numbers) and
      end-to-end runs of `cli/upload.py` and `run_synthetic_graph_experiment`.
  - `build_rule_query` — called from several scripts (`engine/edb.py`,
    `engine/generator.py`); this was always just a "don't delete it, it's
    alive" note, not an actionable TODO.
  - This closes out the `queries.py` module entirely — every sub-item
    tracked under it (function consolidation, `download_graph_raw`
    removal, `copy_graph_sparql`/`clear_graph_sparql` refactor+rename,
    `run_select_query`/`execute_insert_query`/`count_triples` renames,
    and this reorganization pass) is now resolved.

## `engine/`

- [x] **`generator.py`** — Refactor and delete obsolete functions.
  - **No dead functions found** — same pattern as `rules.py`/`queries.py`:
    every function (`decrement_counts`, `update_closed_preds`,
    `is_assignment_solvable`, `create_searchspace`, `GraphSources`,
    `triples_from_bindings`, `apply_rule`) is actually called from
    `edb.py`, `idb.py`, or `completion.py`.
  - **Found and fixed a real correctness bug** while auditing:
    `apply_rule`'s inner `filter_triples` checked
    `profile.range.get(subject, 0)` instead of
    `profile.range.get(obj, 0)`. `range` is object-keyed everywhere else
    in the codebase (`get_range()`, `is_assignment_solvable`,
    `edb.py`'s domain/range usage all pair `domain`↔`subject` /
    `range`↔`obj`), so this line effectively always evaluated to "reject
    the triple" whenever a profile was passed — which `idb.py`'s IDB
    generation loop always does. Before the fix, every run in this
    session's history ended in "Reached stale state" without full
    predicate closure and produced ~10-20 synthetic triples for
    `mario.json`; after the fix, runs consistently reach "All predicates
    closed" and produce ~28-30 triples. Fixed by keying on `obj` instead
    of `subject`.
  - Cleaned up dead commented-out debug-log lines in `filter_triples`,
    and fixed several stale docstrings: `create_searchspace` documented
    nonexistent params (`database_endpoint`/`predicate`/`profile`) and a
    wrong return type (said it returns a URI; it returns `None`);
    `triples_from_bindings` said it returns "a set" but it's a generator;
    `decrement_counts`/`update_closed_preds` called themselves "private"
    despite being public, cross-module functions (plus a "helpter" typo
    in the latter).
  - Verified via `mypy` (identical error set to baseline — zero new
    issues) and multiple end-to-end runs of `cli/upload.py` and
    `run_synthetic_graph_experiment` against `mario.json`.

## `configurations/`

- [ ] **`simpsons.json` fails to load** — verified via `RunConfig.from_json`:
      raises `ValueError: Configuration Error: Invalid 'graph' section:
      GraphConfig.__init__() missing 1 required positional argument:
      'namespace'` — its `graph` section is missing the required `namespace`
      field that `mario.json`/`french_royalty.json` both have. `mario.json`
      and `french_royalty.json` currently load successfully.

## Tooling / process

- [ ] **To be researched: tests + CI.** No `tests/` directory or CI pipeline
      exists yet. `ruff` and `mypy` are already configured in `pyproject.toml`
      but nothing runs them automatically. Decide whether to add a `tests/`
      scaffold (pytest, mocking `SPARQLWrapper` for unit tests that don't need
      a live DB) and a GitHub Actions workflow to run `ruff check .`,
      `mypy .`, and the test suite on every push/PR.
