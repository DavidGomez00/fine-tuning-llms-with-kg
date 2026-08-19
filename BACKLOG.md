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
- [ ] **`queries.py`** — Hay 3 funciones para escribir queries: reducir o
      eliminar hasta sólo tener las que se usan. Además:
  - `insert_triples_sparql` se usa en `edb.py` y en `generator.py`. Sería
    mejor que los triples sólo se inserten desde una función; revisar si esta
    arquitectura es apropiada.
  - `insert_triples_gsp()` (renombrada a `insert_triples_bulk()`) — [DONE]
    ahora soporta GraphDB además de Virtuoso:
    detecta el backend por el endpoint (`/repositories/` ⇒ GraphDB) y hace
    POST de N-Triples en crudo al endpoint REST de bulk-load de cada store
    (`.../statements?context=<grafo>` en GraphDB/RDF4J,
    `sparql-graph-crud-auth` en Virtuoso), evitando el parseo de `INSERT
    DATA` para cientos de miles de triples.
  - `insert_graph_from_nt_sparql` — revisar uso y refactorizar.
  - `download_graph_raw()` — revisar uso y refactorizar. Es la función que
    generó el artefacto de 17MB limpiado del repo (`output_path.mkdir()`
    seguido de `output_path / file_name` produce un directorio anidado si
    `output_path` ya termina en el nombre de archivo); ya no tiene caller
    desde que se eliminó `cli/download.py`, evaluar si eliminarla también.
  - `copy_graph_sparql` — reordenar, revisar uso y refactorizar.
  - Refactorizar nombres de `run_select_query`, `execute_ask_query`,
    `execute_insert_query`.
  - Refactorizar el script en general: ordenar funciones y organizar en
    secciones.
  - `chunk_iter` — helper usado sólo dentro de `queries.py`; renombrar a
    `_chunk_iter()` y agrupar en una sección de "helpers".
  - `build_rule_query` — es llamada desde varios scripts.
  - `build_filtered_query` — es llamada desde `generate_triples_from_rule`,
    que tal vez esté obsoleta; de momento se mantiene comentada para ver si
    se elimina más adelante.

## `engine/`

- [ ] **`generator.py`** — Refactor and delete obsolete functions.

## `configurations/`

- [ ] **`simpsons.json` fails to load** — verified via `RunConfig.from_json`:
      raises `TypeError: GraphConfig.__init__() missing 1 required positional
      argument: 'namespace'` — its `graph` section is missing the required
      `namespace` field that `mario.json`/`french_royalty.json` both have.
      `mario.json` and `french_royalty.json` currently load successfully.

## Tooling / process

- [ ] **To be researched: tests + CI.** No `tests/` directory or CI pipeline
      exists yet. `ruff` and `mypy` are already configured in `pyproject.toml`
      but nothing runs them automatically. Decide whether to add a `tests/`
      scaffold (pytest, mocking `SPARQLWrapper` for unit tests that don't need
      a live DB) and a GitHub Actions workflow to run `ruff check .`,
      `mypy .`, and the test suite on every push/PR.
