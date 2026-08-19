# Backlog

Known issues and pending refactors, organized by module. See `AGENTS.md` for
the current architecture map.

## `cli/`

- [ ] **`main.py`** — No se ejecuta correctamente. `TODO: Debug main.py`.
- [ ] **`upload.py`** — Sin pendientes registrados.

## `config.py`

- [ ] **Dead/unused config variables and classes.** `RunConfig` and its
      sub-configs carry several fields/classes with zero readers anywhere in
      `src/`. For each, decide whether to wire it into the pipeline it was
      built for or delete it until that work starts:
  - `HardwareConfig` (`n_gpus`, `device`, `precision`, `max_memory_mb`) —
    entirely unused; `RunConfig.hardware` is never read outside its own
    `default_factory` construction. Its only live effect today is forcing
    the `torch` import in `config.py` (and the `torch` dependency) at
    config-load time.
  - `DataConfig.crud_endpoint` — defined but never read anywhere.
  - `FineTuningConfig` / `CoTGenerationConfig` — already noted in
    `AGENTS.md` as configuration for pipeline code that "is not present yet
    under `src/`"; confirmed zero reads of `run_config.fine_tuning`/
    `run_config.cot_generation` anywhere in `src/`.
  - **`classification`/`pca_threshold` filtering inconsistency** —
    `RulesConfig.pca_threshold`'s docstring claims rules classified
    `"NEGATIVE"` are "excluded from generation," and `AGENTS.md` references
    a `utils.filter_rules` function for ad hoc PCA/Std-confidence
    filtering — but no such function exists in `utils.py`, and nothing
    downstream actually checks `rule.classification` before feeding rules
    into `generate_edb`/`generate_idb`. Needs investigation: should
    filtering happen in `parse_rule_set`, a new `utils.filter_rules`, or
    elsewhere?

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
