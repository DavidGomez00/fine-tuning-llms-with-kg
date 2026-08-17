# Backlog

Known issues and pending refactors, organized by module. See `CLAUDE.md` for
the current architecture map.

## `cli/`

- [ ] **`download.py`** — Debe ser reescrito para funcionar con el nuevo DBMS
      (referencia campos de config, p.ej. `config.virtuoso`, que ya no existen
      tras el refactor). `TODO: Reescribir download.py para que funcione con
      GraphDB.`
- [ ] **`main.py`** — No se ejecuta correctamente. `TODO: Debug main.py`.
- [ ] **`upload.py`** — Sin pendientes registrados.

## `core/`

- [ ] **`parsing.py`** — Sólo contiene una función (`tsv_to_nt`) que necesita
      ser revisada. Es posible que deba moverse a `utils.py` y eliminar el
      script.
- [ ] **`rules.py`** — Refactor dataclass functions to delete obsolete code;
      refactor and delete obsolete functions.
- [ ] **`queries.py`** — Hay 3 funciones para escribir queries: reducir o
      eliminar hasta sólo tener las que se usan. Además:
  - `insert_triples_sparql` se usa en `edb.py` y en `generator.py`. Sería
    mejor que los triples sólo se inserten desde una función; revisar si esta
    arquitectura es apropiada.
  - `insert_triples_gsp()` no es compatible con GraphDB, buscar una
    alternativa.
  - `insert_graph_from_nt_sparql` — revisar uso y refactorizar.
  - `download_graph_raw()` — revisar uso y refactorizar.
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

## Tooling / process

- [ ] **To be researched: tests + CI.** No `tests/` directory or CI pipeline
      exists yet. `ruff` and `mypy` are already configured in `pyproject.toml`
      but nothing runs them automatically. Decide whether to add a `tests/`
      scaffold (pytest, mocking `SPARQLWrapper` for unit tests that don't need
      a live DB) and a GitHub Actions workflow to run `ruff check .`,
      `mypy .`, and the test suite on every push/PR.
