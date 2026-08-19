# Backlog

Known issues and pending refactors, organized by module. See `AGENTS.md` for
the current architecture map.

## `cli/`

- [ ] **`main.py`** — No se ejecuta correctamente. `TODO: Debug main.py`.
- [ ] **`upload.py`** — Sin pendientes registrados.

## `core/`

- [ ] **`rules.py`** — Refactor dataclass functions to delete obsolete code;
      refactor and delete obsolete functions.
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
