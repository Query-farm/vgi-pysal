# CLAUDE.md — vgi-pysal

Contributor/agent notes for this repo. User-facing docs live in `README.md`;
this file is the "how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://github.com/query-farm/vgi-python) worker exposing
[PySAL](https://pysal.org) (spatial analysis) to DuckDB/SQL. `vgi_pysal.worker`
assembles every function into one `pysal` catalog (single `main` schema) and runs
it over stdio (local) or HTTP (Fly.io). Modeled on the sibling `vgi-scikit-learn`
worker; built on the local `~/Development/vgi-python` + `~/Development/vgi-rpc`
checkouts for dev, PyPI `vgi-python` for the package.

## Layout

```
vgi_pysal/
  worker.py           builds the `pysal` Catalog + PysalWorker; main()/main_http()
  datasets.py         built-in libpysal example datasets as table functions (WKB geom)
  spatial_weights.py  build_weights() shared helper + weights() / weights_summary()
  esda.py             global autocorrelation (moran / geary / getis_ord_g), one-row stats
  lisa.py             local autocorrelation (local_moran / getis_ord_g_local), per-obs
  classify.py         mapclassify choropleth schemes (buffering transforms)
  inequality.py       gini / theil SQL aggregates
  regression.py       spreg fit / coefficient table / predict + registry mgmt
  registry.py         fitted-model store (JSON; local disk now, S3/R2 later)
  buffering.py        shared SinkBuffer + serialize/column helpers
  geometry.py         WKB / coordinate parsing
  schema_utils.py     pa.Field comment helper, name sanitisation, NoArgs
pysal_worker.py       repo-root stdio shim over vgi_pysal.worker
serve.py              repo-root HTTP shim over vgi_pysal.worker
tests/                pytest (in-process harness in tests/harness.py)
test/sql/*.test       DuckDB sqllogictest — the authoritative integration tests
```

To add functions: implement in the relevant `vgi_pysal/*.py`, export a
`*_FUNCTIONS` list, and splice it into `_FUNCTIONS` in `vgi_pysal/worker.py`.

**Entry points / packaging.** Console scripts (`vgi-pysal`, `vgi-pysal-http`)
point at `vgi_pysal.worker:main` / `:main_http` — *inside the package*, so they
ship in the wheel. The repo-root `pysal_worker.py` / `serve.py` are thin shims
for `uv run` / the Fly container / `import` in tests; they are deliberately NOT
in the wheel. Don't point entry points at the root modules — that breaks console
scripts on `pip install`. PyPI publish is `publish.yml` (GitHub Release → CI →
`uv build && uv publish`); bump `version` in `pyproject.toml` before tagging.

## The one spatial-input convention (the core idea)

Everything that needs a spatial weights matrix — `weights`, `esda`, `lisa`,
`regression` — gets it through **one** helper: `spatial_weights.build_weights(
table, args)`. The argument dataclasses all inherit from `WeightsArgs`
(`w_type`, `geom`, `x`, `y`, `k`, `threshold`, `kernel_function`, `transform`),
so the convention is identical everywhere:

- `queen` / `rook` → parse the WKB `geom` column (`geometry.parse_wkb_column`)
  → `libpysal.weights.Queen/Rook.from_iterable(geoms, ids=range(n))`.
- `knn` / `distance_band` / `kernel` → coordinates from the `x`/`y` columns, or
  geometry centroids as a fallback → `.from_array(coords)`.

Observations are identified by **0-based position in the buffered table**.
Per-observation results (LISA, classify) are computed in that same order and
carry an optional `id` column through, so they line up and join back regardless
of how DuckDB ordered the input.

## Which VGI primitive for which job

| Need | Primitive | Example here |
| --- | --- | --- |
| Emit rows, no input | `TableFunctionGenerator` (`@bind_fixed_schema` + `@init_single_worker`) | `datasets.py` |
| Scalar-per-group over one column | `AggregateFunction[State]` | `inequality.py` |
| Needs the whole input (build W, fit, classify) | `TableBufferingFunction` via `buffering.SinkBuffer` | `esda`, `lisa`, `classify`, `weights`, `regression.spreg/fit` |
| Score a stream with an already-fit model | `TableInOutGenerator` | `regression.PredictModel` |

ESDA/weights/regression are buffering functions because a spatial weights matrix
is global — you can't compute Moran's I from a streamed batch. The convention:
the table input is `Arg(0)` as a `(SELECT ...)` subquery; the analysed column is
`value` (ESDA/classify) or `target` (regression); `id` is an optional passthrough.

## Datasets

`datasets.py` builds one fixed-schema `TableFunctionGenerator` per curated
libpysal example (`columbus`, `baltim`, `sids`, `us_states`, `mexico`,
`georgia`) — all of which ship *inside* libpysal (no network download). The
schema is computed once at import from the loaded GeoDataFrame: a 0-based `id`,
snake-cased attribute columns (dtype-mapped to int64/float64/bool/string), and a
`geom` BLOB column holding WKB (`gdf.geometry.to_wkb()`). `_make_dataset` stamps
out the classes with `type(...)` (plain classes, no subscripted-generic base, so
no `types.new_class` dance needed — see sharp edge #4).

## Regression: coefficients vs. registry

- `spreg` returns the **coefficient table** (one row per parameter:
  `variable, coefficient, std_err, statistic, p_value, is_spatial`). This is the
  spatial-stats idiom and the primary output.
- `fit` returns a **one-row summary + a model BLOB** and persists to the registry
  if `model_name` is given (mirrors the sklearn worker's `fit`).
- A fitted model is reduced to its **linear predictor** (`intercept` + per-feature
  `coefficients`) plus diagnostics, all JSON-serializable — so the registry is
  plain JSON, **no pickle/skops** and no trust problem. `predict` does
  `intercept + X·coef`. For OLS that equals spreg's `.predy`; for spatial
  lag/error it's the trend (the spatial feedback `rho`/`lambda` is stored in
  `spatial_coef` but not replayed).
- `spreg` model dispatch is `_MODELS` (`ols`/`ml_lag`/`ml_error`/`gm_lag`). OLS is
  built with `spat_diag=True, moran=True` (gives a residual Moran's I); the others
  take `w=`. `_param_names` aligns names with `betas` by trying `name_z` then
  `name_x` (lengths differ across estimators); the spatial term is named `W_<y>`
  (lag) or `lambda` (error), detected by `_is_spatial_name`.

## Sharp edges (learned the hard way — read before debugging)

1. **Aggregate state: reassign, don't mutate.** `inequality.update()` must do
   `states[g] = ValueState(...)`. The framework persists only groups you
   *assigned* this batch; an in-place mutation of a group first seen in the batch
   is silently dropped → every result NULL. Single-group/whole-table aggregates
   always hit this.
2. **`pa.Float64Array` does not exist** — the class is `pa.DoubleArray`. A bad
   `Param` type hint does NOT error; the framework warns
   (`UserWarning: ... type hints could not be resolved`) and registers the
   function with **zero input columns**. `inequality.py` uses `pa.DoubleArray`.
3. **Table argument syntax is `(SELECT ...)`, not `TABLE(...)`;** a table function
   gets at most ONE subquery parameter (the table input is it). `predict` takes a
   `model` BLOB as a scalar, not a subquery.
4. **Generating function classes:** the dataset classes use plain
   `type(name, (_ExampleDataset,), ns)` because `_ExampleDataset` is
   `TableFunctionGenerator[NoArgs]` — a *concrete* (already-subscripted) base, so
   MRO resolution works. If you ever subclass a still-generic
   `SinkBuffer[TArgs, TState]` dynamically you must use `types.new_class(...)`
   (plain `type()` raises "doesn't support MRO entry resolution"). The classify
   functions sidestep this by making `_ClassifyFn` concrete
   (`SinkBuffer[ClassifyArgs, DrainState]`) and subclassing it with plain classes.
5. **`value` defaults to `y`, and `y` is also the y-coordinate column name.** For
   contiguity weights there's no coordinate `y`, so `value => 'y'` is natural. For
   *distance* weights with `x`/`y` coordinate columns, name the analysed column
   explicitly (`value => 'price'`) so it doesn't collide with the `y` coordinate.
6. **Getis-Ord wants binary weights.** `esda.G` / `G_Local` flip a row-standardised
   `W` to binary (`if w.transform == "R": w.transform = "B"`) before computing.
7. **Permutation reproducibility:** esda permutation p-values use NumPy's RNG. The
   functions `np.random.seed(args.seed)` (default 12345) before computing, and the
   local stats also pass `seed=` to `Moran_Local`/`G_Local`. SQL tests rely on this
   for stable p-values.
8. **`giddy` is not installable on Python 3.13** (its `quantecon` → `numba` chain
   caps at <3.10). It's intentionally not a dependency. Spatial-dynamics functions
   would need a different package or a 3.13-compatible giddy release.
9. **WKB only on the input side — no `require spatial` in the SQL tests.** The
   worker parses WKB itself with shapely, so datasets/weights/esda need only the
   `vgi` extension. The DuckDB spatial extension is only needed if *you* want to
   produce WKB from `GEOMETRY` (`ST_AsWKB`) or read it back (`ST_GeomFromWKB`).

## Packaging & CI

The repo is an installable package (`pyproject.toml`, hatchling, `uv.lock`):
`uv sync` resolves PyPI `vgi-python[http]` + the PySAL stack (libpysal, esda,
spreg, mapclassify, inequality, shapely, geopandas) and exposes the `vgi-pysal`
(stdio) and `vgi-pysal-http` console scripts. The worker uses only **stable**
vgi-python features (no union-typed arguments), so it pins plain PyPI
`vgi-python>=0.8.1` — no vendoring and no version gating. GitHub Actions
(`.github/workflows/ci.yml` + `ci/`) runs a `static` gate (ruff + mypy +
pydoclint) plus the unit + SQL suites on Linux/macOS/Windows against the **signed
community `vgi` extension** via a prebuilt `haybarn-unittest` — no C++ build (see
`ci/README.md`). Keep PyPI deps in `pyproject.toml` in sync with the PEP 723
headers in `pysal_worker.py`/`serve.py` and the Dockerfile `pip install` line.

## Static analysis (adopted from vgi-python)

Tooling config in `pyproject.toml` matches the `vgi-python` repo: **ruff** with
the `D` docstring rules (`select = ["E","F","I","UP","B","SIM","D"]`, google
convention, double-quote format), **strict mypy**, and the **pydoclint**
docstring-consistency gate. `make check` runs all three plus pytest; `make lint`
and `make typecheck` run subsets. Notes:

- Every public class/method/function (including each `Meta` block) has a
  docstring — required by ruff `D`. pydoclint here only validates `Args:`/
  `Attributes:` sections *if present*, so the summary-line docstrings pass it.
- mypy is **strict**. `OutputCollector` must be imported from its canonical home
  `vgi_rpc.rpc` (the `vgi.table_*` modules re-import but don't re-export it, so
  strict no-implicit-reexport rejects importing it from there). The untyped PySAL
  stack (libpysal/esda/spreg/mapclassify/inequality/geopandas/shapely/pandas) is
  declared `ignore_missing_imports` via `[[tool.mypy.overrides]]`; the two
  untyped `pa.ipc` IPC calls in `buffering.py` carry a narrow
  `# type: ignore[no-untyped-call]`. Generic buffering bases bind their TypeVar
  (`_LocalStat[TArgs: LocalEsdaArgs]`) so attribute access on `params.args` type-checks.
- mypy is run over the package + shims (`vgi_pysal/ pysal_worker.py serve.py`),
  not the tests (which use dynamic `SimpleNamespace` fixtures).

## Testing

```sh
uv sync && uv run pytest tests/ -q   # unit tests (CI's unit job)
make venv && make pytest             # unit tests against local vgi checkouts
make test-stdio                      # SQL tests, worker as subprocess (authoritative)
make test-http                       # SQL tests against a local HTTP server
```

- **SQL tests are authoritative.** Unit tests call classmethods directly (the
  `compute`/`classify`/`_fit` logic + schemas) and can pass while the real RPC
  path is broken — that's exactly how the aggregate state-persistence bug (edge
  #1) slips past pytest. Always run the SQL suite.
- SQL tests need a sqllogictest runner with the VGI extension: a DuckDB `unittest`
  built with vgi at `$(VGI_BUILD_DIR)/test/unittest`, or (what CI uses) a
  standalone `haybarn-unittest` + `INSTALL vgi FROM community`.
- For fast local probing with *real* error messages, `pip install haybarn` and
  drive it from Python (`con.execute("INSTALL vgi FROM community; LOAD vgi")`,
  `ATTACH ... LOCATION '.venv/bin/python pysal_worker.py'`) — far better than
  reading sqllogictest diffs while iterating.
- `make test-stdio` / `test-http` point `PYSAL_MODELS_DIR` at an isolated
  `.test-models/` so the registry tests don't pollute `./models`.

## Container image & CI/CD (modelled on vgi-scikit-learn)

Three workflows, modelled on the sibling sklearn worker:

- **`ci.yml`** — `static` (ruff + mypy + pydoclint), `unit` (pytest on
  Linux/macOS/Windows), and `integration` (the SQL suite via `haybarn-unittest` +
  the signed community `vgi` extension). Reusable via `workflow_call`.
- **`publish.yml`** — on a GitHub Release, runs `ci.yml`, verifies the tag matches
  the package version (`ci/check-version.sh`), then `uv build && uv publish` to PyPI.
- **`docker-publish.yml`** — on a `vX.Y.Z` tag or push to `main`, builds the
  multi-arch image on native amd64/arm64 runners, **tests the built image in BOTH
  transports before pushing** (amd64 runs the full SQL suite over stdio *and*
  HTTP; arm64 — no haybarn asset — runs an import + `/health` smoke), pushes by
  digest, then merges into one cosign-signed manifest on ghcr.io.

The single image (`Dockerfile` + `docker-entrypoint.sh`) serves both transports:
`docker run … IMG` → HTTP server; `docker run -i … IMG stdio` → the stdio worker
DuckDB spawns. It installs `pip install '.[serve]'` (the `serve` extra adds
oauth/sentry/authlib for the HTTP server only); geopandas/shapely ship manylinux
wheels so there's no system GEOS/GDAL to build. The `farm.query.vgi.volumes` image
label tells the extension to mount `/data` (registry + WAL-SQLite state). **Do not
import `vgi` in a build-time `RUN`**: importing it opens `VGI_WORKER_SQLITE_PATH`
(`/data/state/...`), which doesn't exist until the entrypoint or the later `mkdir`
creates it — a build-time warm-up step failed exactly this way.

Version is single-sourced from `vgi_pysal/__init__.py` (`dynamic = ["version"]`
via `[tool.hatch.version]`); `ci/check-version.sh` gates releases/tags against it.

```sh
make image                       # build the image locally
make test-docker-stdio           # SQL suite vs the image, stdio transport
make test-docker-http            # SQL suite vs the image, HTTP transport
make deploy TAG=0.1.0            # fly deploy the published ghcr image
fly volumes create pysal_models --size 1 --region iad   # one-time, registry
```

`fly.toml` bumps VM memory to 1 GB (geopandas/scipy are heavy) and mounts a volume
at `/data` for the model registry + framework state.

## Model registry

`registry.get_store()` is the single seam selecting the backend. `LocalDiskStore`
(one `<name>.json` per model under `PYSAL_MODELS_DIR`, default `./models`) is the
only impl today; an `S3Store` for S3/R2 is the planned next backend and drops in
here without touching `regression.py`. Models are plain JSON (intercept,
coefficients, diagnostics) — no pickle, no code execution on load.
