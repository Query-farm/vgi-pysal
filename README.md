<!--
Copyright 2026 Query Farm LLC - https://query.farm
-->

# vgi-pysal

**Spatial analysis in plain SQL.** `vgi-pysal` is a [VGI](https://github.com/query-farm/vgi-python)
worker that exposes [PySAL](https://pysal.org) — the Python Spatial Analysis
Library — to DuckDB. Build spatial weights, measure spatial autocorrelation
(Moran's I, LISA, Getis-Ord), classify choropleth maps, compute inequality
indices, and fit spatial regressions, all as ordinary table functions and
aggregates.

```sql
ATTACH 'pysal' (TYPE vgi, LOCATION 'vgi-pysal');

-- Is neighbourhood crime spatially clustered? (Moran's I)
SELECT statistic AS morans_i, p_value
FROM pysal.moran((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'queen');
```
```
┌──────────┬─────────┐
│ morans_i │ p_value │
├──────────┼─────────┤
│   0.5002 │   0.001 │   -- strong, significant positive spatial autocorrelation
└──────────┴─────────┘
```

It ships with several classic spatial datasets built in, so you can run every
example below with no data of your own.

## How it works (read this first — it's quick)

The worker runs PySAL in a separate process; DuckDB talks to it over Arrow. You
never import Python — you call SQL functions.

**Spatial structure comes from your columns.** Every spatial function builds a
weights matrix `W` from the input table using one convention, controlled by
`w_type`:

| `w_type` | Neighbours are… | Needs |
| --- | --- | --- |
| `queen` *(default)* | polygons sharing an edge **or** a vertex | a WKB `geom` column |
| `rook` | polygons sharing an edge | a WKB `geom` column |
| `knn` | the *k* nearest points | `x`/`y` coordinate columns |
| `distance_band` | points within a distance threshold | `x`/`y` coordinate columns |
| `kernel` | distance-decayed nearby points | `x`/`y` coordinate columns |

- **Geometry** is passed as **WKB bytes** in a column named `geom` (override with
  `geom =>`). From the DuckDB spatial extension that's `ST_AsWKB(geom) AS geom`;
  the built-in datasets already provide it.
- **Coordinates** are two numeric columns `x` and `y` (override with `x =>` /
  `y =>`). Distance weights can also fall back to geometry centroids.
- The weights `transform` defaults to `r` (row-standardised), the right choice
  for Moran's I and spatial regression. Use `b` (binary) for Getis-Ord G.

Observations are identified by their **0-based position in the input** and any
per-observation result lines up with it; pass `id =>` to carry your own key
through so you can join results back.

## Recipes

### Measure global spatial autocorrelation

```sql
-- Moran's I, Geary's C, and Getis-Ord General G of the same variable
SELECT 'moran' AS stat, statistic, p_value
FROM pysal.moran((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'queen')
UNION ALL
SELECT 'geary', statistic, p_value
FROM pysal.geary((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'queen')
UNION ALL
SELECT 'getis_ord_g', statistic, p_value
FROM pysal.getis_ord_g((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'knn', k => 8);
```

Each returns one row: the `statistic`, its `expected` value and `variance`, a
`z_score`, a permutation `p_value`, and an analytical `p_norm`. P-values are
reproducible (fixed `seed =>`, default 12345).

### Find clusters and hot spots (LISA)

```sql
-- Local Moran's I: where are the high-high clusters and spatial outliers?
SELECT id, cluster, local_i, p_value
FROM pysal.local_moran((SELECT id, crime AS y, geom FROM pysal.columbus()),
                       w_type => 'queen', id => 'id')
WHERE p_value < 0.05
ORDER BY cluster;
```

`cluster` is `HH`/`LL` (clusters), `HL`/`LH` (spatial outliers), or `ns`
(not significant at `significance =>`, default 0.05). For hot-spot analysis use
`getis_ord_g_local`, which labels each observation `hot`/`cold`/`ns` with a
`z_score`.

### Inspect the weights graph

```sql
-- The neighbour graph as an edge list, joinable like any table
SELECT focal, neighbor, weight
FROM pysal.weights((SELECT id, geom FROM pysal.columbus()), w_type => 'queen', id => 'id');

-- One-row connectivity summary (islands, neighbour counts)
SELECT * FROM pysal.weights_summary((SELECT geom FROM pysal.columbus()), w_type => 'queen');
```

### Classify a choropleth map

```sql
-- Quantile classes (the binning behind a choropleth), joinable by id
SELECT id, class, upper_bound
FROM pysal.quantiles((SELECT id, crime AS y FROM pysal.columbus()), value => 'y', k => 5);
```

Schemes: `quantiles`, `equal_interval`, `natural_breaks`, `fisher_jenks`,
`std_mean`, `box_plot`, `head_tail_breaks`. Each returns the 0-based `class` and
its inclusive `upper_bound` per row.

### Measure inequality

```sql
-- Gini and Theil indices, composing with GROUP BY like any aggregate
SELECT pysal.gini(pcgdp2000) AS gini, pysal.theil(pcgdp2000) AS theil
FROM pysal.mexico();
```

### Fit a spatial regression

```sql
-- Spatial lag model: does a neighbourhood's crime depend on its neighbours'?
SELECT variable, coefficient, p_value, is_spatial
FROM pysal.spreg((SELECT crime AS target, inc, hoval, geom FROM pysal.columbus()),
                 model => 'ml_lag', target => 'target', w_type => 'queen');
```
```
┌──────────┬─────────────┬─────────┬────────────┐
│ variable │ coefficient │ p_value │ is_spatial │
├──────────┼─────────────┼─────────┼────────────┤
│ CONSTANT │     45.6032 │     0.0 │ false      │
│ inc      │     -1.0487 │  0.0006 │ false      │
│ hoval    │     -0.2663 │  0.0779 │ false      │
│ W_target │      0.4233 │  0.0024 │ true       │   -- the spatial autoregressive term (rho)
└──────────┴─────────────┴─────────┴────────────┘
```

`model` is `ols`, `ml_lag`, `ml_error`, or `gm_lag`. The dependent variable is
named by `target =>`; every other numeric column (except `id` and the
coordinate/geometry columns) is an explanatory variable.

### Store a model and predict with it

```sql
-- Fit + persist to the registry, returning a one-row summary
SELECT model_name, model_type, pseudo_r2, spatial_coef
FROM pysal.fit((SELECT crime AS target, inc, hoval, geom FROM pysal.columbus()),
               model => 'ml_lag', target => 'target', model_name => 'columbus_lag');

-- Score rows with the stored model's linear predictor
SELECT * FROM pysal.predict((SELECT id, inc, hoval FROM pysal.columbus()),
                            model_name := 'columbus_lag', id := 'id');

SELECT * FROM pysal.list_models();
SELECT * FROM pysal.model_info('columbus_lag');
SELECT * FROM pysal.drop_model('columbus_lag');
```

`predict` applies the **linear (systematic) predictor** — intercept plus
coefficients · explanatory columns. For OLS that is the exact fitted value; for
spatial lag/error models it is the trend component (the spatial feedback term is
reported in `spatial_coef` but not replayed against a new neighbourhood graph).

### Get sample data to play with

```sql
SELECT * FROM pysal.columbus();   -- 49 neighbourhood polygons (crime, income, home value)
SELECT * FROM pysal.baltim();     -- 211 house sales (points, with x/y)
SELECT * FROM pysal.sids();       -- 100 NC counties (SIDS counts and rates)
SELECT * FROM pysal.mexico();     -- 32 Mexican states (per-capita GDP 1940–2000)
```

Each dataset has a 0-based `id`, snake-cased attribute columns, and a `geom`
column (WKB). Read the geometry with the DuckDB spatial extension via
`ST_GeomFromWKB(geom)`.

## Function reference

| Function | Kind | Purpose |
| --- | --- | --- |
| `columbus`, `baltim`, `sids`, `us_states`, `mexico`, `georgia` | table | Built-in example datasets |
| `weights` | table | Neighbour graph as an edge list |
| `weights_summary` | table | Connectivity diagnostics (one row) |
| `moran`, `geary`, `getis_ord_g` | table | Global spatial autocorrelation (one row) |
| `local_moran`, `getis_ord_g_local` | table | Local autocorrelation / hot spots (per observation) |
| `quantiles`, `equal_interval`, `natural_breaks`, `fisher_jenks`, `std_mean`, `box_plot`, `head_tail_breaks` | table | Choropleth classification |
| `gini`, `theil` | aggregate | Inequality indices |
| `spreg` | table | Spatial-regression coefficient table |
| `fit` | table | Fit + persist a model, return summary + BLOB |
| `predict` | table | Score rows with a stored model |
| `list_models`, `model_info`, `drop_model` | table | Model registry management |

Every function carries inline docs and runnable examples, surfaced through
`duckdb_functions()` and the catalog metadata.

## Where models live

`fit` returns the model two ways: always as a self-contained `model` BLOB, and —
when you pass `model_name =>` — persisted to a registry. The store is selected by
`get_store()`; today it is a local-disk backend writing one JSON file per model
under `PYSAL_MODELS_DIR` (default `./models`). An S3/R2 backend drops in behind
the same interface without touching `regression.py`. Because a fitted model is
just an intercept, coefficients, and diagnostics, the registry is plain JSON —
there is no pickle and no code-execution risk on load.

## Install

```sh
uv pip install vgi-pysal     # or: pip install vgi-pysal
```

This installs the `vgi-pysal` (stdio) and `vgi-pysal-http` (HTTP) console
scripts. Point DuckDB at the stdio one:

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'pysal' (TYPE vgi, LOCATION 'vgi-pysal');
```

## Local development

```sh
make venv        # venv against the local ~/Development/vgi-{python,rpc} checkouts
make check       # ruff lint + format, strict mypy, pydoclint, and pytest
make test-stdio  # SQL tests with the worker as a subprocess (authoritative)
make test-http   # SQL tests against a local HTTP server
```

Run the worker straight from a checkout with `uv run pysal_worker.py`. The static
gates (`make lint` / `make typecheck`) and tooling config (ruff with the `D`
docstring rules, strict mypy, pydoclint) mirror the `vgi-python` repo.

## Container image & deployment

One image serves both transports — `docker run … IMG` (HTTP) and
`docker run -i … IMG stdio` (the worker DuckDB spawns):

```sh
make image              # build locally
make test-docker-http   # run the SQL suite against the image over HTTP
make test-docker-stdio  # …and over stdio
```

CI (`docker-publish.yml`) builds the multi-arch image on native runners, tests it
in both transports, and pushes a cosign-signed manifest to
`ghcr.io/query-farm/vgi-pysal` on each `vX.Y.Z` tag (and `:edge` on `main`).
Deploy the published image to Fly.io:

```sh
make deploy TAG=0.1.0   # fly deploy the ghcr image
fly volumes create pysal_models --size 1 --region iad   # one-time, model registry
```

`fly.toml` sets the VM to 1 GB (geopandas/scipy are heavy) and mounts a volume at
`/data` for the registry + framework state.

## Layout

```
vgi_pysal/
  worker.py           builds the `pysal` Catalog + PysalWorker; main()/main_http()
  datasets.py         built-in libpysal example datasets as table functions (WKB geom)
  spatial_weights.py  build_weights() shared helper + weights() / weights_summary()
  esda.py             global autocorrelation (moran / geary / getis_ord_g)
  lisa.py             local autocorrelation (local_moran / getis_ord_g_local)
  classify.py         mapclassify choropleth schemes
  inequality.py       gini / theil aggregates
  regression.py       spreg fit / coefficients / predict + registry management
  registry.py         fitted-model store (local disk now, S3/R2 later)
  buffering.py        shared sink/combine/serialize helpers
  geometry.py         WKB / coordinate parsing
  schema_utils.py     pa.Field comment helper, name sanitisation
pysal_worker.py       repo-root stdio shim (uv run / Fly / tests)
serve.py              repo-root HTTP shim (Fly / tests)
tests/                pytest (in-process harness in tests/harness.py)
test/sql/*.test       DuckDB sqllogictest — the authoritative integration tests
```

## License

MIT — see [LICENSE](LICENSE). Built on [PySAL](https://pysal.org) (libpysal,
esda, spreg, mapclassify, inequality) and the
[VGI](https://github.com/query-farm/vgi-python) framework.
