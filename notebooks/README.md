# PySAL Spatial Analysis notebooks

Four executed SQL stories use the public **v0.1.0** release through Haybarn.
Each notebook has an Open in Colab link, visible connection SQL, saved SVG figures,
and an experiment that changes one parameter while keeping source values fixed.

| Notebook | Question | Experiment |
| --- | --- | --- |
| [Neighbors change spatial autocorrelation](01_neighbors_change_spatial_autocorrelation.ipynb) | Do the returned graph and Moran’s I agree with independent SQL calculations? | `K_NEIGHBORS`: 3 → 6 |
| [Classification changes the map](02_classification_changes_the_map.ipynb) | How do quantiles, equal intervals and Fisher–Jenks classify the same historical rates? | `CLASS_COUNT`: 4 → 6 |
| [The population defines inequality](03_the_population_defines_inequality.ipynb) | What changes when the same state is excluded from both years of a distribution comparison? | `EXCLUDE_TOP_2000`: False → True |
| [A saved model has a lifecycle](04_a_saved_model_has_a_lifecycle.ipynb) | Can a stored OLS model’s predictions and deletion be audited explicitly? | `SCORING_SCALE`: 1.0 → 1.5 |

## Run

In Colab, open the badge and choose **Run all**. The first setup cell installs the
shared base package while preserving compatible preloaded notebook libraries.
The worker runs in a separate uv environment. Initial execution needs network
access for packages, extension installation and verified example files. The actual
calculations use bundled teaching data and a stored synthetic fixture; no account
or API credentials are needed. The map notebook also installs DuckDB’s spatial extension.

For local execution, install uv and Git, then run from this directory:

```sh
uv sync --locked
uv run --locked vgi-notebooks --timeout 300
uv run --locked python -m ipykernel install --user --name vgi-pysal-notebooks --display-name 'Python (PySAL examples)'
```

The visible connection uses the release tag, without a local source path or a list
of transitive dependency overrides:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'pysal' (
  TYPE vgi,
  LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-pysal@v0.1.0 vgi-pysal'
);
```

The local environment uses the shared package’s `[runner]` extra. Colab installs
only the base package. Presentation, bootstrap and execution tools come from
[vgi-example-notebooks v0.3.50](https://github.com/Query-farm/vgi-example-notebooks/tree/v0.3.50);
the [shared standard](https://github.com/Query-farm/vgi-example-notebooks/blob/main/docs/standard.md)
and [coverage specification](https://github.com/Query-farm/vgi-example-notebooks/blob/main/docs/coverage.md)
are maintained there. GitHub does not honor notebook cell-folding metadata; setup
is kept separate from the visible connection and analytical SQL.

## Inputs and interpretation

Books 1 and 4 retain the original `fixtures/sensors.csv` and its provenance/hash.
These 25 invented points form a perturbed planar grid with a deliberate reading
gradient, deterministic variation and central bump. Units are arbitrary; they are
not real sensors or geographic coordinates. Directed KNN edges can be asymmetric.
The first book checks the full edge set, outgoing sums, weak connectivity and an
independent reconstruction of Moran’s I. Its permutation value is the tested esda
implementation’s smaller-tail pseudo-p, not a doubled two-sided p-value.

Book 2 uses libpysal’s 49-area Columbus teaching dataset from 1980. The stored rate
combines residential burglary and vehicle theft per 1,000 households. All three
maps retain the same geometry and values; equal ordinal colors across panels do
not mean equal numeric thresholds. These are historical data, not current crime rates.

Book 3 uses the bundled Mexico state per-capita values for 1940 and 2000. Both years
use the same selected roster and equal weights per state. Gini and Theil T are
reconstructed independently in SQL. The Lorenz curve accumulates state per-capita
values; it does not estimate household inequality or population-weighted national GDP.

Book 4 explicitly separates IDs, coordinates, target and predictor columns. A named
OLS model is written to an isolated temporary directory, inspected, used for two
scoring scenarios and deleted. SQL reconstructs every predicted value and change.
Training RMSE is not held-out validation, and changed features are not newly observed
outcomes or causal interventions. Existing user models are never accessed.

The tested release environment used libpysal 4.15.1, esda 2.10.0, mapclassify 2.11.0,
spreg 1.9.2, NumPy 2.5.4 and SciPy 1.18.1. The release declares dependency ranges;
its Git tag alone does not freeze future transitive resolution. Setup uses a headless
backend for the worker, restores previous environment values, and removes the
temporary model directory at cleanup.

## Experiments and validation

Parameter overrides must write into a separate output directory:

```sh
uv run --locked vgi-notebooks 01_neighbors_change_spatial_autocorrelation.ipynb --parameters '{"K_NEIGHBORS":6}' --output-dir artifacts/neighbors-six
uv run --locked vgi-notebooks 02_classification_changes_the_map.ipynb --parameters '{"CLASS_COUNT":6}' --output-dir artifacts/classes-six
uv run --locked vgi-notebooks 03_the_population_defines_inequality.ipynb --parameters '{"EXCLUDE_TOP_2000":true}' --output-dir artifacts/exclude-top
uv run --locked vgi-notebooks 04_a_saved_model_has_a_lifecycle.ipynb --parameters '{"SCORING_SCALE":1.5}' --output-dir artifacts/scaled-features
```

All four defaults and experiments were executed with the released worker. Saved
notebooks include eight SVG figures and numerical assertions. A fresh notebook-only
run with preloaded pandas 2.2.3, Matplotlib 3.10.0, IPython 8.30.0, ipykernel 6.17.1
and jupyter-client 7.4.9 retained those versions and passed `pip check`; this is a
local compatibility test, not a manual hosted-Colab run.

The original graph story’s displayed analytical tables and fixture bytes are retained.
The complete catalog has 28 public functions, including model-registry writes;
these examples explain and execute 16. Release v0.1.0 and public main at authoring
resolve to the same source commit, so coverage applies to that exact revision.
Hash-bound catalogs and successful execution receipts are in `coverage.json` and
`coverage/`. Existing green [source CI](https://github.com/Query-farm/vgi-pysal/actions/runs/43615920989)
was reused; no worker source or root dependency files were modified.

[Worker repository](https://github.com/Query-farm/vgi-pysal) ·
[Query.Farm](https://query.farm/) · [VGI](https://query.farm/vgi/)
