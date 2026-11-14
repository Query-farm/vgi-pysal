# PySAL notebooks

[01 · Neighbors change spatial autocorrelation](01_neighbors_change_spatial_autocorrelation.ipynb)
uses 25 explicitly synthetic sensor locations and readings to examine neighborhood
sensitivity. It audits directed KNN weights, checks them independently against SQL
distances, and reconstructs Moran's I from the returned graph. The end experiment
changes three neighbors to six while holding the input and permutation settings fixed.

## Run from this directory alone

Install [uv](https://docs.astral.sh/uv/) and Git:

```sh
cd notebooks
uv sync --locked
uv run --locked python execute.py
uv run --locked python -m ipykernel install --user --name vgi-pysal-notebooks --display-name 'Python (vgi-pysal notebooks)'
```

Select that kernel in Jupyter and start it in this directory. Python 3.13/3.14 is
supported for the notebook; the worker launch requests Python 3.14 through uv.
Initial setup downloads the locked notebook packages, VGI extension, immutable Git
worker and scientific dependencies. No source checkout, root .venv or worker build
is required. The tested platform is macOS ARM64, Haybarn 2.0.0rc11. Computation uses
only the stored synthetic CSV; there is no replay/live API mode.

The expanded SQL is literal and portable:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'pysal' (
 TYPE vgi,
 LOCATION 'uvx --python 3.14 --from git+https://github.com/Query-farm/vgi-pysal.git@b3ef7feed7e55d3b223cce73f417843bcd3caf24 --constraints worker-constraints.txt vgi-pysal'
);
```

The repository has no published release and PyPI `vgi-pysal` returned 404 during
validation. This launch uses the published Git revision instead. `uvx` caches the
installation. `worker-constraints.txt` pins tested numerical dependencies, including
libpysal 4.14.1, esda 2.10.0, NumPy 2.5.0 and SciPy 1.18.0, plus SDK 0.51.0 and
RPC 0.53.1. The unchanged root lock uses SDK 0.50.0/protocol 3.2; the published
notebook engine requires protocol 3.3. The notebook-only SDK pin resolves that
compatibility issue; no root dependency files are changed.

The worker imports Matplotlib indirectly through mapclassify. Setup sets the
subprocess's `MPLBACKEND=Agg` after configuring this notebook's inline backend,
then restores the prior environment value at cleanup. This prevents the worker
from trying to import a notebook-only backend. To upgrade, change the literal Git
revision and constraints, then revalidate both executions and source SQL suites.

## Synthetic provenance and interpretation

`fixtures/sensors.csv` stores a perturbed 5×5 grid. The reading is a deliberate
planar gradient plus deterministic variation and a central bump. The exact formula,
arbitrary units and SHA256 are in `fixtures/provenance.json`; the notebook verifies
the hash before execution. These are not measurements, geographic coordinates or
evidence about a real population. Coordinates are Euclidean planar units, not
latitude/longitude. No data were collected from sensors or external APIs.

KNN is directed and can be asymmetric. The graph uses explicit sensor IDs and a
stable row order. Every outgoing row is standardized to one; no islands alone does
not imply connectivity, so SQL counts weak components too. An independent distance
ranking checks the full neighbor set and rejects equal-distance ties in this input.
The worker buffers all rows, so this example is deliberately small.

The second chart plots reading z-scores against weighted neighbor means. Its line
is `I × z` through the origin, not a predictive fit. Quadrant signs are not locally
significant clusters. SQL independently reconstructs the worker's statistic.

The pinned esda implementation returns a smaller-tail permutation pseudo-p:
`(min(number >= observed I, number < observed I) + 1)/(permutations + 1)`.
This is not a doubled two-sided p-value, a local test, or a probability that the
null is true. The values are globally permuted over a fixed graph; the reference
assumes exchangeability under spatial randomness. The constructed gradient makes
this a demonstration of computation, not substantive inference. Reproducibility
requires the seed and numerical runtime, as well as the inputs, to stay fixed.

References: [KNN](https://pysal.org/libpysal/v4.14.1/generated/libpysal.weights.KNN.html),
[Moran](https://pysal.org/esda/v2.9.0/generated/esda.Moran.html). The precise pseudo-p
formula above was checked against the installed esda 2.10.0 source.

## One-parameter experiment

```sh
uv run --locked python execute.py --parameters '{"K_NEIGHBORS":6}' --output-dir validation/k-6
```

Keep `PERMUTATIONS=499` and `SEED=1729` fixed. Compare the focal arrows, neighbor
table, directed link counts, scatterplot and Moran's I. The CSV hash, input
fingerprint, readings and color scale must stay unchanged. Restore `K_NEIGHBORS=3`
afterwards. Increasing k does not guarantee a smaller I in every dataset.

The runner uses a fresh kernel and atomically replaces saved outputs only after
success. Parameter overrides require a separate destination; unknown names are
rejected. Experiment copies include the relative assets, fixtures, constraints,
locked notebook environment and documentation. `artifacts/` and `validation/` are
ignored; the main notebook's successful embedded outputs are committed.

## Source validation and baseline issues

The unchanged source passes 69 unit tests with its locked runtime and again with
the notebook's SDK/Arrow override. Ruff lint/format and strict mypy pass. The
Makefile's in-environment pydoclint crashes because `docstring-parser-fork` and
`docstring-parser` overlap (`DocstringYields` import error). The repository CI
already specifies isolated `uvx pydoclint==0.8.7`; that documented route passes.

The original SDK 0.50.0 stdio SQL suite fails the protocol 3.2/3.3 handshake.
The immutable Git worker with the notebook constraints passes the authoritative
112 stdio SQL assertions in eight cases, plus 128 HTTP assertions in eight cases
(including explicit httpfs install/load prerequisites in the staged tests). The
stock preprocessing initially skipped the HTTP suite because httpfs was not loaded;
validation disables ignored-error skips and explicitly loads it. Arrow 24.0.0 is
retained from the source lock, and strict mypy also passes under the SDK override.
Source and root locks remain unchanged.

`vgi-lint-check==2.5.0 --fail-on error` reports 22/100 and exits 2: 18 errors,
154 warnings and two informational findings. Errors include missing agent tasks
(VGI152), undeclared argument choices (VGI317), missing executable examples
(VGI509), eight example-binding failures (VGI901), and five first-row latency
findings (VGI911). These are source metadata/example baseline findings; this
notebook does not edit worker functionality or metadata. See [STANDARD.md](STANDARD.md)
for notebook conventions.
