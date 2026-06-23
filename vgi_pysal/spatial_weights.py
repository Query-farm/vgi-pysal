"""Spatial weights construction -- the shared spatial primitive of the worker.

A spatial weights matrix ``W`` encodes which observations are neighbours and how
strongly. Almost everything else here (Moran's I, LISA, spatial regression)
needs one, so ``build_weights`` is the single place that turns table columns +
arguments into a ``libpysal.weights.W``. The ESDA and regression layers reuse it
so the spatial-input convention is identical everywhere:

* ``w_type => 'queen' | 'rook'`` -- contiguity from a WKB ``geom`` column.
* ``w_type => 'knn' | 'distance_band' | 'kernel'`` -- distance from ``x``/``y``
  coordinate columns (or geometry centroids if only ``geom`` is given).

Two functions expose weights directly:

* ``weights`` -- the neighbour graph as an edge list ``(focal, neighbor, weight)``.
* ``weights_summary`` -- one row of structural diagnostics (connectivity, islands).

    SELECT * FROM pysal.weights((SELECT id, ST_AsWKB(geom) AS geom FROM tracts), w_type => 'queen');
    SELECT * FROM pysal.weights_summary((SELECT x, y FROM points), w_type => 'knn', k => 6);
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
from libpysal import weights as lpw
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, emit_empty, input_schema_of
from .geometry import coords_from_columns, coords_from_geometries, has_column, parse_wkb_column
from .schema_utils import field as sfield

CONTIGUITY = {"queen", "rook"}
DISTANCE = {"knn", "distance_band", "kernel"}
W_TYPES = CONTIGUITY | DISTANCE

# libpysal single-letter transform codes, exposed under friendly names.
_TRANSFORMS = {
    "o": "O",  # original / binary as built
    "original": "O",
    "b": "B",  # binary
    "binary": "B",
    "r": "R",  # row-standardised (weights sum to 1 per row)
    "row": "R",
    "d": "D",  # doubly standardised
    "v": "V",  # variance stabilising
}


@dataclass(slots=True, frozen=True)
class WeightsArgs:
    """Common spatial-weights construction arguments.

    Subclassed by every function that needs a ``W`` so the spatial-input
    convention (``w_type`` + ``geom`` or ``x``/``y``) is identical everywhere.
    """

    data: Annotated[TableInput, Arg(0, doc="Input table with a WKB 'geom' column and/or 'x'/'y' coordinate columns.")]
    w_type: Annotated[
        str, Arg("w_type", default="queen", doc="Weights type: queen, rook, knn, distance_band, or kernel.")
    ]
    geom: Annotated[str, Arg("geom", default="geom", doc="WKB geometry column name (for queen/rook).")]
    x: Annotated[str, Arg("x", default="x", doc="X / longitude coordinate column (for distance weights).")]
    y: Annotated[str, Arg("y", default="y", doc="Y / latitude coordinate column (for distance weights).")]
    k: Annotated[int, Arg("k", default=4, doc="Number of nearest neighbours (knn, and bandwidth for kernel).")]
    threshold: Annotated[
        float, Arg("threshold", default=0.0, doc="Distance band threshold (distance_band; 0 = auto min-nn distance).")
    ]
    kernel_function: Annotated[
        str, Arg("kernel_function", default="triangular", doc="Kernel: triangular, uniform, quadratic, gaussian.")
    ]
    transform: Annotated[
        str, Arg("transform", default="r", doc="Weights transform: r (row-standardised), b (binary), o, d, v.")
    ]


def normalise_w_type(w_type: str) -> str:
    wt = (w_type or "").strip().lower()
    if wt not in W_TYPES:
        raise ValueError(f"unknown w_type {w_type!r}; choose one of: {', '.join(sorted(W_TYPES))}")
    return wt


def normalise_transform(transform: str) -> str:
    code = _TRANSFORMS.get((transform or "").strip().lower())
    if code is None:
        raise ValueError(f"unknown transform {transform!r}; choose one of: r (row), b (binary), o (original), d, v")
    return code


def validate_weights_args(args: Any) -> None:
    """Bind-time validation of weight-construction arguments (raise friendly errors early)."""
    normalise_w_type(args.w_type)
    normalise_transform(args.transform)
    if args.w_type.strip().lower() == "knn" and args.k < 1:
        raise ValueError("knn weights require k >= 1")


def build_weights(table: pa.Table, args: Any) -> lpw.W:
    """Construct a ``libpysal.weights.W`` from a buffered table per the ``WeightsArgs`` convention.

    Observations are identified by 0-based position in ``table`` row order, so the
    weights line up with any per-row results computed from the same table.
    """
    if table.num_rows == 0:
        raise ValueError("cannot build spatial weights from an empty input")
    wt = normalise_w_type(args.w_type)

    if wt in CONTIGUITY:
        geoms = parse_wkb_column(table, args.geom)
        ids = list(range(len(geoms)))
        cls = lpw.Queen if wt == "queen" else lpw.Rook
        w = cls.from_iterable(geoms, ids=ids, silence_warnings=True)
    else:
        coords = _coords(table, args)
        if wt == "knn":
            k = min(args.k, table.num_rows - 1)
            if k < 1:
                raise ValueError("knn weights need at least 2 observations")
            w = lpw.KNN.from_array(coords, k=k, silence_warnings=True)
        elif wt == "distance_band":
            threshold = args.threshold if args.threshold > 0 else lpw.min_threshold_distance(coords)
            w = lpw.DistanceBand.from_array(coords, threshold=threshold, silence_warnings=True)
        else:  # kernel
            k = min(max(args.k, 1), table.num_rows - 1)
            w = lpw.Kernel.from_array(coords, k=k, function=args.kernel_function, fixed=False, silence_warnings=True)

    w.transform = normalise_transform(args.transform)
    return w


def _coords(table: pa.Table, args: Any) -> np.ndarray:
    """Coordinates for distance weights: prefer x/y columns, fall back to geometry centroids."""
    if has_column(table, args.x) and has_column(table, args.y):
        return coords_from_columns(table, args.x, args.y)
    if has_column(table, args.geom):
        return coords_from_geometries(parse_wkb_column(table, args.geom))
    raise ValueError(
        f"distance weights need coordinate columns ({args.x!r}, {args.y!r}) or a geometry column ({args.geom!r}); "
        f"input columns: {', '.join(table.schema.names)}"
    )


def cardinalities(w: lpw.W) -> np.ndarray:
    """Number of neighbours per observation, in id order."""
    return np.array([len(w.neighbors[i]) for i in w.id_order], dtype=float)


# ===========================================================================
# weights() -- the neighbour graph as an edge list
# ===========================================================================


@dataclass(slots=True, frozen=True)
class WeightsEdgeArgs(WeightsArgs):
    id: Annotated[
        str,
        Arg(
            "id", default="", doc="Optional id column; focal/neighbor are mapped to its values (else 0-based position)."
        ),
    ]


_WEIGHTS_SCHEMA = pa.schema(
    [
        sfield("focal", pa.string(), "Focal (source) observation id.", nullable=False),
        sfield("neighbor", pa.string(), "Neighbouring observation id (NULL for an island with no neighbours)."),
        sfield("weight", pa.float64(), "Edge weight after the requested transform (NULL for an island)."),
    ]
)


class WeightsFn(SinkBuffer[WeightsEdgeArgs, DrainState]):
    """Emit the spatial weights graph as an edge list, one row per neighbour link."""

    FunctionArguments: ClassVar[type] = WeightsEdgeArgs

    class Meta:
        name = "weights"
        description = "Build spatial weights and return the neighbour graph as an edge list"
        categories = ["weights", "spatial"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM pysal.weights("
                    "(SELECT polyid AS id, geom FROM pysal.columbus()), w_type => 'queen', id => 'id')"
                ),
                description="Queen-contiguity neighbour graph for the columbus polygons",
            ),
            FunctionExample(
                sql=(
                    "SELECT * FROM pysal.weights("
                    "(SELECT polyid AS id, x, y FROM pysal.columbus()), w_type => 'knn', k => 5, id => 'id')"
                ),
                description="5-nearest-neighbour graph from point coordinates",
            ),
        ]

    @classmethod
    def on_bind(cls, params: BindParams[WeightsEdgeArgs]) -> BindResponse:
        validate_weights_args(params.args)
        return BindResponse(output_schema=_WEIGHTS_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[WeightsEdgeArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[WeightsEdgeArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        table = cls.buffered_table(params, input_schema_of(params))
        if table is None or table.num_rows == 0:
            emit_empty(out, params.output_schema)
            return

        w = build_weights(table, a)
        labels = _id_labels(table, a.id, w.n)

        focal: list[str] = []
        neighbor: list[str | None] = []
        weight: list[float | None] = []
        for i in w.id_order:
            neigh = w.neighbors[i]
            wts = w.weights[i]
            if not neigh:  # island
                focal.append(labels[i])
                neighbor.append(None)
                weight.append(None)
                continue
            for j, wij in zip(neigh, wts, strict=True):
                focal.append(labels[i])
                neighbor.append(labels[j])
                weight.append(float(wij))
        out.emit(
            pa.RecordBatch.from_pydict(
                {"focal": focal, "neighbor": neighbor, "weight": weight}, schema=params.output_schema
            )
        )


def _id_labels(table: pa.Table, id_col: str, n: int) -> list[str]:
    """Map 0-based positions to string labels from ``id_col`` (or the position itself)."""
    if id_col and id_col in table.schema.names:
        return [str(v) for v in table.column(id_col).to_pylist()]
    return [str(i) for i in range(n)]


# ===========================================================================
# weights_summary() -- structural diagnostics, one row
# ===========================================================================


_SUMMARY_SCHEMA = pa.schema(
    [
        sfield("w_type", pa.string(), "Weights type used.", nullable=False),
        sfield("transform", pa.string(), "Applied transform code.", nullable=False),
        sfield("n", pa.int64(), "Number of observations.", nullable=False),
        sfield("n_links", pa.int64(), "Number of directed neighbour links (nonzero entries).", nullable=False),
        sfield("pct_nonzero", pa.float64(), "Percent of the n x n matrix that is nonzero.", nullable=False),
        sfield("n_islands", pa.int64(), "Observations with no neighbours.", nullable=False),
        sfield("min_neighbors", pa.int64(), "Minimum neighbour count.", nullable=False),
        sfield("max_neighbors", pa.int64(), "Maximum neighbour count.", nullable=False),
        sfield("mean_neighbors", pa.float64(), "Average neighbour count.", nullable=False),
    ]
)


class WeightsSummaryFn(SinkBuffer[WeightsArgs, DrainState]):
    """Summarise the structure of a spatial weights matrix in a single row."""

    FunctionArguments: ClassVar[type] = WeightsArgs

    class Meta:
        name = "weights_summary"
        description = "Structural diagnostics for a spatial weights matrix (connectivity, islands)"
        categories = ["weights", "spatial"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM pysal.weights_summary((SELECT geom FROM pysal.columbus()), w_type => 'queen')",
                description="Connectivity summary of queen weights on columbus",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[WeightsArgs]) -> BindResponse:
        validate_weights_args(params.args)
        return BindResponse(output_schema=_SUMMARY_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[WeightsArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[WeightsArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        table = cls.buffered_table(params, input_schema_of(params))
        if table is None or table.num_rows == 0:
            emit_empty(out, params.output_schema)
            return

        w = build_weights(table, a)
        card = cardinalities(w)
        n_links = int(card.sum())
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "w_type": [normalise_w_type(a.w_type)],
                    "transform": [w.transform],
                    "n": [int(w.n)],
                    "n_links": [n_links],
                    "pct_nonzero": [float(w.pct_nonzero)],
                    "n_islands": [int(len(w.islands))],
                    "min_neighbors": [int(card.min())],
                    "max_neighbors": [int(card.max())],
                    "mean_neighbors": [float(card.mean())],
                },
                schema=params.output_schema,
            )
        )


WEIGHTS_FUNCTIONS: list[type] = [WeightsFn, WeightsSummaryFn]
