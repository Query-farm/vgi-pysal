"""Local spatial autocorrelation (LISA) as per-observation DuckDB table functions.

Where the ``esda`` global statistics give one number for the whole map, these
return one row per observation so you can find *where* clusters and outliers are.
The input is buffered, a weights matrix is built from it, the local statistic is
computed, and each observation's result is emitted (carrying an optional ``id``
column through for joining back).

* ``local_moran``          -- Local Moran's I_i with HH/LL/HL/LH cluster labels
* ``getis_ord_g_local``    -- Local Getis-Ord G_i / G_i* hot-spot / cold-spot z-scores

    SELECT * FROM pysal.local_moran((SELECT id, crime AS y, geom FROM pysal.columbus()),
                                    w_type => 'queen', id => 'id');
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import esda
import numpy as np
import pyarrow as pa
from vgi.arguments import Arg
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, emit_empty, input_schema_of, numeric_column
from .schema_utils import field as sfield
from .spatial_weights import WeightsArgs, build_weights, validate_weights_args

# Local Moran quadrant codes -> cluster labels.
_QUADRANT = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


@dataclass(slots=True, frozen=True)
class LocalEsdaArgs(WeightsArgs):
    value: Annotated[str, Arg("value", default="y", doc="Numeric column to analyse for local autocorrelation.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through to the output.")]
    permutations: Annotated[int, Arg("permutations", default=999, doc="Conditional permutations for inference.")]
    seed: Annotated[int, Arg("seed", default=12345, doc="Random seed for reproducible permutation p-values.")]
    significance: Annotated[
        float, Arg("significance", default=0.05, doc="Pseudo p-value cutoff for labelling a location significant.")
    ]


@dataclass(slots=True, frozen=True)
class GLocalArgs(LocalEsdaArgs):
    star: Annotated[bool, Arg("star", default=True, doc="Use G_i* (include the focal observation) rather than G_i.")]


class _LocalStat[TArgs](SinkBuffer[TArgs, DrainState]):
    """Base: buffer input, build W, compute a per-observation statistic, stream rows."""

    @classmethod
    def output_fields(cls) -> list[pa.Field]:
        raise NotImplementedError

    @classmethod
    def compute(cls, y: np.ndarray, w: Any, args: Any) -> dict[str, list[Any]]:
        raise NotImplementedError

    @classmethod
    def _schema(cls, input_schema: pa.Schema, id_col: str) -> pa.Schema:
        fields: list[pa.Field] = []
        if id_col and id_col in input_schema.names:
            fields.append(input_schema.field(id_col))
        fields.extend(cls.output_fields())
        return pa.schema(fields)

    @classmethod
    def on_bind(cls, params: BindParams[TArgs]) -> BindResponse:
        validate_weights_args(params.args)
        assert params.bind_call.input_schema is not None
        return BindResponse(output_schema=cls._schema(params.bind_call.input_schema, params.args.id))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[TArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[TArgs],
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

        np.random.seed(a.seed)
        y = numeric_column(table, a.value, what="value")
        w = build_weights(table, a)
        columns: dict[str, list[Any]] = {}
        if a.id and a.id in table.schema.names:
            columns[a.id] = table.column(a.id).to_pylist()
        columns.update(cls.compute(y, w, a))
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


class LocalMoranFn(_LocalStat[LocalEsdaArgs]):
    """Local Moran's I: per-observation clustering with HH/LL (clusters) and HL/LH (outliers)."""

    FunctionArguments: ClassVar[type] = LocalEsdaArgs

    class Meta:
        name = "local_moran"
        description = "Local Moran's I (LISA): per-observation spatial clusters and outliers"
        categories = ["esda", "lisa", "autocorrelation", "spatial"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT id, cluster, p_value FROM pysal.local_moran("
                    "(SELECT id, crime AS y, geom FROM pysal.columbus()), w_type => 'queen', id => 'id') "
                    "WHERE p_value < 0.05"
                ),
                description="Significant local crime clusters/outliers on columbus",
            )
        ]

    @classmethod
    def output_fields(cls) -> list[pa.Field]:
        return [
            sfield("local_i", pa.float64(), "Local Moran's I_i statistic.", nullable=False),
            sfield("z_score", pa.float64(), "Standardised score (permutation-based)."),
            sfield("p_value", pa.float64(), "Permutation pseudo p-value (one-sided)."),
            sfield("quadrant", pa.int64(), "Moran scatterplot quadrant: 1=HH, 2=LH, 3=LL, 4=HL.", nullable=False),
            sfield(
                "cluster",
                pa.string(),
                "Cluster label at the significance cutoff: HH/LL (clusters), HL/LH (outliers), or 'ns'.",
                nullable=False,
            ),
        ]

    @classmethod
    def compute(cls, y: np.ndarray, w: Any, args: Any) -> dict[str, list[Any]]:
        lm = esda.Moran_Local(y, w, permutations=args.permutations, seed=args.seed)
        p = lm.p_sim
        labels = [_QUADRANT[int(q)] if pv <= args.significance else "ns" for q, pv in zip(lm.q, p, strict=True)]
        return {
            "local_i": [float(v) for v in lm.Is],
            "z_score": [float(v) for v in lm.z_sim],
            "p_value": [float(v) for v in p],
            "quadrant": [int(v) for v in lm.q],
            "cluster": labels,
        }


class GetisOrdGLocalFn(_LocalStat[GLocalArgs]):
    """Local Getis-Ord G_i / G_i*: per-observation hot-spot and cold-spot z-scores."""

    FunctionArguments: ClassVar[type] = GLocalArgs

    class Meta:
        name = "getis_ord_g_local"
        description = "Local Getis-Ord G_i/G_i* hot-spot and cold-spot analysis"
        categories = ["esda", "lisa", "autocorrelation", "spatial"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT id, z_score, hotspot FROM pysal.getis_ord_g_local("
                    "(SELECT id, crime AS y, geom FROM pysal.columbus()), w_type => 'knn', k => 8, id => 'id')"
                ),
                description="Crime hot/cold spots using G_i* and 8-nearest-neighbour weights",
            )
        ]

    @classmethod
    def output_fields(cls) -> list[pa.Field]:
        return [
            sfield("g_local", pa.float64(), "Local G_i (or G_i*) statistic.", nullable=False),
            sfield(
                "z_score", pa.float64(), "Standardised score; large positive = hot spot, large negative = cold spot."
            ),
            sfield("p_value", pa.float64(), "Permutation pseudo p-value (one-sided)."),
            sfield(
                "hotspot",
                pa.string(),
                "'hot' / 'cold' at the significance cutoff (positive/negative z), else 'ns'.",
                nullable=False,
            ),
        ]

    @classmethod
    def compute(cls, y: np.ndarray, w: Any, args: Any) -> dict[str, list[Any]]:
        # Getis-Ord uses binary / distance weights, not row-standardised.
        if w.transform == "R":
            w.transform = "B"
        gl = esda.G_Local(y, w, star=args.star, permutations=args.permutations, seed=args.seed)
        labels = [
            ("hot" if z > 0 else "cold") if pv <= args.significance else "ns"
            for z, pv in zip(gl.Zs, gl.p_sim, strict=True)
        ]
        return {
            "g_local": [float(v) for v in gl.Gs],
            "z_score": [float(v) for v in gl.Zs],
            "p_value": [float(v) for v in gl.p_sim],
            "hotspot": labels,
        }


LISA_FUNCTIONS: list[type] = [LocalMoranFn, GetisOrdGLocalFn]
