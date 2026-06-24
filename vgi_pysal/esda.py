"""Global spatial autocorrelation (ESDA) as one-row DuckDB table functions.

Each function buffers the input, builds a spatial weights matrix from it (via the
shared ``build_weights`` convention), runs an `esda` global statistic over a value
column, and returns a single row of results: the statistic, its expectation and
variance, a z-score, and both a permutation (``p_value``) and an analytical
(``p_norm``) significance.

* ``moran``        -- Moran's I (the standard global autocorrelation measure)
* ``geary``        -- Geary's C (more sensitive to local differences)
* ``getis_ord_g``  -- Getis-Ord General G (concentration of high/low values)

    SELECT * FROM pysal.moran((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'queen');
    SELECT * FROM pysal.getis_ord_g((SELECT y, x AS x, y AS y FROM pts), w_type => 'knn', k => 8);
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
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, emit_empty, input_schema_of, numeric_column
from .schema_utils import field as sfield
from .spatial_weights import WeightsArgs, build_weights, validate_weights_args

_GLOBAL_SCHEMA = pa.schema(
    [
        sfield(
            "statistic", pa.float64(), "The global statistic (Moran's I, Geary's C, or Getis-Ord G).", nullable=False
        ),
        sfield("expected", pa.float64(), "Expected value under spatial randomness.", nullable=False),
        sfield("variance", pa.float64(), "Variance of the statistic under the null."),
        sfield("z_score", pa.float64(), "Standardised score (permutation-based)."),
        sfield("p_value", pa.float64(), "Permutation pseudo p-value (one-sided)."),
        sfield("p_norm", pa.float64(), "Analytical p-value under normality."),
        sfield("n", pa.int64(), "Number of observations.", nullable=False),
        sfield("permutations", pa.int64(), "Number of conditional permutations used.", nullable=False),
    ]
)


@dataclass(slots=True, frozen=True)
class GlobalEsdaArgs(WeightsArgs):
    """Weights arguments plus the value column, permutation count, and seed for global ESDA statistics."""

    value: Annotated[str, Arg("value", default="y", doc="Numeric column to test for spatial autocorrelation.")]
    permutations: Annotated[int, Arg("permutations", default=999, doc="Conditional permutations for inference.")]
    seed: Annotated[int, Arg("seed", default=12345, doc="Random seed for reproducible permutation p-values.")]


class _GlobalStat(SinkBuffer[GlobalEsdaArgs, DrainState]):
    """Base: buffer input, build W, compute one global statistic, emit one row."""

    FunctionArguments: ClassVar[type] = GlobalEsdaArgs

    @classmethod
    def compute(cls, y: np.ndarray, w: Any, permutations: int) -> dict[str, float]:
        """Return the statistic dict from an `esda` result. Subclasses implement."""
        raise NotImplementedError

    @classmethod
    def on_bind(cls, params: BindParams[GlobalEsdaArgs]) -> BindResponse:
        """Validate arguments and declare the global-statistic output schema."""
        validate_weights_args(params.args)
        return BindResponse(output_schema=_GLOBAL_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[GlobalEsdaArgs]
    ) -> DrainState:
        """Create the initial drain state for the finalize phase."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[GlobalEsdaArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Build the weights, compute the global statistic, and emit one row of results."""
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
        res = cls.compute(y, w, a.permutations)
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "statistic": [res["statistic"]],
                    "expected": [res["expected"]],
                    "variance": [res.get("variance")],
                    "z_score": [res.get("z_score")],
                    "p_value": [res.get("p_value")],
                    "p_norm": [res.get("p_norm")],
                    "n": [int(w.n)],
                    "permutations": [int(a.permutations)],
                },
                schema=params.output_schema,
            )
        )


class MoranFn(_GlobalStat):
    """Compute the global Moran's I autocorrelation statistic over a value column."""

    class Meta:
        """Catalog metadata for the moran function."""

        name = "moran"
        description = "Global Moran's I spatial autocorrelation statistic"
        categories = ["esda", "autocorrelation", "spatial"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM pysal.moran((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'queen')",
                description="Moran's I of neighbourhood crime using queen contiguity",
            )
        ]

    @classmethod
    def compute(cls, y: np.ndarray, w: Any, permutations: int) -> dict[str, float]:
        """Run esda.Moran and return its statistic, expectation, variance, and p-values."""
        mi = esda.Moran(y, w, permutations=permutations)
        return {
            "statistic": float(mi.I),
            "expected": float(mi.EI),
            "variance": float(mi.VI_norm),
            "z_score": float(mi.z_sim),
            "p_value": float(mi.p_sim),
            "p_norm": float(mi.p_norm),
        }


class GearyFn(_GlobalStat):
    """Compute the global Geary's C autocorrelation statistic over a value column."""

    class Meta:
        """Catalog metadata for the geary function."""

        name = "geary"
        description = "Global Geary's C spatial autocorrelation statistic"
        categories = ["esda", "autocorrelation", "spatial"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM pysal.geary((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'queen')",
                description="Geary's C of neighbourhood crime",
            )
        ]

    @classmethod
    def compute(cls, y: np.ndarray, w: Any, permutations: int) -> dict[str, float]:
        """Run esda.Geary and return its statistic, expectation, variance, and p-values."""
        gc = esda.Geary(y, w, permutations=permutations)
        return {
            "statistic": float(gc.C),
            "expected": float(gc.EC),
            "variance": float(gc.VC_norm),
            "z_score": float(gc.z_sim),
            "p_value": float(gc.p_sim),
            "p_norm": float(gc.p_norm),
        }


class GetisOrdGFn(_GlobalStat):
    """Compute the Getis-Ord General G statistic over a value column."""

    class Meta:
        """Catalog metadata for the getis_ord_g function."""

        name = "getis_ord_g"
        description = "Getis-Ord General G statistic (global concentration of high/low values)"
        categories = ["esda", "autocorrelation", "spatial"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM pysal.getis_ord_g("
                    "(SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'knn', k => 8)"
                ),
                description="General G of crime using 8-nearest-neighbour weights",
            )
        ]

    @classmethod
    def compute(cls, y: np.ndarray, w: Any, permutations: int) -> dict[str, float]:
        """Run esda.G with binary weights and return its statistic, expectation, variance, and p-values."""
        # Getis-Ord G requires binary (or distance) weights, not row-standardised.
        if w.transform == "R":
            w.transform = "B"
        g = esda.G(y, w, permutations=permutations)
        return {
            "statistic": float(g.G),
            "expected": float(g.EG),
            "variance": float(g.VG),
            "z_score": float(g.z_sim),
            "p_value": float(g.p_sim),
            "p_norm": float(g.p_norm),
        }


ESDA_FUNCTIONS: list[type] = [MoranFn, GearyFn, GetisOrdGFn]
