"""Choropleth classification schemes (mapclassify) as DuckDB table functions.

Each function buckets a numeric column into ``k`` classes and returns, per input
row, the class index and the class's upper bound -- the same binning you would
use to colour a choropleth map. Quantiles, natural breaks, etc. need the whole
column to choose break points, so these are buffering functions.

* ``quantiles``        -- equal-count classes
* ``equal_interval``   -- equal-width classes
* ``natural_breaks``   -- Jenks natural breaks (k-means style)
* ``fisher_jenks``     -- optimal natural breaks
* ``std_mean``         -- breaks at the mean +/- standard deviations
* ``box_plot``         -- quartile/whisker classes (outlier-aware)
* ``head_tail_breaks`` -- recursive head/tail split for heavy-tailed data

    SELECT id, class, upper_bound FROM pysal.quantiles(
        (SELECT id, crime AS y FROM pysal.columbus()), value => 'y', k => 5);
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import mapclassify
import numpy as np
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, emit_empty, input_schema_of, numeric_column
from .schema_utils import field as sfield


@dataclass(slots=True, frozen=True)
class ClassifyArgs:
    """Arguments shared by every classification table function."""

    data: Annotated[TableInput, Arg(0, doc="Input table containing the value column (and optional id).")]
    value: Annotated[str, Arg("value", default="y", doc="Numeric column to classify.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through to the output.")]
    k: Annotated[int, Arg("k", default=5, doc="Number of classes (ignored by schemes with data-driven class counts).")]


_CLASSIFY_OUTPUT = [
    sfield("class", pa.int64(), "0-based class index (0 = lowest class).", nullable=False),
    sfield("upper_bound", pa.float64(), "Upper bound (inclusive) of the assigned class.", nullable=False),
]


class _ClassifyFn(SinkBuffer[ClassifyArgs, DrainState]):
    """Buffer the whole value column, fit a mapclassify scheme, emit per-row class + bound."""

    FunctionArguments: ClassVar[type] = ClassifyArgs
    SCHEME: ClassVar[Any]
    USES_K: ClassVar[bool] = True

    @classmethod
    def classify(cls, y: np.ndarray, k: int) -> Any:
        """Fit this function's mapclassify scheme to the value array."""
        return cls.SCHEME(y, k=k) if cls.USES_K else cls.SCHEME(y)

    @classmethod
    def _schema(cls, input_schema: pa.Schema, id_col: str) -> pa.Schema:
        """Build the output schema, optionally carrying through the id column."""
        fields: list[pa.Field] = []
        if id_col and id_col in input_schema.names:
            fields.append(input_schema.field(id_col))
        fields.extend(_CLASSIFY_OUTPUT)
        return pa.schema(fields)

    @classmethod
    def on_bind(cls, params: BindParams[ClassifyArgs]) -> BindResponse:
        """Validate ``k`` and resolve the output schema at bind time."""
        if cls.USES_K and params.args.k < 1:
            raise ValueError(f"{cls.Meta.name} requires k >= 1")
        assert params.bind_call.input_schema is not None
        return BindResponse(output_schema=cls._schema(params.bind_call.input_schema, params.args.id))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[ClassifyArgs]) -> DrainState:
        """Return the initial drain state for the finalize phase."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[ClassifyArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Classify the buffered column and emit one row per input row."""
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        table = cls.buffered_table(params, input_schema_of(params))
        if table is None or table.num_rows == 0:
            emit_empty(out, params.output_schema)
            return

        y = numeric_column(table, a.value, what="value")
        scheme = cls.classify(y, a.k)
        bins = np.asarray(scheme.bins, dtype=float)
        yb = np.asarray(scheme.yb, dtype=int)
        columns: dict[str, list[Any]] = {}
        if a.id and a.id in table.schema.names:
            columns[a.id] = table.column(a.id).to_pylist()
        columns["class"] = [int(v) for v in yb]
        columns["upper_bound"] = [float(bins[min(v, len(bins) - 1)]) for v in yb]
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


def _ex(name: str, extra: str = "") -> list[FunctionExample]:
    """Build a one-item catalog example list for a classification function."""
    args = f", {extra}" if extra else ""
    return [
        FunctionExample(
            sql=(
                f"SELECT id, class, upper_bound FROM pysal.{name}("
                f"(SELECT id, crime AS y FROM pysal.columbus()), value => 'y'{args})"
            ),
            description=f"Classify columbus crime with {name}",
        )
    ]


class QuantilesFn(_ClassifyFn):
    """Classify values into equal-count quantile classes."""

    SCHEME = mapclassify.Quantiles

    class Meta:
        """Catalog metadata for the quantiles function."""

        name = "quantiles"
        description = "Quantile classification (equal number of observations per class)"
        categories = ["classify", "choropleth"]
        examples = _ex("quantiles", "k => 5")


class EqualIntervalFn(_ClassifyFn):
    """Classify values into equal-width interval classes."""

    SCHEME = mapclassify.EqualInterval

    class Meta:
        """Catalog metadata for the equal_interval function."""

        name = "equal_interval"
        description = "Equal-interval classification (equal value width per class)"
        categories = ["classify", "choropleth"]
        examples = _ex("equal_interval", "k => 5")


class NaturalBreaksFn(_ClassifyFn):
    """Classify values using Jenks natural breaks (k-means style)."""

    SCHEME = mapclassify.NaturalBreaks

    class Meta:
        """Catalog metadata for the natural_breaks function."""

        name = "natural_breaks"
        description = "Jenks natural-breaks classification (k-means style)"
        categories = ["classify", "choropleth"]
        examples = _ex("natural_breaks", "k => 5")


class FisherJenksFn(_ClassifyFn):
    """Classify values using optimal Fisher-Jenks natural breaks."""

    SCHEME = mapclassify.FisherJenks

    class Meta:
        """Catalog metadata for the fisher_jenks function."""

        name = "fisher_jenks"
        description = "Fisher-Jenks optimal natural-breaks classification"
        categories = ["classify", "choropleth"]
        examples = _ex("fisher_jenks", "k => 5")


class StdMeanFn(_ClassifyFn):
    """Classify values by standard deviations from the mean."""

    SCHEME = mapclassify.StdMean
    USES_K = False

    class Meta:
        """Catalog metadata for the std_mean function."""

        name = "std_mean"
        description = "Standard-deviation classification (breaks at mean +/- multiples of std)"
        categories = ["classify", "choropleth"]
        examples = _ex("std_mean")


class BoxPlotFn(_ClassifyFn):
    """Classify values into box-plot quartile and whisker classes."""

    SCHEME = mapclassify.BoxPlot
    USES_K = False

    class Meta:
        """Catalog metadata for the box_plot function."""

        name = "box_plot"
        description = "Box-plot classification (quartiles and whiskers; flags outliers)"
        categories = ["classify", "choropleth"]
        examples = _ex("box_plot")


class HeadTailBreaksFn(_ClassifyFn):
    """Classify heavy-tailed values using recursive head/tail breaks."""

    SCHEME = mapclassify.HeadTailBreaks
    USES_K = False

    class Meta:
        """Catalog metadata for the head_tail_breaks function."""

        name = "head_tail_breaks"
        description = "Head/tail-breaks classification for heavy-tailed (long-tail) distributions"
        categories = ["classify", "choropleth"]
        examples = _ex("head_tail_breaks")


CLASSIFY_FUNCTIONS: list[type] = [
    QuantilesFn,
    EqualIntervalFn,
    NaturalBreaksFn,
    FisherJenksFn,
    StdMeanFn,
    BoxPlotFn,
    HeadTailBreaksFn,
]
