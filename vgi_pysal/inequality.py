"""Inequality measures exposed as DuckDB aggregate functions.

Each measure is an aggregate over a single numeric column, so it composes with
``GROUP BY``:

    SELECT region, pysal.gini(per_capita_gdp) AS gini
    FROM regions GROUP BY region;

Most inequality indices need the full distribution (not streaming sufficient
statistics), so each group buffers its values and the measure is computed once in
``finalize``. NULL values are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import pyarrow as pa
from inequality.gini import Gini
from inequality.theil import Theil
from vgi.aggregate_function import AggregateFunction
from vgi.arguments import Param, Returns
from vgi.metadata import FunctionExample
from vgi.table_function import ProcessParams
from vgi_rpc import ArrowSerializableDataclass


@dataclass(kw_only=True)
class ValueState(ArrowSerializableDataclass):
    """Buffered values for one group."""

    values: list[float] = field(default_factory=list)


class _BufferedInequality(AggregateFunction[ValueState]):
    """Base: buffer all values per group, compute the measure once in finalize.

    Subclasses implement ``compute_measure`` and declare a ``Meta``.
    """

    @classmethod
    def initial_state(cls, params: ProcessParams[None]) -> ValueState:
        return ValueState()

    @classmethod
    def update(
        cls,
        states: dict[int, ValueState],
        group_ids: pa.Int64Array,
        value: Annotated[pa.DoubleArray, Param(doc="Numeric value whose distribution is measured")],
    ) -> None:
        # Accumulate this batch per group, then *reassign* states[g]: the framework
        # only persists groups assigned in this batch (see the aggregate
        # state-persistence sharp edge documented in the sklearn worker).
        batch: dict[int, list[float]] = {}
        for g, v in zip(group_ids.to_pylist(), value.to_pylist(), strict=False):
            if v is None:
                continue
            batch.setdefault(g, []).append(v)
        for g, vals in batch.items():
            s = states[g]
            states[g] = ValueState(values=s.values + vals)

    @classmethod
    def combine(cls, source: ValueState, target: ValueState, params: ProcessParams[None]) -> ValueState:
        return ValueState(values=source.values + target.values)

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, ValueState],
        params: ProcessParams[None],
    ) -> Annotated[pa.RecordBatch, Returns(pa.float64())]:
        results: list[float | None] = []
        for gid in group_ids:
            s = states.get(gid.as_py())
            if s is None or len(s.values) < 2:
                results.append(None)
                continue
            try:
                results.append(float(cls.compute_measure(np.asarray(s.values, dtype=float))))
            except Exception:
                results.append(None)
        return pa.record_batch({"result": pa.array(results, type=pa.float64())})

    @classmethod
    def compute_measure(cls, values: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError


class GiniFn(_BufferedInequality):
    class Meta:
        name = "gini"
        description = "Gini coefficient of inequality (0 = perfect equality, 1 = maximal inequality)"
        categories = ["inequality", "spatial"]
        examples = [
            FunctionExample(
                sql="SELECT pysal.gini(pcgdp2000) FROM pysal.mexico()",
                description="Gini coefficient of Mexican state per-capita GDP in 2000",
            )
        ]

    @classmethod
    def compute_measure(cls, values: np.ndarray) -> float:
        return float(Gini(values).g)


class TheilFn(_BufferedInequality):
    class Meta:
        name = "theil"
        description = "Theil's T entropy index of inequality"
        categories = ["inequality", "spatial"]
        examples = [
            FunctionExample(
                sql="SELECT pysal.theil(pcgdp2000) FROM pysal.mexico()",
                description="Theil index of Mexican state per-capita GDP in 2000",
            )
        ]

    @classmethod
    def compute_measure(cls, values: np.ndarray) -> float:
        return float(Theil(values).T)


INEQUALITY_FUNCTIONS: list[type] = [GiniFn, TheilFn]
