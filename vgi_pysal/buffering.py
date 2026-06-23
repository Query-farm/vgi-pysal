"""Shared plumbing for table-buffering functions.

Spatial analysis is inherently global: a spatial weights matrix, Moran's I, a
choropleth classification or a regression all need the *whole* input before they
can produce output. Buffering functions handle this -- the sink phase serializes
each input batch to execution-scoped storage; finalize reassembles the full
table. This module holds the serialization, storage, and column-extraction
helpers plus the single-bucket sink/combine implementation so each function only
writes its finalize logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi_rpc import ArrowSerializableDataclass

_DATA_KEY = b"input_batches"


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Per-finalize-stream cursor: emit the result once, then finish."""

    done: bool = False


def serialize_batch(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def deserialize_batches(value: bytes) -> list[pa.RecordBatch]:
    reader = pa.ipc.open_stream(pa.BufferReader(value))
    return reader.read_all().to_batches()


def numeric_column(table: pa.Table, name: str, *, what: str = "value") -> np.ndarray:
    """Return one named column as a 1D float64 array, with a clear error if absent/non-numeric.

    Raises a friendly ``ValueError`` -- rather than an opaque pyarrow KeyError or
    numpy ValueError -- when the column is missing or not numeric. ``what`` labels
    the column in error messages (e.g. "value", "target").
    """
    if name not in table.schema.names:
        raise ValueError(f"missing required {what} column {name!r}; input has columns: {', '.join(table.schema.names)}")
    ftype = table.schema.field(name).type
    if not (pa.types.is_floating(ftype) or pa.types.is_integer(ftype) or pa.types.is_boolean(ftype)):
        raise ValueError(
            f"{what} column {name!r} must be numeric, but is {ftype}. Select a numeric column or cast it first."
        )
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=float)


def matrix(table: pa.Table, feature_names: list[str], *, what: str = "feature") -> np.ndarray:
    """Assemble the named columns (in the given order) into a 2D float64 array.

    Selects ``feature_names`` by name, so input column order does not matter and
    extra columns are ignored. Raises a clear error when a column is missing or
    not numeric. ``what`` labels the columns in error messages.
    """
    present = set(table.schema.names)
    missing = [n for n in feature_names if n not in present]
    if missing:
        raise ValueError(
            f"missing required {what} column(s): {', '.join(missing)}; "
            f"input has columns: {', '.join(table.schema.names)}"
        )
    non_numeric = [
        n
        for n in feature_names
        if not pa.types.is_floating(table.schema.field(n).type)
        and not pa.types.is_integer(table.schema.field(n).type)
        and not pa.types.is_boolean(table.schema.field(n).type)
    ]
    if non_numeric:
        raise ValueError(
            f"{what} column(s) must be numeric, but these are not: "
            + ", ".join(f"{n} ({table.schema.field(n).type})" for n in non_numeric)
            + ". Select only numeric columns, or encode/scale them first."
        )
    cols = [np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=float) for name in feature_names]
    if not cols:
        return np.empty((table.num_rows, 0), dtype=float)
    return np.column_stack(cols)


class SinkBuffer[TArgs, TState](TableBufferingFunction[TArgs, TState]):
    """Single-bucket sink/combine: buffer every input batch under one key.

    Subclasses implement ``on_bind``, ``initial_finalize_state``, and
    ``finalize`` (calling ``buffered_table(params)`` to get the full input).
    """

    @classmethod
    def process(cls, batch: pa.RecordBatch, params: TableBufferingParams[TArgs]) -> bytes:
        if batch.num_rows:
            params.storage.state_append(_DATA_KEY, b"", serialize_batch(batch))
        return params.execution_id

    @classmethod
    def combine(cls, state_ids: list[bytes], params: TableBufferingParams[TArgs]) -> list[bytes]:
        return [params.execution_id]

    @classmethod
    def buffered_table(cls, params: TableBufferingParams[TArgs], input_schema: pa.Schema) -> pa.Table | None:
        batches: list[pa.RecordBatch] = []
        for _sid, value in params.storage.state_log_scan(_DATA_KEY, b""):
            batches.extend(deserialize_batches(value))
        if not batches:
            return None
        return pa.Table.from_batches(batches, schema=input_schema)


def input_schema_of(params: Any) -> pa.Schema:
    """Input schema from a process/finalize params object."""
    schema = params.init_call.bind_call.input_schema
    assert schema is not None
    return schema


def emit_empty(out: Any, schema: pa.Schema) -> None:
    """Emit a single empty batch matching ``schema`` (the no-input finalize case)."""
    out.emit(pa.RecordBatch.from_pydict({n: [] for n in schema.names}, schema=schema))
