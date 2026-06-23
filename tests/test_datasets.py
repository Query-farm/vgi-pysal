"""Unit tests for the built-in example datasets as table functions."""

from __future__ import annotations

import pyarrow as pa
import pytest

from tests.harness import invoke_table_function
from vgi_pysal.datasets import DATASET_FUNCTIONS

_EXPECTED_ROWS = {
    "columbus": 49,
    "baltim": 211,
    "sids": 100,
    "us_states": 48,
    "mexico": 32,
    "georgia": 159,
}


@pytest.mark.parametrize("func", DATASET_FUNCTIONS, ids=lambda f: f.Meta.name)
def test_dataset_loads_with_expected_shape(func: type) -> None:
    table = invoke_table_function(func)
    name = func.Meta.name
    assert table.num_rows == _EXPECTED_ROWS[name]
    # every dataset exposes an id (first column) and a WKB geom (last column)
    assert table.schema.names[0] == "id"
    assert table.schema.names[-1] == "geom"
    assert pa.types.is_binary(table.schema.field("geom").type)


def test_id_is_dense_zero_based() -> None:
    table = invoke_table_function(DATASET_FUNCTIONS[0])
    ids = table.column("id").to_pylist()
    assert ids == list(range(table.num_rows))


def test_geometry_is_non_null_wkb() -> None:
    table = invoke_table_function(DATASET_FUNCTIONS[0])
    geoms = table.column("geom").to_pylist()
    assert all(isinstance(g, bytes) and g for g in geoms)


def test_columbus_has_expected_attributes() -> None:
    table = invoke_table_function(DATASET_FUNCTIONS[0])
    for col in ("crime", "inc", "hoval", "x", "y"):
        assert col in table.schema.names


def test_dataset_names_are_unique() -> None:
    names = [f.Meta.name for f in DATASET_FUNCTIONS]
    assert len(names) == len(set(names))
