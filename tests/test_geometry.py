"""Unit tests for WKB / coordinate parsing."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import shapely

from vgi_pysal.geometry import (
    coords_from_columns,
    coords_from_geometries,
    has_column,
    is_binary_column,
    parse_wkb_column,
)


def _wkb_table() -> pa.Table:
    pts = [shapely.Point(0, 0), shapely.Point(1, 1), shapely.Point(2, 4)]
    return pa.table({"geom": pa.array([p.wkb for p in pts], type=pa.binary())})


def test_parse_wkb_roundtrips() -> None:
    geoms = parse_wkb_column(_wkb_table(), "geom")
    assert len(geoms) == 3
    assert [shapely.get_x(g) for g in geoms] == [0.0, 1.0, 2.0]


def test_parse_wkb_missing_column() -> None:
    with pytest.raises(ValueError, match="geometry column 'geom' not found"):
        parse_wkb_column(pa.table({"x": [1.0]}), "geom")


def test_parse_wkb_rejects_non_binary() -> None:
    with pytest.raises(ValueError, match="must be WKB BLOB"):
        parse_wkb_column(pa.table({"geom": ["not-wkb"]}), "geom")


def test_parse_wkb_rejects_null() -> None:
    t = pa.table({"geom": pa.array([shapely.Point(0, 0).wkb, None], type=pa.binary())})
    with pytest.raises(ValueError, match="NULL geometries"):
        parse_wkb_column(t, "geom")


def test_coords_from_columns() -> None:
    t = pa.table({"x": [0.0, 3.0], "y": [1.0, 4.0]})
    coords = coords_from_columns(t, "x", "y")
    assert coords.tolist() == [[0.0, 1.0], [3.0, 4.0]]


def test_coords_from_columns_missing() -> None:
    with pytest.raises(ValueError, match="coordinate column"):
        coords_from_columns(pa.table({"x": [1.0]}), "x", "y")


def test_coords_from_geometries_uses_centroids() -> None:
    geoms = parse_wkb_column(_wkb_table(), "geom")
    coords = coords_from_geometries(geoms)
    assert np.allclose(coords, [[0, 0], [1, 1], [2, 4]])


def test_has_and_is_binary_column() -> None:
    t = _wkb_table()
    assert has_column(t, "geom")
    assert not has_column(t, "nope")
    assert is_binary_column(t, "geom")
    assert not is_binary_column(pa.table({"geom": ["x"]}), "geom")
