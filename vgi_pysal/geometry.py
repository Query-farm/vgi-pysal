"""Turn DuckDB spatial input into the geometries / coordinates PySAL needs.

The worker accepts spatial structure two ways, both as ordinary table columns so
nothing requires the DuckDB ``spatial`` extension on the *input* side beyond
producing WKB:

* **A WKB geometry column** (default name ``geom``) -- bytes produced by
  ``ST_AsWKB(geom)``. Parsed with shapely; used for contiguity weights
  (Queen/Rook) and, via centroids, for distance weights too.
* **Coordinate columns** (default names ``x`` and ``y``) -- numeric longitude /
  latitude (or projected easting / northing). Used for distance weights
  (KNN/DistanceBand/Kernel).

This module isolates the parsing so the weights/ESDA/regression layers never
touch WKB or geopandas directly.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import shapely


def has_column(table: pa.Table, name: str) -> bool:
    """Report whether the table has a column with the given name."""
    return name in table.schema.names


def is_binary_column(table: pa.Table, name: str) -> bool:
    """Report whether the named column exists and holds binary (WKB-capable) bytes."""
    if name not in table.schema.names:
        return False
    t = table.schema.field(name).type
    return bool(pa.types.is_binary(t) or pa.types.is_large_binary(t))


def parse_wkb_column(table: pa.Table, name: str) -> list[shapely.Geometry]:
    """Parse a WKB (BLOB) column into a list of shapely geometries.

    Raises a clear error if the column is absent, not binary, or contains a NULL
    or unparseable value (every observation must have a geometry to be wired into
    a spatial weights matrix).
    """
    if name not in table.schema.names:
        raise ValueError(
            f"geometry column {name!r} not found; provide a WKB column (e.g. ST_AsWKB(geom) AS {name}) "
            f"or x/y coordinate columns. input columns: {', '.join(table.schema.names)}"
        )
    if not is_binary_column(table, name):
        raise ValueError(
            f"geometry column {name!r} must be WKB BLOB bytes (use ST_AsWKB(geom)); got {table.schema.field(name).type}"
        )
    raw = table.column(name).to_pylist()
    if any(v is None for v in raw):
        raise ValueError(f"geometry column {name!r} contains NULL geometries; every row must have a geometry")
    try:
        geoms = shapely.from_wkb(raw)
    except Exception as exc:  # pragma: no cover - shapely error surface
        raise ValueError(f"could not parse WKB geometry column {name!r}: {exc}") from exc
    return list(geoms)


def coords_from_columns(table: pa.Table, x: str, y: str) -> np.ndarray:
    """Stack two numeric coordinate columns into an ``(n, 2)`` float array."""
    missing = [c for c in (x, y) if c not in table.schema.names]
    if missing:
        raise ValueError(
            f"coordinate column(s) {', '.join(missing)} not found; input columns: {', '.join(table.schema.names)}"
        )
    xs = np.asarray(table.column(x).to_numpy(zero_copy_only=False), dtype=float)
    ys = np.asarray(table.column(y).to_numpy(zero_copy_only=False), dtype=float)
    return np.column_stack([xs, ys])


def coords_from_geometries(geoms: list[shapely.Geometry]) -> np.ndarray:
    """Representative point (centroid) coordinates for a list of geometries."""
    pts = shapely.centroid(np.asarray(geoms, dtype=object))
    return np.column_stack([shapely.get_x(pts), shapely.get_y(pts)])
