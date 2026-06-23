"""libpysal example datasets exposed as DuckDB table functions.

A curated set of the spatial example datasets that ship *inside* libpysal (no
network download) are surfaced as zero-argument table functions. Each row is one
geographic observation: a 0-based ``id``, the dataset's attribute columns
(snake-cased), and a ``geom`` column holding the geometry as WKB bytes. Feed
``geom`` straight into the spatial functions, or read it with the DuckDB spatial
extension via ``ST_GeomFromWKB(geom)``.

    SELECT id, crime, inc, hoval FROM pysal.columbus();
    SELECT * FROM pysal.moran((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'queen');
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar

import pyarrow as pa
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi_rpc.rpc import OutputCollector

from .schema_utils import NoArgs, dedupe_names, field, snake_case


@dataclass(slots=True, frozen=True)
class _Column:
    """Plan for one output column: its name, Arrow type, and how to read it."""

    name: str
    type: pa.DataType
    source: str  # original GeoDataFrame column name, or "" for the synthetic id
    kind: str  # "id" | "int" | "float" | "bool" | "str" | "geom"


def _arrow_type(dtype: Any) -> tuple[pa.DataType, str]:
    s = str(dtype)
    if s.startswith(("int", "uint")):
        return pa.int64(), "int"
    if s.startswith("float"):
        return pa.float64(), "float"
    if s.startswith("bool"):
        return pa.bool_(), "bool"
    return pa.string(), "str"


@cache
def _load(name: str, shp: str) -> Any:
    """Load a libpysal example shapefile as a GeoDataFrame (cached)."""
    import geopandas as gpd
    from libpysal import examples

    return gpd.read_file(examples.load_example(name).get_path(shp))


def _plan(gdf: Any) -> list[_Column]:
    """Build the ordered output-column plan for a GeoDataFrame: id + attributes + geom."""
    geom_name = gdf.geometry.name
    raw_names = [c for c in gdf.columns if c != geom_name]
    out_names = dedupe_names(["id", *[snake_case(c) for c in raw_names]])
    cols = [_Column(name=out_names[0], type=pa.int64(), source="", kind="id")]
    for out, src in zip(out_names[1:], raw_names, strict=True):
        atype, kind = _arrow_type(gdf[src].dtype)
        cols.append(_Column(name=out, type=atype, source=src, kind=kind))
    cols.append(_Column(name="geom", type=pa.binary(), source=geom_name, kind="geom"))
    return cols


def _schema(gdf: Any, plan: list[_Column], description: str) -> pa.Schema:
    comments = {
        "id": "0-based observation index (stable row order; usable as a join key).",
        "geom": "Geometry as WKB bytes (read with ST_GeomFromWKB).",
    }
    fields = [
        field(c.name, c.type, comments.get(c.kind, f"{description}: column '{c.source}'."), nullable=(c.kind != "id"))
        for c in plan
    ]
    return pa.schema(fields)


def _emit(gdf: Any, plan: list[_Column], out: OutputCollector, output_schema: pa.Schema) -> None:
    import pandas as pd

    n = len(gdf)
    columns: dict[str, list[Any]] = {}
    for c in plan:
        if c.kind == "id":
            columns[c.name] = list(range(n))
        elif c.kind == "geom":
            columns[c.name] = list(gdf.geometry.to_wkb())
        else:
            series = gdf[c.source]
            if c.kind == "int":
                columns[c.name] = [None if pd.isna(v) else int(v) for v in series]
            elif c.kind == "float":
                columns[c.name] = [None if pd.isna(v) else float(v) for v in series]
            elif c.kind == "bool":
                columns[c.name] = [None if pd.isna(v) else bool(v) for v in series]
            else:
                columns[c.name] = [None if pd.isna(v) else str(v) for v in series]
    out.emit(pa.RecordBatch.from_pydict(columns, schema=output_schema))
    out.finish()


class _ExampleDataset(TableFunctionGenerator[NoArgs]):
    """Base for a fixed-schema libpysal example dataset. Subclasses set EXAMPLE + SHP."""

    EXAMPLE: ClassVar[str]
    SHP: ClassVar[str]
    PLAN: ClassVar[list[_Column]]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        n = len(_load(cls.EXAMPLE, cls.SHP))
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        _emit(_load(cls.EXAMPLE, cls.SHP), cls.PLAN, out, params.output_schema)


def _make_dataset(cls_name: str, example: str, shp: str, fn_name: str, description: str, cats: list[str]) -> type:
    """Construct a configured, fixed-schema dataset function class at import time."""
    gdf = _load(example, shp)
    plan = _plan(gdf)
    schema = _schema(gdf, plan, description)
    example_sql = FunctionExample(sql=f"SELECT * FROM pysal.{fn_name}()", description=f"Load the {fn_name} dataset")

    meta = type(
        "Meta",
        (),
        {
            "name": fn_name,
            "description": description,
            "categories": ["datasets", *cats],
            "projection_pushdown": True,
            "examples": [example_sql],
        },
    )
    cls = type(
        cls_name,
        (_ExampleDataset,),
        {
            "__doc__": description,
            "EXAMPLE": example,
            "SHP": shp,
            "PLAN": plan,
            "FIXED_SCHEMA": schema,
            "Meta": meta,
        },
    )
    return init_single_worker(bind_fixed_schema(cls))


ColumbusFunction = _make_dataset(
    "ColumbusFunction",
    "columbus",
    "columbus.shp",
    "columbus",
    "Columbus, OH neighbourhood crime, income and home value (49 polygons)",
    ["polygon", "regression"],
)
BaltimFunction = _make_dataset(
    "BaltimFunction",
    "baltim",
    "baltim.shp",
    "baltim",
    "Baltimore house sales prices and hedonic attributes (211 points)",
    ["point", "regression"],
)
SidsFunction = _make_dataset(
    "SidsFunction",
    "sids2",
    "sids2.shp",
    "sids",
    "North Carolina SIDS (sudden infant death) counts and rates by county (100 polygons)",
    ["polygon", "rates"],
)
UsIncomeFunction = _make_dataset(
    "UsIncomeFunction",
    "us_income",
    "us48.shp",
    "us_states",
    "Lower-48 US states boundaries and identifiers (48 polygons)",
    ["polygon"],
)
MexicoFunction = _make_dataset(
    "MexicoFunction",
    "mexico",
    "mexicojoin.shp",
    "mexico",
    "Mexican states regional per-capita GDP, 1940-2000 (32 polygons)",
    ["polygon", "regional"],
)
GeorgiaFunction = _make_dataset(
    "GeorgiaFunction",
    "georgia",
    "G_utm.shp",
    "georgia",
    "Georgia counties educational attainment and demographics (159 polygons)",
    ["polygon", "regression"],
)


DATASET_FUNCTIONS: list[type] = [
    ColumbusFunction,
    BaltimFunction,
    SidsFunction,
    UsIncomeFunction,
    MexicoFunction,
    GeorgiaFunction,
]
