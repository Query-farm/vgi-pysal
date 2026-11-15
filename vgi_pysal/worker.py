"""VGI worker exposing PySAL to DuckDB/SQL.

Assembles the per-area implementation modules in ``vgi_pysal`` into a single
``pysal`` catalog and provides the process entry points. The repo-root
``pysal_worker.py`` / ``serve.py`` are thin shims over this module for ``uv run``
and the Fly.io container; installed users get the ``vgi-pysal`` and
``vgi-pysal-http`` console scripts, which call ``main`` / ``main_http`` here.

    ATTACH 'pysal' (TYPE vgi, LOCATION 'vgi-pysal');
    SELECT * FROM pysal.moran((SELECT crime AS y, geom FROM pysal.columbus()), w_type => 'queen');
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_pysal import __version__
from vgi_pysal.classify import CLASSIFY_FUNCTIONS
from vgi_pysal.datasets import DATASET_FUNCTIONS
from vgi_pysal.esda import ESDA_FUNCTIONS
from vgi_pysal.examples import AGENT_TEST_TASKS, SCHEMA_EXAMPLE_QUERIES
from vgi_pysal.inequality import INEQUALITY_FUNCTIONS
from vgi_pysal.lisa import LISA_FUNCTIONS
from vgi_pysal.regression import REGRESSION_FUNCTIONS
from vgi_pysal.spatial_weights import WEIGHTS_FUNCTIONS

log = logging.getLogger(__name__)

DATA_VERSION = __version__
GIT_COMMIT = os.environ.get("VGI_PYSAL_GIT_COMMIT") or "unknown"

# Every callable the worker exposes, grouped by spatial-analysis area.
_FUNCTIONS: list[type] = [
    *DATASET_FUNCTIONS,
    *WEIGHTS_FUNCTIONS,
    *ESDA_FUNCTIONS,
    *LISA_FUNCTIONS,
    *CLASSIFY_FUNCTIONS,
    *INEQUALITY_FUNCTIONS,
    *REGRESSION_FUNCTIONS,
]

# Long-form catalog docs (the catalog comment stays the one-line summary).
CATALOG_DOC_LLM = (
    "Run PySAL spatial statistics from SQL. Bundled example datasets (for instance columbus) return "
    "attribute columns plus a WKB geom column. Analysis functions take a subquery of rows with a numeric "
    "value column and a geom column, build spatial weights (queen, rook, k-nearest, distance band) from the "
    "geometries, and return global or local statistics: Moran's I and Geary's C autocorrelation, local "
    "LISA clusters, map classifications (quantiles, natural breaks, ...), inequality measures (Gini, Theil) "
    "and spatial regression (OLS diagnostics, spatial lag and error models). Permutation-based p-values "
    "depend on the permutation count and seed. Everything runs locally; no network access or API key is needed."
)
CATALOG_DOC_MD = (
    "# PySAL spatial analysis for SQL\n\n"
    "This worker exposes the [PySAL](https://pysal.org/) Python spatial analysis library as DuckDB table "
    "functions. Load a bundled example dataset or pass any query with a numeric value and a geometry, and "
    "the worker builds spatial weights from the geometries and returns the statistic as rows: global and "
    "local spatial autocorrelation, map classification, inequality measures and spatial regression. "
    "Permutation p-values depend on the permutation count and seed; all computation is local.\n\n"
    "## Sources and references\n\n"
    "- [PySAL](https://pysal.org/): the Python Spatial Analysis Library federation this worker wraps.\n"
    "- [libpysal](https://pysal.org/libpysal/stable/): spatial weights and the bundled example datasets.\n"
    "- [esda](https://pysal.org/esda/): Moran's I, Geary's C and local (LISA) autocorrelation statistics.\n"
    "- [spreg](https://pysal.org/spreg/), [mapclassify](https://pysal.org/mapclassify/stable/) and "
    "[inequality](https://pysal.org/inequality/): spatial regression, choropleth classification and "
    "inequality measures."
)

_PYSAL_CATALOG = Catalog(
    name="pysal",
    default_schema="main",
    comment="PySAL spatial statistics for SQL: datasets, weights, autocorrelation, classification, and regression.",
    source_url="https://github.com/Query-farm/vgi-pysal",
    tags={
        "vgi.title": "PySAL Spatial Analysis",
        "vgi.subtitle": (
            "Spatial weights, autocorrelation tests, map classification, inequality measures, "
            "and spatial regression from PySAL, callable from SQL."
        ),
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC",
        "vgi.license": "MIT",
        "vgi.doc_llm": CATALOG_DOC_LLM,
        "vgi.doc_md": CATALOG_DOC_MD,
        # Analyst tasks for `vgi-lint --agent-check`; graders live in vgi-agent-tests.yaml.
        "vgi.agent_test_tasks": AGENT_TEST_TASKS,
    },
    schemas=[
        Schema(
            path=["main"],
            comment="PySAL spatial datasets, weights, autocorrelation, classification, and regression for SQL",
            tags={"vgi.title": "PySAL Functions", "vgi.example_queries": SCHEMA_EXAMPLE_QUERIES},
            functions=list(_FUNCTIONS),
        ),
    ],
)


class PysalCatalog(ReadOnlyCatalogInterface):
    """Advertises the worker's data + implementation version on ATTACH."""

    catalog = _PYSAL_CATALOG
    catalog_name = _PYSAL_CATALOG.name

    def catalogs(self) -> list[CatalogInfo]:
        """Advertise the pysal catalog with its data and implementation versions."""
        return [
            CatalogInfo(
                name=self._effective_catalog_name,
                implementation_version=GIT_COMMIT,
                data_version_spec=DATA_VERSION,
                attach_option_specs=[spec.serialize() for spec in self.attach_option_specs],
            )
        ]

    def catalog_attach(self, **kwargs: Any) -> CatalogAttachResult:
        """Attach the catalog, stamping the resolved data and implementation versions."""
        result = super().catalog_attach(**kwargs)
        return dataclasses.replace(
            result,
            resolved_data_version=DATA_VERSION,
            resolved_implementation_version=GIT_COMMIT,
        )


class PysalWorker(Worker):
    """Worker process hosting the PySAL catalog."""

    catalog = _PYSAL_CATALOG
    catalog_interface = PysalCatalog


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    PysalWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    PysalWorker.main()
