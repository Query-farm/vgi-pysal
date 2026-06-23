"""PySAL as a VGI worker: spatial datasets, weights, autocorrelation, classification, and regression for DuckDB/SQL.

The implementation is split by spatial-analysis area so each module stays focused:

- ``datasets``         -- built-in libpysal example datasets as table functions (geometry as WKB)
- ``spatial_weights``  -- spatial weights construction (Queen/Rook/KNN/DistanceBand/Kernel) + shared ``build_weights``
- ``esda``             -- global spatial autocorrelation (Moran's I, Geary's C, Getis-Ord G) as one-row stats
- ``lisa``             -- local spatial autocorrelation (Local Moran, Getis-Ord G/G*) per observation
- ``classify``         -- mapclassify choropleth classification schemes as per-row transforms
- ``inequality``       -- inequality measures (Gini, Theil) as SQL aggregates
- ``regression``       -- spatial regression (spreg OLS/ML_Lag/ML_Error/GM_Lag) + a model registry
- ``registry``         -- pluggable fitted-model store (local disk now, S3/R2 later)

``vgi_pysal.worker`` assembles these into the ``pysal`` catalog and runs the worker; the
repo-root ``pysal_worker.py`` / ``serve.py`` are thin shims for ``uv run`` and the Fly.io container.
"""

from __future__ import annotations

__version__ = "0.1.0"
