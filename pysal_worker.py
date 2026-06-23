# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]",
#     "vgi-rpc[sentry]",
#     "libpysal>=4.12",
#     "esda>=2.5",
#     "spreg>=1.6",
#     "mapclassify>=2.6",
#     "inequality>=1.0",
#     "numpy",
#     "shapely>=2.0",
#     "geopandas>=1.0",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
# vgi-rpc = { path = "../vgi-rpc" }
#
# [tool.uv]
# # Use the local vgi-rpc checkout even if it lags vgi-python's pinned lower bound.
# override-dependencies = ["vgi-rpc>=0.20.3"]
# ///
"""Stdio entry shim for the PySAL VGI worker.

Lets the worker run straight from a source checkout (``uv run pysal_worker.py``)
and from the Fly.io container, and keeps ``import pysal_worker`` working for
tests. The implementation lives in ``vgi_pysal.worker``; installed users invoke
the ``vgi-pysal`` console script (which points at ``vgi_pysal.worker:main``).

    ATTACH 'pysal' (TYPE vgi, LOCATION 'uv run pysal_worker.py');
    SELECT * FROM pysal.columbus();
"""

from vgi_pysal.worker import PysalWorker, main

__all__ = ["PysalWorker", "main"]

if __name__ == "__main__":
    main()
