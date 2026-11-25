# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "vgi-python[http,oauth]>=0.55.2,<0.56",
#     "vgi-rpc[sentry]>=0.57.1",
#     "libpysal>=4.12",
#     "esda>=2.5",
#     "spreg>=1.6",
#     "mapclassify>=2.6",
#     "inequality>=1.0",
#     "numpy",
#     "shapely>=2.0",
#     "geopandas>=1.0",
# ]
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
