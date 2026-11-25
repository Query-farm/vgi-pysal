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
"""HTTP entry shim for the PySAL VGI worker (used by the Fly.io container).

Forces the worker CLI into HTTP mode. The implementation lives in
``vgi_pysal.worker``; installed users invoke the ``vgi-pysal-http`` console
script (which points at ``vgi_pysal.worker:main_http``) instead.
"""

from vgi_pysal.worker import PysalWorker, main_http

__all__ = ["PysalWorker", "main_http"]


def main() -> None:
    """Run the worker over HTTP."""
    main_http()


if __name__ == "__main__":
    main()
