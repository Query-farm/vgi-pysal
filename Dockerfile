FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# geopandas/shapely/pyproj ship manylinux wheels; no system GEOS/GDAL needed.
# Install the framework (from PyPI) and the PySAL stack. The worker uses only
# stable vgi-python features, so no local checkout / vendoring is required.
RUN pip install --no-cache-dir \
        "vgi-python[http,oauth]>=0.8.1" \
        authlib \
        "libpysal>=4.12" "esda>=2.5" "spreg>=1.6" "mapclassify>=2.6" \
        "inequality>=1.0" "shapely>=2.0" "geopandas>=1.0" numpy \
    && pip uninstall -y pip

COPY vgi_pysal /app/vgi_pysal
COPY pysal_worker.py /app/pysal_worker.py
COPY serve.py /app/serve.py

ARG GIT_COMMIT=unknown
ENV VGI_PYSAL_GIT_COMMIT=${GIT_COMMIT}
ENV SENTRY_RELEASE=${GIT_COMMIT}

# Where the local-disk model registry persists (mount a Fly volume here in prod).
ENV PYSAL_MODELS_DIR=/data/models

# Cache the bundled libpysal example datasets into the image (warms the loaders
# and confirms the imports resolve at build time).
RUN python -c "import vgi_pysal.worker as w; print('functions:', len(w._FUNCTIONS))"

EXPOSE 8000
CMD ["sh", "-c", "python /app/serve.py --host 0.0.0.0 --port ${PORT:-8000}"]
