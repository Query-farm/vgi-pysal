# PySAL VGI worker — dev, test, and deploy targets.
#
# Usage:
#   make venv         # local venv against the ~/Development/vgi-{python,rpc} checkouts
#   make pytest       # unit tests
#   make test         # pytest + SQL (stdio/http)
#   make test-stdio   # SQL tests with the worker as a subprocess
#   make test-http    # start a local HTTP server, run SQL tests, stop it
#   make test-cloud   # SQL tests against the deployed Fly.io service
#   make deploy       # build locally, smoke-test, push, deploy to Fly.io

VGI_PYTHON_SRC ?= $(HOME)/Development/vgi-python
VGI_RPC_SRC    ?= $(HOME)/Development/vgi-rpc

VGI_BUILD_DIR  ?= $(HOME)/Development/vgi/build/release
TEST_RUNNER     = $(VGI_BUILD_DIR)/test/unittest
TEST_DIR        = .
TEST_PATTERN    = test/sql/*

# Worker paths (overridable)
WORKER_STDIO   ?= uv run --python 3.13 pysal_worker.py
WORKER_HTTP    ?= http://localhost:8000
WORKER_CLOUD   ?= https://$(FLY_APP).fly.dev
HTTP_PORT      ?= 8000

# Fly.io config
FLY_APP        ?= vgi-pysal

# Isolated model registry for local SQL tests (stdio/http workers inherit this).
TEST_MODELS_DIR ?= $(CURDIR)/.test-models

.PHONY: test pytest lint typecheck check test-stdio test-http test-cloud smoke-test deploy venv

venv:
	uv venv --python 3.13
	uv pip install --python .venv \
		"vgi-python[http,oauth] @ $(VGI_PYTHON_SRC)" \
		"vgi-rpc[sentry] @ $(VGI_RPC_SRC)" \
		"libpysal>=4.12" "esda>=2.5" "spreg>=1.6" "mapclassify>=2.6" \
		"inequality>=1.0" "shapely>=2.0" "geopandas>=1.0" numpy pytest mypy pydoclint

# Static gates (config in pyproject.toml, adopted from vgi-python): ruff lint +
# format, strict mypy, and the pydoclint docstring gate.
PKG_SOURCES = vgi_pysal/ pysal_worker.py serve.py

lint:
	uvx ruff check .
	uvx ruff format --check .

typecheck:
	.venv/bin/mypy $(PKG_SOURCES)
	.venv/bin/pydoclint $(PKG_SOURCES)

check: lint typecheck pytest

pytest:
	.venv/bin/pytest tests/ --rootdir=. -o "addopts=" -q

test: check test-stdio test-http

test-stdio:
	rm -rf "$(TEST_MODELS_DIR)"
	PYSAL_MODELS_DIR="$(TEST_MODELS_DIR)" VGI_PYSAL_WORKER="$(WORKER_STDIO)" \
		$(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

test-http:
	@if lsof -iTCP:$(HTTP_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "ERROR: port $(HTTP_PORT) is already in use" >&2; \
		echo "  Kill the existing process: kill $$(lsof -iTCP:$(HTTP_PORT) -sTCP:LISTEN -t)" >&2; \
		exit 1; \
	fi
	@rm -rf "$(TEST_MODELS_DIR)"
	@PYSAL_MODELS_DIR="$(TEST_MODELS_DIR)" VGI_SIGNING_KEY=dev .venv/bin/python serve.py --port $(HTTP_PORT) & \
		SERVER_PID=$$!; \
		for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
			curl -fsS -o /dev/null "http://localhost:$(HTTP_PORT)/health" 2>/dev/null && break; \
			sleep 1; \
		done; \
		VGI_PYSAL_WORKER="$(WORKER_HTTP)" $(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"; \
		TEST_EXIT=$$?; \
		kill $$SERVER_PID 2>/dev/null; \
		wait $$SERVER_PID 2>/dev/null; \
		exit $$TEST_EXIT

test-cloud:
	VGI_PYSAL_WORKER="$(WORKER_CLOUD)" $(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

# ---------------------------------------------------------------------------
# Container image. CI (docker-publish.yml) builds, tests, and pushes the signed
# multi-arch image to ghcr.io; these targets are for local builds and probing
# the same image. The single image serves both transports (HTTP default, `stdio`).
# ---------------------------------------------------------------------------

GIT_COMMIT     := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
# Single source of truth: the __version__ literal the package advertises over VGI.
VERSION        := $(shell sed -nE 's/^__version__ = "([^"]+)".*/\1/p' vgi_pysal/__init__.py)

GHCR_IMAGE     ?= ghcr.io/query-farm/vgi-pysal
# Tag Fly.io pulls. Defaults to the released version; override for edge/sha tags.
TAG            ?= $(VERSION)

# Locally-built image for the image test targets below.
DOCKER_IMAGE     ?= vgi-pysal:dev
DOCKER_STATE_VOL ?= vgi_pysal_state_test

.PHONY: image test-docker-stdio test-docker-http

image:
	docker build --build-arg VERSION=$(VERSION) --build-arg GIT_COMMIT=$(GIT_COMMIT) \
		-t $(DOCKER_IMAGE) .

# Run the authoritative SQL suite against the built image, stdio transport: the
# extension spawns the container per ATTACH, sharing a throwaway named volume.
test-docker-stdio: image
	-docker volume rm $(DOCKER_STATE_VOL) >/dev/null 2>&1
	VGI_PYSAL_WORKER="docker run -i --rm -v $(DOCKER_STATE_VOL):/data $(DOCKER_IMAGE) stdio" \
		$(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"
	-docker volume rm $(DOCKER_STATE_VOL) >/dev/null 2>&1

# Same suite against the built image over HTTP: start one container, run, stop.
test-docker-http: image
	-docker volume rm $(DOCKER_STATE_VOL) >/dev/null 2>&1
	@CID=$$(docker run -d -p $(HTTP_PORT):8000 -v $(DOCKER_STATE_VOL):/data \
			-e VGI_SIGNING_KEY=dev $(DOCKER_IMAGE)); \
		trap "docker rm -f $$CID >/dev/null 2>&1; docker volume rm $(DOCKER_STATE_VOL) >/dev/null 2>&1" EXIT; \
		for i in $$(seq 1 30); do \
			curl -fsS -o /dev/null "http://localhost:$(HTTP_PORT)/health" 2>/dev/null && break; \
			sleep 1; \
		done; \
		VGI_PYSAL_WORKER="$(WORKER_HTTP)" $(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

# Quick local probe of the built image's HTTP mode (CI does the full image suite).
smoke-test: image
	@CID=$$(docker run -d -e VGI_SIGNING_KEY=dev -p 18000:8000 $(DOCKER_IMAGE)); \
		trap "docker rm -f $$CID >/dev/null 2>&1" EXIT; \
		for i in 1 2 3 4 5 6 7 8 9 10; do \
			if curl -fsS -o /dev/null http://localhost:18000/health 2>/dev/null; then \
				echo "HTTP server responding"; exit 0; \
			fi; \
			sleep 1; \
		done; \
		echo "ERROR: container did not respond on /health within 10s" >&2; \
		docker logs $$CID >&2; \
		exit 1

# Deploy the published ghcr image to Fly.io (CI built + signed it on release).
deploy:
	fly deploy --image $(GHCR_IMAGE):$(TAG) --app $(FLY_APP)
