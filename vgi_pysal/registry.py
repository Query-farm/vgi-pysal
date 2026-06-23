"""Fitted-model registry for spatial regression behind a swappable storage backend.

A fitted spreg model reduces, for prediction purposes, to its *systematic
(linear) predictor*: an intercept plus a coefficient per explanatory variable.
That is fully JSON-serializable (just names and floats), so -- unlike the
scikit-learn worker, which must serialize live estimator objects -- there is no
pickle/skops trust problem here. Each model is a single ``<name>.json`` artifact.

The ``ModelStore`` interface is the seam where an S3/R2 backend drops in later
(selected by ``get_store()``) without touching ``regression.py``.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ModelNameError(ValueError):
    """Raised for model names that are empty or unsafe as a filename."""


class ModelNotFoundError(KeyError):
    """Raised when a requested model is not in the registry."""


def validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name) or "/" in name or ".." in name:
        raise ModelNameError(
            f"invalid model name {name!r}: use letters, digits, '_', '-', '.' and do not start with a separator"
        )
    return name


@dataclass(kw_only=True)
class SpatialModel:
    """A fitted spatial-regression model: everything needed to describe and predict it.

    Prediction uses the linear (systematic) predictor only -- ``intercept`` plus
    ``coefficients`` dotted with the aligned explanatory columns -- which is exact
    for OLS and is the trend component for spatial lag/error models (the spatial
    feedback term is reported separately in ``spatial_coef`` but not replayed).
    """

    name: str
    model_type: str
    target: str
    feature_names: list[str]
    coefficients: list[float]
    has_constant: bool = True
    intercept: float = 0.0
    spatial_coef: float | None = None
    spatial_coef_name: str | None = None
    n: int = 0
    k: int = 0
    r2: float | None = None
    pseudo_r2: float | None = None
    log_likelihood: float | None = None
    aic: float | None = None
    sigma2: float | None = None
    pysal_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpatialModel:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})


class ModelStore:
    """Abstract model store. Implementations persist a ``SpatialModel`` by name."""

    def save(self, model: SpatialModel) -> None:
        raise NotImplementedError

    def load(self, name: str) -> SpatialModel:
        raise NotImplementedError

    def list(self) -> list[SpatialModel]:
        raise NotImplementedError

    def delete(self, name: str) -> bool:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        raise NotImplementedError


class LocalDiskStore(ModelStore):
    """Stores each model as ``<root>/<name>.json``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        validate_name(name)
        return self.root / f"{name}.json"

    def save(self, model: SpatialModel) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(model.name).write_text(json.dumps(model.to_dict(), indent=2, default=str))

    def load(self, name: str) -> SpatialModel:
        path = self._path(name)
        if not path.exists():
            raise ModelNotFoundError(name)
        return SpatialModel.from_dict(json.loads(path.read_text()))

    def list(self) -> list[SpatialModel]:
        if not self.root.exists():
            return []
        out: list[SpatialModel] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                out.append(SpatialModel.from_dict(json.loads(path.read_text())))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def delete(self, name: str) -> bool:
        path = self._path(name)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed

    def exists(self, name: str) -> bool:
        return self._path(name).exists()


_store: ModelStore | None = None


def get_store() -> ModelStore:
    """Return the process-wide model store, configured from the environment.

    ``PYSAL_MODELS_DIR`` selects the local-disk root (default ``./models``). A
    future S3/R2 backend would be selected here behind the same interface.
    """
    global _store
    if _store is None:
        root = os.environ.get("PYSAL_MODELS_DIR", "models")
        _store = LocalDiskStore(root)
    return _store


def set_store(store: ModelStore | None) -> None:
    """Override the process-wide store (used by tests)."""
    global _store
    _store = store


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Self-contained model BLOB (a model in one value): 4-byte length || JSON.
# Lets a fitted model flow through SQL as a single BLOB column and live inside a
# DuckDB table instead of (or alongside) the on-disk registry.
# ---------------------------------------------------------------------------


def pack_model(model: SpatialModel) -> bytes:
    """Serialize a model into one self-describing BLOB."""
    payload = json.dumps(model.to_dict(), default=str).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def unpack_model(blob: bytes) -> SpatialModel:
    """Read a model back from a BLOB produced by ``pack_model``."""
    if len(blob) < 4:
        raise ValueError("not a valid pysal model BLOB (too short)")
    (n,) = struct.unpack(">I", blob[:4])
    if len(blob) < 4 + n:
        raise ValueError("not a valid pysal model BLOB (truncated)")
    return SpatialModel.from_dict(json.loads(blob[4 : 4 + n]))
