"""Unit tests for the fitted-model registry and BLOB pack/unpack."""

from __future__ import annotations

import pytest

from vgi_pysal.registry import (
    LocalDiskStore,
    ModelNameError,
    ModelNotFoundError,
    SpatialModel,
    pack_model,
    unpack_model,
    validate_name,
)


def _model(name: str = "m") -> SpatialModel:
    return SpatialModel(
        name=name,
        model_type="ml_lag",
        target="crime",
        feature_names=["inc", "hoval"],
        coefficients=[-1.05, -0.27],
        has_constant=True,
        intercept=45.6,
        spatial_coef=0.42,
        spatial_coef_name="W_crime",
        n=49,
        k=4,
        pseudo_r2=0.65,
    )


class TestBlob:
    def test_pack_unpack_roundtrips(self) -> None:
        m = _model()
        back = unpack_model(pack_model(m))
        assert back.name == m.name
        assert back.coefficients == m.coefficients
        assert back.spatial_coef_name == "W_crime"

    def test_truncated_blob_errors(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            unpack_model(b"\x00\x00")


class TestStore:
    def test_save_load_list_delete(self, tmp_path) -> None:
        store = LocalDiskStore(tmp_path)
        store.save(_model("a"))
        store.save(_model("b"))
        assert store.exists("a")
        assert {m.name for m in store.list()} == {"a", "b"}
        loaded = store.load("a")
        assert loaded.target == "crime"
        assert store.delete("a") is True
        assert store.delete("a") is False
        assert {m.name for m in store.list()} == {"b"}

    def test_load_missing_raises(self, tmp_path) -> None:
        with pytest.raises(ModelNotFoundError):
            LocalDiskStore(tmp_path).load("ghost")

    def test_empty_store_lists_nothing(self, tmp_path) -> None:
        assert LocalDiskStore(tmp_path / "nope").list() == []


class TestNameValidation:
    @pytest.mark.parametrize("bad", ["", "../escape", "a/b", "..", "-leading"])
    def test_rejects_unsafe_names(self, bad: str) -> None:
        with pytest.raises(ModelNameError):
            validate_name(bad)

    @pytest.mark.parametrize("good", ["model1", "a_b-c.2", "Columbus"])
    def test_accepts_safe_names(self, good: str) -> None:
        assert validate_name(good) == good
