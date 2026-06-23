"""Unit tests for the spatial-regression fitting + normalisation logic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.harness import invoke_table_function
from vgi_pysal.datasets import ColumbusFunction
from vgi_pysal.regression import _features, _fit, _to_model, normalise_model

COLUMBUS = invoke_table_function(ColumbusFunction)


def _args(**over) -> SimpleNamespace:
    base = dict(
        model="ols", target="crime", id="id", w_type="queen", geom="geom", x="x", y="y",
        k=4, threshold=0.0, kernel_function="triangular", transform="r",
    )
    base.update(over)
    return SimpleNamespace(**base)


# Restrict to a small, well-known design: crime ~ inc + hoval.
SUBSET = COLUMBUS.select(["id", "crime", "inc", "hoval", "geom"])


def test_features_excludes_target_id_and_spatial_columns() -> None:
    feats = _features(SUBSET, _args())
    assert feats == ["inc", "hoval"]


class TestFit:
    def test_ols_constant_and_slopes(self) -> None:
        fitted = _fit(SUBSET, _args(model="ols"))
        assert fitted.param_names == ["CONSTANT", "inc", "hoval"]
        const = fitted.betas[fitted.param_names.index("CONSTANT")]
        assert abs(const - 68.6189) < 1e-3
        assert all(fitted.betas[fitted.param_names.index(f)] < 0 for f in ("inc", "hoval"))

    def test_ml_lag_adds_spatial_term(self) -> None:
        fitted = _fit(SUBSET, _args(model="ml_lag"))
        assert any(n.startswith("W_") for n in fitted.param_names)
        assert len(fitted.betas) == 4

    def test_ml_error_has_lambda(self) -> None:
        fitted = _fit(SUBSET, _args(model="ml_error"))
        assert "lambda" in fitted.param_names

    def test_unknown_model(self) -> None:
        with pytest.raises(ValueError, match="unknown model"):
            normalise_model("wizardry")

    def test_no_explanatory_columns(self) -> None:
        only_target = COLUMBUS.select(["id", "crime", "geom"])
        with pytest.raises(ValueError, match="at least one explanatory"):
            _fit(only_target, _args(model="ols"))


class TestToModel:
    def test_ols_linear_predictor_matches_spreg(self) -> None:
        import spreg

        from vgi_pysal.spatial_weights import build_weights

        feats = ["inc", "hoval"]
        x = np.column_stack([SUBSET.column(f).to_numpy(zero_copy_only=False) for f in feats]).astype(float)
        y = np.asarray(SUBSET.column("crime").to_numpy(zero_copy_only=False), dtype=float).reshape(-1, 1)
        w = build_weights(SUBSET, _args())
        m = spreg.OLS(y, x, w=w, spat_diag=True, moran=True, name_y="crime", name_x=feats)
        fitted = _fit(SUBSET, _args(model="ols"))
        model = _to_model(fitted, m, "t")
        # reconstruct the linear predictor and compare to spreg's fitted values
        pred = (x @ np.asarray(model.coefficients)) + model.intercept
        assert np.allclose(pred, m.predy.flatten(), atol=1e-6)

    def test_spatial_coef_recorded_for_lag(self) -> None:
        import spreg

        from vgi_pysal.spatial_weights import build_weights

        feats = ["inc", "hoval"]
        x = np.column_stack([SUBSET.column(f).to_numpy(zero_copy_only=False) for f in feats]).astype(float)
        y = np.asarray(SUBSET.column("crime").to_numpy(zero_copy_only=False), dtype=float).reshape(-1, 1)
        w = build_weights(SUBSET, _args())
        m = spreg.ML_Lag(y, x, w=w, name_y="crime", name_x=feats)
        model = _to_model(_fit(SUBSET, _args(model="ml_lag")), m, "t")
        assert model.spatial_coef is not None
        assert model.spatial_coef_name.startswith("W_")
        assert len(model.coefficients) == 2  # spatial term excluded from the linear predictor
