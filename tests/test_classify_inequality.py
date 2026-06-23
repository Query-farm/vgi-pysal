"""Unit tests for classification schemes and inequality aggregates."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from vgi_pysal.classify import EqualIntervalFn, QuantilesFn, StdMeanFn
from vgi_pysal.inequality import GiniFn, TheilFn

Y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])


class TestClassify:
    def test_quantiles_k_classes(self) -> None:
        scheme = QuantilesFn.classify(Y, 5)
        assert len(scheme.bins) == 5
        assert set(scheme.yb.tolist()) == {0, 1, 2, 3, 4}

    def test_equal_interval_widths(self) -> None:
        scheme = EqualIntervalFn.classify(Y, 5)
        widths = np.diff([Y.min(), *scheme.bins])
        assert np.allclose(widths, widths[0])

    def test_every_value_within_its_bound(self) -> None:
        scheme = QuantilesFn.classify(Y, 4)
        bins = np.asarray(scheme.bins, dtype=float)
        for v, b in zip(Y, scheme.yb, strict=True):
            assert v <= bins[min(int(b), len(bins) - 1)] + 1e-9

    def test_std_mean_ignores_k(self) -> None:
        scheme = StdMeanFn.classify(Y, 5)
        assert len(scheme.yb) == len(Y)


def _gini(values: list[float]) -> float | None:
    states = {0: GiniFn.initial_state(None)}
    GiniFn.update(states, pa.array([0] * len(values), pa.int64()), pa.array(values, pa.float64()))
    return GiniFn.finalize(pa.array([0], pa.int64()), states, None).column("result")[0].as_py()


def _theil(values: list[float]) -> float | None:
    states = {0: TheilFn.initial_state(None)}
    TheilFn.update(states, pa.array([0] * len(values), pa.int64()), pa.array(values, pa.float64()))
    return TheilFn.finalize(pa.array([0], pa.int64()), states, None).column("result")[0].as_py()


class TestInequality:
    def test_gini_equal_distribution_is_zero(self) -> None:
        assert abs(_gini([5.0, 5, 5, 5])) < 1e-9

    def test_gini_in_unit_range(self) -> None:
        g = _gini([1.0, 2, 3, 10, 50])
        assert 0 < g < 1

    def test_theil_equal_distribution_is_zero(self) -> None:
        assert abs(_theil([3.0, 3, 3])) < 1e-9

    def test_single_value_yields_null(self) -> None:
        assert _gini([10.0]) is None

    def test_nulls_are_skipped(self) -> None:
        states = {0: GiniFn.initial_state(None)}
        GiniFn.update(states, pa.array([0, 0, 0], pa.int64()), pa.array([5.0, None, 5.0], pa.float64()))
        # two equal values -> Gini 0 (the NULL is ignored, not treated as 0)
        assert abs(GiniFn.finalize(pa.array([0], pa.int64()), states, None).column("result")[0].as_py()) < 1e-9

    def test_per_group_state_is_independent(self) -> None:
        states = {0: GiniFn.initial_state(None), 1: GiniFn.initial_state(None)}
        GiniFn.update(
            states,
            pa.array([0, 0, 1, 1], pa.int64()),
            pa.array([5.0, 5.0, 1.0, 9.0], pa.float64()),
        )
        out = GiniFn.finalize(pa.array([0, 1], pa.int64()), states, None).column("result").to_pylist()
        assert abs(out[0]) < 1e-9  # group 0 equal
        assert out[1] > 0  # group 1 unequal
