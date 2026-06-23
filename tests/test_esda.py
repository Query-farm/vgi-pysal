"""Unit tests for the ESDA global and local statistic compute logic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tests.harness import invoke_table_function
from vgi_pysal.datasets import ColumbusFunction
from vgi_pysal.esda import GearyFn, GetisOrdGFn, MoranFn
from vgi_pysal.lisa import GetisOrdGLocalFn, LocalMoranFn
from vgi_pysal.spatial_weights import build_weights

COLUMBUS = invoke_table_function(ColumbusFunction)
CRIME = np.asarray(COLUMBUS.column("crime").to_numpy(zero_copy_only=False), dtype=float)


def _w(**over):
    base = dict(
        w_type="queen", geom="geom", x="x", y="y", k=8, threshold=0.0, kernel_function="triangular", transform="r"
    )
    base.update(over)
    np.random.seed(12345)
    return build_weights(COLUMBUS, SimpleNamespace(**base))


def _args(**over) -> SimpleNamespace:
    base = dict(
        w_type="queen", geom="geom", x="x", y="y", k=8, threshold=0.0, kernel_function="triangular",
        transform="r", value="crime", id="id", permutations=999, seed=12345, significance=0.05, star=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestGlobal:
    def test_moran_matches_reference(self) -> None:
        np.random.seed(12345)
        res = MoranFn.compute(CRIME, _w(), 999)
        assert res["statistic"] == 0.5001885571828611
        assert abs(res["expected"] - (-1.0 / 48)) < 1e-12
        assert res["p_value"] < 0.05

    def test_geary_expectation_is_one(self) -> None:
        res = GearyFn.compute(CRIME, _w(), 999)
        assert abs(res["expected"] - 1.0) < 1e-12
        assert 0.5 < res["statistic"] < 0.6

    def test_getis_ord_g_positive_significant(self) -> None:
        np.random.seed(12345)
        res = GetisOrdGFn.compute(CRIME, _w(w_type="knn"), 999)
        assert res["statistic"] > 0
        assert res["p_value"] < 0.05


class TestLocal:
    def test_local_moran_shape_and_labels(self) -> None:
        out = LocalMoranFn.compute(CRIME, _w(), _args())
        assert len(out["local_i"]) == 49
        assert set(out["quadrant"]) <= {1, 2, 3, 4}
        assert set(out["cluster"]) <= {"HH", "LH", "LL", "HL", "ns"}

    def test_local_moran_significance_labelling(self) -> None:
        out = LocalMoranFn.compute(CRIME, _w(), _args(significance=0.05))
        for cluster, p in zip(out["cluster"], out["p_value"], strict=True):
            assert (cluster == "ns") == (p > 0.05)

    def test_g_local_hot_cold_sign(self) -> None:
        out = GetisOrdGLocalFn.compute(CRIME, _w(w_type="knn"), _args(w_type="knn"))
        assert set(out["hotspot"]) <= {"hot", "cold", "ns"}
        for label, z in zip(out["hotspot"], out["z_score"], strict=True):
            if label == "hot":
                assert z > 0
            elif label == "cold":
                assert z < 0
