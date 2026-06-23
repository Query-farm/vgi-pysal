"""Unit tests for spatial weights construction and the source datasets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.harness import invoke_table_function
from vgi_pysal.datasets import ColumbusFunction
from vgi_pysal.spatial_weights import (
    build_weights,
    cardinalities,
    normalise_transform,
    normalise_w_type,
    validate_weights_args,
)

COLUMBUS = invoke_table_function(ColumbusFunction)


def _args(**over) -> SimpleNamespace:
    base = dict(
        w_type="queen", geom="geom", x="x", y="y", k=4, threshold=0.0, kernel_function="triangular", transform="r"
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_columbus_loads_with_geom() -> None:
    assert COLUMBUS.num_rows == 49
    assert "geom" in COLUMBUS.schema.names
    assert COLUMBUS.column("geom")[0].as_py()  # non-empty WKB bytes


class TestContiguity:
    def test_queen_structure(self) -> None:
        w = build_weights(COLUMBUS, _args(w_type="queen"))
        assert w.n == 49
        assert len(w.islands) == 0
        assert w.transform == "R"

    def test_rook_is_subset_of_queen(self) -> None:
        rook = build_weights(COLUMBUS, _args(w_type="rook", transform="b"))
        queen = build_weights(COLUMBUS, _args(w_type="queen", transform="b"))
        # rook adjacency (shared edges) is a subset of queen (shared edge or vertex)
        assert cardinalities(rook).sum() <= cardinalities(queen).sum()

    def test_row_standardised_weights_sum_to_one(self) -> None:
        w = build_weights(COLUMBUS, _args(w_type="queen", transform="r"))
        for i in w.id_order:
            if w.neighbors[i]:
                assert abs(sum(w.weights[i]) - 1.0) < 1e-9


class TestDistance:
    def test_knn_fixed_cardinality(self) -> None:
        w = build_weights(COLUMBUS, _args(w_type="knn", k=5, transform="b"))
        card = cardinalities(w)
        assert card.min() == 5 and card.max() == 5

    def test_distance_band_builds(self) -> None:
        w = build_weights(COLUMBUS, _args(w_type="distance_band"))
        assert w.n == 49

    def test_kernel_builds(self) -> None:
        w = build_weights(COLUMBUS, _args(w_type="kernel", k=6))
        assert w.n == 49

    def test_knn_caps_k_at_n_minus_one(self) -> None:
        w = build_weights(COLUMBUS, _args(w_type="knn", k=999, transform="b"))
        assert cardinalities(w).max() == 48


class TestValidation:
    def test_unknown_w_type(self) -> None:
        with pytest.raises(ValueError, match="unknown w_type"):
            normalise_w_type("hexagon")

    def test_unknown_transform(self) -> None:
        with pytest.raises(ValueError, match="unknown transform"):
            normalise_transform("zzz")

    def test_transform_aliases(self) -> None:
        assert normalise_transform("row") == "R"
        assert normalise_transform("binary") == "B"

    def test_validate_weights_args_rejects_bad_knn(self) -> None:
        with pytest.raises(ValueError, match="k >= 1"):
            validate_weights_args(_args(w_type="knn", k=0))
