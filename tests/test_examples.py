"""Static checks on the catalog example metadata."""

from __future__ import annotations

import json

from vgi_pysal.classify import CLASSIFY_FUNCTIONS, QuantilesFn
from vgi_pysal.examples import COMPOSED_EXAMPLES, INCOME_QUINTILE_CRIME, LISA_PROFILE, SPATIAL_LAG
from vgi_pysal.lisa import LocalMoranFn
from vgi_pysal.spatial_weights import WeightsFn
from vgi_pysal.worker import _PYSAL_CATALOG


def test_classify_examples_carry_the_id_they_select() -> None:
    """Regression: classify examples selected `id` without passing `id => 'id'` (Binder Error)."""
    for fn in CLASSIFY_FUNCTIONS:
        for ex in fn.Meta.examples:
            if ex.sql.startswith("SELECT id,"):
                assert "id => 'id'" in ex.sql, fn.Meta.name


def test_composed_examples_lead_their_driving_function() -> None:
    """Each composed example sits within the first five examples of the function that drives it."""
    for fn, ex in ((WeightsFn, SPATIAL_LAG), (LocalMoranFn, LISA_PROFILE), (QuantilesFn, INCOME_QUINTILE_CRIME)):
        assert ex in fn.Meta.examples[:5], fn.Meta.name


def test_schema_example_queries_list_every_composed_example() -> None:
    """The schema-level vgi.example_queries tag carries every composed example, each with a description."""
    tags = _PYSAL_CATALOG.schemas[0].tags
    entries = json.loads(tags["vgi.example_queries"])
    assert [e["sql"] for e in entries] == [e.sql for e in COMPOSED_EXAMPLES]
    assert all(e["description"] for e in entries)
    assert len(tags["vgi.example_queries"]) <= 8000
