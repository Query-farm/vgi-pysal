"""Composed example queries that combine several of the worker's functions.

Each example feeds one function's output into another (or joins an analysis
result back to its source dataset) — the way the functions are used in practice.
Every SQL string is defined once here and referenced from the example list of the
function that drives the query (within its first five examples) and from the
schema-level ``vgi.example_queries`` tag in ``vgi_pysal.worker``.
"""

from __future__ import annotations

import json

from vgi.metadata import FunctionExample

# weights + columbus: the spatial lag of crime computed in SQL from the edge list.
SPATIAL_LAG_SQL = (
    "WITH c AS (SELECT id, crime FROM pysal.main.columbus()) "
    "SELECT w.focal::BIGINT AS id, own.crime AS crime, "
    "round(sum(w.weight * nb.crime), 2) AS neighbour_crime, count(*) AS neighbours "
    "FROM pysal.main.weights((SELECT id, geom FROM pysal.main.columbus()), w_type => 'queen', id => 'id') AS w "
    "JOIN c AS own ON own.id = w.focal::BIGINT "
    "JOIN c AS nb ON nb.id = w.neighbor::BIGINT "
    "GROUP BY w.focal, own.crime "
    "ORDER BY neighbour_crime DESC, id "
    "LIMIT 5"
)
SPATIAL_LAG = FunctionExample(
    sql=SPATIAL_LAG_SQL,
    description=(
        "Spatial lag of columbus crime: join the row-standardised queen weights edge list to the "
        "dataset and average each neighbourhood's neighbours' crime; the five highest-lag "
        "neighbourhoods, with their own crime and neighbour count"
    ),
)

# local_moran + columbus: profile each LISA cluster type by the dataset's attributes.
LISA_PROFILE_SQL = (
    "SELECT lm.cluster, count(*) AS neighbourhoods, round(avg(c.crime), 1) AS avg_crime, "
    "round(avg(c.inc), 1) AS avg_income, round(avg(c.hoval), 1) AS avg_house_value "
    "FROM pysal.main.local_moran((SELECT id, crime AS y, geom FROM pysal.main.columbus()), "
    "w_type => 'queen', id => 'id') AS lm "
    "JOIN pysal.main.columbus() AS c USING (id) "
    "GROUP BY lm.cluster "
    "ORDER BY avg_crime DESC"
)
LISA_PROFILE = FunctionExample(
    sql=LISA_PROFILE_SQL,
    description=(
        "Profile local Moran crime clusters: join local_moran back to columbus by id and compare "
        "average crime, income, and house value per cluster type (high-high clusters are the "
        "poorer, lower-value neighbourhoods)"
    ),
)

# quantiles + columbus: classify one variable, summarise another per class.
INCOME_QUINTILE_CRIME_SQL = (
    "SELECT q.class AS income_quintile, round(max(q.upper_bound), 2) AS income_upper_bound, "
    "count(*) AS neighbourhoods, round(avg(c.crime), 1) AS avg_crime "
    "FROM pysal.main.quantiles((SELECT id, inc AS y FROM pysal.main.columbus()), "
    "value => 'y', k => 5, id => 'id') AS q "
    "JOIN pysal.main.columbus() AS c USING (id) "
    "GROUP BY q.class "
    "ORDER BY income_quintile"
)
INCOME_QUINTILE_CRIME = FunctionExample(
    sql=INCOME_QUINTILE_CRIME_SQL,
    description=(
        "Average columbus crime per household-income quintile: classify income with quantiles, "
        "join the classes back to the dataset by id, and aggregate crime per class"
    ),
)

COMPOSED_EXAMPLES: list[FunctionExample] = [SPATIAL_LAG, LISA_PROFILE, INCOME_QUINTILE_CRIME]

# Schema-level ``vgi.example_queries`` tag value.
SCHEMA_EXAMPLE_QUERIES = json.dumps([{"description": e.description, "sql": e.sql} for e in COMPOSED_EXAMPLES])

# Public agent tasks (names + prompts only); graders live in vgi-agent-tests.yaml.
AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "mexico_highest_neighbour_gdp",
            "prompt": (
                "Using queen-contiguity spatial weights between the Mexican states in the built-in "
                "mexico dataset (row-standardised, the default transform), compute each state's "
                "spatial lag of 2000 per-capita GDP (pcgdp2000) — the weighted average over its "
                "neighbours. Which state has the highest spatial lag? Return one column: the "
                "state name."
            ),
        },
        {
            "name": "columbus_hot_spot_house_value",
            "prompt": (
                "Run a Getis-Ord G_i* hot-spot analysis of crime on the built-in columbus "
                "neighbourhoods using 8-nearest-neighbour weights. How many neighbourhoods does "
                "the worker label as crime hot spots, and what is their average house value "
                "(hoval)? Return two columns: the hot-spot count (integer) and the average house "
                "value rounded to 1 decimal place."
            ),
        },
    ]
)
