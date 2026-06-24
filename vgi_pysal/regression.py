"""Spatial regression (PySAL spreg) as DuckDB table functions + a model registry.

The convention mirrors the rest of the worker: the input is a table whose
``target`` column is the dependent variable, whose ``geom`` (or ``x``/``y``)
columns define the spatial weights, and whose remaining numeric columns are the
explanatory variables. Four estimators are exposed via the ``model`` argument:

* ``ols``       -- ordinary least squares (baseline, with a residual Moran's I diagnostic)
* ``ml_lag``    -- maximum-likelihood spatial lag (spatially lagged dependent variable)
* ``ml_error``  -- maximum-likelihood spatial error (spatially correlated errors)
* ``gm_lag``    -- GMM/IV spatial lag

Functions:

* ``spreg``        -- the coefficient table (one row per variable + the spatial term)
* ``fit``          -- one-row fit summary + a self-contained model BLOB; persists to the registry
* ``predict``      -- per-observation linear prediction from a stored / BLOB model
* ``list_models`` / ``model_info`` / ``drop_model`` -- registry management

    SELECT * FROM pysal.spreg((SELECT crime AS target, inc, hoval, geom FROM pysal.columbus()),
                              model => 'ml_lag', target => 'target', w_type => 'queen');
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import libpysal
import numpy as np
import pyarrow as pa
import spreg
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi.table_in_out_function import TableInOutGenerator
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, emit_empty, input_schema_of, matrix, numeric_column
from .registry import (
    ModelNotFoundError,
    SpatialModel,
    get_store,
    now_iso,
    pack_model,
    unpack_model,
    validate_name,
)
from .schema_utils import NoArgs
from .schema_utils import field as sfield
from .spatial_weights import WeightsArgs, build_weights, validate_weights_args

# model name -> (spreg class, needs_weights, builds spatial term)
_MODELS: dict[str, type] = {
    "ols": spreg.OLS,
    "ml_lag": spreg.ML_Lag,
    "ml_error": spreg.ML_Error,
    "gm_lag": spreg.GM_Lag,
}


def normalise_model(model: str) -> str:
    """Normalise an estimator name to its canonical key, raising for unknown models."""
    m = (model or "").strip().lower()
    if m not in _MODELS:
        raise ValueError(f"unknown model {model!r}; choose one of: {', '.join(sorted(_MODELS))}")
    return m


def _is_spatial_name(name: str) -> bool:
    """Return whether a parameter name denotes a spatial autoregressive term (rho / lambda)."""
    return name.startswith("W_") or name in ("lambda", "rho")


def _features(table_or_schema: Any, args: Any) -> list[str]:
    """Numeric explanatory columns: everything except the target, id, and spatial columns."""
    schema = table_or_schema.schema if isinstance(table_or_schema, pa.Table) else table_or_schema
    reserved = {args.target, args.id, args.geom, args.x, args.y}
    feats: list[str] = []
    for name in schema.names:
        if name in reserved:
            continue
        t = schema.field(name).type
        if pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_boolean(t):
            feats.append(name)
    return feats


def _scalar(value: Any) -> float | None:
    """Coerce a possibly 0-d / 1-element numpy value to a python float."""
    if value is None:
        return None
    arr = np.ravel(np.asarray(value, dtype=float))
    return float(arr[0]) if arr.size else None


@dataclass(slots=True, frozen=True)
class _Fitted:
    """Normalised view of a fitted spreg model (estimator-agnostic)."""

    model_type: str
    target: str
    feature_names: list[str]
    param_names: list[str]
    betas: np.ndarray
    std_err: np.ndarray
    stat: list[tuple[float, float]]
    predy: np.ndarray
    n: int
    k: int


def _fit(table: pa.Table, args: Any) -> _Fitted:
    """Fit the requested estimator and normalise its outputs."""
    model_type = normalise_model(args.model)
    feats = _features(table, args)
    if not feats:
        raise ValueError("spatial regression needs at least one explanatory column besides the target")
    y = numeric_column(table, args.target, what="target").reshape(-1, 1)
    x = matrix(table, feats, what="explanatory")
    w = build_weights(table, args)

    cls = _MODELS[model_type]
    if model_type == "ols":
        m = cls(y, x, w=w, spat_diag=True, moran=True, name_y=args.target, name_x=feats)
    else:
        m = cls(y, x, w=w, name_y=args.target, name_x=feats)

    betas = m.betas.flatten()
    names = _param_names(m, betas)
    stat = getattr(m, "z_stat", None) or getattr(m, "t_stat", None) or []
    return _Fitted(
        model_type=model_type,
        target=args.target,
        feature_names=feats,
        param_names=names,
        betas=betas,
        std_err=np.ravel(np.asarray(m.std_err, dtype=float)),
        stat=[(float(s), float(p)) for s, p in stat],
        predy=np.ravel(np.asarray(m.predy, dtype=float)),
        n=int(m.n),
        k=int(m.k),
    )


def _param_names(model: Any, betas: np.ndarray) -> list[str]:
    """The parameter names aligned with ``betas`` (handles the spatial term)."""
    for attr in ("name_z", "name_x"):
        names = getattr(model, attr, None)
        if names is not None and len(names) == len(betas):
            return [str(n) for n in names]
    return [f"beta_{i}" for i in range(len(betas))]


def _to_model(fitted: _Fitted, model_obj: Any, name: str) -> SpatialModel:
    """Build a persistable ``SpatialModel`` (linear predictor + diagnostics) from a fit."""
    names = fitted.param_names
    betas = fitted.betas
    has_constant = "CONSTANT" in names
    intercept = float(betas[names.index("CONSTANT")]) if has_constant else 0.0
    coefficients = [float(betas[names.index(f)]) for f in fitted.feature_names]
    spatial_coef = None
    spatial_name = None
    for i, nm in enumerate(names):
        if _is_spatial_name(nm):
            spatial_coef = float(betas[i])
            spatial_name = nm
            break
    return SpatialModel(
        name=name,
        model_type=fitted.model_type,
        target=fitted.target,
        feature_names=fitted.feature_names,
        coefficients=coefficients,
        has_constant=has_constant,
        intercept=intercept,
        spatial_coef=spatial_coef,
        spatial_coef_name=spatial_name,
        n=fitted.n,
        k=fitted.k,
        r2=_scalar(getattr(model_obj, "r2", None)),
        pseudo_r2=_scalar(getattr(model_obj, "pr2", None)),
        log_likelihood=_scalar(getattr(model_obj, "logll", None)),
        aic=_scalar(getattr(model_obj, "aic", None)),
        sigma2=_scalar(getattr(model_obj, "sig2", None)),
        pysal_version=libpysal.__version__,
        created_at=now_iso(),
    )


@dataclass(slots=True, frozen=True)
class RegArgs(WeightsArgs):
    """Arguments shared by the spatial-regression table functions."""

    model: Annotated[str, Arg("model", default="ols", doc="Estimator: ols, ml_lag, ml_error, or gm_lag.")]
    target: Annotated[str, Arg("target", default="", doc="Dependent-variable column name (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column excluded from the explanatory variables.")]


def _validate_reg_bind(params: BindParams[Any]) -> None:
    """Validate the model, weights, and target arguments against the input schema at bind time."""
    a = params.args
    normalise_model(a.model)
    validate_weights_args(a)
    if not a.target:
        raise ValueError("spatial regression requires 'target' (the dependent-variable column, e.g. target := 'price')")
    input_schema = params.bind_call.input_schema
    assert input_schema is not None
    if a.target not in input_schema.names:
        raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")


# ===========================================================================
# spreg() -- coefficient table
# ===========================================================================


_COEF_SCHEMA = pa.schema(
    [
        sfield(
            "variable",
            pa.string(),
            "Parameter name (CONSTANT, an explanatory variable, or the spatial term).",
            nullable=False,
        ),
        sfield("coefficient", pa.float64(), "Estimated coefficient.", nullable=False),
        sfield("std_err", pa.float64(), "Standard error of the estimate."),
        sfield("statistic", pa.float64(), "z- or t-statistic (estimate / std error)."),
        sfield("p_value", pa.float64(), "Two-sided significance of the statistic."),
        sfield("is_spatial", pa.bool_(), "True for the spatial autoregressive term (rho / lambda).", nullable=False),
    ]
)


class SpregFn(SinkBuffer[RegArgs, DrainState]):
    """Fit a spatial regression and return its coefficient table."""

    FunctionArguments: ClassVar[type] = RegArgs

    class Meta:
        """Catalog metadata for the spreg function."""

        name = "spreg"
        description = "Fit a spatial regression (OLS/ML_Lag/ML_Error/GM_Lag) and return the coefficient table"
        categories = ["regression", "spreg", "spatial"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM pysal.spreg("
                    "(SELECT crime AS target, inc, hoval, geom FROM pysal.columbus()), "
                    "model => 'ml_lag', target => 'target', w_type => 'queen')"
                ),
                description="Maximum-likelihood spatial lag model of columbus crime",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[RegArgs]) -> BindResponse:
        """Validate the regression arguments and return the coefficient-table output schema."""
        _validate_reg_bind(params)
        return BindResponse(output_schema=_COEF_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[RegArgs]) -> DrainState:
        """Create the initial finalize-time drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[RegArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Fit the model on the buffered rows and emit one coefficient row per parameter."""
        if state.done:
            out.finish()
            return
        state.done = True

        table = cls.buffered_table(params, input_schema_of(params))
        if table is None or table.num_rows == 0:
            emit_empty(out, params.output_schema)
            return

        fitted = _fit(table, params.args)
        names = fitted.param_names
        se = fitted.std_err
        stat = fitted.stat
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "variable": list(names),
                    "coefficient": [float(b) for b in fitted.betas],
                    "std_err": [float(se[i]) if i < len(se) else None for i in range(len(names))],
                    "statistic": [stat[i][0] if i < len(stat) else None for i in range(len(names))],
                    "p_value": [stat[i][1] if i < len(stat) else None for i in range(len(names))],
                    "is_spatial": [_is_spatial_name(n) for n in names],
                },
                schema=params.output_schema,
            )
        )


# ===========================================================================
# fit() -- one-row summary + model BLOB, persisted to the registry
# ===========================================================================


@dataclass(slots=True, frozen=True)
class FitArgs(RegArgs):
    """Arguments for the fit function, adding the optional registry name."""

    model_name: Annotated[str, Arg("model_name", default="", doc="Name to store the fitted model under (optional).")]


_FIT_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Name the model was stored under ('' if not persisted).", nullable=False),
        sfield("model_type", pa.string(), "Estimator used.", nullable=False),
        sfield("target", pa.string(), "Dependent variable.", nullable=False),
        sfield("n", pa.int64(), "Number of observations.", nullable=False),
        sfield("k", pa.int64(), "Number of estimated parameters.", nullable=False),
        sfield("r2", pa.float64(), "R-squared (OLS)."),
        sfield("pseudo_r2", pa.float64(), "Pseudo R-squared (spatial models)."),
        sfield("log_likelihood", pa.float64(), "Log-likelihood (ML models)."),
        sfield("aic", pa.float64(), "Akaike information criterion."),
        sfield("spatial_coef", pa.float64(), "Spatial autoregressive coefficient (rho / lambda; NULL for OLS)."),
        sfield("features", pa.list_(pa.string()), "Ordered explanatory-variable names.", nullable=False),
        sfield("model", pa.binary(), "The fitted model as a self-contained BLOB.", nullable=False),
    ]
)


class FitModel(SinkBuffer[FitArgs, DrainState]):
    """Fit a spatial regression, persist it (if named), and return a one-row summary + BLOB."""

    FunctionArguments: ClassVar[type] = FitArgs

    class Meta:
        """Catalog metadata for the fit function."""

        name = "fit"
        description = "Fit a spatial regression model, store it in the registry, and return a summary + model BLOB"
        categories = ["regression", "spreg", "models"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT model_name, model_type, pseudo_r2 FROM pysal.fit("
                    "(SELECT crime AS target, inc, hoval, geom FROM pysal.columbus()), "
                    "model => 'ml_error', target => 'target', model_name => 'columbus_err')"
                ),
                description="Fit a spatial error model and store it as 'columbus_err'",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FitArgs]) -> BindResponse:
        """Validate the model name and regression arguments and return the fit-summary output schema."""
        if params.args.model_name:
            validate_name(params.args.model_name)
        _validate_reg_bind(params)
        return BindResponse(output_schema=_FIT_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[FitArgs]) -> DrainState:
        """Create the initial finalize-time drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FitArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Fit and normalise the model, persist it when named, and emit a one-row summary plus a model BLOB."""
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        table = cls.buffered_table(params, input_schema_of(params))
        if table is None or table.num_rows == 0:
            raise ValueError("fit received no training rows")

        # Refit to capture the live model object's diagnostics, then normalise.
        feats = _features(table, a)
        if not feats:
            raise ValueError("spatial regression needs at least one explanatory column besides the target")
        y = numeric_column(table, a.target, what="target").reshape(-1, 1)
        x = matrix(table, feats, what="explanatory")
        w = build_weights(table, a)
        model_type = normalise_model(a.model)
        if model_type == "ols":
            model_obj = _MODELS[model_type](y, x, w=w, spat_diag=True, moran=True, name_y=a.target, name_x=feats)
        else:
            model_obj = _MODELS[model_type](y, x, w=w, name_y=a.target, name_x=feats)

        betas = model_obj.betas.flatten()
        fitted = _Fitted(
            model_type=model_type,
            target=a.target,
            feature_names=feats,
            param_names=_param_names(model_obj, betas),
            betas=betas,
            std_err=np.ravel(np.asarray(model_obj.std_err, dtype=float)),
            stat=[],
            predy=np.ravel(np.asarray(model_obj.predy, dtype=float)),
            n=int(model_obj.n),
            k=int(model_obj.k),
        )
        model = _to_model(fitted, model_obj, a.model_name)
        if a.model_name:
            get_store().save(model)

        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "model_name": [a.model_name],
                    "model_type": [model.model_type],
                    "target": [model.target],
                    "n": [model.n],
                    "k": [model.k],
                    "r2": [model.r2],
                    "pseudo_r2": [model.pseudo_r2],
                    "log_likelihood": [model.log_likelihood],
                    "aic": [model.aic],
                    "spatial_coef": [model.spatial_coef],
                    "features": [model.feature_names],
                    "model": [pack_model(model)],
                },
                schema=params.output_schema,
            )
        )


# ===========================================================================
# predict() -- per-observation linear prediction
# ===========================================================================


@dataclass(slots=True, frozen=True)
class PredictArgs:
    """Arguments for the predict function."""

    data: Annotated[TableInput, Arg(0, doc="Table to score (must contain the model's explanatory columns).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")
    ]
    model: Annotated[
        bytes, Arg("model", default=b"", doc="A model BLOB (as returned by fit). Provide this OR model_name.")
    ]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]


_PREDICT_CACHE: dict[bytes, SpatialModel] = {}


class PredictModel(TableInOutGenerator[PredictArgs]):
    """Score a table with a stored model's linear predictor (intercept + coefficients . X)."""

    FunctionArguments: ClassVar[type] = PredictArgs

    class Meta:
        """Catalog metadata for the predict function."""

        name = "predict"
        description = "Predict with a stored spatial-regression model (linear/systematic predictor)"
        categories = ["regression", "spreg", "inference"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM pysal.predict("
                    "(SELECT id, inc, hoval FROM pysal.columbus()), model_name := 'columbus_err', id := 'id')"
                ),
                description="Linear prediction from the stored 'columbus_err' model",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[PredictArgs]) -> BindResponse:
        """Load the model, check its explanatory columns exist in the input, and build the prediction schema."""
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("predict requires either 'model_name' (a registry name) or 'model' (a model BLOB)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        model = cls._load(a)
        missing = [f for f in model.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model requires explanatory column(s) {', '.join(missing)} not present in the input; "
                f"model features: {', '.join(model.feature_names)}; input columns: {', '.join(input_schema.names)}"
            )
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.append(sfield("prediction", pa.float64(), "Predicted value (linear predictor).", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _load(cls, args: PredictArgs) -> SpatialModel:
        """Load the model from the registry by name or unpack it from the supplied BLOB."""
        if args.model_name:
            try:
                return get_store().load(args.model_name)
            except ModelNotFoundError as exc:
                raise ValueError(f"model {args.model_name!r} not found in the registry") from exc
        return unpack_model(args.model)

    @classmethod
    def _cached(cls, params: ProcessParams[PredictArgs]) -> SpatialModel:
        """Return the model for this execution, loading and caching it on first use."""
        assert params.init_response is not None
        key = params.init_response.execution_id
        model = _PREDICT_CACHE.get(key)
        if model is None:
            model = cls._load(params.args)
            _PREDICT_CACHE[key] = model
        return model

    @classmethod
    def process(
        cls,
        params: ProcessParams[PredictArgs],
        state: None,
        batch: pa.RecordBatch,
        out: OutputCollector,
    ) -> None:
        """Compute the linear predictor for each row in the batch and emit predictions."""
        a = params.args
        model = cls._cached(params)
        table = pa.Table.from_batches([batch])
        x = matrix(table, model.feature_names, what="explanatory")
        coefs = np.asarray(model.coefficients, dtype=float)
        preds = (x @ coefs) + (model.intercept if model.has_constant else 0.0)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = batch.column(a.id).to_pylist()
        columns["prediction"] = [float(v) for v in preds]
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# Registry management: list_models / model_info / drop_model
# ===========================================================================

_MODEL_INFO_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Stored model name.", nullable=False),
        sfield("model_type", pa.string(), "Estimator.", nullable=False),
        sfield("target", pa.string(), "Dependent variable.", nullable=False),
        sfield("n", pa.int64(), "Number of observations.", nullable=False),
        sfield("k", pa.int64(), "Number of parameters.", nullable=False),
        sfield("r2", pa.float64(), "R-squared (OLS)."),
        sfield("pseudo_r2", pa.float64(), "Pseudo R-squared (spatial models)."),
        sfield("aic", pa.float64(), "Akaike information criterion."),
        sfield("spatial_coef", pa.float64(), "Spatial autoregressive coefficient."),
        sfield("spatial_coef_name", pa.string(), "Name of the spatial term (rho / lambda)."),
        sfield("features", pa.list_(pa.string()), "Ordered explanatory-variable names.", nullable=False),
        sfield("created_at", pa.string(), "UTC timestamp the model was stored."),
        sfield("pysal_version", pa.string(), "libpysal version used to fit."),
    ]
)


def _model_rows(models: list[SpatialModel]) -> dict[str, list[Any]]:
    """Project a list of stored models into the column-wise dict for the model-info schema."""
    return {
        "model_name": [m.name for m in models],
        "model_type": [m.model_type for m in models],
        "target": [m.target for m in models],
        "n": [m.n for m in models],
        "k": [m.k for m in models],
        "r2": [m.r2 for m in models],
        "pseudo_r2": [m.pseudo_r2 for m in models],
        "aic": [m.aic for m in models],
        "spatial_coef": [m.spatial_coef for m in models],
        "spatial_coef_name": [m.spatial_coef_name for m in models],
        "features": [m.feature_names for m in models],
        "created_at": [m.created_at for m in models],
        "pysal_version": [m.pysal_version for m in models],
    }


@init_single_worker
@bind_fixed_schema
class ListModels(TableFunctionGenerator[NoArgs]):
    """List every spatial-regression model stored in the registry."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _MODEL_INFO_SCHEMA

    class Meta:
        """Catalog metadata for the list_models function."""

        name = "list_models"
        description = "List all spatial-regression models in the registry"
        categories = ["regression", "registry"]
        examples = [FunctionExample(sql="SELECT * FROM pysal.list_models()", description="List stored models")]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        """Return the estimated and maximum row counts for the registry listing."""
        return TableCardinality(estimate=10, max=10000)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: Any) -> None:
        """Emit one row per stored model and finish."""
        out.emit(pa.RecordBatch.from_pydict(_model_rows(get_store().list()), schema=params.output_schema))
        out.finish()


@dataclass(slots=True, frozen=True)
class ModelInfoArgs:
    """Arguments for the model_info function."""

    model_name: Annotated[str, Arg(0, doc="Name of a stored model.")]


@init_single_worker
@bind_fixed_schema
class ModelInfo(TableFunctionGenerator[ModelInfoArgs]):
    """Describe a single stored model, returning one row or none if it is absent."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _MODEL_INFO_SCHEMA

    class Meta:
        """Catalog metadata for the model_info function."""

        name = "model_info"
        description = "Describe a single stored model (one row, empty if absent)"
        categories = ["regression", "registry"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM pysal.model_info('columbus_err')", description="Show one model's metadata"
            )
        ]

    @classmethod
    def cardinality(cls, params: BindParams[ModelInfoArgs]) -> TableCardinality:
        """Return the single-row cardinality of a model lookup."""
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[ModelInfoArgs], state: None, out: Any) -> None:
        """Look up the named model and emit its metadata row, or nothing if it is absent."""
        try:
            models = [get_store().load(params.args.model_name)]
        except ModelNotFoundError:
            models = []
        out.emit(pa.RecordBatch.from_pydict(_model_rows(models), schema=params.output_schema))
        out.finish()


@dataclass(slots=True, frozen=True)
class DropModelArgs:
    """Arguments for the drop_model function."""

    model_name: Annotated[str, Arg(0, doc="Name of the model to delete.")]


_DROP_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Name of the model.", nullable=False),
        sfield("dropped", pa.bool_(), "True if a model was deleted, False if it did not exist.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class DropModel(TableFunctionGenerator[DropModelArgs]):
    """Delete a model from the registry and report whether it existed."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _DROP_SCHEMA

    class Meta:
        """Catalog metadata for the drop_model function."""

        name = "drop_model"
        description = "Delete a model from the registry"
        categories = ["regression", "registry"]
        examples = [
            FunctionExample(sql="SELECT * FROM pysal.drop_model('columbus_err')", description="Delete a stored model")
        ]

    @classmethod
    def cardinality(cls, params: BindParams[DropModelArgs]) -> TableCardinality:
        """Return the single-row cardinality of a delete operation."""
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[DropModelArgs], state: None, out: Any) -> None:
        """Delete the named model and emit a row reporting whether it was removed."""
        name = params.args.model_name
        dropped = get_store().delete(name)
        out.emit(pa.RecordBatch.from_pydict({"model_name": [name], "dropped": [dropped]}, schema=params.output_schema))
        out.finish()


REGRESSION_FUNCTIONS: list[type] = [
    SpregFn,
    FitModel,
    PredictModel,
    ListModels,
    ModelInfo,
    DropModel,
]
