"""Resample a pySIPNET Dataset to a coarser timestep, one explicit method per variable.

There is no default method.  How consecutive steps should be combined is not a
property of a variable but of what you want from it: daily soil water can be
the value at the end of the day or the mean over it, and those are different
quantities with different CF ``cell_methods``.  What *is* a property of the
variable is which methods are meaningful at all — a total adds, a pool does
not, a running total is its last value, and a mean or rate must be weighted by
the step length because SIPNET's steps are not all the same length (the Niwot
record alternates day and night steps between 0.29 and 0.63 days).  So the
caller names the method, and :func:`resample` refuses one the variable's kind
does not support, saying why and what would be valid.

::

    from pysipnet import resample

    daily = resample(result.outputs[["nee", "gpp"]], "1D", how="sum")
    annual = resample(result.outputs[["wood_carbon", "nee"]], "YS",
                      how={"wood_carbon": "last", "net_ecosystem_exchange": "sum"})

The result has the same layout as the input — ``time`` is the end of each
coarser step, ``time_step_start``, ``time_step_length`` and ``time_bounds``
describe the interval it covers — and every variable's ``kind``,
``time_reference`` and ``cell_methods`` are rewritten to describe what it now
is.  ``time_step_length`` is the sum of the steps that went into each cell, so a
cell that the record only partly covers (the first or last day of a run) can be
recognized by comparing it with the span of its bounds.

Cells are aligned to the calendar with the step-end convention: a step ending
at midnight belongs to the day that ended, so ``"1D"`` cells are ``(00:00,
24:00]`` and a daily record resamples to itself.  *freq* is any pandas offset
alias (``"1D"``, ``"7D"``, ``"MS"``, ``"YS"``) naming a period at least as long
as the shortest step; a finer one would return the input under a false label.

A variable may carry dimensions besides ``time`` — an ensemble's ``member``, a
stack's ``site`` — and they come through untouched, with every coordinate that
does not depend on ``time`` (``lon``/``lat`` on ``site``, a scalar ``member``).
What cannot vary along them is the time axis itself: the cells are one set for
the whole Dataset, so ``time_step_start`` and ``time_step_length`` must be on
``time`` alone.

:func:`check_resampling_method` is the check :func:`resample` applies to each
variable, for code that reduces over time some other way and wants the same
refusal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import pandas as pd

from pysipnet.dataset import (
    BOUNDS_DIMENSION,
    TIME_DIMENSION,
    TIME_ZONE_UNDECLARED,
    assemble_time_coords,
    unfilled_coordinates,
)
from pysipnet.variables import (
    CELL_METHODS_FOR_KIND,
    CLIMATE_VARIABLES_BY_NAME,
    OUTPUT_VARIABLES_BY_NAME,
    RESAMPLED_KIND,
    RESAMPLING_METHODS_FOR_KIND,
    TIME_REFERENCE_FOR_KIND,
    ResamplingMethod,
    VariableKind,
)

if TYPE_CHECKING:
    import xarray as xr

STEP_LENGTH_RESAMPLED = "sum of the declared lengths of the steps combined into each cell"

_METHODS = ("sum", "mean", "last")

_KIND_IN_WORDS: dict[VariableKind, str] = {
    VariableKind.TIMESTEP_START_COORDINATE: "a time coordinate",
    VariableKind.TIMESTEP_END_STATE: "a pool reported at the end of the timestep",
    VariableKind.TIMESTEP_TOTAL: "a total over the timestep",
    VariableKind.DAILY_RATE: "a rate during the timestep",
    VariableKind.TIMESTEP_MEAN: "a mean over the timestep",
    VariableKind.CUMULATIVE: "a running total from the start of the run",
}

# What each valid (kind, method) pair produces, for the error message's menu.
_WHAT_YOU_GET: dict[tuple[VariableKind, str], str] = {
    (VariableKind.TIMESTEP_END_STATE, "last"): "the pool at the end of the coarser step",
    (VariableKind.TIMESTEP_END_STATE, "mean"): (
        "the time-weighted mean of the pool over the coarser step, which makes it a "
        "timestep_mean rather than a state"
    ),
    (VariableKind.TIMESTEP_TOTAL, "sum"): "the total over the coarser step",
    (VariableKind.DAILY_RATE, "mean"): "the time-weighted mean rate over the coarser step",
    (VariableKind.TIMESTEP_MEAN, "mean"): "the time-weighted mean over the coarser step",
    (VariableKind.CUMULATIVE, "last"): "the running total as of the end of the coarser step",
}

# Why an invalid (kind, method) pair is refused rather than computed.
_WHY_NOT: dict[tuple[VariableKind, str], str] = {
    (VariableKind.TIMESTEP_END_STATE, "sum"): (
        "a pool is not additive across steps; adding end-of-step values counts the same "
        "stock once per step"
    ),
    (VariableKind.TIMESTEP_TOTAL, "mean"): (
        "the mean of per-step totals depends on how long the steps happened to be; sum "
        "them, and for a rate divide by the step length "
        "(pysipnet.arithmetic.divide_with_units with step_length())"
    ),
    (VariableKind.TIMESTEP_TOTAL, "last"): (
        "the last step's total is not the total over the coarser step"
    ),
    (VariableKind.DAILY_RATE, "sum"): (
        "adding rates over steps of unequal length is not a total; multiply by the "
        "step length first (pysipnet.arithmetic.multiply_with_units with step_length()), "
        "then the variable is a total and sums"
    ),
    (VariableKind.DAILY_RATE, "last"): "a rate at the last step does not represent the whole",
    (VariableKind.TIMESTEP_MEAN, "sum"): "means do not add",
    (VariableKind.TIMESTEP_MEAN, "last"): (
        "the mean over the last step does not represent the whole"
    ),
    (VariableKind.CUMULATIVE, "sum"): (
        "a running total already includes every earlier step; summing it double counts"
    ),
    (VariableKind.CUMULATIVE, "mean"): (
        "the mean of a running total is not a quantity anyone wants; take 'last' for the "
        "total so far, or resample net_ecosystem_exchange with 'sum'"
    ),
}


@overload
def resample(
    data: xr.Dataset, freq: str, *, how: ResamplingMethod | Mapping[str, ResamplingMethod]
) -> xr.Dataset: ...


@overload
def resample(data: xr.DataArray, freq: str, *, how: ResamplingMethod) -> xr.DataArray: ...


def resample(
    data: xr.Dataset | xr.DataArray,
    freq: str,
    *,
    how: ResamplingMethod | Mapping[str, ResamplingMethod],
) -> xr.Dataset | xr.DataArray:
    """Combine consecutive timesteps into coarser ones, by an explicit method.

    Parameters
    ----------
    data:
        A Dataset from :class:`~pysipnet.output.SIPNETOutput` or
        :class:`~pysipnet.climate.ClimateDrivers`, or one DataArray taken from
        such a Dataset.  Anything with the same coordinates works, including a
        stack of runs: a variable may have dimensions besides ``time``, in any
        order, provided ``time_step_start`` and ``time_step_length`` are on
        ``time`` alone.
    freq:
        A pandas offset alias for the new step: ``"1D"``, ``"7D"``, ``"MS"``
        for calendar months, ``"YS"`` for calendar years.  It must name a
        positive period no shorter than the shortest step.
    how:
        ``"sum"``, ``"mean"`` or ``"last"`` for every data variable, or a
        mapping from variable name to method.  With a mapping, only the named
        variables are kept.  ``"mean"`` is weighted by ``time_step_length``.
        There is no default; see the module docstring for why.

    Returns
    -------
    The same type as *data*, with the same time layout and with every
    variable's ``kind``, ``time_reference`` and ``cell_methods`` rewritten.
    Only ``time`` is reduced: every other dimension keeps its size, its order
    within each variable and its index coordinate, and every coordinate that
    does not depend on ``time`` is carried over unchanged.  The coordinates on
    ``time`` are rebuilt for the coarser cells.  Rows whose
    ``time_step_start`` or ``time_step_length`` is ``NaT`` — the padding
    xarray leaves when runs on different time axes are aligned and one of
    them is then selected — are not steps, and are dropped first.

    Raises
    ------
    ValueError
        If a method is not meaningful for a variable's kind, saying why and
        listing the methods that are (see :func:`check_resampling_method`); if
        a variable's kind cannot be determined or it has no ``time``
        dimension; if *data* lacks the pySIPNET time coordinates, or they vary
        along a dimension other than ``time``; if *freq* is not a positive
        period; or if its periods are all shorter than the shortest step.
    """
    import xarray as xr

    if isinstance(data, xr.DataArray):
        if isinstance(how, Mapping):
            raise TypeError("Pass how as a single method for a DataArray, e.g. how='sum'.")
        # An arithmetic result has no name; it resamples under its derivation, which
        # is what a refusal should call it.
        name = data.name if data.name is not None else data.attrs.get("derivation") or "array"
        resampled = resample(data.to_dataset(name=name), freq, how=how)[name]
        return resampled.rename(data.name)

    _require_time_layout(data)
    offset = _checked_frequency(freq)
    methods = _methods_for(data, how)
    if not methods:
        raise ValueError("Nothing to resample: the Dataset has no data variables.")
    data = _only_real_steps(data)

    def grouped(obj: xr.DataArray | xr.Dataset) -> Any:
        # The step-end labeling makes a step ending on a boundary part of the
        # cell that ends there, hence right-closed cells.
        return obj.resample({TIME_DIMENSION: freq}, closed="right", label="right")

    length_ns = data["time_step_length"].values.astype("timedelta64[ns]").astype("int64")
    weight_days = _as_data_array(length_ns / 86_400e9, data)
    n_steps = grouped(xr.ones_like(weight_days)).sum()
    keep = (n_steps > 0).values
    cell_ends = n_steps[TIME_DIMENSION].values.astype("datetime64[ns]")
    _check_not_upsampling(freq, offset, cell_ends, length_ns)

    by_method: dict[str, list[str]] = {m: [] for m in _METHODS}
    for name, method in methods.items():
        by_method[method].append(name)

    pieces: list[xr.Dataset] = []
    if by_method["sum"]:
        pieces.append(grouped(data[by_method["sum"]]).sum(skipna=False))
    if by_method["last"]:
        pieces.append(grouped(data[by_method["last"]]).last(skipna=False))
    if by_method["mean"]:
        weighted = grouped(data[by_method["mean"]] * weight_days).sum(skipna=False)
        pieces.append(weighted / grouped(weight_days).sum())

    start = grouped(_as_data_array(data["time_step_start"].values, data)).min()
    end = grouped(_as_data_array(data[TIME_DIMENSION].values, data)).max()

    merged = xr.merge(pieces, compat="no_conflicts", join="exact")
    merged = merged[list(methods)].isel({TIME_DIMENSION: keep})

    def attributes_for(name: str) -> dict[str, Any]:
        return dict(data[name].attrs) if name in data.coords else {}

    coords: dict[str, Any] = assemble_time_coords(
        start=start.values[keep].astype("datetime64[ns]"),
        end=end.values[keep].astype("datetime64[ns]"),
        length=_summed_lengths(data, cell_ends, length_ns)[keep].view("timedelta64[ns]"),
        attributes_for=attributes_for,
        length_source=STEP_LENGTH_RESAMPLED,
        time_zone=data[TIME_DIMENSION].attrs.get("time_zone", TIME_ZONE_UNDECLARED),
    )
    for name, coord in data.coords.items():
        if TIME_DIMENSION not in coord.dims and BOUNDS_DIMENSION not in coord.dims:
            coords[str(name)] = coord.variable

    data_vars = {
        name: (
            data[name].dims,
            merged[name].transpose(*data[name].dims).values,
            _resampled_attrs(data[name].attrs, _kind_of(data, name), methods[name], freq),
        )
        for name in methods
    }
    attrs = dict(data.attrs)
    attrs["time_step_length_source"] = STEP_LENGTH_RESAMPLED
    attrs["resampling_frequency"] = freq
    return unfilled_coordinates(xr.Dataset(data_vars, coords=coords, attrs=attrs))


def check_resampling_method(kind: VariableKind | str, method: str, *, name: str) -> None:
    """Raise unless *method* is a meaningful way to combine steps of a variable of *kind*.

    The check :func:`resample` applies to each variable, for code that reduces
    over time some other way — a window reduction that also offers ``min`` or
    ``max``, say — and wants to refuse ``sum``, ``mean`` or ``last`` in the
    same words.  *name* is the variable the message names.

    Raises
    ------
    ValueError
        If *method* is not one of ``"sum"``, ``"mean"``, ``"last"``; if *kind*
        is not a :class:`~pysipnet.variables.VariableKind`; or if the kind does
        not admit the method, saying what the variable is, why the method is
        meaningless for it, and which methods would be valid.
    """
    _check_method_name(method, name)
    try:
        kind = VariableKind(kind)
    except ValueError:
        raise ValueError(
            f"{name!r} has kind {kind!r}, which is not one of {[k.value for k in VariableKind]}."
        ) from None
    valid = RESAMPLING_METHODS_FOR_KIND[kind]
    if method in valid:
        return
    menu = "; ".join(f"'{m}' gives {_WHAT_YOU_GET[(kind, m)]}" for m in _METHODS if m in valid)
    why = _WHY_NOT.get((kind, method), "that combination has no meaning")
    options = f" Valid for this kind: {menu}." if menu else " Nothing is valid for this kind."
    raise ValueError(
        f"Cannot resample {name!r} with {method!r}: it is {_KIND_IN_WORDS[kind]} "
        f"(kind {kind.value!r}), and {why}.{options}"
    )


def _as_data_array(values: np.ndarray, like: xr.Dataset) -> xr.DataArray:
    import xarray as xr

    return xr.DataArray(values, dims=TIME_DIMENSION, coords={TIME_DIMENSION: like[TIME_DIMENSION]})


def _require_time_layout(ds: xr.Dataset) -> None:
    needed = (TIME_DIMENSION, "time_step_start", "time_step_length")
    missing = [c for c in needed if c not in ds.coords]
    if missing or TIME_DIMENSION not in ds.dims:
        raise ValueError(
            f"Dataset lacks the pySIPNET time coordinates {missing}. resample() needs the "
            "layout SIPNETOutput.xarray, SIPNETOutput.select or ClimateDrivers.xarray produce: "
            "'time' at the step end with 'time_step_start' and 'time_step_length' alongside."
        )
    if BOUNDS_DIMENSION in ds.dims and "time_bounds" not in ds.coords:
        raise ValueError("Dataset has a 'bounds' dimension but no 'time_bounds' coordinate.")
    expected = {
        "time_step_start": (TIME_DIMENSION,),
        "time_step_length": (TIME_DIMENSION,),
        "time_bounds": (TIME_DIMENSION, BOUNDS_DIMENSION),
    }
    offenders = {
        name: tuple(map(str, ds[name].dims))
        for name, dims in expected.items()
        if name in ds.coords and ds[name].dims != dims
    }
    if offenders:
        raise ValueError(
            f"The interval coordinates vary along more than time: {offenders}. xarray gives "
            "them an extra dimension when runs on different time axes are concatenated, and "
            "then the runs no longer share one set of cells. Resample each run, or each "
            "site, separately; resample() drops the padding rows that selecting one leaves."
        )


def _checked_frequency(freq: str) -> pd.DateOffset:
    """*freq* as a pandas offset, refusing anything that does not name a positive period."""
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"freq must be a pandas offset alias such as '1D', 'MS' or 'YS', not {freq!r}."
        ) from error
    if offset is None:
        raise ValueError(
            f"freq must be a pandas offset alias such as '1D', 'MS' or 'YS', not {freq!r}."
        )
    if offset.n <= 0:
        raise ValueError(
            f"freq={freq!r} names a period of {offset.n} units, which cannot hold a step; "
            "pass a positive frequency."
        )
    return offset


def _only_real_steps(ds: xr.Dataset) -> xr.Dataset:
    """*ds* without the rows that are not steps: those with no start or no length.

    Selecting one run out of a stack of runs on different time axes leaves the
    union of those axes, padded with ``NaT``.  Left in, a padding row turns the
    cell it falls in to ``NaN`` and, because a ``NaT`` length casts to the
    ``int64`` minimum, gives that cell a length of minus 292 years.
    """
    real = ~(np.isnat(ds["time_step_start"].values) | np.isnat(ds["time_step_length"].values))
    if real.all():
        return ds
    if not real.any():
        raise ValueError(
            "Every row's time_step_start or time_step_length is NaT, so there is no step "
            "to resample; a selection from a stack of runs matched a run with no record."
        )
    return ds.isel({TIME_DIMENSION: real})


def _check_not_upsampling(
    freq: str, offset: pd.DateOffset, cell_ends: np.ndarray, length_ns: np.ndarray
) -> None:
    """Refuse a *freq* whose every cell is shorter than the shortest step.

    Each step lands in a cell of its own then, so the result is the input under
    a label that says it was resampled to *freq*.  The longest cell is measured
    on the record's own cells, so calendar periods of varying length are
    compared as they fall.  Equal lengths pass: daily data to ``"1D"`` is
    itself.
    """
    first_edge = (pd.Timestamp(cell_ends[0]) - offset).to_datetime64().astype("datetime64[ns]")
    edges = np.concatenate([[first_edge], cell_ends]).astype("int64")
    longest_cell = int(np.diff(edges).max())
    shortest_step = int(length_ns.min())
    if shortest_step > longest_cell:
        raise ValueError(
            f"freq={freq!r} makes cells of at most {pd.Timedelta(longest_cell, 'ns')}, shorter "
            f"than the shortest step in the data ({pd.Timedelta(shortest_step, 'ns')}). Every "
            "step would land in a cell of its own and come back unchanged, labeled as "
            f"resampled to {freq!r}. Pass a frequency at least as long as the steps."
        )


def _summed_lengths(ds: xr.Dataset, cell_ends: np.ndarray, length_ns: np.ndarray) -> np.ndarray:
    """The declared step lengths summed per cell, exactly, as ``int64`` nanoseconds.

    Summed here rather than by xarray, which casts to float64 to fill empty
    cells; a float64 holds whole nanoseconds exactly only to about 104 days.
    A step belongs to the right-closed cell its end falls in, which is the
    first cell ending at or after it.
    """
    cell = np.searchsorted(cell_ends, ds[TIME_DIMENSION].values.astype("datetime64[ns]"), "left")
    totals = np.zeros(cell_ends.size, dtype="int64")
    np.add.at(totals, cell, length_ns)
    return totals


def _methods_for(ds: xr.Dataset, how: str | Mapping[str, str]) -> dict[str, str]:
    """Resolve *how* into one validated method per variable to keep."""
    if isinstance(how, str):
        requested = {str(name): how for name in ds.data_vars}
    elif isinstance(how, Mapping):
        requested = {str(k): v for k, v in how.items()}
        unknown = [name for name in requested if name not in ds.data_vars]
        if unknown:
            raise KeyError(
                f"{unknown} are not data variables of this Dataset; it has "
                f"{list(map(str, ds.data_vars))}. Variable names are the registry names "
                "('net_ecosystem_exchange', not 'nee'); select by alias before resampling."
            )
    else:
        raise TypeError(
            f"how must be 'sum', 'mean', 'last' or a mapping from variable to one of those, "
            f"not {type(how).__name__}."
        )

    for name, method in requested.items():
        _check_method_name(method, name)
        check_resampling_method(_kind_of(ds, name), method, name=name)
        if TIME_DIMENSION not in ds[name].dims:
            raise ValueError(
                f"{name!r} has dims {tuple(map(str, ds[name].dims))} and no {TIME_DIMENSION!r}, "
                "so there is nothing to resample; leave it out of how."
            )
    return requested


def _check_method_name(method: str, name: str) -> None:
    if method not in _METHODS:
        raise ValueError(
            f"Unknown resampling method {method!r} for {name!r}; choose from {list(_METHODS)}."
        )


def _kind_of(ds: xr.Dataset, name: str) -> VariableKind:
    """The variable's kind, from its attributes or, failing that, the registries."""
    value = ds[name].attrs.get("kind")
    if value is not None:
        return VariableKind(value)
    spec = OUTPUT_VARIABLES_BY_NAME.get(name) or CLIMATE_VARIABLES_BY_NAME.get(name)
    if spec is not None:
        return spec.kind
    raise ValueError(
        f"Cannot resample {name!r}: neither its attributes nor the variable registry say "
        "what kind of quantity it is. Set attrs['kind'] to one of "
        f"{[k.value for k in VariableKind]} so the method can be checked against it."
    )


def _resampled_attrs(
    attrs: Mapping[str, Any], kind: VariableKind, method: str, freq: str
) -> dict[str, Any]:
    new_kind = RESAMPLED_KIND[(kind, method)]
    out = dict(attrs)
    out["kind"] = new_kind.value
    out["time_reference"] = TIME_REFERENCE_FOR_KIND[new_kind]
    cell_methods = CELL_METHODS_FOR_KIND[new_kind]
    if cell_methods is None:
        out.pop("cell_methods", None)
    else:
        out["cell_methods"] = cell_methods
    weighting = ", weighted by time_step_length" if method == "mean" else ""
    out["resampling"] = f"{method} of {kind.value} values over {freq}{weighting}"
    # The file's printf precision no longer describes the combined value.
    out.pop("output_decimals", None)
    return out


__all__ = ["check_resampling_method", "resample"]
