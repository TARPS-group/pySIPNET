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

A record without the interval coordinates — an observation, which has
``time`` labels and nothing saying where each step began — resamples too, on
calendar cells alone.  With nothing to say where its steps end, each cell is
labeled at its right edge, the result has no interval coordinates, and a mean
weighs every step equally, so it is refused unless the labels are equally
spaced.

The rules :func:`resample` applies are public on their own, for code that
reduces over time some other way (over arbitrary windows, say, or with ``min``
and ``max``) and wants the same refusals in the same words:
:func:`check_resampling_method` for a kind and a method, :func:`check_frequency`
and :func:`check_not_upsampling` for *freq*, :func:`drop_padding` for the rows a
stack of runs leaves behind, :func:`resampled_attributes` for what a combined
value's attributes should say, and :func:`~pysipnet.variables.variable_kind`
for the kind all of them start from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset

from pysipnet.dataset import (
    BOUNDS_DIMENSION,
    NS_PER_DAY,
    STEP_TOLERANCE,
    TIME_AXIS_ATTRIBUTES,
    TIME_DIMENSION,
    TIME_ZONE_UNDECLARED,
    assemble_time_coords,
    unfilled_coordinates,
    without_absent_bounds,
)
from pysipnet.variables import (
    CELL_METHODS_FOR_KIND,
    RESAMPLED_KIND,
    RESAMPLING_METHODS_FOR_KIND,
    TIME_REFERENCE_FOR_KIND,
    ResamplingMethod,
    VariableKind,
    parse_variable_kind,
    variable_kind,
    variable_label,
)

if TYPE_CHECKING:
    import xarray as xr

STEP_LENGTH_RESAMPLED = "sum of the declared lengths of the steps combined into each cell"

_INTERVAL_COORDINATES = ("time_step_start", "time_step_length")

_Xarray = TypeVar("_Xarray", "xr.DataArray", "xr.Dataset")

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
        ``time`` alone.  A record with a datetime ``time`` and none of the
        interval coordinates (``time_step_start``, ``time_step_length``,
        ``time_bounds``) resamples on calendar cells alone; see Returns.
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
    them is then selected — are not steps, and are dropped first (see
    :func:`drop_padding`).  A DataArray's ``time`` has no ``bounds``
    attribute, since it cannot carry the ``time_bounds`` it would name.

    A record without interval coordinates comes back without them too, and
    without any other coordinate on ``time``.  Its
    ``time`` is the right edge of each calendar cell, since nothing says where
    the steps in it end, cells no step falls in are dropped, and ``"mean"``
    weighs every step equally, which is why it needs equally spaced labels.
    Each variable's ``resampling`` attribute says the cells are calendar
    cells.  Such a record's first and last cells may be only partly covered,
    and nothing in the result says so.

    Either way a cell holding a ``NaN`` is ``NaN`` under ``"sum"`` and
    ``"mean"``; ``"last"`` is the cell's last value, ``NaN`` only if that is.

    Raises
    ------
    ValueError
        If a method is not meaningful for a variable's kind, saying why and
        listing the methods that are (see :func:`check_resampling_method`); if
        a variable's kind cannot be determined (see
        :func:`~pysipnet.variables.variable_kind`) or it has no ``time``
        dimension; if *data* has only some of the pySIPNET time coordinates, or
        they vary along a dimension other than ``time``; if *freq* is not a
        positive period (:func:`check_frequency`) or its periods are all
        shorter than the shortest step (:func:`check_not_upsampling`); if a row
        with a value has a ``NaT`` interval (:func:`drop_padding`); or, for a
        record without interval coordinates, if its labels are not datetimes
        that strictly increase, or are unequally spaced and a mean is asked
        for.
    """
    import xarray as xr

    if isinstance(data, xr.DataArray):
        if isinstance(how, Mapping):
            raise TypeError("Pass how as a single method for a DataArray, e.g. how='sum'.")
        name = variable_label(data)
        resampled = resample(data.to_dataset(name=name), freq, how=how)[name]
        return without_absent_bounds(resampled.rename(data.name))

    if _on_calendar_cells_only(data):
        return _resample_on_calendar_cells(data, freq, how)

    _require_time_layout(data)
    offset = check_frequency(freq)
    methods = _methods_for(data, how)
    data = _drop_padding(data, _valued_among(list(methods)))

    length_ns = data["time_step_length"].values.astype("timedelta64[ns]").astype("int64")
    weight_days = _as_data_array(length_ns / NS_PER_DAY, data)
    cell_ends, n_steps = _steps_per_cell(data, freq)
    cell = _cell_of_each_step(data, cell_ends, n_steps)
    keep = n_steps > 0
    _check_not_upsampling(freq, offset, cell_ends, int(length_ns.min()))

    combined = _combined(data, freq, methods, weight_days, keep)
    start = _grouped(_as_data_array(data["time_step_start"].values, data), freq).min()
    end = _grouped(_as_data_array(data[TIME_DIMENSION].values, data), freq).max()

    def attributes_for(name: str) -> dict[str, Any]:
        return dict(data[name].attrs) if name in data.coords else {}

    coords: dict[str, Any] = assemble_time_coords(
        start=start.values[keep].astype("datetime64[ns]"),
        end=end.values[keep].astype("datetime64[ns]"),
        length=_summed_lengths(cell, cell_ends.size, length_ns)[keep].view("timedelta64[ns]"),
        attributes_for=attributes_for,
        length_source=STEP_LENGTH_RESAMPLED,
        time_zone=data[TIME_DIMENSION].attrs.get("time_zone", TIME_ZONE_UNDECLARED),
    )
    coords.update(_coordinates_off_time(data))

    data_vars = _resampled_variables(
        data, combined, methods, over=f"over {freq}", weighting="weighted by time_step_length"
    )
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
    if method not in _METHODS:
        raise ValueError(
            f"Unknown resampling method {method!r} for {name!r}; choose from {list(_METHODS)}."
        )
    kind = parse_variable_kind(kind, name=name)
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


def check_frequency(freq: str) -> BaseOffset:
    """*freq* as a pandas offset, refusing anything that does not name a positive period.

    The first check :func:`resample` makes, for code that wants to refuse a
    bad *freq* before there is any data to resample: when a forward model is
    configured, say, rather than on a worker after a queue wait.  Whether the
    period is long enough needs the data; that is :func:`check_not_upsampling`.

    Raises
    ------
    ValueError
        If pandas does not read *freq* as an offset, with pandas' own reason
        (that ``'M'`` is now ``'ME'``, say), or if the period is not positive.
    """
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        if offset is None:
            raise ValueError("no frequency given")
    except (TypeError, ValueError) as error:
        # pandas' own reason carries the useful part, e.g. that 'M' is now 'ME'.
        raise ValueError(
            f"freq must be a pandas offset alias such as '1D', 'MS' or 'YS', not {freq!r}: {error}"
        ) from error
    if offset.n <= 0:
        raise ValueError(
            f"freq={freq!r} names a period of {offset.n} units, which cannot hold a step; "
            "pass a positive frequency."
        )
    return offset


def check_not_upsampling(data: xr.Dataset | xr.DataArray, freq: str) -> None:
    """Raise if every cell of *freq* on *data*'s time axis is shorter than its shortest step.

    Each step would land in a cell of its own, and a resampled record would be
    the input under a label saying it was resampled to *freq*.  The check
    :func:`resample` makes once the data is at hand, for code that combines
    steps on the same cells itself.

    The shortest step is the shortest ``time_step_length`` where *data* has
    one, ignoring padding, and otherwise the shortest gap between ``time``
    labels; a record of one label has no gap and always passes.  The longest
    cell is measured on the record's own right-closed cells, so calendar
    periods of varying length are compared as they fall.  Equal lengths pass,
    to within :data:`~pysipnet.dataset.STEP_TOLERANCE`, so hourly data
    declared as ``0.0416667`` days resamples to ``"1h"`` as itself.

    Raises
    ------
    ValueError
        If *freq* is not a positive period (see :func:`check_frequency`); if
        *data* has no datetime ``time`` labels that strictly increase; or if
        *freq* is finer than the steps.
    """
    offset = check_frequency(freq)
    _require_increasing_labels(data)
    if "time_step_length" in data.coords:
        lengths = data["time_step_length"].values.astype("timedelta64[ns]")
        steps = lengths[~np.isnat(lengths)].astype("int64")
    else:
        steps = np.diff(_labels_ns(data))
    if steps.size:
        cell_ends, _ = _steps_per_cell(data, freq)
        _check_not_upsampling(freq, offset, cell_ends, int(steps.min()))


def drop_padding(data: _Xarray) -> _Xarray:
    """*data* without its padding rows: no interval, and no value in any variable.

    Selecting one run out of a stack of runs on different time axes leaves the
    union of those axes, padded with ``NaT`` in ``time_step_start`` and
    ``time_step_length`` and ``NaN`` in every value.  Those rows are not
    steps.  Left in, a padding row turns whatever cell or window it falls in to
    ``NaN`` and, because a ``NaT`` length casts to the ``int64`` minimum,
    gives it a length of minus 292 years.  :func:`resample` drops them first;
    this is the same rule, for code that combines steps some other way.

    A row with a value but no interval is not padding: it is a step whose
    extent is unknown, and dropping it would lose the value silently, so it
    is refused.  Only variables on ``time`` are consulted; data with neither
    interval coordinate has no padding and comes back unchanged.

    Raises
    ------
    ValueError
        If a row whose ``time_step_start`` or ``time_step_length`` is ``NaT``
        holds a value, naming the variables that hold one; if every row is
        padding; or if an interval coordinate varies along a dimension other
        than ``time``, as it does in the stack itself before one run is
        selected.
    """
    import xarray as xr

    if isinstance(data, xr.DataArray):
        name = variable_label(data)
        return _drop_padding(data, lambda rows: [name] if bool(rows.notnull().any()) else [])
    names = [str(name) for name, array in data.data_vars.items() if TIME_DIMENSION in array.dims]
    return _drop_padding(data, _valued_among(names))


def resampled_attributes(
    attrs: Mapping[str, Any], kind: VariableKind | str, method: str, *, name: str
) -> dict[str, Any]:
    """*attrs* rewritten for values of *kind* combined over a coarser step by *method*.

    ``kind`` becomes the kind of the result (a pool averaged is a
    ``timestep_mean``; see :data:`~pysipnet.variables.RESAMPLED_KIND`), and
    ``time_reference`` and ``cell_methods`` follow it; ``cell_methods`` is
    removed where the new kind has none.  ``output_decimals`` is dropped,
    since SIPNET's printf precision does not describe a combined value.
    Everything else is kept.  :func:`resample` adds a ``resampling`` sentence
    saying how the cells were formed; code combining steps over its own
    windows says that itself.

    *name* is the variable a refusal names.

    Raises
    ------
    ValueError
        As :func:`check_resampling_method` does, if *method* is not
        meaningful for *kind*.
    """
    check_resampling_method(kind, method, name=name)
    new_kind = RESAMPLED_KIND[(parse_variable_kind(kind, name=name), method)]
    out = dict(attrs)
    out["kind"] = new_kind.value
    out["time_reference"] = TIME_REFERENCE_FOR_KIND[new_kind]
    cell_methods = CELL_METHODS_FOR_KIND[new_kind]
    if cell_methods is None:
        out.pop("cell_methods", None)
    else:
        out["cell_methods"] = cell_methods
    out.pop("output_decimals", None)
    return out


def _as_data_array(values: np.ndarray, like: xr.Dataset | xr.DataArray) -> xr.DataArray:
    import xarray as xr

    return xr.DataArray(values, dims=TIME_DIMENSION, coords={TIME_DIMENSION: like[TIME_DIMENSION]})


def _grouped(obj: xr.DataArray | xr.Dataset, freq: str) -> Any:
    """*obj* grouped into cells of *freq*, right-closed and labeled at the right edge.

    ``time`` is a step's end, so a step ending on a boundary belongs to the
    cell that ends there.
    """
    return obj.resample({TIME_DIMENSION: freq}, closed="right", label="right")


def _steps_per_cell(data: xr.Dataset | xr.DataArray, freq: str) -> tuple[np.ndarray, np.ndarray]:
    """The end of every cell of *freq* the record spans, and how many steps fall in each."""
    ones = _as_data_array(np.ones(data.sizes[TIME_DIMENSION]), data)
    n_steps = _grouped(ones, freq).sum()
    # xarray reports an empty cell's count as NaN, not 0.
    counts = np.nan_to_num(n_steps.values).astype("int64")
    return n_steps[TIME_DIMENSION].values.astype("datetime64[ns]"), counts


def _labels_ns(data: xr.Dataset | xr.DataArray) -> np.ndarray:
    return np.asarray(data[TIME_DIMENSION].values.astype("datetime64[ns]").astype("int64"))


def _combined(
    data: xr.Dataset,
    freq: str,
    methods: Mapping[str, tuple[str, VariableKind]],
    weights: xr.DataArray,
    keep: np.ndarray,
) -> xr.Dataset:
    """Each variable in *methods* combined over the cells of *freq* that *keep* marks.

    ``"mean"`` is weighted by *weights*, one per step.  ``skipna=False``
    throughout, so a ``NaN`` in a cell reaches its sum and mean; ``"last"``
    reads only the cell's last step.
    """
    import xarray as xr

    by_method: dict[str, list[str]] = {m: [] for m in _METHODS}
    for name, (method, _) in methods.items():
        by_method[method].append(name)
    pieces: list[xr.Dataset] = []
    if by_method["sum"]:
        pieces.append(_grouped(data[by_method["sum"]], freq).sum(skipna=False))
    if by_method["last"]:
        pieces.append(_grouped(data[by_method["last"]], freq).last(skipna=False))
    if by_method["mean"]:
        weighted = _grouped(data[by_method["mean"]] * weights, freq).sum(skipna=False)
        pieces.append(weighted / _grouped(weights, freq).sum())
    merged = xr.merge(pieces, compat="no_conflicts", join="exact")
    return merged[list(methods)].isel({TIME_DIMENSION: keep})


def _resampled_variables(
    data: xr.Dataset,
    combined: xr.Dataset,
    methods: Mapping[str, tuple[str, VariableKind]],
    *,
    over: str,
    weighting: str,
) -> dict[str, Any]:
    """The result's data variables: *combined*'s values in *data*'s order of dimensions.

    Attributes are :func:`resampled_attributes` plus a ``resampling`` sentence:
    *over* says what the cells were and *weighting* how a mean weighed the
    steps.
    """
    variables: dict[str, Any] = {}
    for name, (method, kind) in methods.items():
        attrs = resampled_attributes(data[name].attrs, kind, method, name=name)
        how = over + (f", {weighting}" if method == "mean" else "")
        attrs["resampling"] = f"{method} of {kind.value} values {how}"
        variables[name] = (
            data[name].dims,
            combined[name].transpose(*data[name].dims).values,
            attrs,
        )
    return variables


def _coordinates_off_time(data: xr.Dataset) -> dict[str, Any]:
    """Every coordinate of *data* that does not depend on ``time``, to carry over as is."""
    return {
        str(name): coord.variable
        for name, coord in data.coords.items()
        if TIME_DIMENSION not in coord.dims and BOUNDS_DIMENSION not in coord.dims
    }


def _on_calendar_cells_only(ds: xr.Dataset) -> bool:
    """Whether *ds* has none of the interval coordinates, and so only its labels to go on."""
    return not any(name in ds.coords for name in (*_INTERVAL_COORDINATES, "time_bounds"))


def _require_time_layout(ds: xr.Dataset) -> None:
    needed = (TIME_DIMENSION, *_INTERVAL_COORDINATES)
    missing = [c for c in needed if c not in ds.coords]
    if missing or TIME_DIMENSION not in ds.dims:
        raise ValueError(
            f"Dataset lacks the pySIPNET time coordinates {missing}. resample() needs the "
            "layout SIPNETOutput.xarray, SIPNETOutput.select or ClimateDrivers.xarray produce: "
            "'time' at the step end with 'time_step_start' and 'time_step_length' alongside. "
            "A record with none of the interval coordinates resamples on calendar cells."
        )
    if BOUNDS_DIMENSION in ds.dims and "time_bounds" not in ds.coords:
        raise ValueError("Dataset has a 'bounds' dimension but no 'time_bounds' coordinate.")
    expected_dtype = {TIME_DIMENSION: "M", "time_step_start": "M", "time_step_length": "m"}
    mistyped = {
        name: str(ds[name].dtype)
        for name, kind in expected_dtype.items()
        if ds[name].dtype.kind != kind
    }
    if mistyped:
        raise ValueError(
            f"The time coordinates have dtypes {mistyped}: 'time' and 'time_step_start' must "
            "be datetime64 and 'time_step_length' timedelta64, as pySIPNET builds them. A "
            "length in days is converted with pysipnet.dataset.days_to_timedelta."
        )
    # Order is not checked: a transposed time_bounds is rebuilt, never read.
    _require_intervals_on_time_alone(ds, {"time_bounds": {TIME_DIMENSION, BOUNDS_DIMENSION}})


def _require_intervals_on_time_alone(
    data: xr.Dataset | xr.DataArray, also: Mapping[str, set[str]] | None = None
) -> None:
    expected = {name: {TIME_DIMENSION} for name in _INTERVAL_COORDINATES} | dict(also or {})
    offenders = {
        name: tuple(map(str, data[name].dims))
        for name, dims in expected.items()
        if name in data.coords and set(data[name].dims) != dims
    }
    if offenders:
        raise ValueError(
            f"The interval coordinates vary along more than time: {offenders}. xarray gives "
            "them an extra dimension when runs on different time axes are concatenated, and "
            "then the runs no longer share one set of cells. Resample each run, or each "
            "site, separately; resample() drops the padding rows that selecting one leaves."
        )


def _drop_padding(data: _Xarray, valued: Callable[[Any], list[str]]) -> _Xarray:
    """*data* without its padding rows; *valued* names what holds a value on given rows.

    :func:`resample` consults only the variables it was asked for, so that one
    it will not return cannot refuse the call.
    """
    present = [name for name in _INTERVAL_COORDINATES if name in data.coords]
    if not present:
        return data
    _require_intervals_on_time_alone(data)
    no_interval = np.zeros(data.sizes[TIME_DIMENSION], dtype=bool)
    for name in present:
        no_interval |= np.isnat(data[name].values)
    if not no_interval.any():
        return data
    with_values = valued(data.isel({TIME_DIMENSION: no_interval}))
    if with_values:
        raise ValueError(
            f"{with_values} have values on {int(no_interval.sum())} rows whose "
            "time_step_start or time_step_length is NaT, so which cell those values belong "
            "to, and how much they weigh in a mean, is unknown. Rows of pure padding (NaT "
            "interval and every value missing) are dropped; these are not padding."
        )
    if no_interval.all():
        raise ValueError(
            "Every row's time_step_start or time_step_length is NaT, so there is no step "
            "to resample; a selection from a stack of runs matched a run with no record."
        )
    return data.isel({TIME_DIMENSION: ~no_interval})


def _valued_among(names: list[str]) -> Callable[[xr.Dataset], list[str]]:
    """Which of the variables *names* hold a value somewhere on the rows given."""
    return lambda rows: [name for name in names if bool(rows[name].notnull().any())]


def _check_not_upsampling(
    freq: str, offset: BaseOffset, cell_ends: np.ndarray, shortest_step_ns: int
) -> None:
    """The rule of :func:`check_not_upsampling`, on cells and a step already measured."""
    first_edge = (pd.Timestamp(cell_ends[0]) - offset).to_datetime64().astype("datetime64[ns]")
    edges = np.concatenate([[first_edge], cell_ends]).astype("int64")
    longest_cell = int(np.diff(edges).max())
    tolerance = int(STEP_TOLERANCE.astype("timedelta64[ns]").astype("int64"))
    if shortest_step_ns > longest_cell + tolerance:
        raise ValueError(
            f"freq={freq!r} makes cells of at most {pd.Timedelta(longest_cell, 'ns')}, shorter "
            f"than the shortest step in the data ({pd.Timedelta(shortest_step_ns, 'ns')}). "
            "Every step would land in a cell of its own and come back unchanged, labeled as "
            f"resampled to {freq!r}. Pass a frequency at least as long as the steps."
        )


def _cell_of_each_step(ds: xr.Dataset, cell_ends: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The index of the cell each step falls in, checked against xarray's grouping.

    A step belongs to the right-closed cell its end falls in, which is the
    first cell ending at or after it.  xarray groups the values; this groups
    the lengths, which must be summed outside xarray (see
    :func:`_summed_lengths`).  The two must agree step for step, so the per-cell
    counts are compared rather than trusted.
    """
    ends = ds[TIME_DIMENSION].values.astype("datetime64[ns]")
    cell = np.searchsorted(cell_ends, ends, "left")
    if not np.array_equal(np.bincount(cell, minlength=cell_ends.size), counts):
        raise RuntimeError(
            "pySIPNET's cell assignment disagrees with xarray's resample grouping, so the "
            "combined lengths would not describe the combined values. Please report this."
        )
    return cell


def _summed_lengths(cell: np.ndarray, n_cells: int, length_ns: np.ndarray) -> np.ndarray:
    """The declared step lengths summed per cell, exactly, as ``int64`` nanoseconds.

    Summed here rather than by xarray, which casts to float64 to fill empty
    cells; a float64 holds whole nanoseconds exactly only to about 104 days.
    """
    totals = np.zeros(n_cells, dtype="int64")
    np.add.at(totals, cell, length_ns)
    return totals


def _methods_for(
    ds: xr.Dataset, how: str | Mapping[str, str]
) -> dict[str, tuple[str, VariableKind]]:
    """Resolve *how* into one validated method per variable to keep, with its kind."""
    if isinstance(how, str):
        requested = {str(name): how for name in ds.data_vars}
    elif isinstance(how, Mapping):
        requested = {str(k): v for k, v in how.items()}
        unknown = [name for name in requested if name not in ds.data_vars]
        if unknown:
            raise KeyError(
                f"{unknown} are not data variables of this Dataset; it has "
                f"{list(map(str, ds.data_vars))}. The keys of how are the Dataset's own "
                "variable names, which for pySIPNET output are the registry names "
                "('net_ecosystem_exchange', not 'nee')."
            )
    else:
        raise TypeError(
            f"how must be 'sum', 'mean', 'last' or a mapping from variable to one of those, "
            f"not {type(how).__name__}."
        )

    methods: dict[str, tuple[str, VariableKind]] = {}
    for name, method in requested.items():
        kind = variable_kind(ds[name])
        check_resampling_method(kind, method, name=name)
        if TIME_DIMENSION not in ds[name].dims:
            raise ValueError(
                f"{name!r} has dims {tuple(map(str, ds[name].dims))} and no {TIME_DIMENSION!r}, "
                "so there is nothing to resample; leave it out of how."
            )
        methods[name] = (method, kind)
    if not methods:
        raise ValueError("Nothing to resample: the Dataset has no data variables.")
    return methods


# ── Records without interval coordinates ─────────────────────────────────────


def _resample_on_calendar_cells(
    ds: xr.Dataset, freq: str, how: str | Mapping[str, str]
) -> xr.Dataset:
    """:func:`resample` for a record with ``time`` labels and no interval coordinates."""
    import xarray as xr

    _require_increasing_labels(ds)
    offset = check_frequency(freq)
    methods = _methods_for(ds, how)
    spacing = np.diff(_labels_ns(ds))
    averaged = [name for name, (method, _) in methods.items() if method == "mean"]
    tolerance = int(STEP_TOLERANCE.astype("timedelta64[ns]").astype("int64"))
    if averaged and spacing.size and int(spacing.max() - spacing.min()) > tolerance:
        raise ValueError(
            f"Cannot average {averaged} over calendar cells: the record has no "
            "time_step_length to weight its steps by, and its time labels are not equally "
            f"spaced (gaps from {pd.Timedelta(int(spacing.min()), 'ns')} to "
            f"{pd.Timedelta(int(spacing.max()), 'ns')}), so an equally weighted mean would "
            "weigh short steps as much as long ones. Attach time_step_start and "
            "time_step_length, or resample a record that carries them."
        )
    cell_ends, n_steps = _steps_per_cell(ds, freq)
    if spacing.size:
        _check_not_upsampling(freq, offset, cell_ends, int(spacing.min()))
    keep = n_steps > 0
    equal_weights = _as_data_array(np.ones(ds.sizes[TIME_DIMENSION]), ds)
    combined = _combined(ds, freq, methods, equal_weights, keep)

    coords: dict[str, Any] = {
        TIME_DIMENSION: (
            TIME_DIMENSION,
            cell_ends[keep],
            {
                **TIME_AXIS_ATTRIBUTES,
                "long_name": "End of calendar cell",
                "description": (
                    f"The right edge of the calendar cell of {freq} the values were combined "
                    "over, on the input's clock. The record says nothing about where its "
                    "steps end, so the last one in a cell may end before its edge, and the "
                    "first and last cells may be only partly covered."
                ),
                "time_zone": ds[TIME_DIMENSION].attrs.get("time_zone", TIME_ZONE_UNDECLARED),
            },
        ),
        **_coordinates_off_time(ds),
    }
    data_vars = _resampled_variables(
        ds,
        combined,
        methods,
        over=f"over calendar cells of {freq}, labeled at each cell's right edge",
        weighting="weighted equally",
    )
    attrs = dict(ds.attrs)
    # Whatever axis the record had, these no longer describe the calendar cells.
    for stale in ("time_axis_source", "time_step_length_source"):
        attrs.pop(stale, None)
    attrs["time_convention"] = (
        f"'time' is the right edge of each right-closed calendar cell of {freq}; the "
        "record has no step intervals, so the cells' coverage is not recorded."
    )
    attrs["resampling_frequency"] = freq
    return unfilled_coordinates(xr.Dataset(data_vars, coords=coords, attrs=attrs))


def _require_increasing_labels(ds: xr.Dataset | xr.DataArray) -> None:
    """A record's ``time`` labels are datetimes that strictly increase, and there are some."""
    if TIME_DIMENSION not in ds.dims or TIME_DIMENSION not in ds.coords:
        raise ValueError(
            f"The data has no {TIME_DIMENSION!r} coordinate along a {TIME_DIMENSION!r} "
            "dimension, so there is no time axis to resample."
        )
    labels = ds[TIME_DIMENSION].values
    if labels.dtype.kind != "M":
        raise ValueError(
            f"'time' has dtype {labels.dtype}; resample() needs datetime64 labels to place "
            "each step in a calendar cell."
        )
    if labels.size == 0:
        raise ValueError("The record has no timesteps to resample.")
    if np.isnat(labels).any():
        raise ValueError(
            f"{int(np.isnat(labels).sum())} 'time' labels are NaT, so which cell those rows "
            "belong to is unknown."
        )
    spacing = np.diff(_labels_ns(ds))
    if (spacing <= 0).any():
        row = int(np.flatnonzero(spacing <= 0)[0]) + 1
        raise ValueError(
            f"The 'time' labels do not strictly increase: row {row} ({labels[row]}) does not "
            f"follow row {row - 1} ({labels[row - 1]}). Sort the record, and drop or combine "
            "duplicates; two rows sharing a label would be combined as though they were "
            "consecutive steps."
        )


__all__ = [
    "check_frequency",
    "check_not_upsampling",
    "check_resampling_method",
    "drop_padding",
    "resample",
    "resampled_attributes",
]
