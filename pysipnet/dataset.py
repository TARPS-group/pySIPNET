"""Turn a per-timestep DataFrame into a self-describing :class:`xarray.Dataset`.

Shared by :class:`~pysipnet.output.SIPNETOutput` and
:class:`~pysipnet.climate.ClimateDrivers`.  When an output knows the drivers
it was run on, both are built on the drivers' own time axis, row for row, so
they can be aligned or merged directly.

SIPNET has no time zone
-----------------------
SIPNET has no clock of its own.  It never computes solar geometry, and it
reads a row's ``year``, ``day`` and ``time`` only to echo them into the
output, to compare against ``leafOnDay`` / ``leafOffDay`` and to check restart
boundaries; every flux is integrated over the row's declared ``length``.  A
row's labels are therefore the start of its step **on whatever clock the
climate drivers use**, and SIPNET passes that clock through untouched.  It is
the :class:`~pysipnet.climate.ClimateDrivers` that say which clock it is,
through their optional ``time_zone``; this module only records the
declaration, on the ``time`` coordinate and on the Dataset, and converts
nothing.  An undeclared clock is recorded as ``"undeclared"``.

Time convention
---------------
The ``time`` coordinate is the **end** of each timestep.  SIPNET labels its
rows with the *start* (``year``, ``day_of_year``, ``hour_of_day``), but it
writes pools after the step has been applied and fluxes as totals over the
step, so the end is the one instant at which every row's Climate and Forecast
(CF) ``cell_methods`` is literally true: a pool is ``time: point`` at ``time``,
a total is ``time: sum`` over the step's ``time_bounds``.  This is the
convention of the CLM and ELM land models, and it is the natural one for data
assimilation, where ``ds.sel(time=t)`` should give the state valid at the
analysis time and the fluxes over the interval that led up to it.

Every row also describes the interval it covers: ``timestep_start`` and
``timestep_length`` coordinates, and a CF ``time_bounds`` variable giving
``[timestep_start, time]``.  ``time`` carries the standard ``bounds``
attribute naming it, which is what lets CF-aware tooling decide whether an
observation falls inside a step and how to combine steps.  ``year``,
``day_of_year`` and ``hour_of_day`` are the start of the step, and say so in
their attributes.

Where the axis comes from
-------------------------
When the climate drivers are known, the axis is built from **them**: their
start times, their step lengths, their clock.  SIPNET writes exactly one output
row per climate row, echoing that row's labels, so this is the same axis the
labels in the output describe, without the rounding: SIPNET prints
``hour_of_day`` to 0.01 h, which is up to 18 s off the driver it came from and
cannot represent a step such as 20 minutes at all.  An output whose row count
differs from the drivers', or whose printed labels do not round from the
drivers', is refused rather than aligned by guesswork.  ``year``,
``day_of_year`` and ``hour_of_day`` then carry the drivers' unrounded values;
the output's :attr:`~pysipnet.output.SIPNETOutput.pandas` view still has the
printed ones, being a faithful view of the file.

With no drivers — an output file re-opened on its own — the axis falls back
to the printed labels, and the step lengths are inferred from consecutive
labels (exact for every step but the last, which is assumed to repeat the one
before it).  A single row with no declared length cannot be placed on the axis
at all and is refused.  The Dataset's ``time_axis_source`` and
``timestep_length_source`` attributes say which happened.

Labels and lengths must agree
-----------------------------
SIPNET never checks that a row's start plus its length is the next row's
start.  :func:`check_step_continuity` does, once for every set of drivers:
when a :class:`~pysipnet.climate.ClimateDrivers` loads its data (see
:meth:`~pysipnet.climate.ClimateDrivers.validate`), or, for a bare frame that
carries its own ``timestep_length`` column, when its axis is built.  An axis
built from a ``ClimateDrivers`` does not repeat it.  The rule:

- an **overlap**, a row starting more than a minute before the previous one
  ends, is refused;
- a **drift**, labels that wander from the running sum of the declared lengths
  by more than five minutes, is refused, because a per-step tolerance loose
  enough for rounded lengths cannot see a slow drift at all;
- a **gap**, a row starting more than a minute after the previous one ends, is
  a genuine hole in the record.  It is allowed, shows as a gap between
  ``time_bounds``, and restarts the drift reconstruction.

The minute is what three-decimal day lengths need: SIPNET's own Niwot fixture
declares ``0.292`` days where its labels say exactly 7 hours, 29 seconds
apart, and its worst step is 43 s off.  Over its 800 rows the labels never
wander more than 101 s from the lengths' running sum.  An output re-opened
without its drivers has no declared lengths to check its labels against.

The end of a step is its start plus its **declared** length, which is the
duration SIPNET integrates fluxes over — except that when the next row starts
within the same minute of that instant, the end snaps to the next start, so
that rounded lengths do not leave ``time`` a few seconds off the clock and the
cells overlapping or gapped by the rounding error.  Because the continuity
check has already refused anything the snap could hide, the snap only ever
moves a boundary by an amount that is rounding.  ``timestep_length`` is never
adjusted, because the declared length is what the model actually used; only
the placement of the boundary is.

``time_bounds`` is a coordinate rather than a data variable, so that selecting
variables gives a Dataset whose ``data_vars`` are exactly what was asked for.
The cost is a second dimension, ``bounds``, which appears in ``ds.sizes`` and
makes ``ds.to_dataframe()`` produce two rows per timestep; use the container's
``.pandas`` view for a flat frame.  A single variable cannot carry it either:
``ds["nee"]`` keeps ``time``'s attributes, ``bounds`` included, but not the
two-dimensional ``time_bounds`` they name.  The DataArrays pySIPNET hands back
(``output["nee"]``, :func:`~pysipnet.resample.resample` of a DataArray) drop
the attribute; :func:`without_absent_bounds` does the same for one taken with
plain xarray indexing.

Every data variable carries the attributes its registry spec provides,
including ``kind`` and ``time_reference`` in words.  The Dataset declares
``Conventions = "CF-1.11"``; no coordinate is given a ``_FillValue`` on
encoding, as CF requires.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from pysipnet.variables import TIME_COORDINATE_NAMES, VariableKind

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from pysipnet.climate import ClimateDrivers

_Xarray = TypeVar("_Xarray", "xr.DataArray", "xr.Dataset")

TIME_DIMENSION = "time"
BOUNDS_DIMENSION = "bounds"

TIME_AXIS_ATTRIBUTES: dict[str, str] = {"standard_name": "time", "axis": "T"}
"""What identifies ``time`` as the time axis to Climate and Forecast readers."""

CF_CONVENTIONS = "CF-1.11"

TIME_CONVENTION = (
    "The 'time' coordinate is the END of each timestep, the instant at which state "
    "variables are valid. 'timestep_start' is the start, which is how SIPNET labels its "
    "rows ('year', 'day_of_year', 'hour_of_day'). Fluxes are totals over "
    "[timestep_start, time], also given as 'time_bounds'. Each variable's 'kind', "
    "'time_reference' and 'cell_methods' attributes say which applies to it. SIPNET "
    "has no time zone: the times are on whatever clock the climate drivers use, which "
    "'time_zone' names when the drivers declare it."
)

# The time_zone attribute when the drivers do not say which clock they use.
TIME_ZONE_UNDECLARED = "undeclared"

# Where the step lengths in a TimeAxis came from, recorded on the Dataset so a
# consumer can tell a measured interval from a reconstructed one.
STEP_LENGTH_FROM_DRIVERS = "climate drivers"
STEP_LENGTH_INFERRED = (
    "inferred from consecutive timestamps; the last step repeats the one before it"
)

# Where the step starts in a TimeAxis came from.
TIME_AXIS_FROM_DRIVERS = "climate drivers"
TIME_AXIS_FROM_PRINTED_LABELS = (
    "row labels printed in the SIPNET output, which rounds hour_of_day to 0.01 h"
)

NS_PER_DAY = 86_400_000_000_000
_NS_PER_HOUR = 3_600_000_000_000

# How far a step's declared end may miss the next row's start and still be
# rounding. Three-decimal day lengths are off by at most 43 s.
STEP_TOLERANCE = np.timedelta64(60, "s")

# How far labels may wander from the running sum of the declared lengths. The
# Niwot fixture's never exceed 101 s; a 2.5 s-per-step drift passes it in a
# little over a hundred steps.
DRIFT_TOLERANCE = np.timedelta64(300, "s")

# SIPNET prints hour_of_day as %5.2f, so a label read back from its output is
# within 0.005 h of the driver it came from; the rest is room for the 6
# significant figures pySIPNET writes the driver with.
_PRINTED_HOUR_TOLERANCE = 0.005 + 1e-4

# datetime64[ns] spans 1677-09-21 to 2262-04-11. A year outside this range would
# wrap around silently, so it is refused instead.
_MIN_YEAR = 1678
_MAX_YEAR = 2261


def _finite(values: np.ndarray, column: str) -> np.ndarray:
    """Refuse non-finite values before they reach an integer cast.

    ``np.rint(nan).astype("int64")`` is an undefined cast: numpy warns and
    produces whatever the platform's instruction happens to yield — 0 on
    arm64, INT64_MIN on x86 — so the same file would give different timestamps
    on different machines.
    """
    if not np.isfinite(values).all():
        bad = int(np.argmin(np.isfinite(values)))
        raise ValueError(
            f"{column!r} contains a non-finite value ({values[bad]!r}) at row {bad}. "
            "Timestamps cannot be built from missing data; fix or drop the row."
        )
    return values


def timestep_start(df: pd.DataFrame) -> np.ndarray:
    """Calendar time at the start of each row, as ``datetime64[ns]``.

    Built from ``year``, ``day_of_year`` and ``hour_of_day`` with integer
    ``datetime64`` arithmetic rather than a string round-trip through
    :func:`pandas.to_datetime`.  The round-trip costs roughly an order of
    magnitude more, and is paid once per Dataset.  The two agree to the
    nanosecond, except that this one rounds a fractional hour to the nearest
    nanosecond where :func:`pandas.to_datetime` truncates the float error.
    """
    missing = [c for c in TIME_COORDINATE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot build a time coordinate: data is missing {missing}. "
            "Was the file read without a header row?"
        )

    year = _finite(df["year"].to_numpy().astype("float64"), "year").astype("int64")
    day_of_year = _finite(df["day_of_year"].to_numpy().astype("float64"), "day_of_year").astype(
        "int64"
    )
    hour_of_day = _finite(df["hour_of_day"].to_numpy().astype("float64"), "hour_of_day")

    if len(year) and (year.min() < _MIN_YEAR or year.max() > _MAX_YEAR):
        raise ValueError(
            f"Years {year.min()}-{year.max()} fall outside {_MIN_YEAR}-{_MAX_YEAR}, the range "
            "nanosecond datetimes can represent. A year beyond it would wrap silently to an "
            "unrelated date rather than fail."
        )

    january_first = (year - 1970).astype("datetime64[Y]").astype("datetime64[ns]")
    days = (day_of_year - 1).astype("timedelta64[D]").astype("timedelta64[ns]")
    hours = np.rint(hour_of_day * _NS_PER_HOUR).astype("int64").astype("timedelta64[ns]")
    return january_first + days + hours


def sipnet_row_labels(start: np.ndarray) -> dict[str, np.ndarray]:
    """``year``, ``day_of_year`` and ``hour_of_day`` for step starts, as SIPNET writes them."""
    import pandas as pd

    index = pd.DatetimeIndex(start)
    hour = (
        index.hour
        + index.minute / 60
        + index.second / 3600
        + index.microsecond / 3.6e9
        + index.nanosecond / 3.6e12
    )
    return {
        "year": index.year.to_numpy().astype("int64"),
        "day_of_year": index.dayofyear.to_numpy().astype("int64"),
        "hour_of_day": np.asarray(hour, dtype="float64"),
    }


def days_to_timedelta(days: np.ndarray) -> np.ndarray:
    """Lengths in days as ``timedelta64[ns]``, rounded to the nearest nanosecond."""
    values = _finite(np.asarray(days, dtype=float), "timestep_length")
    return np.rint(values * NS_PER_DAY).astype("int64").view("timedelta64[ns]")


def _gaps_in_days(start: np.ndarray) -> np.ndarray:
    """Differences between consecutive starts, refusing any that do not move forward."""
    gaps = np.diff(start).astype("timedelta64[ns]").astype("int64") / NS_PER_DAY
    if (gaps <= 0).any():
        row = int(np.argmax(gaps <= 0))
        raise ValueError(
            f"Timestamps do not increase at row {row + 1}: the step starting there is "
            f"{gaps[row]} days after the one before it. Rows must be in strictly ascending "
            "time order for the interval each one covers to mean anything."
        )
    return gaps


def _infer_step_length(start: np.ndarray) -> np.ndarray | None:
    """Step lengths in days from consecutive starts, or ``None`` if undeterminable.

    SIPNET labels each row with the start of its step and the steps are
    contiguous, so a difference of starts is the exact length of every step but
    the last, which has no successor to measure against and is assumed to repeat
    its predecessor.  On a record with irregular steps that assumption is simply
    wrong for the final row, which is one reason the drivers' own lengths are
    preferred whenever they are available.
    """
    if len(start) < 2:
        return None
    gaps = _gaps_in_days(start)
    return np.concatenate([gaps, gaps[-1:]])


def _as_ns(duration: np.timedelta64) -> int:
    return int(duration.astype("timedelta64[ns]").astype("int64"))


def describe_duration(duration: np.timedelta64 | float) -> str:
    """A duration, given as ``timedelta64`` or in nanoseconds, in the unit a reader would use."""
    if isinstance(duration, np.timedelta64):
        duration = _as_ns(duration)
    seconds = abs(float(duration)) / 1e9
    if seconds < 120:
        return f"{seconds:.3g} s"
    if seconds < 7200:
        return f"{seconds / 60:.3g} min"
    if seconds < 172_800:
        return f"{seconds / 3600:.3g} h"
    return f"{seconds / 86_400:.3g} days"


def check_step_continuity(
    start: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    """Check that each row starts where the one before it ends, and return where it does not.

    Compares every row's label with the previous row's start plus its declared
    length, and with the running sum of the declared lengths since the last
    gap, because a per-step tolerance loose enough for rounded lengths is blind
    to a slow drift.  See the module docstring for the rule and the tolerances.

    Parameters
    ----------
    start:
        Step starts, ``datetime64[ns]``, in ascending order.
    length:
        Declared step lengths, ``timedelta64[ns]``, one per row.

    Returns
    -------
    numpy.ndarray
        The rows followed by a gap, where the next row starts more than
        :data:`STEP_TOLERANCE` after this one ends.  A gap is a genuine hole in
        the record rather than an error.

    Raises
    ------
    ValueError
        On an overlap, a row starting more than :data:`STEP_TOLERANCE` before
        the previous one ends, or a drift, a label further than
        :data:`DRIFT_TOLERANCE` from the running sum of the lengths before it.
    """
    if len(start) < 2:
        return np.empty(0, dtype=np.intp)

    step_tolerance = _as_ns(STEP_TOLERANCE)
    drift_tolerance = _as_ns(DRIFT_TOLERANCE)

    end = start + length
    # Positive: the next row starts after this one ends. Negative: before.
    mismatch = (start[1:] - end[:-1]).astype("timedelta64[ns]").astype("int64")

    overlaps = np.flatnonzero(mismatch < -step_tolerance)
    gaps = mismatch > step_tolerance
    running = np.cumsum(np.where(gaps, 0, mismatch))
    last_gap = np.maximum.accumulate(np.where(gaps, np.arange(len(mismatch)), -1))
    # drift[i]: how far row i + 1's label is from the start of its gap-free
    # stretch plus the declared lengths of every row in between.
    drift = running - np.where(last_gap >= 0, running[np.maximum(last_gap, 0)], 0)
    drifted = np.flatnonzero(np.abs(drift) > drift_tolerance)

    # Report whichever comes first: a drift usually ends in an overlap when
    # the labels are reset, and the drift is the cause.
    if len(overlaps) and (not len(drifted) or overlaps[0] <= drifted[0]):
        row = int(overlaps[0])
        raise ValueError(
            f"Row {row + 1} starts at {start[row + 1]}, {describe_duration(mismatch[row])} "
            f"before row {row} ends at {end[row]} (its start, {start[row]}, plus its declared "
            f"length of {length[row] / np.timedelta64(1, 'D'):.6g} days); {len(overlaps)} "
            "row(s) overlap the one before like this. Two rows cannot both cover the same "
            "instant. SIPNET does not check this — it integrates each row over its declared "
            "length and uses the labels only to name the rows — so either the labels or the "
            "lengths are wrong, or the rows are not one continuous record."
        )
    if len(drifted):
        row = int(drifted[0]) + 1
        first = int(last_gap[row - 1]) + 1
        amount = int(drift[row - 1])
        reconstructed = start[row] - np.timedelta64(amount, "ns")
        overlap_note = (
            f" Later, {len(overlaps)} row(s) also start before the previous one ends."
            if len(overlaps)
            else ""
        )
        raise ValueError(
            f"The labels drift from the declared step lengths: row {row} is labelled "
            f"{start[row]}, but row {first} ({start[first]}) plus the declared lengths of the "
            f"{row - first} row(s) from there comes to {reconstructed}, "
            f"{describe_duration(amount)} {'earlier' if amount > 0 else 'later'}. Each label "
            f"moves on average {describe_duration(amount / (row - first))} "
            f"{'more' if amount > 0 else 'less'} than the length before it.{overlap_note} "
            "SIPNET integrates each row over its declared length and never reads the labels "
            "back, so a time axis built from them would put steps where the model did not. "
            "Either the labels do not advance by the step length, or the lengths are rounded "
            "too coarsely for this many steps; write labels and lengths that agree. The "
            f"tolerance is {describe_duration(drift_tolerance)} cumulative and "
            f"{describe_duration(step_tolerance)} per step."
        )

    return np.flatnonzero(gaps)


def _step_end(start: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Start plus declared length, snapped to the next row's start within :data:`STEP_TOLERANCE`."""
    end = start + length
    if len(start) > 1:
        next_start = start[1:]
        off = np.abs((next_start - end[:-1]).astype("timedelta64[ns]"))
        snap = off <= STEP_TOLERANCE
        end[:-1] = np.where(snap, next_start, end[:-1])
    if len(start) and (end <= start).any():
        row = int(np.argmax(end <= start))
        raise ValueError(
            f"Row {row} ends at or before it starts ({end[row]} vs {start[row]}): its declared "
            "step length is within rounding of zero and the next row starts before it ends, "
            "so there is no interval for it to cover."
        )
    return end


def assemble_time_coords(
    *,
    start: np.ndarray,
    end: np.ndarray,
    length: np.ndarray,
    attributes_for: Callable[[str], dict[str, Any]],
    length_source: str,
    time_zone: str,
) -> dict[str, Any]:
    """The full set of time coordinates for rows with the given starts, ends and lengths.

    ``length`` is ``timedelta64[ns]``; ``time_zone`` is the clock the drivers
    declare, or :data:`TIME_ZONE_UNDECLARED`.  Used both for a fresh Dataset
    and for one that :func:`pysipnet.resample.resample` has coarsened, so the
    two carry the same coordinates with the same attributes.
    """
    coords: dict[str, Any] = {
        TIME_DIMENSION: (
            TIME_DIMENSION,
            end,
            {
                **TIME_AXIS_ATTRIBUTES,
                "long_name": "End of timestep",
                "description": (
                    "Calendar time at the end of the timestep, on the climate drivers' clock: "
                    "the instant state variables are valid, and the close of the interval "
                    "fluxes are totals over."
                ),
                "bounds": "time_bounds",
                "time_zone": time_zone,
            },
        ),
        "timestep_start": (
            TIME_DIMENSION,
            start,
            {
                "long_name": "Start of timestep",
                "description": "Calendar time at the start of the timestep, on the climate "
                "drivers' clock, as the drivers label the row.",
            },
        ),
        "timestep_length": (
            TIME_DIMENSION,
            length,
            {
                "long_name": "Timestep length",
                "description": "Duration of the timestep, as declared to SIPNET.",
                "kind": VariableKind.TIMESTEP_TOTAL.value,
                "cell_methods": "time: sum",
                "source": length_source,
            },
        ),
        "time_bounds": (
            (TIME_DIMENSION, BOUNDS_DIMENSION),
            np.stack([start, end], axis=1),
            {
                "long_name": "Timestep bounds",
                "description": "The interval [timestep_start, time] each row covers, in the "
                "Climate and Forecast conventions' bounds form.",
            },
        ),
    }
    for name, values in sipnet_row_labels(start).items():
        coords[name] = (TIME_DIMENSION, values, attributes_for(name))
    return coords


@dataclass(frozen=True)
class TimeAxis:
    """The time coordinates every view of one run's rows shares.

    Built once and reused, because it does not depend on which variables are
    selected: every view of a run has the same rows.
    """

    coords: dict[str, Any]
    n_rows: int
    step_length_source: str
    axis_source: str
    time_zone: str


def _declared_lengths(values: np.ndarray) -> np.ndarray:
    """Step lengths in days, refusing a value no interval can have."""
    length = _finite(np.asarray(values, dtype=float), "timestep_length")
    if len(length) and length.min() <= 0:
        row = int(np.argmin(length))
        raise ValueError(
            f"timestep_length is {length[row]} days at row {row}. A step must have a "
            "positive duration for the interval it covers to mean anything."
        )
    return length


def _check_labels_match(df: pd.DataFrame, drivers: pd.DataFrame) -> None:
    """Refuse rows whose labels do not round from the drivers' they supposedly echo.

    SIPNET writes one output row per climate row with that row's ``year`` and
    ``day`` verbatim and its ``time`` to 0.01 h, so anything further apart is
    an output of some other drivers.
    """
    missing = [c for c in TIME_COORDINATE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot check the rows against their climate drivers: data is missing {missing}. "
            "Was the file read without a header row?"
        )
    same_year = df["year"].to_numpy(dtype=float) == drivers["year"].to_numpy(dtype=float)
    same_day = df["day_of_year"].to_numpy(dtype=float) == drivers["day_of_year"].to_numpy(
        dtype=float
    )
    hour_off = np.abs(
        df["hour_of_day"].to_numpy(dtype=float) - drivers["hour_of_day"].to_numpy(dtype=float)
    )
    agree = same_year & same_day & (hour_off <= _PRINTED_HOUR_TOLERANCE)
    if not agree.all():
        row = int(np.argmin(agree))
        printed = df.iloc[row]
        driver = drivers.iloc[row]
        raise ValueError(
            f"Row {row} is labelled year {printed['year']:g}, day {printed['day_of_year']:g}, "
            f"hour {printed['hour_of_day']:g}, but row {row} of the climate drivers is year "
            f"{driver['year']:g}, day {driver['day_of_year']:g}, hour "
            f"{driver['hour_of_day']:g}. SIPNET echoes each climate row's labels into its "
            "output row, to 0.01 h, so these rows were not produced from these drivers."
        )


def build_time_axis(
    df: pd.DataFrame,
    *,
    attributes_for: Callable[[str], dict[str, Any]],
    drivers: ClimateDrivers | None = None,
) -> TimeAxis:
    """Build the shared time coordinates for a frame's rows.

    The step starts and lengths come from exactly one of three places:

    1. *drivers*, when given.  Their starts, their lengths and their declared
       ``time_zone`` become the axis, and *df*'s own labels are only checked
       against them, row for row.  Their labels and lengths are not checked
       against each other again: a :class:`~pysipnet.climate.ClimateDrivers`
       did that when its data was loaded.
    2. *df*'s own ``timestep_length`` column, when it has one: the frame
       carries its whole clock, as a ``.clim`` file does.  Nothing has checked
       it yet, so :func:`check_step_continuity` runs here.
    3. Otherwise the lengths are inferred from consecutive labels, which
       agree with them by construction, so there is nothing to check; see
       :func:`_infer_step_length`.

    Parameters
    ----------
    df:
        The rows to place.  Must contain ``year``, ``day_of_year`` and
        ``hour_of_day``.
    attributes_for:
        Returns the ``.attrs`` for a column name; ``{}`` for unknown columns.
    drivers:
        The climate drivers the rows were produced from, one row each.

    Raises
    ------
    ValueError
        If the rows cannot be placed: a row count or labels that do not match
        the drivers, timestamps that do not increase, or a frame's own lengths
        that disagree with its labels.
    """
    time_zone = None
    if drivers is not None:
        clock = drivers.pandas
        if len(clock) != len(df):
            raise ValueError(
                f"The data has {len(df)} rows but its climate drivers have {len(clock)}. SIPNET "
                "writes exactly one output row per climate row, so these are not the drivers "
                "the rows were produced from, or the run stopped early. Refusing to guess how "
                "they line up."
            )
        if clock is not df:
            _check_labels_match(df, clock)
        start = timestep_start(clock)
        length = clock["timestep_length"].to_numpy(dtype=float)
        axis_source = TIME_AXIS_FROM_DRIVERS
        length_source = STEP_LENGTH_FROM_DRIVERS
        time_zone = drivers.time_zone
    elif "timestep_length" in df.columns:
        start = timestep_start(df)
        length = _declared_lengths(df["timestep_length"].to_numpy())
        if len(start) > 1:
            _gaps_in_days(start)
        check_step_continuity(start, days_to_timedelta(length))
        axis_source = TIME_AXIS_FROM_DRIVERS
        length_source = STEP_LENGTH_FROM_DRIVERS
    else:
        start = timestep_start(df)
        inferred = _infer_step_length(start)
        if inferred is None:
            raise ValueError(
                f"Cannot place {len(df)} row(s) on a time axis without knowing the step length: "
                "the 'time' coordinate is the end of each step, and with fewer than two rows "
                "there is nothing to infer it from. Pass the climate drivers the rows were "
                "produced from, or give the frame a timestep_length column (days)."
            )
        length = inferred
        axis_source = TIME_AXIS_FROM_PRINTED_LABELS
        length_source = STEP_LENGTH_INFERRED

    length_td = days_to_timedelta(length)
    zone = TIME_ZONE_UNDECLARED if time_zone is None else time_zone
    coords = assemble_time_coords(
        start=start,
        end=_step_end(start, length_td),
        length=length_td,
        attributes_for=attributes_for,
        length_source=length_source,
        time_zone=zone,
    )
    return TimeAxis(
        coords=coords,
        n_rows=len(df),
        step_length_source=length_source,
        axis_source=axis_source,
        time_zone=zone,
    )


def unfilled_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Mark every coordinate to be encoded without a ``_FillValue``, as CF requires.

    xarray otherwise writes ``_FillValue = NaN`` on any floating or datetime
    coordinate, which CF forbids on coordinate variables.  Modifies the
    encoding in place and returns the same Dataset for chaining.
    """
    for name in ds.coords:
        ds[name].encoding["_FillValue"] = None
    return ds


def without_absent_bounds(data: _Xarray) -> _Xarray:
    """*data* without a ``bounds`` attribute on ``time`` that names a variable it lacks.

    CF readers look the name up, so a ``bounds`` naming nothing is an error in
    the file, not a harmless leftover.  Returns *data* itself when there is
    nothing to drop, and otherwise a shallow copy, so the Dataset an array was
    taken from keeps its own attribute.
    """
    if TIME_DIMENSION not in data.coords:
        return data
    bounds = data[TIME_DIMENSION].attrs.get("bounds")
    # A Dataset may hold the bounds as a data variable; a DataArray only as a coordinate.
    if bounds is None or bounds in getattr(data, "variables", data.coords):
        return data
    data = data.copy(deep=False)
    del data[TIME_DIMENSION].attrs["bounds"]
    return data


def dataset_from_dataframe(
    df: pd.DataFrame,
    axis: TimeAxis,
    *,
    attributes_for: Callable[[str], dict[str, Any]],
    source: str,
    extra_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Assemble a Dataset from a frame's data columns and an already-built *axis*.

    The half of :func:`build_xarray_dataset` that does not depend on the time
    columns: pass an axis built once by :func:`build_time_axis` and this adds
    the data variables to it. Every column that is not part of the axis becomes
    a data variable, so a frame holding only some variables gives a Dataset
    holding only those — the whole point of building the axis separately.
    """
    import xarray as xr

    if len(df) != axis.n_rows:
        raise ValueError(f"Frame has {len(df)} rows but the time axis was built for {axis.n_rows}.")

    # The time columns and a timestep_length column are the axis, not data.
    as_coordinate = {*TIME_COORDINATE_NAMES, "timestep_length"}
    data_vars = {
        name: (TIME_DIMENSION, df[name].to_numpy(), attributes_for(name))
        for name in df.columns
        if name not in as_coordinate
    }

    attrs: dict[str, Any] = {
        "Conventions": CF_CONVENTIONS,
        "source": source,
        "time_convention": TIME_CONVENTION,
        "time_zone": axis.time_zone,
        "time_axis_source": axis.axis_source,
        "timestep_length_source": axis.step_length_source,
    }
    attrs.update(extra_attrs or {})

    return unfilled_coordinates(xr.Dataset(data_vars, coords=axis.coords, attrs=attrs))


def build_xarray_dataset(
    df: pd.DataFrame,
    *,
    attributes_for: Callable[[str], dict[str, Any]],
    drivers: ClimateDrivers | None = None,
    source: str,
    extra_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Build a Dataset with one ``time`` dimension from a per-timestep DataFrame.

    The whole job in one call: :func:`build_time_axis` followed by
    :func:`dataset_from_dataframe`.  Call those two separately when several
    Datasets are built from the same rows, so the time axis is paid for once —
    which is what :class:`~pysipnet.output.SIPNETOutput` does.

    Parameters
    ----------
    df:
        Must contain ``year``, ``day_of_year`` and ``hour_of_day`` columns
        marking the start of each step.  Every other column becomes a data
        variable.
    attributes_for:
        Returns the ``.attrs`` for a column name; ``{}`` for unknown columns.
    drivers:
        The climate drivers the rows came from; see :func:`build_time_axis`
        for where the axis comes from with and without them.
    source:
        Free text for the Dataset's ``source`` attribute.
    extra_attrs:
        Further Dataset-level attributes, e.g. run provenance.
    """
    import xarray as xr

    if df.empty:
        return xr.Dataset()

    axis = build_time_axis(df, attributes_for=attributes_for, drivers=drivers)
    return dataset_from_dataframe(
        df, axis, attributes_for=attributes_for, source=source, extra_attrs=extra_attrs
    )
