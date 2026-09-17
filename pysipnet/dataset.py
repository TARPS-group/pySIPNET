"""Turn a per-timestep DataFrame into a self-describing :class:`xarray.Dataset`.

Shared by :class:`~pysipnet.output.SIPNETOutput` and
:class:`~pysipnet.climate.ClimateDrivers`, so outputs and drivers have the same
time axis and can be aligned or merged directly.

Time convention
---------------
The ``time`` coordinate is the **start** of each timestep, as SIPNET labels
its rows.  Every row also describes the interval it covers: ``time_step_end``
and ``time_step_length`` coordinates, and a CF ``time_bounds`` variable giving
the half-open interval ``[time, time_step_end)``.  ``time`` carries the
standard ``bounds`` attribute naming it, which is what lets CF-aware tooling
decide whether an observation falls inside a step and how to combine steps.

Step lengths come from the climate drivers when they are known.  When they are
not, they are inferred from consecutive timestamps — exact for every step but
the last, which is assumed to repeat the one before it.  The Dataset's
``time_step_length_source`` attribute always says which happened, so nothing
about the interval is assumed silently.

The end of a step is its start plus its **declared** length, which is the
duration SIPNET integrates fluxes over.  A ``.clim`` file whose lengths are
rounded more coarsely than its timestamps therefore yields cells that overlap
or leave gaps by that rounding error — SIPNET's own Niwot fixture declares
``0.292`` days where its timestamps say exactly 7 hours, a 29-second
discrepancy.  The cells are not silently squared up, because the declared
length is what the model actually used; code that bins observations into steps
should expect intervals to be contiguous only to the precision of the drivers.

``time_bounds`` is a coordinate rather than a data variable, so that selecting
variables gives a Dataset whose ``data_vars`` are exactly what was asked for.
The cost is a second dimension, ``bounds``, which appears in ``ds.sizes`` and
makes ``ds.to_dataframe()`` produce two rows per timestep; use the container's
``.pandas`` view for a flat frame.

Every data variable carries the attributes its registry spec provides,
including ``time_reference`` in words.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pysipnet.variables import TIME_COORDINATE_NAMES

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

TIME_DIMENSION = "time"
BOUNDS_DIMENSION = "bounds"

TIME_CONVENTION = (
    "The 'time' coordinate is the START of each timestep, as SIPNET labels its rows. "
    "State variables are values at the END of the timestep; fluxes are totals OVER "
    "the timestep. Each variable's 'time_reference' attribute says which. The interval "
    "a row covers is the half-open [time, time_step_end), also given as 'time_bounds'."
)

TIME_ZONE = (
    "naive: SIPNET has no concept of a time zone, so this axis carries whatever "
    "convention the .clim drivers used (commonly local standard time). Check it "
    "before aligning against observations stamped in UTC."
)

# Where the step lengths in a TimeAxis came from, recorded on the Dataset so a
# consumer can tell a measured interval from a reconstructed one.
STEP_LENGTH_FROM_DRIVERS = "climate drivers"
STEP_LENGTH_INFERRED = (
    "inferred from consecutive timestamps; the last step repeats the one before it"
)
STEP_LENGTH_UNKNOWN = "unknown"

_NS_PER_DAY = 86_400_000_000_000
_NS_PER_HOUR = 3_600_000_000_000


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


def _days_to_timedelta(days: np.ndarray) -> np.ndarray:
    values = _finite(np.asarray(days, dtype=float), "time_step_length")
    return np.rint(values * _NS_PER_DAY).astype("int64").view("timedelta64[ns]")


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
    gaps = np.diff(start).astype("timedelta64[ns]").astype("int64") / _NS_PER_DAY
    if (gaps <= 0).any():
        row = int(np.argmax(gaps <= 0))
        raise ValueError(
            f"Timestamps do not increase at row {row + 1}: the step length inferred there "
            f"is {gaps[row]} days. Rows must be in strictly ascending time order for the "
            "interval each one covers to be reconstructed."
        )
    return np.concatenate([gaps, gaps[-1:]])


@dataclass(frozen=True)
class TimeAxis:
    """The time coordinates every view of one run's rows shares.

    Built once and reused, because it does not depend on which variables are
    selected: every view of a run has the same rows.
    """

    coords: dict[str, Any]
    n_rows: int
    step_length_source: str

    @property
    def has_step_length(self) -> bool:
        """Whether the axis knows how long each step is."""
        return "time_step_length" in self.coords


def build_time_axis(
    df: pd.DataFrame,
    *,
    attributes_for: Callable[[str], dict[str, Any]],
    time_step_length: np.ndarray | None = None,
) -> TimeAxis:
    """Build the shared time coordinates from a frame's time columns.

    Parameters
    ----------
    df:
        Must contain ``year``, ``day_of_year`` and ``hour_of_day``.
    attributes_for:
        Returns the ``.attrs`` for a column name; ``{}`` for unknown columns.
    time_step_length:
        Step lengths in days, one per row.  When ``None`` the frame's own
        ``time_step_length`` column is used if it has one, and failing that the
        lengths are inferred from consecutive timestamps; see
        :func:`_infer_step_length`.
    """
    start = timestep_start(df)

    if time_step_length is None and "time_step_length" in df.columns:
        # The drivers state their own step lengths; measuring the gaps between
        # timestamps instead would discard the last row's true length.
        time_step_length = df["time_step_length"].to_numpy()

    if time_step_length is None:
        length = _infer_step_length(start)
        source = STEP_LENGTH_INFERRED if length is not None else STEP_LENGTH_UNKNOWN
    else:
        length = _finite(np.asarray(time_step_length, dtype=float), "time_step_length")
        source = STEP_LENGTH_FROM_DRIVERS
        if len(length) != len(df):
            raise ValueError(
                f"time_step_length has {len(length)} values but the data has {len(df)} rows."
            )
        if len(length) and length.min() <= 0:
            row = int(np.argmin(length))
            raise ValueError(
                f"time_step_length is {length[row]} days at row {row}. A step must have a "
                "positive duration for the interval it covers to mean anything."
            )

    coords: dict[str, Any] = {
        TIME_DIMENSION: (
            TIME_DIMENSION,
            start,
            {
                "long_name": "Start of timestep",
                "description": "Calendar time at the start of the timestep, as SIPNET labels it.",
                "time_zone": TIME_ZONE,
            },
        ),
    }
    for name in TIME_COORDINATE_NAMES:
        coords[name] = (TIME_DIMENSION, df[name].to_numpy(), attributes_for(name))

    if length is not None:
        length_td = _days_to_timedelta(length)
        end = start + length_td
        coords[TIME_DIMENSION][2]["bounds"] = "time_bounds"
        coords["time_step_length"] = (
            TIME_DIMENSION,
            length_td,
            {
                "long_name": "Timestep length",
                "description": "Duration of the timestep.",
                "source": source,
            },
        )
        coords["time_step_end"] = (
            TIME_DIMENSION,
            end,
            {
                "long_name": "End of timestep",
                "description": "Calendar time at the end of the timestep; state variables "
                "are valid at this instant.",
            },
        )
        coords["time_bounds"] = (
            (TIME_DIMENSION, BOUNDS_DIMENSION),
            np.stack([start, end], axis=1),
            {
                "long_name": "Timestep bounds",
                "description": "The half-open interval [time, time_step_end) each row "
                "covers, in the Climate and Forecast conventions' bounds form.",
            },
        )

    return TimeAxis(coords=coords, n_rows=len(df), step_length_source=source)


def dataset_from_dataframe(
    df: pd.DataFrame,
    axis: TimeAxis,
    *,
    attributes_for: Callable[[str], dict[str, Any]],
    source: str,
    extra_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Assemble a Dataset from a frame's data columns and an already-built axis.

    Every column that is not part of the time axis becomes a data variable, so
    passing a frame holding only some variables gives a Dataset holding only
    those — the whole point of building the axis separately.
    """
    import xarray as xr

    if len(df) != axis.n_rows:
        raise ValueError(f"Frame has {len(df)} rows but the time axis was built for {axis.n_rows}.")

    # A time_step_length column becomes the coordinate of that name when the axis
    # has lengths; otherwise it stays an ordinary variable.
    as_coordinate = set(TIME_COORDINATE_NAMES)
    if axis.has_step_length:
        as_coordinate.add("time_step_length")
    data_vars = {
        name: (TIME_DIMENSION, df[name].to_numpy(), attributes_for(name))
        for name in df.columns
        if name not in as_coordinate
    }

    attrs: dict[str, Any] = {
        "source": source,
        "time_convention": TIME_CONVENTION,
        "time_zone": TIME_ZONE,
        "time_step_length_source": axis.step_length_source,
    }
    attrs.update(extra_attrs or {})

    return xr.Dataset(data_vars, coords=axis.coords, attrs=attrs)


def dataframe_to_dataset(
    df: pd.DataFrame,
    *,
    attributes_for: Callable[[str], dict[str, Any]],
    time_step_length: np.ndarray | None = None,
    source: str,
    extra_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Build a Dataset with one ``time`` dimension from a per-timestep DataFrame.

    The one-shot form of :func:`build_time_axis` followed by
    :func:`dataset_from_dataframe`.  Use the two separately when several Datasets
    are built from the same rows, so the time axis is paid for once.

    Parameters
    ----------
    df:
        Must contain ``year``, ``day_of_year`` and ``hour_of_day`` columns
        marking the start of each step.  Every other column becomes a data
        variable.
    attributes_for:
        Returns the ``.attrs`` for a column name; ``{}`` for unknown columns.
    time_step_length:
        Step lengths in days, one per row.  Inferred from the timestamps when
        omitted.
    source:
        Free text for the Dataset's ``source`` attribute.
    extra_attrs:
        Further Dataset-level attributes, e.g. run provenance.
    """
    import xarray as xr

    if df.empty:
        return xr.Dataset()

    axis = build_time_axis(df, attributes_for=attributes_for, time_step_length=time_step_length)
    return dataset_from_dataframe(
        df, axis, attributes_for=attributes_for, source=source, extra_attrs=extra_attrs
    )
