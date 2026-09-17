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


def timestep_start(df: pd.DataFrame) -> np.ndarray:
    """Calendar time at the start of each row, as ``datetime64[ns]``.

    Built from ``year``, ``day_of_year`` and ``hour_of_day`` with integer
    ``datetime64`` arithmetic rather than a string round-trip through
    :func:`pandas.to_datetime`, which costs about 12 ms for a half-hourly year
    — comparable to parsing the whole output file.  The two agree exactly.
    """
    missing = [c for c in TIME_COORDINATE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot build a time coordinate: data is missing {missing}. "
            "Was the file read without a header row?"
        )

    year = df["year"].to_numpy().astype("int64")
    day_of_year = df["day_of_year"].to_numpy().astype("int64")
    hour_of_day = df["hour_of_day"].to_numpy().astype("float64")

    january_first = (year - 1970).astype("datetime64[Y]").astype("datetime64[ns]")
    days = (day_of_year - 1).astype("timedelta64[D]").astype("timedelta64[ns]")
    hours = np.rint(hour_of_day * _NS_PER_HOUR).astype("int64").astype("timedelta64[ns]")
    return january_first + days + hours


def _days_to_timedelta(days: np.ndarray) -> np.ndarray:
    return (
        np.rint(np.asarray(days, dtype=float) * _NS_PER_DAY).astype("int64").view("timedelta64[ns]")
    )


def _infer_step_length(start: np.ndarray) -> np.ndarray | None:
    """Step lengths in days from consecutive starts, or ``None`` if undeterminable.

    SIPNET labels each row with the start of its step and the steps are
    contiguous, so a difference of starts is the exact length of every step but
    the last, which has no successor to measure against.
    """
    if len(start) < 2:
        return None
    gaps = np.diff(start).astype("timedelta64[ns]").astype("int64") / _NS_PER_DAY
    return np.concatenate([gaps, gaps[-1:]])


@dataclass(frozen=True)
class TimeAxis:
    """The time coordinates every view of one run's rows shares.

    Built once and reused, because constructing it is a third of the cost of
    reading the output file and it does not depend on which variables are
    selected.
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
        Step lengths in days, one per row.  When ``None`` they are inferred
        from consecutive timestamps; see :func:`_infer_step_length`.
    """
    start = timestep_start(df)

    if time_step_length is None:
        length = _infer_step_length(start)
        source = STEP_LENGTH_INFERRED if length is not None else STEP_LENGTH_UNKNOWN
    else:
        length = np.asarray(time_step_length, dtype=float)
        source = STEP_LENGTH_FROM_DRIVERS
        if len(length) != len(df):
            raise ValueError(
                f"time_step_length has {len(length)} values but the data has {len(df)} rows."
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


def dataset_from_frame(
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
    :func:`dataset_from_frame`.  Use the two separately when several Datasets
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
    return dataset_from_frame(
        df, axis, attributes_for=attributes_for, source=source, extra_attrs=extra_attrs
    )
