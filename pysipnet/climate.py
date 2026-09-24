"""Climate driver data structure and I/O.

The :class:`ClimateDrivers` class holds the meteorological forcing time series
required by SIPNET.  It is layout-aware: the number of columns differs between
the 14-column and 12-column layouts, both of which the pinned SIPNET reads.

Column conventions
------------------
Python-side column names are the registry names in
:data:`pysipnet.variables.CLIMATE_VARIABLES`, which also carry units,
descriptions and the conversions SIPNET applies on read. The I/O layer maps
them to the positional layout SIPNET expects. The old short names (``tair``,
``vpd``, ...) and SIPNET's own column names are accepted as aliases by
:meth:`ClimateDrivers.from_dataframe`, which renames them; stored columns are
always the registry names.

+----+-------------------------------------+---------+----------------------------------+
| Col| Name                                | Unit    | Notes                            |
+====+=====================================+=========+==================================+
|  1 | year                                | —       | Integer year, start of step      |
+----+-------------------------------------+---------+----------------------------------+
|  2 | day_of_year                         | —       | 1 = Jan 1, start of step         |
+----+-------------------------------------+---------+----------------------------------+
|  3 | hour_of_day                         | h       | After midnight, start of step    |
+----+-------------------------------------+---------+----------------------------------+
|  4 | time_step_length                    | d       | Timestep duration                |
+----+-------------------------------------+---------+----------------------------------+
|  5 | air_temperature                     | °C      | Mean over the step               |
+----+-------------------------------------+---------+----------------------------------+
|  6 | soil_temperature                    | °C      | Mean over the step               |
+----+-------------------------------------+---------+----------------------------------+
|  7 | photosynthetically_active_radiation | mol m⁻² | Total over the step (Einstein)   |
+----+-------------------------------------+---------+----------------------------------+
|  8 | precipitation                       | mm      | Total over the step              |
+----+-------------------------------------+---------+----------------------------------+
|  9 | vapor_pressure_deficit             | Pa      | Mean over the step; must be > 0  |
+----+-------------------------------------+---------+----------------------------------+
| 10 | soil_vapor_pressure_deficit        | Pa      | Mean over the step               |
+----+-------------------------------------+---------+----------------------------------+
| 11 | vapor_pressure                     | Pa      | Mean over the step               |
+----+-------------------------------------+---------+----------------------------------+
| 12 | wind_speed                          | m s⁻¹   | Mean over the step; must be > 0  |
+----+-------------------------------------+---------+----------------------------------+

Column numbers are for the 12-column file layout; the 14-column layout wraps
the same values in a leading site identifier and a trailing soil wetness
value, both of which SIPNET ignores and neither of which is stored here.

``year``, ``day_of_year`` and ``hour_of_day`` label the start of the step on
whatever clock the drivers were written in.

The drivers declare the clock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SIPNET has no time zone.  It computes no solar geometry and reads a row's
labels only to name the row, to compare against ``leafOnDay`` /
``leafOffDay`` and to check restart boundaries, so which clock the labels are
on is a property of the drivers, not of the model.  ``ClimateDrivers`` can
declare it: ``time_zone="UTC"``, or a fixed offset such as ``"UTC-07:00"``.
The declaration is metadata only — nothing is converted — and it travels on
the ``time`` coordinate of :attr:`ClimateDrivers.xarray` and of every output
run on these drivers.  It is undeclared by default.  A named zone is refused,
because one with daylight saving time has a clock that jumps and SIPNET's rows
cannot.

Labels and lengths
~~~~~~~~~~~~~~~~~~
SIPNET integrates each row over ``time_step_length`` and never checks that a
row's start plus its length is the next row's start.
:meth:`ClimateDrivers.validate` does: it refuses a row that starts before the
previous one ends and labels that drift away from the running sum of the
lengths, and warns about a gap in the record.  See
:func:`pysipnet.dataset.check_step_continuity` for the tolerances.

PAR units note
~~~~~~~~~~~~~~
``photosynthetically_active_radiation`` holds the **total** over the timestep
in mol photons m⁻² ground. When converting from an instantaneous flux
(µmol m⁻² s⁻¹), multiply by ``time_step_length × 86400 / 1e6``.

VPD and wind speed
~~~~~~~~~~~~~~~~~~
SIPNET clamps VPD and wind speed up to a tiny positive value when they fall
below it — note the test is "below", so small negatives are silently clamped
too, not just zeros, to avoid division by zero.  :meth:`ClimateDrivers.validate` flags
non-positive values as warnings rather than errors, matching SIPNET's
behavior while making the issue visible to the user.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from pysipnet.variables import (
    CLIMATE_COLUMN_NAMES,
    CLIMATE_VARIABLES_BY_NAME,
    resolve_climate_variable,
)

if TYPE_CHECKING:
    import xarray as xr

# Canonical column names for the Python representation, from the registry.
# The loc and soil_wetness columns of the 14-column layout are not included:
# they are written/read by the IO layer as padding, not stored in the DataFrame.
CLIMATE_COLUMNS: list[str] = list(CLIMATE_COLUMN_NAMES)

_UTC_OFFSET = re.compile(r"UTC(?:([+-])(\d{2}):(\d{2}))?")


def normalize_time_zone(time_zone: str | None) -> str | None:
    """Check a clock declaration and return it in canonical form.

    Accepts ``"UTC"`` or a fixed offset from it, ``"UTC+HH:MM"`` /
    ``"UTC-HH:MM"``; a zero offset is returned as ``"UTC"``.  ``None`` means
    undeclared and is returned unchanged.
    """
    if time_zone is None:
        return None
    match = _UTC_OFFSET.fullmatch(time_zone)
    if match is None:
        raise ValueError(
            f"time_zone must be 'UTC' or a fixed offset such as 'UTC-07:00', not {time_zone!r}. "
            "A named zone like 'America/Denver' includes daylight saving time, whose clock "
            "jumps twice a year; SIPNET's rows cannot jump, so drivers labeled on such a clock "
            "would overlap or leave a gap at each change. Name the offset the labels actually "
            "use (for local standard time in Denver, 'UTC-07:00')."
        )
    sign, hours, minutes = match.groups()
    if sign is None:
        return "UTC"
    if int(hours) > 14 or int(minutes) >= 60:
        raise ValueError(
            f"time_zone {time_zone!r} is not a UTC offset: offsets run from UTC-12:00 to "
            "UTC+14:00, with minutes below 60."
        )
    if int(hours) == 0 and int(minutes) == 0:
        return "UTC"
    return f"UTC{sign}{hours}:{minutes}"


class ClimateDrivers:
    """Meteorological forcing time series for a SIPNET run.

    Instances are either *memory-backed* (holding a full DataFrame) or
    *file-backed* (holding only a path, with data loaded lazily on first
    access).  Use the factory methods to construct:

    - :meth:`from_dataframe` — in-memory, validated at construction.
    - :meth:`from_file` — reads an existing ``.clim`` file fully into memory,
      validated at construction.
    - :meth:`from_path` — file-backed, defers loading until ``.pandas`` is
      accessed, and **validation with it**.  Use this in ensemble workflows
      where the file already exists on disk and you want to avoid a redundant
      read/write cycle.

    Validation (:meth:`validate`) runs exactly once for any set of data, at the
    moment the data is loaded: at construction when data is passed in, and on
    first access for a file-backed instance.  Nothing that uses the data
    afterwards — a run, a time axis, an output Dataset — repeats it, so the
    DataFrame behind :attr:`pandas` must not be modified in place.

    Parameters
    ----------
    data:
        One row per model timestep with every column in :data:`CLIMATE_COLUMNS`,
        under its registry name or an alias; aliases are renamed and extra
        columns dropped, as in :meth:`from_dataframe`.  The frame is copied and
        validated.  Mutually exclusive with *source_path*.
    source_path:
        Path to an existing ``.clim`` file.  Mutually exclusive with *data*.
    n_columns:
        Which climate file layout this object writes: 12 or 14 columns. Both
        carry the same 12 values; the 14-column layout adds two SIPNET
        ignores.
    loc:
        Location index written to column 1 of v1 climate files (memory-backed
        only).  SIPNET ignores this value; it exists for backward compatibility.
    time_zone:
        The clock the ``year`` / ``day_of_year`` / ``hour_of_day`` labels are
        on: ``"UTC"`` or a fixed offset such as ``"UTC-07:00"``.  Metadata
        only, recorded on the time axis of these drivers and of any output run
        on them; nothing is converted.  ``None`` (the default) leaves it
        undeclared.  See the module docstring.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | None = None,
        source_path: Path | None = None,
        n_columns: Literal[12, 14] = 14,
        loc: int = 0,
        time_zone: str | None = None,
    ) -> None:
        if (data is None) == (source_path is None):
            raise ValueError(
                "Exactly one of 'data' or 'source_path' must be provided, not both or neither."
            )
        self.source_path: Path | None = source_path
        self.n_columns: Literal[12, 14] = n_columns
        self.loc: int = loc
        self.time_zone: str | None = normalize_time_zone(time_zone)
        self._n_timesteps: int | None = None
        self._date_range: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._data: pd.DataFrame | None = None
        if data is not None:
            self._data = _standard_columns(data)
            self._run_checks()

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        n_columns: Literal[12, 14] = 14,
        *,
        time_zone: str | None = None,
    ) -> ClimateDrivers:
        """Read a SIPNET climate file fully into memory.

        Parameters
        ----------
        path:
            Path to the ``.clim`` file.
        n_columns:
            Which layout to expect.  ``14`` expects 14 columns (site index
            in col 1, soil-wetness in col 14); ``12`` expects 12 columns.
        time_zone:
            The clock the file's labels are on; see the class docstring.  A
            ``.clim`` file cannot say, so only the caller can.
        """
        from pysipnet.io.clim_io import read_clim_file

        return read_clim_file(Path(path), n_columns=n_columns, time_zone=time_zone)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        n_columns: Literal[12, 14] = 14,
        loc: int = 0,
        *,
        time_zone: str | None = None,
    ) -> ClimateDrivers:
        """Construct from a pre-built DataFrame.

        The DataFrame must contain every column in :data:`CLIMATE_COLUMNS`,
        under its registry name or an alias (``tair`` for ``air_temperature``,
        ``vpd`` for ``vapor_pressure_deficit``, ...). Aliased columns are
        renamed; extra columns are ignored.

        Parameters
        ----------
        df:
            Input DataFrame with climate variables.
        n_columns:
            Which layout to use when this object is written to a file.
        loc:
            Location index (v1 only).
        time_zone:
            The clock the labels are on; see the class docstring.
        """
        return cls(data=df, n_columns=n_columns, loc=loc, time_zone=time_zone)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        n_columns: Literal[12, 14] = 14,
        *,
        time_zone: str | None = None,
    ) -> ClimateDrivers:
        """Create a file-backed instance without loading data into memory.

        The file is not read until :attr:`pandas` is accessed.  Lightweight
        validation checks the column count of the first and last rows, and
        caches :attr:`n_timesteps` and :attr:`date_range` from those rows.

        .. note::
            **Validation is deferred along with the read.**  The full checks in
            :meth:`validate` — missing values, ordering, and labels against
            step lengths — run the first time the data is loaded, not here.  A
            run stages the file without reading it, so a file that fails them
            still runs; the failure surfaces when something reads the drivers,
            such as the output's Dataset.  Call :meth:`validate` to load and
            check the file up front.  Until then, :attr:`date_range` is taken
            from the first and last rows and assumes the file is sorted.

        Parameters
        ----------
        path:
            Path to an existing ``.clim`` file.
        n_columns:
            Which layout to expect.
        time_zone:
            The clock the file's labels are on; see the class docstring.
        """
        from pysipnet.io.clim_io import peek_clim_file

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Climate file not found: {path}")

        n_rows, start, end = peek_clim_file(path, n_columns=n_columns)
        obj = cls(source_path=path, n_columns=n_columns, time_zone=time_zone)
        obj._n_timesteps = n_rows
        obj._date_range = (start, end)
        return obj

    # ── Data access ────────────────────────────────────────────────────────────

    @property
    def pandas(self) -> pd.DataFrame:
        """The climate time series as a :class:`pandas.DataFrame`.

        For file-backed instances, the first access reads, validates and caches
        the full file from :attr:`source_path`, so it raises if the file fails
        :meth:`validate`.  Subsequent accesses return the cached copy at no
        cost.  Do not modify the frame in place: it was validated once, and
        nothing checks it again.
        """
        if self._data is None:
            from pysipnet.io.clim_io import read_clim_file

            if self.source_path is None:
                raise ValueError(
                    "This ClimateDrivers has neither loaded data nor a file to read from."
                )
            self._data = read_clim_file(
                self.source_path, n_columns=self.n_columns, time_zone=self.time_zone
            ).pandas
        return self._data

    @property
    def xarray(self) -> xr.Dataset:
        """The drivers as an :class:`xarray.Dataset` on the same ``time`` axis as outputs.

        ``time`` is the end of each step, as for outputs, and an output run on
        these drivers has exactly this axis; ``time_step_start``,
        ``time_step_length`` and ``time_bounds`` are coordinates; ``time``
        carries the declared ``time_zone``; every variable carries its units,
        description and time reference from
        :data:`pysipnet.variables.CLIMATE_VARIABLES`.
        """
        from pysipnet.dataset import build_xarray_dataset

        return build_xarray_dataset(
            self.pandas,
            attributes_for=_attributes_for,
            drivers=self,
            source="SIPNET climate drivers, via pySIPNET",
        )

    def head(self, n: int) -> ClimateDrivers:
        """The first *n* steps, as memory-backed drivers with the same layout and clock.

        Not validated again: every check :meth:`validate` makes holds for a
        prefix of a record that passed it, the drift reconstruction included,
        since it starts from the same row. Loads a file-backed instance.
        """
        prefix = copy.copy(self)
        prefix.source_path = None
        prefix._data = self.pandas.head(n).copy()
        prefix._n_timesteps = None
        prefix._date_range = None
        return prefix

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Check the climate data for common errors.

        The checks run by themselves whenever data is loaded, so calling this
        is only needed to load a file-backed instance now, and have its file
        checked, rather than on first use.  On data already loaded it runs the
        checks again.

        Raises
        ------
        ValueError
            On any condition that would cause SIPNET to crash or produce
            silently wrong results, including labels that disagree with the
            step lengths: a row that starts before the previous one ends, or
            labels that drift from the running sum of the lengths.

        Warns
        -----
        UserWarning
            On a gap in the record, and on non-positive vapor pressure deficit
            or wind speed.
        """
        if self._data is None:
            _ = self.pandas  # loading runs the checks
        else:
            self._run_checks()

    def _run_checks(self) -> None:
        self._check_no_nulls()
        self._check_positive_length()
        self._check_monotonic_time()
        self._check_step_continuity()
        self._check_vpd_wind()

    def _check_no_nulls(self) -> None:
        null_cols = self.pandas.columns[self.pandas.isnull().any()].tolist()
        if null_cols:
            raise ValueError(
                f"Missing values (NaN) found in climate columns: {null_cols}. "
                "SIPNET requires complete climate data."
            )

    def _check_positive_length(self) -> None:
        if (self.pandas["time_step_length"] <= 0).any():
            raise ValueError(
                "All 'time_step_length' values must be > 0 (timestep duration in days)."
            )

    def _check_monotonic_time(self) -> None:
        # Construct a monotone scalar: days from an arbitrary epoch
        d = self.pandas
        doy = d["year"] * 366 + d["day_of_year"] + d["hour_of_day"] / 24.0
        if not doy.is_monotonic_increasing:
            raise ValueError(
                "Climate timesteps are not in chronological order. "
                "Rows must be sorted by (year, day_of_year, hour_of_day)."
            )

    def _check_step_continuity(self) -> None:
        import warnings

        from pysipnet.dataset import (
            check_step_continuity,
            days_to_timedelta,
            describe_duration,
            timestep_start,
        )

        d = self.pandas
        start = timestep_start(d)
        length = days_to_timedelta(d["time_step_length"].to_numpy())
        gaps = check_step_continuity(start, length)
        if len(gaps):
            row = int(gaps[0])
            end = start[row] + length[row]
            missing = describe_duration(start[row + 1] - end)
            warnings.warn(
                f"{len(gaps)} gap(s) in the climate record. The first follows row {row}, which "
                f"ends at {end}, and row {row + 1} starts {missing} later, at {start[row + 1]}. "
                "SIPNET does not see gaps: it runs the next row from the state the previous one "
                "left, so no time passes for the model across one. The time axis shows it as a "
                "gap between time_bounds.",
                stacklevel=3,
            )

    def _check_vpd_wind(self) -> None:
        import warnings

        vpd = self.pandas["vapor_pressure_deficit"]
        if (vpd <= 0).any():
            warnings.warn(
                f"{(vpd <= 0).sum()} timestep(s) have vapor_pressure_deficit ≤ 0 Pa. "
                "SIPNET clamps these to a tiny positive value to avoid division by zero, "
                "but this may indicate a data issue.",
                stacklevel=3,
            )
        wind = self.pandas["wind_speed"]
        if (wind <= 0).any():
            warnings.warn(
                f"{(wind <= 0).sum()} timestep(s) have wind_speed ≤ 0 m s⁻¹. "
                "SIPNET clamps these internally, but this may indicate bad data.",
                stacklevel=3,
            )

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_file(self, path: str | Path) -> None:
        """Write the climate data to a SIPNET-format ``.clim`` file."""
        from pysipnet.io.clim_io import write_clim_file

        write_clim_file(self, Path(path))

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps in the driving data.

        For file-backed instances created via :meth:`from_path`, this is
        available without triggering a full data load.
        """
        if self._data is not None:
            return len(self._data)
        if self._n_timesteps is not None:
            return self._n_timesteps
        return len(self.pandas)

    @property
    def date_range(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """``((start_year, start_doy), (end_year, end_doy))`` of the time series.

        For file-backed instances created via :meth:`from_path`, this is
        available without triggering a full data load.  The values are read
        from the first and last rows of the file and assume chronological order.
        """
        if self._data is not None:
            first = self._data.iloc[0]
            last = self._data.iloc[-1]
            return (
                (int(first["year"]), int(first["day_of_year"])),
                (int(last["year"]), int(last["day_of_year"])),
            )
        if self._date_range is not None:
            return self._date_range
        first = self.pandas.iloc[0]
        last = self.pandas.iloc[-1]
        return (
            (int(first["year"]), int(first["day_of_year"])),
            (int(last["year"]), int(last["day_of_year"])),
        )

    def __repr__(self) -> str:
        (y0, d0), (y1, d1) = self.date_range
        zone = "" if self.time_zone is None else f", time_zone={self.time_zone!r}"
        return (
            f"ClimateDrivers(n_columns={self.n_columns!r}, "
            f"timesteps={self.n_timesteps}, "
            f"range={y0}-{d0:03d} to {y1}-{d1:03d}{zone})"
        )


def _standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of *df* with exactly :data:`CLIMATE_COLUMNS`, aliases renamed."""
    df = _rename_aliases(df)
    missing = set(CLIMATE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
    return df[CLIMATE_COLUMNS].copy()


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns given under an alias or SIPNET name to the registry name."""
    renames: dict[str, str] = {}
    for column in df.columns:
        if column in CLIMATE_VARIABLES_BY_NAME:
            continue
        try:
            target = resolve_climate_variable(str(column)).name
        except KeyError:
            continue  # an extra column; from_dataframe drops it
        if target in df.columns or target in renames.values():
            raise ValueError(
                f"Climate DataFrame has column {column!r} and its canonical name {target!r} "
                "(or another alias of it). Keep one of them; there is no way to know which "
                "holds the intended values."
            )
        renames[column] = target
    return df.rename(columns=renames) if renames else df


def _attributes_for(name: str) -> dict[str, Any]:
    spec = CLIMATE_VARIABLES_BY_NAME.get(name)
    return {} if spec is None else spec.xarray_attributes()
