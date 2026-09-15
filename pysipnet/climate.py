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
|  9 | vapour_pressure_deficit             | Pa      | Mean over the step; must be > 0  |
+----+-------------------------------------+---------+----------------------------------+
| 10 | soil_vapour_pressure_deficit        | Pa      | Mean over the step               |
+----+-------------------------------------+---------+----------------------------------+
| 11 | vapour_pressure                     | Pa      | Mean over the step               |
+----+-------------------------------------+---------+----------------------------------+
| 12 | wind_speed                          | m s⁻¹   | Mean over the step; must be > 0  |
+----+-------------------------------------+---------+----------------------------------+

Column numbers are for the 12-column file layout; the 14-column layout wraps
the same values in a leading site identifier and a trailing soil wetness
value, both of which SIPNET ignores and neither of which is stored here.

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
behaviour while making the issue visible to the user.
"""

from __future__ import annotations

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


class ClimateDrivers:
    """Meteorological forcing time series for a SIPNET run.

    Instances are either *memory-backed* (holding a full DataFrame) or
    *file-backed* (holding only a path, with data loaded lazily on first
    access).  Use the factory methods to construct:

    - :meth:`from_dataframe` — in-memory, with full column and data validation.
    - :meth:`from_file` — reads an existing ``.clim`` file fully into memory.
    - :meth:`from_path` — file-backed, defers loading until ``.data`` is
      accessed.  Use this in ensemble workflows where the file already exists
      on disk and you want to avoid a redundant read/write cycle.

    Parameters
    ----------
    data:
        One row per model timestep with columns matching :data:`CLIMATE_COLUMNS`.
        Mutually exclusive with *source_path*.
    source_path:
        Path to an existing ``.clim`` file.  Mutually exclusive with *data*.
    n_columns:
        Which climate file layout this object writes: 12 or 14 columns. Both
        carry the same 12 values; the 14-column layout adds two SIPNET
        ignores.
    loc:
        Location index written to column 1 of v1 climate files (memory-backed
        only).  SIPNET ignores this value; it exists for backward compatibility.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | None = None,
        source_path: Path | None = None,
        n_columns: Literal[12, 14] = 14,
        loc: int = 0,
    ) -> None:
        if (data is None) == (source_path is None):
            raise ValueError(
                "Exactly one of 'data' or 'source_path' must be provided, not both or neither."
            )
        self._data: pd.DataFrame | None = data
        self.source_path: Path | None = source_path
        self.n_columns: Literal[12, 14] = n_columns
        self.loc: int = loc
        self._n_timesteps: int | None = None
        self._date_range: tuple[tuple[int, int], tuple[int, int]] | None = None

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path, n_columns: Literal[12, 14] = 14) -> ClimateDrivers:
        """Read a SIPNET climate file fully into memory.

        Parameters
        ----------
        path:
            Path to the ``.clim`` file.
        n_columns:
            Which layout to expect.  ``14`` expects 14 columns (site index
            in col 1, soil-wetness in col 14); ``12`` expects 12 columns.
        """
        from pysipnet.io.clim_io import read_clim_file

        return read_clim_file(Path(path), n_columns=n_columns)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        n_columns: Literal[12, 14] = 14,
        loc: int = 0,
    ) -> ClimateDrivers:
        """Construct from a pre-built DataFrame.

        The DataFrame must contain every column in :data:`CLIMATE_COLUMNS`,
        under its registry name or an alias (``tair`` for ``air_temperature``,
        ``vpd`` for ``vapour_pressure_deficit``, ...). Aliased columns are
        renamed; extra columns are ignored.

        Parameters
        ----------
        df:
            Input DataFrame with climate variables.
        n_columns:
            Which layout to use when this object is written to a file.
        loc:
            Location index (v1 only).
        """
        df = _rename_aliases(df)
        missing = set(CLIMATE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
        obj = cls(data=df[CLIMATE_COLUMNS].copy(), n_columns=n_columns, loc=loc)
        obj.validate()
        return obj

    @classmethod
    def from_path(cls, path: str | Path, n_columns: Literal[12, 14] = 14) -> ClimateDrivers:
        """Create a file-backed instance without loading data into memory.

        The file is not read until :attr:`data` is accessed.  Lightweight
        validation checks the column count of the first and last rows, and
        caches :attr:`n_timesteps` and :attr:`date_range` from those rows.

        .. note::
            Chronological ordering is **assumed but not verified**.  The first
            and last rows are used to populate :attr:`date_range`; if the file
            is not sorted those values will be wrong.  Call :meth:`validate`
            to perform a complete check, which will trigger a full data load.

        Parameters
        ----------
        path:
            Path to an existing ``.clim`` file.
        n_columns:
            Which layout to expect.
        """
        from pysipnet.io.clim_io import peek_clim_file

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Climate file not found: {path}")

        n_rows, start, end = peek_clim_file(path, n_columns=n_columns)
        obj = cls(source_path=path, n_columns=n_columns)
        obj._n_timesteps = n_rows
        obj._date_range = (start, end)
        return obj

    # ── Data access ────────────────────────────────────────────────────────────

    @property
    def data(self) -> pd.DataFrame:
        """The climate time series as a DataFrame.

        For file-backed instances, the first access reads and caches the full
        file from :attr:`source_path`.  Subsequent accesses return the cached
        copy at no cost.
        """
        if self._data is None:
            from pysipnet.io.clim_io import read_clim_file

            if self.source_path is None:
                raise ValueError(
                    "This ClimateDrivers has neither loaded data nor a file to read from."
                )
            self._data = read_clim_file(self.source_path, n_columns=self.n_columns).data
        return self._data

    @property
    def dataset(self) -> xr.Dataset:
        """The drivers as an :class:`xarray.Dataset` on the same ``time`` axis as outputs.

        ``time`` is the start of each step; ``time_step_end`` and
        ``time_step_length`` are coordinates; every variable carries its units,
        description and time reference from :data:`pysipnet.variables.CLIMATE_VARIABLES`.
        """
        from pysipnet.dataset import dataframe_to_dataset

        return dataframe_to_dataset(
            self.data,
            attributes_for=_attributes_for,
            time_step_length=self.data["time_step_length"].to_numpy(),
            source="SIPNET climate drivers, via pySIPNET",
        )

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Check the climate data for common errors.

        For file-backed instances, calling this method triggers a full data
        load from :attr:`source_path`.

        Raises
        ------
        ValueError
            On any condition that would cause SIPNET to crash or produce
            silently wrong results.
        """
        self._check_no_nulls()
        self._check_positive_length()
        self._check_monotonic_time()
        self._check_vpd_wind()

    def _check_no_nulls(self) -> None:
        null_cols = self.data.columns[self.data.isnull().any()].tolist()
        if null_cols:
            raise ValueError(
                f"Missing values (NaN) found in climate columns: {null_cols}. "
                "SIPNET requires complete climate data."
            )

    def _check_positive_length(self) -> None:
        if (self.data["time_step_length"] <= 0).any():
            raise ValueError(
                "All 'time_step_length' values must be > 0 (timestep duration in days)."
            )

    def _check_monotonic_time(self) -> None:
        # Construct a monotone scalar: days from an arbitrary epoch
        d = self.data
        doy = d["year"] * 366 + d["day_of_year"] + d["hour_of_day"] / 24.0
        if not doy.is_monotonic_increasing:
            raise ValueError(
                "Climate timesteps are not in chronological order. "
                "Rows must be sorted by (year, day_of_year, hour_of_day)."
            )

    def _check_vpd_wind(self) -> None:
        import warnings

        vpd = self.data["vapour_pressure_deficit"]
        if (vpd <= 0).any():
            warnings.warn(
                f"{(vpd <= 0).sum()} timestep(s) have vapour_pressure_deficit ≤ 0 Pa. "
                "SIPNET clamps these to a tiny positive value to avoid division by zero, "
                "but this may indicate a data issue.",
                stacklevel=3,
            )
        wind = self.data["wind_speed"]
        if (wind <= 0).any():
            warnings.warn(
                f"{(wind <= 0).sum()} timestep(s) have wind_speed ≤ 0 m s⁻¹. "
                "SIPNET clamps these internally, but this may indicate bad data.",
                stacklevel=3,
            )

    # ── Serialisation ──────────────────────────────────────────────────────────

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
        return len(self.data)

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
        first = self.data.iloc[0]
        last = self.data.iloc[-1]
        return (
            (int(first["year"]), int(first["day_of_year"])),
            (int(last["year"]), int(last["day_of_year"])),
        )

    def __repr__(self) -> str:
        (y0, d0), (y1, d1) = self.date_range
        return (
            f"ClimateDrivers(n_columns={self.n_columns!r}, "
            f"timesteps={self.n_timesteps}, "
            f"range={y0}-{d0:03d} to {y1}-{d1:03d})"
        )


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
