"""Climate driver data structure and I/O.

The :class:`ClimateDrivers` class holds the meteorological forcing time series
required by SIPNET.  It is version-aware: the column layout differs between
SIPNET v1 (14 columns) and v2 (12 columns).

Column conventions
------------------
Python-side column names use ``snake_case`` and match the table below.  The
I/O layer maps these to the positional format SIPNET expects.

SIPNET v1 climate file (14 columns, space-delimited, no header):

+-----+----------------+---------+-------------------------------------------+
| Col | Name           | Unit    | Notes                                     |
+=====+================+=========+===========================================+
|  1  | loc            | —       | Integer location index; ignored by SIPNET |
+-----+----------------+---------+-------------------------------------------+
|  2  | year           | —       | Integer year                              |
+-----+----------------+---------+-------------------------------------------+
|  3  | day            | —       | Integer day-of-year (1 = Jan 1)           |
+-----+----------------+---------+-------------------------------------------+
|  4  | time           | hours   | Fractional hours at start of timestep     |
+-----+----------------+---------+-------------------------------------------+
|  5  | length         | days    | Timestep duration in days                 |
+-----+----------------+---------+-------------------------------------------+
|  6  | tair           | °C      | Mean air temperature                      |
+-----+----------------+---------+-------------------------------------------+
|  7  | tsoil          | °C      | Mean soil temperature                     |
+-----+----------------+---------+-------------------------------------------+
|  8  | par            | mol m⁻² | PAR integrated over the full timestep (mol photons; 1 Einstein = 1 mol) |
+-----+----------------+---------+-------------------------------------------+
|  9  | precip         | mm      | Total precipitation over the timestep     |
+-----+----------------+---------+-------------------------------------------+
| 10  | vpd            | Pa      | Vapour pressure deficit (must be > 0)     |
+-----+----------------+---------+-------------------------------------------+
| 11  | vpd_soil       | Pa      | Soil–air VPD                              |
+-----+----------------+---------+-------------------------------------------+
| 12  | vpress         | Pa      | Vapour pressure in canopy airspace        |
+-----+----------------+---------+-------------------------------------------+
| 13  | wspd           | m s⁻¹  | Mean wind speed (must be > 0)             |
+-----+----------------+---------+-------------------------------------------+
| 14  | soil_wetness   | —       | Ignored by SIPNET (legacy column)         |
+-----+----------------+---------+-------------------------------------------+

PAR units note
~~~~~~~~~~~~~~
The ``par`` column holds the **total** PAR integrated over the timestep
interval, in Einstein m⁻² ground.  When converting from an instantaneous
flux (µmol m⁻² s⁻¹), multiply by ``length × 86400 / 1e6`` to obtain
the per-timestep total in Einstein m⁻².

VPD and wind speed
~~~~~~~~~~~~~~~~~~
SIPNET internally adds a tiny positive value to VPD and wind speed if they
are zero, to avoid division by zero.  :meth:`ClimateDrivers.validate` flags
non-positive values as warnings rather than errors, matching SIPNET's
behaviour while making the issue visible to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from pysipnet.version import CLIM_COLS_V1, CLIM_COLS_V2

# Canonical column names for the Python representation.
# The loc and soil_wetness columns from v1 are not included here —
# they are written/read by the IO layer as padding, not stored in the DataFrame.
CLIM_COLUMNS_V1: list[str] = [
    "year", "day", "time", "length",
    "tair", "tsoil", "par", "precip",
    "vpd", "vpd_soil", "vpress", "wspd",
]

CLIM_COLUMNS_V2: list[str] = CLIM_COLUMNS_V1  # same logical columns, different file format


@dataclass
class ClimateDrivers:
    """Meteorological forcing time series for a SIPNET run.

    The ``data`` DataFrame uses :data:`CLIM_COLUMNS_V1` column names.  Rows
    must be ordered chronologically with no missing values.

    Parameters
    ----------
    data:
        One row per model timestep.  Columns must match :data:`CLIM_COLUMNS_V1`.
    version:
        SIPNET version this object is formatted for.  Controls the file format
        used by :meth:`to_file`.
    loc:
        Location index written to column 1 of v1 climate files.  SIPNET
        ignores this value; it exists for backward compatibility.
    """

    data: pd.DataFrame
    version: Literal["v1", "v2"] = "v1"
    loc: int = 0

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path, version: Literal["v1", "v2"] = "v1") -> ClimateDrivers:
        """Read a SIPNET climate file.

        Parameters
        ----------
        path:
            Path to the ``.clim`` file.
        version:
            File format version.  ``"v1"`` expects 14 columns (location index
            in col 1, soil-wetness in col 14); ``"v2"`` expects 12 columns.
        """
        from pysipnet.io.clim_io import read_clim_file

        return read_clim_file(Path(path), version=version)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        version: Literal["v1", "v2"] = "v1",
        loc: int = 0,
    ) -> ClimateDrivers:
        """Construct from a pre-built DataFrame.

        The DataFrame must contain columns matching :data:`CLIM_COLUMNS_V1`.
        Extra columns are ignored.

        Parameters
        ----------
        df:
            Input DataFrame with climate variables.
        version:
            Target file format version for serialisation.
        loc:
            Location index (v1 only).
        """
        missing = set(CLIM_COLUMNS_V1) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
        obj = cls(data=df[CLIM_COLUMNS_V1].copy(), version=version, loc=loc)
        obj.validate()
        return obj

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Check the climate data for common errors.

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
        if (self.data["length"] <= 0).any():
            raise ValueError("All 'length' values must be > 0 (timestep duration in days).")

    def _check_monotonic_time(self) -> None:
        # Construct a monotone scalar: days from an arbitrary epoch
        doy = self.data["year"] * 366 + self.data["day"] + self.data["time"] / 24.0
        if not doy.is_monotonic_increasing:
            raise ValueError(
                "Climate timesteps are not in chronological order. "
                "Rows must be sorted by (year, day, time)."
            )

    def _check_vpd_wind(self) -> None:
        if (self.data["vpd"] <= 0).any():
            n = (self.data["vpd"] <= 0).sum()
            import warnings
            warnings.warn(
                f"{n} timestep(s) have vpd ≤ 0 Pa. "
                "SIPNET adds a tiny value internally to avoid division by zero, "
                "but this may indicate a data issue.",
                stacklevel=3,
            )
        if (self.data["wspd"] <= 0).any():
            n = (self.data["wspd"] <= 0).sum()
            import warnings
            warnings.warn(
                f"{n} timestep(s) have wspd ≤ 0 m s⁻¹. "
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
        """Number of timesteps in the driving data."""
        return len(self.data)

    @property
    def date_range(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """``((start_year, start_doy), (end_year, end_doy))`` of the time series."""
        first = self.data.iloc[0]
        last = self.data.iloc[-1]
        return (
            (int(first["year"]), int(first["day"])),
            (int(last["year"]), int(last["day"])),
        )

    def __repr__(self) -> str:
        (y0, d0), (y1, d1) = self.date_range
        return (
            f"ClimateDrivers(version={self.version!r}, "
            f"timesteps={self.n_timesteps}, "
            f"range={y0}-{d0:03d} to {y1}-{d1:03d})"
        )
