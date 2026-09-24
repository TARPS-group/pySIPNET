"""Read and write SIPNET climate (``.clim``) files.

SIPNET climate files are space-delimited with **no header row** — a convention
shared by all SIPNET input files.  The file format constants below are the
single source of truth for column counts and the positions of the year and day
columns; all readers, writers, and peeks in this module derive their structure
from those constants rather than encoding it locally.

Layouts
-------
The pinned SIPNET reads two layouts, carrying the same 12 values:

* **12 columns**, the standard since SIPNET v2.0.0: ``year | day | time |
  length | tair | tsoil | par | precip | vpd | vpdSoil | vPress | wspd``
  (SIPNET's names; the Python column names are the registry names in
  :data:`pysipnet.variables.CLIMATE_VARIABLES`, e.g. ``air_temperature`` for
  ``tair``).
* **14 columns**, the legacy layout: the same 12 wrapped in a leading ``loc``
  (site identifier) and a trailing ``soilWetness``.  SIPNET v2.0.0 removed
  multi-site runs and the soil-wetness mode, so it reads both and ignores
  them, with a log line — except that it still errors if ``loc`` changes
  between rows.

**The file says which layout it is.**  SIPNET counts the fields on the first
line (``readClimData`` in ``src/sipnet/sipnet.c``) and accepts exactly 12 or
14; anything else, 13 included, is a hard error.  :func:`detect_clim_layout`
does the same, and every reader here uses it, so a caller never states the
layout of a file it is reading.  The layout pySIPNET *writes* is a separate
choice, :attr:`~pysipnet.climate.ClimateDrivers.n_columns`: 12 for new drivers,
and the file's own layout for drivers read from one, because the runner
stages a file-backed climate by copying it unchanged.

When writing 14 columns, ``loc`` is the drivers' :attr:`~pysipnet.climate.ClimateDrivers.loc`,
constant, and ``soilWetness`` a placeholder; see :data:`_SOIL_WETNESS_FILL`.

Column 8 (``par``) units
~~~~~~~~~~~~~~~~~~~~~~~~
``photosynthetically_active_radiation`` is the **total** PAR over the timestep
in Einstein m⁻². SIPNET divides by ``time_step_length`` to obtain the per-day
rate. Ensure values are consistent with the timestep length.

Column 10 (``vpd``) and column 13 (``wspd``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SIPNET requires ``vapor_pressure_deficit`` > 0 and ``wind_speed`` > 0. Values ≤ 0 are
silently clamped by SIPNET internally.  :class:`~pysipnet.climate.ClimateDrivers` warns but does
not error on non-positive values, matching SIPNET's own tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from pysipnet.climate import CLIMATE_COLUMNS, ClimateDrivers

# ── Climate file layouts ──────────────────────────────────────────────────────
#
# SIPNET works out which layout it has been handed by counting the columns on
# the first line, so the column count is the thing that identifies a layout and
# these constants are named for it.
#
#   12 columns  year day time length tair tsoil par precip vpd vpdSoil vPress wspd
#   14 columns  loc, the 12 above, soilWetness

ClimLayout = Literal[12, 14]

_N_COLS_12 = 12
_N_COLS_14 = 14

# Zero-indexed positions of year and day, which differ between layouts because
# of the leading site-identifier column.
_YEAR_COL_IN_14 = 1
_DAY_COL_IN_14 = 2
_YEAR_COL_IN_12 = 0
_DAY_COL_IN_12 = 1

# Slice of a 14-column row holding the 12 values we actually use.
_LOC_COL_IN_14 = 0
_DATA_START_IN_14 = 1
_DATA_END_IN_14 = 13

# ── Padding values ─────────────────────────────────────────────────────────────

# The soilWetness column (col 14 of the 14-column layout) is required by that layout but
# is never used by SIPNET.  Any float is valid; 0.6 is written as an innocuous
# placeholder so files look plausible on manual inspection.
_SOIL_WETNESS_FILL = 0.6


def write_clim_file(climate: ClimateDrivers, path: Path) -> None:
    """Write a SIPNET climate file in the format matching ``climate.n_columns``.

    Parameters
    ----------
    climate:
        :class:`~pysipnet.climate.ClimateDrivers` to serialize.
    path:
        Output path (typically ``<workdir>/sipnet.clim``).
    """
    if climate.n_columns == 14:
        _write_14_column(climate, path)
    elif climate.n_columns == 12:
        _write_12_column(climate, path)
    else:
        raise ValueError(
            f"Unsupported climate file layout: {climate.n_columns} columns. "
            "SIPNET reads 12- or 14-column files."
        )


def detect_clim_layout(path: Path) -> ClimLayout:
    """Which layout a climate file is in, from the fields on its first line.

    The same test SIPNET makes (``countFields`` on the first line in
    ``readClimData``), so a file this accepts is one SIPNET reads, and a file
    it refuses is one SIPNET would refuse.
    """
    with path.open() as fh:
        first_line = fh.readline()
    n_fields = len(first_line.split())
    if n_fields in (_N_COLS_12, _N_COLS_14):
        return n_fields  # type: ignore[return-value]
    if n_fields == 0:
        raise ValueError(
            f"{path} is empty or starts with a blank line. SIPNET reads the layout from the "
            "first line, so it refuses both."
        )
    if n_fields == 13:
        raise ValueError(
            f"{path} has 13 columns. The pinned SIPNET reads only 12 or 14 and refuses 13, "
            "though an earlier version accepted it. A 13-column file is normally the legacy "
            "14-column layout without its leading site column; drop the trailing soil-wetness "
            "column to make it the standard 12-column layout."
        )
    raise ValueError(
        f"{path} has {n_fields} columns on its first line. SIPNET reads a climate file only "
        f"in the {_N_COLS_12}-column layout or the legacy {_N_COLS_14}-column one."
    )


def peek_clim_file(path: Path) -> tuple[ClimLayout, int, tuple[int, int], tuple[int, int]]:
    """The layout, the row count and the first and last dates, without a full read.

    Used by :meth:`~pysipnet.climate.ClimateDrivers.from_path` to populate
    metadata without loading the whole file.  SIPNET climate files have no
    header row, so every non-blank line is a data row and the count is exact.

    Returns
    -------
    tuple
        ``(n_columns, n_rows, (start_year, start_doy), (end_year, end_doy))``.
    """
    n_columns = detect_clim_layout(path)
    with path.open() as fh:
        n_rows = sum(1 for line in fh if line.strip())

    first = pd.read_csv(path, sep=r"\s+", header=None, nrows=1, dtype=float)
    last = pd.read_csv(path, sep=r"\s+", header=None, skiprows=n_rows - 1, nrows=1, dtype=float)

    if n_columns == _N_COLS_14:
        year_col, day_col = _YEAR_COL_IN_14, _DAY_COL_IN_14
    else:
        year_col, day_col = _YEAR_COL_IN_12, _DAY_COL_IN_12

    start = (int(first.iloc[0, year_col]), int(first.iloc[0, day_col]))
    end = (int(last.iloc[0, year_col]), int(last.iloc[0, day_col]))
    return n_columns, n_rows, start, end


def read_clim_file(path: Path, *, time_zone: str | None = None) -> ClimateDrivers:
    """Read a SIPNET climate file, in whichever layout it is.

    The layout is detected from the file (:func:`detect_clim_layout`) and
    becomes the drivers' :attr:`~pysipnet.climate.ClimateDrivers.n_columns`, so
    writing them out again reproduces it.  A 14-column file's ``loc`` becomes
    their :attr:`~pysipnet.climate.ClimateDrivers.loc`.

    Parameters
    ----------
    path:
        Path to the ``.clim`` file.
    time_zone:
        The clock the file's labels are on, which the file itself cannot say;
        see :class:`~pysipnet.climate.ClimateDrivers`.

    Raises
    ------
    ValueError
        If the file is in neither layout, if a 14-column file names more than
        one location (SIPNET refuses both), or if the data fails
        :meth:`~pysipnet.climate.ClimateDrivers.validate`.
    """
    n_columns = detect_clim_layout(path)
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    if raw.shape[1] != n_columns:
        raise ValueError(
            f"{path} starts with {n_columns} columns but has {raw.shape[1]} in a later row; "
            "every row must have the same layout."
        )

    loc = 0
    if n_columns == _N_COLS_14:
        locations = raw.iloc[:, _LOC_COL_IN_14].unique()
        if len(locations) > 1:
            raise ValueError(
                f"{path} names {len(locations)} locations in its site column "
                f"({sorted(locations)[:5]}). SIPNET runs one site per file and refuses a "
                "legacy climate file whose site identifier changes between rows."
            )
        loc = int(locations[0])
        data = raw.iloc[:, _DATA_START_IN_14:_DATA_END_IN_14].copy()
    else:
        data = raw.copy()

    data.columns = CLIMATE_COLUMNS
    for col in ("year", "day_of_year"):
        data[col] = data[col].astype(int)
    return ClimateDrivers.from_dataframe(data, n_columns=n_columns, loc=loc, time_zone=time_zone)


def _write_14_column(climate: ClimateDrivers, path: Path) -> None:
    df = climate.pandas
    rows: list[str] = []
    for _, row in df.iterrows():
        parts = [
            str(climate.loc),
            str(int(row["year"])),
            str(int(row["day_of_year"])),
            f"{row['hour_of_day']:.6g}",
            f"{row['time_step_length']:.6g}",
            f"{row['air_temperature']:.6g}",
            f"{row['soil_temperature']:.6g}",
            f"{row['photosynthetically_active_radiation']:.10g}",
            f"{row['precipitation']:.6g}",
            f"{row['vapor_pressure_deficit']:.6g}",
            f"{row['soil_vapor_pressure_deficit']:.6g}",
            f"{row['vapor_pressure']:.6g}",
            f"{row['wind_speed']:.6g}",
            f"{_SOIL_WETNESS_FILL:.2f}",
        ]
        rows.append(" ".join(parts))
    path.write_text("\n".join(rows) + "\n")


def _write_12_column(climate: ClimateDrivers, path: Path) -> None:
    df = climate.pandas
    rows: list[str] = []
    for _, row in df.iterrows():
        parts = [
            str(int(row["year"])),
            str(int(row["day_of_year"])),
            f"{row['hour_of_day']:.6g}",
            f"{row['time_step_length']:.6g}",
            f"{row['air_temperature']:.6g}",
            f"{row['soil_temperature']:.6g}",
            f"{row['photosynthetically_active_radiation']:.10g}",
            f"{row['precipitation']:.6g}",
            f"{row['vapor_pressure_deficit']:.6g}",
            f"{row['soil_vapor_pressure_deficit']:.6g}",
            f"{row['vapor_pressure']:.6g}",
            f"{row['wind_speed']:.6g}",
        ]
        rows.append(" ".join(parts))
    path.write_text("\n".join(rows) + "\n")
