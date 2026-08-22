"""Read and write SIPNET climate (``.clim``) files.

SIPNET climate files are space-delimited with **no header row** — a convention
shared by all SIPNET input files.  The file format constants below are the
single source of truth for column counts and the positions of the year and day
columns; all readers, writers, and peeks in this module derive their structure
from those constants rather than encoding it locally.

14-column layout
----------------
Two column-count variants are accepted on read:

* **14 columns (what the writer produces)**: ``loc | year | day | time | length | tair |
  tsoil | par | precip | vpd | vpd_soil | vpress | wspd | soil_wetness``
* **13 columns**: the same layout without the leading ``loc`` column.

The writer always produces 14 columns.  The ``loc`` column (col 1) and the
``soil_wetness`` column (col 14) are required by the file format but are never
read by SIPNET; see :data:`_SOIL_WETNESS_FILL` for details.

Column 8 (``par``) units
~~~~~~~~~~~~~~~~~~~~~~~~~
The ``par`` column is the **total** PAR over the timestep in Einstein m⁻².
SIPNET divides by the ``length`` column to obtain the per-day rate.  Ensure
values are consistent with the timestep length.

Column 10 (``vpd``) and column 13 (``wspd``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SIPNET requires vpd > 0 and wspd > 0.  Values ≤ 0 are silently clamped by
SIPNET internally.  :class:`~pysipnet.climate.ClimateDrivers` warns but does
not error on non-positive values, matching SIPNET's own tolerance.

12-column layout
----------------
The ``loc`` and ``soil_wetness`` columns are absent. The remaining 12 values
and their order are unchanged. This is what current SIPNET writes and the
leaner choice for new files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from pysipnet.climate import CLIM_COLUMNS, ClimateDrivers

# ── Climate file layouts ──────────────────────────────────────────────────────
#
# SIPNET works out which layout it has been handed by counting the columns on
# the first line, so the column count is the thing that identifies a layout and
# these constants are named for it.
#
# All three carry the same 12 values. They differ only in what is wrapped
# around them:
#
#   12 columns  year day time length tair tsoil par precip vpd vpdSoil vPress wspd
#   13 columns  the 12 above, plus a trailing soil wetness value SIPNET ignores
#   14 columns  the 13 above, plus a leading site identifier SIPNET ignores

_N_COLS_12 = 12
_N_COLS_13 = 13
_N_COLS_14 = 14

# Zero-indexed positions of year and day, which differ between layouts because
# of the leading site-identifier column.
_YEAR_COL_IN_14 = 1
_DAY_COL_IN_14 = 2
_YEAR_COL_IN_13 = 0
_DAY_COL_IN_13 = 1
_YEAR_COL_IN_12 = 0
_DAY_COL_IN_12 = 1

# Slice of a 14-column row holding the 12 values we actually use.
_DATA_START_IN_14 = 1
_DATA_END_IN_14 = 13

# ── Padding values ─────────────────────────────────────────────────────────────

# The soilWetness column (col 14 in v1) is required by the v1 file format but
# is never used by SIPNET.  Any float is valid; 0.6 is written as an innocuous
# placeholder so files look plausible on manual inspection.
_SOIL_WETNESS_FILL = 0.6


def write_clim_file(climate: ClimateDrivers, path: Path) -> None:
    """Write a SIPNET climate file in the format matching ``climate.n_columns``.

    Parameters
    ----------
    climate:
        :class:`~pysipnet.climate.ClimateDrivers` to serialise.
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


def peek_clim_file(
    path: Path, n_columns: Literal[12, 14] = 14
) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """Read only the first and last rows of a climate file plus the row count.

    This is a lightweight alternative to a full read, used by
    :meth:`~pysipnet.climate.ClimateDrivers.from_path` to populate metadata
    without loading the whole file.

    SIPNET climate files have no header row, so every line in the file is a
    data row.  The row count returned is therefore exact.

    The year and day column positions are determined by the module-level format
    constants (:data:`_YEAR_COL_IN_14`, :data:`_DAY_COL_IN_14`, etc.); this function
    does not encode that structure locally.

    Parameters
    ----------
    path:
        Path to the ``.clim`` file.
    n_columns:
        Which file layout to expect: 12 or 14 columns.

    Returns
    -------
    tuple
        ``(n_rows, (start_year, start_doy), (end_year, end_doy))``.
    """
    with path.open() as fh:
        n_rows = sum(1 for line in fh if line.strip())

    if n_rows == 0:
        raise ValueError(f"Climate file is empty: {path}")

    first = pd.read_csv(path, sep=r"\s+", header=None, nrows=1, dtype=float)
    last = pd.read_csv(path, sep=r"\s+", header=None, skiprows=n_rows - 1, nrows=1, dtype=float)

    n_cols = first.shape[1]

    if n_columns == 14:
        if n_cols == _N_COLS_14:
            year_col, day_col = _YEAR_COL_IN_14, _DAY_COL_IN_14
        elif n_cols == _N_COLS_13:
            year_col, day_col = _YEAR_COL_IN_13, _DAY_COL_IN_13
        else:
            raise ValueError(
                f"Expected {_N_COLS_13} or {_N_COLS_14} columns in a 'v1' climate "
                f"file at {path}, got {n_cols}."
            )
    elif n_columns == 12:
        if n_cols != _N_COLS_12:
            raise ValueError(
                f"Expected {_N_COLS_12} columns in a 'v2' climate file at {path}, got {n_cols}."
            )
        year_col, day_col = _YEAR_COL_IN_12, _DAY_COL_IN_12
    else:
        raise ValueError(
        f"Unsupported climate file layout: {n_columns} columns. "
        "SIPNET reads 12- or 14-column files."
    )

    start = (int(first.iloc[0, year_col]), int(first.iloc[0, day_col]))
    end = (int(last.iloc[0, year_col]), int(last.iloc[0, day_col]))
    return n_rows, start, end


def read_clim_file(path: Path, n_columns: Literal[12, 14] = 14) -> ClimateDrivers:
    """Read a SIPNET climate file.

    Parameters
    ----------
    path:
        Path to the ``.clim`` file.
    n_columns:
        Which file layout to expect. 14 also accepts a 13-column file, which
        is the same layout without the leading site-identifier column.
    """
    if n_columns == 14:
        return _read_14_column(path)
    if n_columns == 12:
        return _read_12_column(path)
    raise ValueError(
        f"Unsupported climate file layout: {n_columns} columns. "
        "SIPNET reads 12- or 14-column files."
    )


def _write_14_column(climate: ClimateDrivers, path: Path) -> None:
    df = climate.data
    rows: list[str] = []
    for _, row in df.iterrows():
        parts = [
            str(climate.loc),
            str(int(row["year"])),
            str(int(row["day"])),
            f"{row['time']:.6g}",
            f"{row['length']:.6g}",
            f"{row['tair']:.6g}",
            f"{row['tsoil']:.6g}",
            f"{row['par']:.10g}",
            f"{row['precip']:.6g}",
            f"{row['vpd']:.6g}",
            f"{row['vpd_soil']:.6g}",
            f"{row['vpress']:.6g}",
            f"{row['wspd']:.6g}",
            f"{_SOIL_WETNESS_FILL:.2f}",
        ]
        rows.append(" ".join(parts))
    path.write_text("\n".join(rows) + "\n")


def _write_12_column(climate: ClimateDrivers, path: Path) -> None:
    df = climate.data
    rows: list[str] = []
    for _, row in df.iterrows():
        parts = [
            str(int(row["year"])),
            str(int(row["day"])),
            f"{row['time']:.6g}",
            f"{row['length']:.6g}",
            f"{row['tair']:.6g}",
            f"{row['tsoil']:.6g}",
            f"{row['par']:.10g}",
            f"{row['precip']:.6g}",
            f"{row['vpd']:.6g}",
            f"{row['vpd_soil']:.6g}",
            f"{row['vpress']:.6g}",
            f"{row['wspd']:.6g}",
        ]
        rows.append(" ".join(parts))
    path.write_text("\n".join(rows) + "\n")


def _read_14_column(path: Path) -> ClimateDrivers:
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    n_cols = raw.shape[1]
    if n_cols == _N_COLS_14:
        data = raw.iloc[:, _DATA_START_IN_14:_DATA_END_IN_14].copy()
    elif n_cols == _N_COLS_13:
        data = raw.iloc[:, :12].copy()
    else:
        raise ValueError(
            f"Expected {_N_COLS_13} or {_N_COLS_14} columns in a 'v1' climate file, "
            f"got {n_cols}. Expected the 14- or 13-column layout."
        )
    data.columns = CLIM_COLUMNS  # type: ignore[assignment]
    for col in ("year", "day"):
        data[col] = data[col].astype(int)
    return ClimateDrivers.from_dataframe(data, n_columns=14)


def _read_12_column(path: Path) -> ClimateDrivers:
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    n_cols = raw.shape[1]
    if n_cols != _N_COLS_12:
        raise ValueError(
            f"Expected {_N_COLS_12} columns in a 'v2' climate file, got {n_cols}. "
            "Expected the 12-column layout."
        )
    data = raw.copy()
    data.columns = CLIM_COLUMNS  # type: ignore[assignment]
    for col in ("year", "day"):
        data[col] = data[col].astype(int)
    return ClimateDrivers.from_dataframe(data, n_columns=12)
