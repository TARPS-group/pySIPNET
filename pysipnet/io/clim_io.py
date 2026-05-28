"""Read and write SIPNET climate (``.clim``) files.

SIPNET climate files are space-delimited with **no header row** — a convention
shared by all SIPNET input files.  The file format constants below are the
single source of truth for column counts and the positions of the year and day
columns; all readers, writers, and peeks in this module derive their structure
from those constants rather than encoding it locally.

v1 format (14 columns)
-----------------------
Two column-count variants are accepted on read:

* **14 columns (canonical)**: ``loc | year | day | time | length | tair |
  tsoil | par | precip | vpd | vpd_soil | vpress | wspd | soil_wetness``
* **13 columns (legacy)**: same layout but without the leading ``loc`` column.

The writer always produces 14 columns.  The ``loc`` column (col 1) and the
``soil_wetness`` column (col 14) are required by the file format but are never
read by SIPNET; see :data:`_V1_SOIL_WETNESS_FILL` for details.

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

v2 format (12 columns)
-----------------------
The ``loc`` and ``soil_wetness`` columns are absent.  Column order is otherwise
the same for the 12 shared variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from pysipnet.climate import CLIM_COLUMNS_V1, ClimateDrivers

# ── v1 file format constants ───────────────────────────────────────────────────

_V1_N_COLS = 14  # canonical column count (with loc and soil_wetness)
_V1_N_COLS_NO_LOC = 13  # accepted read variant (no leading loc column)

# 0-indexed positions of year and day in each variant.
_V1_YEAR_COL = 1
_V1_DAY_COL = 2
_V1_YEAR_COL_NO_LOC = 0
_V1_DAY_COL_NO_LOC = 1

# Slice of the 14-col row that contains the 12 logical data columns.
_V1_DATA_START = 1
_V1_DATA_END = 13

# ── v2 file format constants ───────────────────────────────────────────────────

_V2_N_COLS = 12
_V2_YEAR_COL = 0
_V2_DAY_COL = 1

# ── Padding values ─────────────────────────────────────────────────────────────

# The soilWetness column (col 14 in v1) is required by the v1 file format but
# is never used by SIPNET.  Any float is valid; 0.6 is written as an innocuous
# placeholder so files look plausible on manual inspection.
_V1_SOIL_WETNESS_FILL = 0.6


def write_clim_file(climate: ClimateDrivers, path: Path) -> None:
    """Write a SIPNET climate file in the format matching ``climate.version``.

    Parameters
    ----------
    climate:
        :class:`~pysipnet.climate.ClimateDrivers` to serialise.
    path:
        Output path (typically ``<workdir>/sipnet.clim``).
    """
    if climate.version == "v1":
        _write_v1(climate, path)
    elif climate.version == "v2":
        _write_v2(climate, path)
    else:
        raise ValueError(f"Unknown climate version: {climate.version!r}")


def peek_clim_file(
    path: Path, version: Literal["v1", "v2"] = "v1"
) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """Read only the first and last rows of a climate file plus the row count.

    This is a lightweight alternative to a full read, used by
    :meth:`~pysipnet.climate.ClimateDrivers.from_path` to populate metadata
    without loading the whole file.

    SIPNET climate files have no header row, so every line in the file is a
    data row.  The row count returned is therefore exact.

    The year and day column positions are determined by the module-level format
    constants (:data:`_V1_YEAR_COL`, :data:`_V1_DAY_COL`, etc.); this function
    does not encode that structure locally.

    Parameters
    ----------
    path:
        Path to the ``.clim`` file.
    version:
        File format version.

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

    if version == "v1":
        if n_cols == _V1_N_COLS:
            year_col, day_col = _V1_YEAR_COL, _V1_DAY_COL
        elif n_cols == _V1_N_COLS_NO_LOC:
            year_col, day_col = _V1_YEAR_COL_NO_LOC, _V1_DAY_COL_NO_LOC
        else:
            raise ValueError(
                f"Expected {_V1_N_COLS_NO_LOC} or {_V1_N_COLS} columns in v1 climate "
                f"file at {path}, got {n_cols}."
            )
    elif version == "v2":
        if n_cols != _V2_N_COLS:
            raise ValueError(
                f"Expected {_V2_N_COLS} columns in v2 climate file at {path}, got {n_cols}."
            )
        year_col, day_col = _V2_YEAR_COL, _V2_DAY_COL
    else:
        raise ValueError(f"Unknown climate version: {version!r}")

    start = (int(first.iloc[0, year_col]), int(first.iloc[0, day_col]))
    end = (int(last.iloc[0, year_col]), int(last.iloc[0, day_col]))
    return n_rows, start, end


def read_clim_file(path: Path, version: Literal["v1", "v2"] = "v1") -> ClimateDrivers:
    """Read a SIPNET climate file.

    Parameters
    ----------
    path:
        Path to the ``.clim`` file.
    version:
        File format version: ``"v1"`` (14 cols) or ``"v2"`` (12 cols).
    """
    if version == "v1":
        return _read_v1(path)
    if version == "v2":
        return _read_v2(path)
    raise ValueError(f"Unknown climate version: {version!r}")


def _write_v1(climate: ClimateDrivers, path: Path) -> None:
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
            f"{_V1_SOIL_WETNESS_FILL:.2f}",
        ]
        rows.append(" ".join(parts))
    path.write_text("\n".join(rows) + "\n")


def _write_v2(climate: ClimateDrivers, path: Path) -> None:
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


def _read_v1(path: Path) -> ClimateDrivers:
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    n_cols = raw.shape[1]
    if n_cols == _V1_N_COLS:
        data = raw.iloc[:, _V1_DATA_START:_V1_DATA_END].copy()
    elif n_cols == _V1_N_COLS_NO_LOC:
        data = raw.iloc[:, :12].copy()
    else:
        raise ValueError(
            f"Expected {_V1_N_COLS_NO_LOC} or {_V1_N_COLS} columns in v1 climate file, "
            f"got {n_cols}. Ensure the file is in SIPNET v1 format."
        )
    data.columns = CLIM_COLUMNS_V1  # type: ignore[assignment]
    for col in ("year", "day"):
        data[col] = data[col].astype(int)
    return ClimateDrivers.from_dataframe(data, version="v1")


def _read_v2(path: Path) -> ClimateDrivers:
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    n_cols = raw.shape[1]
    if n_cols != _V2_N_COLS:
        raise ValueError(
            f"Expected {_V2_N_COLS} columns in v2 climate file, got {n_cols}. "
            "Ensure the file is in SIPNET v2 format."
        )
    data = raw.copy()
    data.columns = CLIM_COLUMNS_V1  # type: ignore[assignment]
    for col in ("year", "day"):
        data[col] = data[col].astype(int)
    return ClimateDrivers.from_dataframe(data, version="v2")
