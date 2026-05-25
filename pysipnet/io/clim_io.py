"""Read and write SIPNET climate (``.clim``) files.

v1 format (14 columns, space-delimited, no header)
---------------------------------------------------
The first column (location index) and the last column (soilWetness) are
required by the file format but ignored by SIPNET.  They are written as
constant values (``loc=0``, ``soil_wetness=0.6`` by convention) and discarded
on read.

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
The location and soilWetness columns are absent.  Column order is otherwise
the same for the 12 shared variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from pysipnet.climate import CLIM_COLUMNS_V1, ClimateDrivers

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
    raise NotImplementedError


def _read_v1(path: Path) -> ClimateDrivers:
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    n_cols = raw.shape[1]
    if n_cols == 14:
        data = raw.iloc[:, 1:13].copy()
    elif n_cols == 13:
        data = raw.iloc[:, :12].copy()
    else:
        raise ValueError(
            f"Expected 13 or 14 columns in v1 climate file, got {n_cols}. "
            "Ensure the file is in SIPNET v1 format."
        )
    data.columns = CLIM_COLUMNS_V1  # type: ignore[assignment]
    for col in ("year", "day"):
        data[col] = data[col].astype(int)
    return ClimateDrivers.from_dataframe(data, version="v1")


def _read_v2(path: Path) -> ClimateDrivers:
    raise NotImplementedError
