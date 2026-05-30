"""Parse SIPNET ``.out`` output files.

Output file format
------------------
When compiled with ``HEADER=1`` (as done by the pySIPNET build system), the
``.out`` file begins with a space-delimited header row followed by one data
row per timestep.

The output column names use SIPNET's camelCase conventions.  This module
translates them to snake_case for the Python side.

Column layout (v1, HEADER=1)
-----------------------------
All columns are always present regardless of active compile-time flags;
inactive features output zeros.

+----+-------------------------+--------+----------------------------------------+
| #  | SIPNET name             | Unit   | Notes                                  |
+====+=========================+========+========================================+
|  1 | year                    | —      |                                        |
+----+-------------------------+--------+----------------------------------------+
|  2 | day                     | —      | Day of year                            |
+----+-------------------------+--------+----------------------------------------+
|  3 | time                    | hours  | Fractional                             |
+----+-------------------------+--------+----------------------------------------+
|  4 | plantWoodC              | g C/m² |                                        |
+----+-------------------------+--------+----------------------------------------+
|  5 | plantLeafC              | g C/m² |                                        |
+----+-------------------------+--------+----------------------------------------+
|  6 | woodCreation            | g C/m² | C allocated to wood this step          |
+----+-------------------------+--------+----------------------------------------+
|  7 | soil                    | g C/m² |                                        |
+----+-------------------------+--------+----------------------------------------+
|  8 | coarseRootC             | g C/m² |                                        |
+----+-------------------------+--------+----------------------------------------+
|  9 | fineRootC               | g C/m² |                                        |
+----+-------------------------+--------+----------------------------------------+
| 10 | litter                  | g C/m² | 0 when LITTER_POOL=0                  |
+----+-------------------------+--------+----------------------------------------+
| 11 | soilWater               | cm     |                                        |
+----+-------------------------+--------+----------------------------------------+
| 12 | soilWetnessFrac         | —      | soilWater / soilWHC                    |
+----+-------------------------+--------+----------------------------------------+
| 13 | snow                    | cm     | Water equivalent; 0 when SNOW=0       |
+----+-------------------------+--------+----------------------------------------+
| 14 | npp                     | g C/m² | Per timestep                           |
+----+-------------------------+--------+----------------------------------------+
| 15 | nee                     | g C/m² | Positive = flux to atmosphere          |
+----+-------------------------+--------+----------------------------------------+
| 16 | cumNEE                  | g C/m² | Cumulative                             |
+----+-------------------------+--------+----------------------------------------+
| 17 | gpp                     | g C/m² |                                        |
+----+-------------------------+--------+----------------------------------------+
| 18 | rAboveground            | g C/m² | Aboveground autotrophic respiration    |
+----+-------------------------+--------+----------------------------------------+
| 19 | rSoil                   | g C/m² | Soil respiration (Rh + root)           |
+----+-------------------------+--------+----------------------------------------+
| 20 | rRoot                   | g C/m² | Root respiration                       |
+----+-------------------------+--------+----------------------------------------+
| 21 | ra                      | g C/m² | Total autotrophic respiration          |
+----+-------------------------+--------+----------------------------------------+
| 22 | rh                      | g C/m² | Heterotrophic respiration              |
+----+-------------------------+--------+----------------------------------------+
| 23 | rtot                    | g C/m² | Total ecosystem respiration (Ra + Rh)  |
+----+-------------------------+--------+----------------------------------------+
| 24 | evapotranspiration      | cm     |                                        |
+----+-------------------------+--------+----------------------------------------+
| 25 | fluxestranspiration     | cm/day | Transpiration component                |
+----+-------------------------+--------+----------------------------------------+
| 26 | fPAR                    | —      | Fraction of PAR absorbed               |
+----+-------------------------+--------+----------------------------------------+

Note: ``fPAR`` and ``microbeC`` appear in v1 output but not v2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Maps SIPNET camelCase output column names → snake_case Python names.
SIPNET_TO_PYTHON_OUTPUT: dict[str, str] = {
    "year": "year",
    "day": "day",
    "time": "time",
    "plantWoodC": "plant_wood_c",
    "plantLeafC": "plant_leaf_c",
    "woodCreation": "wood_creation",
    "soil": "soil_c",
    "coarseRootC": "coarse_root_c",
    "fineRootC": "fine_root_c",
    "litter": "litter_c",
    "soilWater": "soil_water",
    "soilWetnessFrac": "soil_wetness_frac",
    "snow": "snow",
    "npp": "npp",
    "nee": "nee",
    "cumNEE": "cum_nee",
    "gpp": "gpp",
    "rAboveground": "r_aboveground",
    "rSoil": "r_soil",
    "rRoot": "r_root",
    "ra": "ra",
    "rh": "rh",
    "rtot": "rtot",
    "evapotranspiration": "evapotranspiration",
    "fluxestranspiration": "transpiration",
    "fPAR": "f_par",
    # v1 only
    "microbeC": "microbe_c",
    "litterWater": "litter_water",
}


def read_output_file(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Parse a SIPNET ``.out`` file into a DataFrame.

    Handles both ``HEADER=1`` (column names present) and ``HEADER=0`` (no
    header, positional columns assumed) output formats.  The pySIPNET build
    system always compiles with ``HEADER=1``, so the no-header path is
    provided only for compatibility with externally compiled binaries.

    Column names in the returned DataFrame use the snake_case names from
    :data:`SIPNET_TO_PYTHON_OUTPUT`.

    Parameters
    ----------
    path:
        Path to the ``.out`` file.
    columns:
        Subset of snake_case column names to return.  The time-coordinate
        columns ``year``, ``day``, and ``time`` are always included regardless
        of this argument.  ``None`` returns all columns.  Column filtering is
        only applied for ``HEADER=1`` output; ``HEADER=0`` files always return
        all columns.
    """
    from io import StringIO

    lines = path.read_text().splitlines()
    if not lines:
        return pd.DataFrame()

    if lines[0].startswith("Notes:"):
        # HEADER=1: line 0 is "Notes: ...", line 1 is space-separated column names
        sipnet_cols = lines[1].split()
        python_cols = [SIPNET_TO_PYTHON_OUTPUT.get(c, c) for c in sipnet_cols]
        data_text = "\n".join(lines[2:])

        if columns is not None:
            time_coords = {"year", "day", "time"}
            requested = time_coords | set(columns)
            usecols = [c for c in python_cols if c in requested]
        else:
            usecols = None
    else:
        # HEADER=0: no header, use positional integer column indices
        python_cols = None
        usecols = None
        data_text = "\n".join(lines)

    df = pd.read_csv(
        StringIO(data_text),
        sep=r"\s+",
        header=None,
        names=python_cols,
        usecols=usecols,
    )
    return df
