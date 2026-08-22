"""Read SIPNET's ``.out`` output file into a pandas DataFrame.

File layout
-----------
pySIPNET always asks SIPNET for a header row (``PRINT_HEADER = 1`` in the
generated ``sipnet.in``), so the file is a row of column names followed by one
row per timestep, all space-separated.

Columns are matched by name rather than by position. That matters because the
set of columns depends on which processes are switched on, and it changes
between SIPNET versions, so anything positional would silently read the wrong
values after a change.

Older SIPNET versions, up to and including v2.0.0, wrote an extra ``Notes:``
line above the header. v2.1.0 removed it. Both layouts are handled, along with
files that have no header at all, which is what an externally compiled binary
run with ``--no-print-header`` produces.

Columns at the pinned version
-----------------------------
36 columns, always all present: SIPNET writes a column even for a process that
is switched off, filling it with zeros. Grouped by what they describe:

- **Time**: ``year``, ``day``, ``time``
- **Carbon pools**: ``plantWoodC``, ``plantLeafC``, ``soil``, ``coarseRootC``,
  ``fineRootC``, ``litter``
- **Carbon fluxes**: ``woodCreation``, ``npp``, ``nee``, ``cumNEE``, ``gpp``,
  ``nppStorage``
- **Respiration**: ``rAboveground``, ``rSoil``, ``rRoot``, ``ra``, ``rh``,
  ``rtot``
- **Water**: ``soilWater``, ``soilWetnessFrac``, ``snow``,
  ``evapotranspiration``, ``fluxestranspiration``
- **Nitrogen** (zero unless the nitrogen cycle is on): ``minN``, ``soilOrgN``,
  ``litterN``, ``n2o``, ``nLeaching``, ``nFixation``, ``nUptake``
- **Methane** (zero unless anaerobic processes are on): ``ch4``
- **Mass-balance checks**: ``bcdeltaC``, ``bcdeltaN``, the carbon and nitrogen
  closure errors SIPNET computes for itself. Both should stay at or near zero;
  a drift away from zero indicates a problem inside the model run.

``microbeC``, ``litterWater`` and ``fPAR`` were written by older versions and
are retained in the name mapping so previously saved output still reads.
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
    # Nitrogen cycle; zero unless ModelFlags.nitrogen_cycle is on.
    "minN": "mineral_n",
    "soilOrgN": "soil_organic_n",
    "litterN": "litter_n",
    "n2o": "n2o",
    "nLeaching": "n_leaching",
    "nFixation": "n_fixation",
    "nUptake": "n_uptake",
    # Methane; zero unless ModelFlags.anaerobic is on.
    "ch4": "ch4",
    # Carbon held back from allocation to represent storage lag.
    "nppStorage": "npp_storage",
    # SIPNET's own mass-balance closure errors; should stay near zero.
    "bcdeltaC": "balance_delta_c",
    "bcdeltaN": "balance_delta_n",
    # Written by SIPNET versions older than the pinned one; kept so that
    # previously saved output files still read.
    "fPAR": "f_par",
    "microbeC": "microbe_c",
    "litterWater": "litter_water",
}


def _split_header(lines: list[str]) -> tuple[list[str] | None, int]:
    """Work out where the header is and where the data starts.

    Returns the column names (or ``None`` when the file has no header) and the
    index of the first data row.

    Three layouts occur in practice:

    - a ``Notes:`` line, then the header, then data (SIPNET up to v2.0.0)
    - the header, then data (the pinned version)
    - data only, no header (a binary run with ``--no-print-header``)

    The first field of a data row is the year, so a first field that does not
    parse as a number means the line is a header.
    """
    if lines[0].startswith("Notes:"):
        return lines[1].split(), 2

    first_field = lines[0].split()[0] if lines[0].split() else ""
    try:
        float(first_field)
    except ValueError:
        return lines[0].split(), 1

    return None, 0


def read_output_file(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a SIPNET ``.out`` file into a DataFrame.

    Column names in the result are the snake_case names from
    :data:`SIPNET_TO_PYTHON_OUTPUT`. A SIPNET column with no entry in that
    mapping keeps its original name, so a column added by a future SIPNET
    version is still readable.

    Parameters
    ----------
    path:
        The ``.out`` file to read.
    columns:
        Which columns to keep, using the snake_case names. ``year``, ``day``
        and ``time`` are always kept because they identify each row. Pass
        ``None`` for everything. Ignored for files with no header row, where
        columns cannot be selected by name.

    Returns
    -------
    pandas.DataFrame
        One row per timestep. Empty if the file is empty.
    """
    from io import StringIO

    lines = path.read_text().splitlines()
    if not lines:
        return pd.DataFrame()

    sipnet_cols, data_start = _split_header(lines)
    data_text = "\n".join(lines[data_start:])

    if sipnet_cols is None:
        # No header row: fall back to positional integer column labels, since
        # there is no reliable way to name columns whose order we cannot check.
        python_cols = None
        usecols = None
    else:
        python_cols = [SIPNET_TO_PYTHON_OUTPUT.get(c, c) for c in sipnet_cols]
        if columns is not None:
            # The time coordinates identify each row, so they are always kept.
            requested = {"year", "day", "time"} | set(columns)
            usecols = [c for c in python_cols if c in requested]
        else:
            usecols = None

    df = pd.read_csv(
        StringIO(data_text),
        sep=r"\s+",
        header=None,
        names=python_cols,
        usecols=usecols,
    )
    return df
