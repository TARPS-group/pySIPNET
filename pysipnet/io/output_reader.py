"""Read SIPNET's ``.out`` output file into a pandas DataFrame.

File layout
-----------
pySIPNET always asks SIPNET for a header row (``PRINT_HEADER = 1`` in the
generated ``sipnet.in``), so the file is a row of column names followed by one
row per timestep, all space-separated.

Columns are matched by name rather than by position. That matters because the
set of columns changes between SIPNET versions, so anything positional would
silently read the wrong values after a change.

Older SIPNET versions, up to and including v2.0.0, wrote an extra ``Notes:``
line above the header. v2.1.0 removed it. Both layouts are handled, along with
files that have no header at all, which is what an externally compiled binary
run with ``--no-print-header`` produces.

Column names
------------
SIPNET's header tokens (``plantWoodC``, ``rSoil``, ``nppStorage``) are renamed
to the registry names in :mod:`pysipnet.variables` (``wood_carbon``,
``soil_respiration``, ``wood_storage_carbon``). The registry is the single
statement of that mapping and of what each column means; this module only
applies it.

A header token the registry does not know is kept under its SIPNET name and
reported with a :class:`UnknownOutputColumnWarning`. That is deliberately loud:
a column added upstream should be modelled, not silently passed through, and
``tests/test_variables.py`` asserts the pinned binary produces none.
"""

from __future__ import annotations

import warnings
from io import StringIO
from pathlib import Path

import pandas as pd

from pysipnet.variables import (
    LEGACY_OUTPUT_COLUMNS,
    OUTPUT_VARIABLES_BY_SIPNET_NAME,
    TIME_COORDINATE_NAMES,
    resolve_output_variable_names,
)


class UnknownOutputColumnWarning(UserWarning):
    """SIPNET wrote a column the variable registry does not describe."""


def sipnet_column_to_variable_name(sipnet_name: str) -> str:
    """Map one SIPNET header token to its pySIPNET variable name.

    Unknown tokens are returned unchanged, with a warning.
    """
    spec = OUTPUT_VARIABLES_BY_SIPNET_NAME.get(sipnet_name)
    if spec is not None:
        return spec.name
    legacy = LEGACY_OUTPUT_COLUMNS.get(sipnet_name)
    if legacy is not None:
        return legacy
    warnings.warn(
        f"SIPNET output column {sipnet_name!r} is not in the variable registry; "
        "keeping its SIPNET name. Add it to pysipnet.variables.OUTPUT_VARIABLES.",
        UnknownOutputColumnWarning,
        stacklevel=3,
    )
    return sipnet_name


def _split_header(lines: list[str]) -> tuple[list[str] | None, int]:
    """Work out where the header is and where the data starts.

    Returns the SIPNET column names (or ``None`` when the file has no header)
    and the index of the first data row.

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


def read_output_file(path: Path, variables: list[str] | None = None) -> pd.DataFrame:
    """Read a SIPNET ``.out`` file into a DataFrame with registry column names.

    Parameters
    ----------
    path:
        The ``.out`` file to read.
    variables:
        Which variables to keep, by registry name or alias (``"nee"`` and
        ``"net_ecosystem_exchange"`` both work). The time coordinates
        ``year``, ``day_of_year`` and ``hour_of_day`` are always kept because
        they identify each row. ``None`` keeps everything. Ignored for files
        with no header row, where columns cannot be selected by name.

    Returns
    -------
    pandas.DataFrame
        One row per timestep. Empty if the file is empty.
    """
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
        python_cols = [sipnet_column_to_variable_name(c) for c in sipnet_cols]
        if variables is not None:
            requested = set(TIME_COORDINATE_NAMES) | set(resolve_output_variable_names(variables))
            usecols = [c for c in python_cols if c in requested]
        else:
            usecols = None

    return pd.read_csv(
        StringIO(data_text),
        sep=r"\s+",
        header=None,
        names=python_cols,
        usecols=usecols,
    )
