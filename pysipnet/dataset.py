"""Turn a per-timestep DataFrame into a self-describing :class:`xarray.Dataset`.

Shared by :class:`~pysipnet.output.SIPNETOutput` and
:class:`~pysipnet.climate.ClimateDrivers`, so outputs and drivers have the same
time axis and can be aligned or merged directly.

Time convention
---------------
The ``time`` coordinate is the **start** of each timestep, as SIPNET labels
its rows. ``time_step_end`` and ``time_step_length`` coordinates are added
when the step lengths are known. Every data variable carries the attributes
its registry spec provides, including ``time_reference`` in words.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from pysipnet.variables import TIME_COORDINATE_NAMES

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

TIME_DIMENSION = "time"

TIME_CONVENTION = (
    "The 'time' coordinate is the START of each timestep, as SIPNET labels its rows. "
    "State variables are values at the END of the timestep; fluxes are totals OVER "
    "the timestep. Each variable's 'time_reference' attribute says which."
)


def dataframe_to_dataset(
    df: pd.DataFrame,
    *,
    attributes_for: Callable[[str], dict[str, Any]],
    time_step_length: np.ndarray | None = None,
    source: str,
) -> xr.Dataset:
    """Build a Dataset with one ``time`` dimension from a per-timestep DataFrame.

    Parameters
    ----------
    df:
        Must contain ``year``, ``day_of_year`` and ``hour_of_day`` columns
        marking the start of each step. Every other column becomes a data
        variable.
    attributes_for:
        Returns the ``.attrs`` for a column name; ``{}`` for unknown columns.
    time_step_length:
        Step lengths in days, one per row. Adds ``time_step_length`` and
        ``time_step_end`` coordinates.
    source:
        Free text for the Dataset's ``source`` attribute.
    """
    import pandas as pd
    import xarray as xr

    if df.empty:
        return xr.Dataset()

    missing = [c for c in TIME_COORDINATE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot build a time coordinate: data is missing {missing}. "
            "Was the file read without a header row?"
        )

    start = (
        pd.to_datetime(df["year"].astype(int).astype(str), format="%Y")
        + pd.to_timedelta(df["day_of_year"].astype(int) - 1, unit="D")
        + pd.to_timedelta(df["hour_of_day"].astype(float), unit="h")
    )

    coords: dict[str, Any] = {
        TIME_DIMENSION: (
            TIME_DIMENSION,
            start.to_numpy(),
            {
                "long_name": "Start of timestep",
                "description": "Calendar time at the start of the timestep, as SIPNET labels it.",
            },
        ),
    }
    for name in TIME_COORDINATE_NAMES:
        coords[name] = (TIME_DIMENSION, df[name].to_numpy(), attributes_for(name))

    if time_step_length is not None:
        length = np.asarray(time_step_length, dtype=float)
        if len(length) != len(df):
            raise ValueError(
                f"time_step_length has {len(length)} values but the data has {len(df)} rows."
            )
        length_td = pd.to_timedelta(length, unit="D").to_numpy()
        coords["time_step_length"] = (
            TIME_DIMENSION,
            length_td,
            {"long_name": "Timestep length", "description": "Duration of the timestep."},
        )
        coords["time_step_end"] = (
            TIME_DIMENSION,
            start.to_numpy() + length_td,
            {
                "long_name": "End of timestep",
                "description": "Calendar time at the end of the timestep; state variables "
                "are valid at this instant.",
            },
        )

    # A time_step_length column becomes the coordinate of that name when lengths
    # are given; otherwise it stays an ordinary variable.
    as_coordinate = set(TIME_COORDINATE_NAMES)
    if time_step_length is not None:
        as_coordinate.add("time_step_length")
    data_vars = {
        name: (TIME_DIMENSION, df[name].to_numpy(), attributes_for(name))
        for name in df.columns
        if name not in as_coordinate
    }

    return xr.Dataset(
        data_vars,
        coords=coords,
        attrs={"source": source, "time_convention": TIME_CONVENTION},
    )
