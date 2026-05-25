"""SIPNET run result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pysipnet.parameters.v1 import ModelFlagsV1, SIPNETParametersV1
    from pysipnet.climate import ClimateDrivers


@dataclass
class SIPNETResult:
    """Output from a single SIPNET run.

    Attributes
    ----------
    timeseries:
        Parsed ``.out`` file as a DataFrame.  One row per model timestep,
        columns named by SIPNET output variable (e.g., ``nee``, ``gpp``,
        ``plant_wood_c``).  Column names use snake_case translations of
        SIPNET's camelCase output headers.
    success:
        Whether the SIPNET binary exited with returncode 0.
    returncode:
        Raw process return code.
    stdout:
        Captured standard output from the SIPNET process.
    stderr:
        Captured standard error from the SIPNET process.
    workdir:
        Directory in which the run was executed.  Contains all input and
        output files produced by SIPNET.
    parameters:
        The :class:`~pysipnet.parameters.v1.SIPNETParametersV1` used for
        this run (for provenance).
    climate:
        The :class:`~pysipnet.climate.ClimateDrivers` used for this run.
    flags:
        The :class:`~pysipnet.parameters.v1.ModelFlagsV1` compiled into the
        binary used for this run.
    """

    timeseries: pd.DataFrame
    success: bool
    returncode: int
    stdout: str
    stderr: str
    workdir: Path
    parameters: SIPNETParametersV1
    climate: ClimateDrivers
    flags: ModelFlagsV1

    @classmethod
    def from_workdir(
        cls,
        workdir: Path,
        parameters: SIPNETParametersV1,
        climate: ClimateDrivers,
        flags: ModelFlagsV1,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> SIPNETResult:
        """Parse SIPNET output files in *workdir* and construct a result object."""
        from pysipnet.io.output_reader import read_output_file

        out_path = workdir / "sipnet.out"
        if returncode == 0 and out_path.exists():
            timeseries = read_output_file(out_path)
        else:
            timeseries = pd.DataFrame()

        return cls(
            timeseries=timeseries,
            success=(returncode == 0),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            workdir=workdir,
            parameters=parameters,
            climate=climate,
            flags=flags,
        )

    def to_xarray(self):
        """Convert ``timeseries`` to an :class:`xarray.Dataset`.

        The returned Dataset uses ``(year, day, time)`` as a multi-level
        index so they appear as coordinates rather than data variables.

        Requires the optional ``xarray`` dependency::

            pip install pysipnet[xarray]
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "xarray is required for SIPNETResult.to_xarray(). "
                "Install with: pip install pysipnet[xarray]"
            ) from exc
        return xr.Dataset.from_dataframe(
            self.timeseries.set_index(["year", "day", "time"])
        )

    def nee(self) -> pd.Series:
        """Net ecosystem exchange time series (g C m⁻² per timestep, + = to atmosphere)."""
        return self.timeseries["nee"]

    def gpp(self) -> pd.Series:
        """Gross primary production time series (g C m⁻² per timestep)."""
        return self.timeseries["gpp"]

    def et(self) -> pd.Series:
        """Evapotranspiration time series (cm per timestep)."""
        return self.timeseries["evapotranspiration"]
