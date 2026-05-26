"""SIPNET run result container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.parameters.v1 import ModelFlagsV1, SIPNETParametersV1
    from pysipnet.runner import ModelPreset


@dataclass
class RunProvenance:
    """Execution provenance for a single SIPNET run.

    Attributes
    ----------
    preset:
        The :class:`~pysipnet.runner.ModelPreset` used for this run.
    binary_path:
        Absolute path to the SIPNET binary that was executed.
    run_id:
        Identifier string for the working directory (UUID hex by default).
    workdir:
        Directory in which the run was executed.  Contains all input and
        output files produced by SIPNET when ``keep_workdir=True``.
    returncode:
        Raw process return code from the SIPNET binary.
    success:
        Whether the SIPNET binary exited with returncode 0.
    stdout:
        Captured standard output from the SIPNET process.
    stderr:
        Captured standard error from the SIPNET process.
    """

    preset: ModelPreset
    binary_path: Path
    run_id: str
    workdir: Path
    returncode: int
    success: bool
    stdout: str
    stderr: str


@dataclass
class SIPNETResult:
    """Output from a single SIPNET run.

    Attributes
    ----------
    outputs:
        Parsed ``.out`` file as a DataFrame.  One row per model timestep,
        columns named by SIPNET output variable (e.g., ``nee``, ``gpp``,
        ``plant_wood_c``).  Column names use snake_case translations of
        SIPNET's camelCase output headers.
    parameters:
        The :class:`~pysipnet.parameters.v1.SIPNETParametersV1` used for
        this run.
    climate:
        The :class:`~pysipnet.climate.ClimateDrivers` used for this run.
    flags:
        The :class:`~pysipnet.parameters.v1.ModelFlagsV1` compiled into the
        binary used for this run.
    provenance:
        Execution metadata: binary, run ID, working directory, return code,
        stdout/stderr.
    """

    outputs: pd.DataFrame
    parameters: SIPNETParametersV1
    climate: ClimateDrivers
    flags: ModelFlagsV1
    provenance: RunProvenance

    @classmethod
    def from_workdir(
        cls,
        parameters: SIPNETParametersV1,
        climate: ClimateDrivers,
        flags: ModelFlagsV1,
        provenance: RunProvenance,
    ) -> SIPNETResult:
        """Parse SIPNET output files and construct a result object.

        Parameters
        ----------
        parameters:
            Model parameter set used for the run.
        climate:
            Climate drivers used for the run.
        flags:
            Compile-time model flags for the run.
        provenance:
            Execution provenance (binary path, return code, stdout/stderr, etc.).
            ``provenance.workdir`` is used to locate the ``sipnet.out`` file.
        """
        from pysipnet.io.output_reader import read_output_file

        out_path = provenance.workdir / "sipnet.out"
        if provenance.returncode == 0 and out_path.exists():
            outputs = read_output_file(out_path)
        else:
            outputs = pd.DataFrame()

        return cls(
            outputs=outputs,
            parameters=parameters,
            climate=climate,
            flags=flags,
            provenance=provenance,
        )

    def to_xarray(self):
        """Convert ``outputs`` to an :class:`xarray.Dataset`.

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
        return xr.Dataset.from_dataframe(self.outputs.set_index(["year", "day", "time"]))

    def nee(self) -> pd.Series:
        """Net ecosystem exchange time series (g C m⁻² per timestep, + = to atmosphere)."""
        return self.outputs["nee"]

    def gpp(self) -> pd.Series:
        """Gross primary production time series (g C m⁻² per timestep)."""
        return self.outputs["gpp"]

    def et(self) -> pd.Series:
        """Evapotranspiration time series (cm per timestep)."""
        return self.outputs["evapotranspiration"]
