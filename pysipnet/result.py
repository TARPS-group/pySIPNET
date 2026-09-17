"""SIPNET run result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.events import EventSequence
    from pysipnet.output import SIPNETOutput
    from pysipnet.parameters.model import ModelFlags, SIPNETParameters


@dataclass
class RunProvenance:
    """Execution provenance for a single SIPNET run.

    Attributes
    ----------
    flags:
        The :class:`~pysipnet.parameters.model.ModelFlags` this run used.
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

    flags: ModelFlags
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
        Parsed ``.out`` file as a :class:`~pysipnet.output.SIPNETOutput`.
        ``result.outputs.pandas`` is a DataFrame, ``result.outputs.xarray`` an
        xarray Dataset with units and descriptions attached, and
        ``result.outputs["nee"]`` one variable by name or alias. Use
        ``result.outputs.load(variables=[...])`` to read only a subset from a
        file-backed instance.
    parameters:
        The :class:`~pysipnet.parameters.model.SIPNETParameters` used for
        this run.
    climate:
        The :class:`~pysipnet.climate.ClimateDrivers` used for this run.
    flags:
        The :class:`~pysipnet.parameters.model.ModelFlags` used for this run.
        Model options are chosen at run time and written into ``sipnet.in``,
        so these are a property of the run, not of the binary.
    provenance:
        Execution metadata: binary, run ID, working directory, return code,
        stdout/stderr.
    events:
        The :class:`~pysipnet.events.EventSequence` used for this run, or
        ``None`` if no events were supplied.
    """

    outputs: SIPNETOutput
    parameters: SIPNETParameters
    climate: ClimateDrivers
    flags: ModelFlags
    provenance: RunProvenance
    events: EventSequence | None = field(default=None)
