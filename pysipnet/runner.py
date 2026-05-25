"""SIPNET binary runner.

The :class:`SIPNETRunner` manages subprocess execution of the SIPNET binary.
Each call to :meth:`~SIPNETRunner.run` is fully isolated: inputs are written
to a fresh temporary directory, SIPNET is executed there, and the outputs are
parsed back into a :class:`~pysipnet.result.SIPNETResult`.

This design has a key property: **runs are stateless and share no resources**,
making it trivial to parallelise them with any executor (``concurrent.futures``,
Dask, Parsl, Ray, etc.)::

    from concurrent.futures import ProcessPoolExecutor
    from pysipnet.runner import SIPNETRunner, ModelPreset

    runner = SIPNETRunner(preset=ModelPreset.FOREST)

    def run_one(config_dict):
        from pysipnet.parameters.v1 import SIPNETParametersV1
        from pysipnet.climate import ClimateDrivers
        params  = SIPNETParametersV1.model_validate(config_dict["params"])
        climate = ClimateDrivers.from_dataframe(pd.DataFrame(config_dict["climate"]))
        return runner.run(params, climate).timeseries.to_dict()

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_one, ensemble_configs))

Presets and binaries
--------------------
A :class:`ModelPreset` selects a pre-compiled SIPNET binary.  Binaries are
stored in ``.sipnet_cache/`` at the repo root and are built with::

    make sipnet   # builds all presets
    make sipnet-standard
    make sipnet-forest

The cache directory can be overridden via ``SIPNETRunner(cache_dir=...)``.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.events import EventSequence
    from pysipnet.parameters.v1 import ModelFlagsV1, SIPNETParametersV1
    from pysipnet.result import SIPNETResult

_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".sipnet_cache"


class ModelPreset(str, Enum):
    """Named SIPNET v1 binary presets.

    Each preset corresponds to a fixed set of compile-time flags (see
    ``Makefile`` for the exact ``-D`` values).  The preset selects the
    binary from ``.sipnet_cache/`` and determines which
    :class:`~pysipnet.parameters.v1.ModelFlagsV1` fields are active.

    +----------+--------------------------------------------+
    | Preset   | Active flags                               |
    +==========+============================================+
    | STANDARD | SNOW=1 GDD=1 WATER_HRESP=1                 |
    +----------+--------------------------------------------+
    | FOREST   | STANDARD + LITTER_POOL=1                   |
    +----------+--------------------------------------------+
    """

    STANDARD = "standard"
    """Default v1 configuration: snow, GDD phenology, moisture-sensitive Rh."""

    FOREST = "forest"
    """Standard + explicit litter C pool (required for sites with distinct
    litter dynamics, e.g. boreal or deciduous forest)."""

    @property
    def flags(self) -> ModelFlagsV1:
        """Return the :class:`~pysipnet.parameters.v1.ModelFlagsV1` for this preset."""
        from pysipnet.parameters.v1 import ModelFlagsV1

        if self == ModelPreset.STANDARD:
            return ModelFlagsV1.standard()
        if self == ModelPreset.FOREST:
            return ModelFlagsV1.forest()
        raise NotImplementedError(f"Flags not defined for preset {self!r}")

    @property
    def binary_name(self) -> str:
        """Filename of the compiled binary in the cache directory."""
        return f"sipnet_{self.value}"


class SIPNETRunner:
    """Execute a single SIPNET run in an isolated working directory.

    Parameters
    ----------
    preset:
        Which compiled binary preset to use.
    cache_dir:
        Directory containing pre-compiled SIPNET binaries (default:
        ``.sipnet_cache/`` at the repo root).
    workdir_base:
        Parent directory for per-run temporary working directories.  Defaults
        to the system temp directory.
    keep_workdir:
        If ``True``, do not delete the working directory after the run.
        Useful for debugging.  Default is ``False``.
    timeout:
        Maximum wall-clock time (seconds) allowed for a single SIPNET run.
        Raises :class:`subprocess.TimeoutExpired` if exceeded.
    """

    def __init__(
        self,
        preset: ModelPreset = ModelPreset.STANDARD,
        *,
        cache_dir: Path | str | None = None,
        workdir_base: Path | str | None = None,
        keep_workdir: bool = False,
        timeout: float = 300.0,
    ) -> None:
        self.preset = preset
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.workdir_base = Path(workdir_base) if workdir_base else Path(tempfile.gettempdir())
        self.keep_workdir = keep_workdir
        self.timeout = timeout

    @property
    def binary_path(self) -> Path:
        """Absolute path to the SIPNET binary for the selected preset."""
        return self.cache_dir / self.preset.binary_name

    def _check_binary(self) -> None:
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"SIPNET binary not found at {self.binary_path}. "
                "Run 'make sipnet' from the repo root to build it."
            )

    def run(
        self,
        parameters: SIPNETParametersV1,
        climate: ClimateDrivers,
        *,
        run_id: str | None = None,
        events: EventSequence | None = None,
    ) -> SIPNETResult:
        """Execute SIPNET and return a parsed result.

        Each call writes inputs to a fresh directory, runs the binary there,
        and parses the output.  The working directory is deleted on success
        unless ``keep_workdir=True``.

        Parameters
        ----------
        parameters:
            Model parameter set.  Must be compatible with the active preset's
            flags (validated via
            :meth:`~pysipnet.parameters.v1.SIPNETParametersV1.validate_for_flags`).
        climate:
            Meteorological forcing.
        run_id:
            Optional identifier for the working directory name.  Defaults to
            a random UUID hex string.
        events:
            Optional :class:`~pysipnet.events.EventSequence`.  When provided,
            the sequence is written to ``events.in`` in the working directory
            and SIPNET is run with ``EVENTS = 1``.

        Returns
        -------
        SIPNETResult
            Contains the parsed output timeseries, run provenance, and process
            metadata.
        """
        import shutil

        from pysipnet.io.clim_io import write_clim_file
        from pysipnet.io.param_io import write_param_file
        from pysipnet.result import SIPNETResult

        self._check_binary()
        flags = self.preset.flags

        run_id = run_id or uuid.uuid4().hex
        workdir = self.workdir_base / f"sipnet_{run_id}"
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            write_param_file(parameters, flags, workdir / "sipnet.param")
            write_clim_file(climate, workdir / "sipnet.clim")

            if events is not None:
                events.to_file(workdir / "events.in")
                events_flag = "1"
            else:
                events_flag = "0"

            (workdir / "sipnet.in").write_text(
                f"fileName = sipnet\n"
                f"EVENTS = {events_flag}\n"
            )

            proc = subprocess.run(
                [str(self.binary_path)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            result = SIPNETResult.from_workdir(
                workdir=workdir,
                parameters=parameters,
                climate=climate,
                flags=flags,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

        return result
