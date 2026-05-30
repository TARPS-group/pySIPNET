"""SIPNET binary runner.

The :class:`SIPNETRunner` manages subprocess execution of the SIPNET binary.
Each call to :meth:`~SIPNETRunner.run` is fully isolated: inputs are written
to a fresh temporary directory, SIPNET is executed there, and the outputs are
returned as a :class:`~pysipnet.result.SIPNETResult`.

This design has a key property: **runs are stateless and share no resources**,
making it trivial to parallelise them with any executor (``concurrent.futures``,
Dask, Parsl, Ray, etc.)::

    from concurrent.futures import ProcessPoolExecutor
    from pysipnet.runner import SIPNETRunner, ModelPreset

    runner = SIPNETRunner(preset=ModelPreset.FOREST)

    def run_one(config_dict):
        from pysipnet.parameters.v1 import SIPNETParametersV1
        from pysipnet.climate import ClimateDrivers
        import pandas as pd
        params  = SIPNETParametersV1.model_validate(config_dict["params"])
        climate = ClimateDrivers.from_dataframe(pd.DataFrame(config_dict["climate"]))
        return runner.run(params, climate).outputs.data.to_dict()

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

Output persistence
------------------
By default the output is parsed eagerly and the working directory is deleted.
Set ``output_dir`` to copy ``sipnet.out`` to a stable location before cleanup
and return a file-backed :class:`~pysipnet.output.SIPNETOutput` instead::

    runner = SIPNETRunner(
        preset=ModelPreset.STANDARD,
        output_dir=Path("ensemble_out"),
    )
    results = [runner.run(params_i, climate, run_id=f"m{i}") for i in range(1000)]
    # No DataFrames in memory yet.
    nee = pd.concat([r.outputs.load(columns=["nee"]) for r in results])

Each run writes ``sipnet_<run_id>.out`` inside ``output_dir``.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.events import EventSequence
    from pysipnet.output import SIPNETOutput
    from pysipnet.parameters.v1 import ModelFlagsV1, SIPNETParametersV1
    from pysipnet.result import RunProvenance, SIPNETResult

# Sentinel used to distinguish "not passed" from None in output_dir overrides.
_UNSET = object()

_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".sipnet_cache"


class ClimateStaging(StrEnum):
    """How the runner stages the climate file into each run's working directory.

    +----------+--------------------------------------------------------------+
    | Value    | Behaviour                                                    |
    +==========+==============================================================+
    | COPY     | Copies the source file with :func:`shutil.copy2`.  Safe on  |
    |          | all platforms and across filesystem boundaries.  Default.    |
    +----------+--------------------------------------------------------------+
    | SYMLINK  | Creates a symbolic link pointing at the resolved source path.|
    |          | Zero I/O overhead; requires the source file to remain        |
    |          | accessible for the duration of the run.  Falls back to COPY  |
    |          | with a warning if :func:`os.symlink` raises :class:`OSError`.|
    +----------+--------------------------------------------------------------+

    Only applies to file-backed :class:`~pysipnet.climate.ClimateDrivers`
    instances created via :meth:`~pysipnet.climate.ClimateDrivers.from_path`.
    In-memory instances are always written via the I/O layer regardless of
    this setting.
    """

    COPY = "copy"
    SYMLINK = "symlink"


class ModelPreset(StrEnum):
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
    output_dir:
        Directory where output files are copied after each run.  When set,
        ``sipnet.out`` is copied to ``<output_dir>/sipnet_<run_id>.out``
        before the working directory is deleted, and the returned
        :class:`~pysipnet.result.SIPNETResult` holds a file-backed
        :class:`~pysipnet.output.SIPNETOutput` pointing at the persistent
        copy.  When ``None`` (default), the output is parsed eagerly into
        memory and no file is retained.  Can be overridden per-call via the
        ``output_dir`` argument to :meth:`run`.
    cache_dir:
        Directory containing pre-compiled SIPNET binaries (default:
        ``.sipnet_cache/`` at the repo root).
    climate_staging:
        How file-backed climate instances are staged into the working
        directory.  See :class:`ClimateStaging`.
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
        output_dir: Path | str | None = None,
        climate_staging: ClimateStaging = ClimateStaging.COPY,
        cache_dir: Path | str | None = None,
        workdir_base: Path | str | None = None,
        keep_workdir: bool = False,
        timeout: float = 300.0,
    ) -> None:
        self.preset = preset
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.climate_staging = climate_staging
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

    def _check_output_dir(self, output_dir: Path, workdir: Path) -> None:
        """Raise ValueError if output_dir is inside the run's working directory."""
        output_dir_resolved = output_dir.resolve()
        workdir_resolved = workdir.resolve()
        if output_dir_resolved.is_relative_to(workdir_resolved):
            raise ValueError(
                f"output_dir '{output_dir}' is inside the run's working directory "
                f"'{workdir}', which is deleted after each run. "
                "Specify a path outside the working directory. "
                "If you need to keep the working directory, set keep_workdir=True "
                "and read the output from provenance.workdir directly."
            )

    def _stage_clim_file(self, climate: ClimateDrivers, dest: Path) -> None:
        """Write or link the climate file into the run working directory."""
        import shutil
        import warnings

        from pysipnet.io.clim_io import write_clim_file

        if climate.source_path is None:
            write_clim_file(climate, dest)
            return

        if self.climate_staging == ClimateStaging.SYMLINK:
            try:
                dest.symlink_to(climate.source_path.resolve())
                return
            except OSError:
                warnings.warn(
                    f"Symlinking climate file failed; falling back to copy. "
                    f"Source: {climate.source_path}",
                    stacklevel=2,
                )
        shutil.copy2(climate.source_path, dest)

    def run(
        self,
        parameters: SIPNETParametersV1,
        climate: ClimateDrivers,
        *,
        run_id: str | None = None,
        events: EventSequence | None = None,
        output_dir: Path | str | None | object = _UNSET,
    ) -> SIPNETResult:
        """Execute SIPNET and return a parsed result.

        Each call writes inputs to a fresh directory, runs the binary there,
        and packages the output.  The working directory is deleted on
        completion unless ``keep_workdir=True``.

        Working directory
        -----------------
        The working directory is ``<workdir_base>/sipnet_<run_id>/``.  It is
        created with ``exist_ok=True``, so re-using the same ``run_id`` simply
        overwrites the previous run's input files.  Use distinct ``run_id``
        values if you need to compare runs.

        Parameters
        ----------
        parameters:
            Model parameter set.
        climate:
            Meteorological forcing.
        run_id:
            Optional identifier for the working directory and output file name.
            Defaults to a random UUID hex string.
        events:
            Optional :class:`~pysipnet.events.EventSequence`.
        output_dir:
            Override the runner-level ``output_dir`` for this call.  Pass
            ``None`` to suppress output persistence even when the runner has a
            default ``output_dir`` set.  If not passed, the runner-level
            default is used.

            The path must not be the same as, or a subdirectory of, the run's
            working directory — this is validated before the run starts.

        Returns
        -------
        SIPNETResult
            Contains the output (file-backed if ``output_dir`` is set, otherwise
            in-memory), run provenance, and process metadata.

        Raises
        ------
        ValueError
            If *output_dir* is inside the run's working directory.
        FileNotFoundError
            If the SIPNET binary cannot be found.
        subprocess.TimeoutExpired
            If the run exceeds *timeout* seconds.
        """
        import shutil

        from pysipnet.io.param_io import write_param_file
        from pysipnet.result import RunProvenance, SIPNETResult

        self._check_binary()
        flags = self.preset.flags

        # Resolve effective output_dir (per-call overrides runner-level default).
        if output_dir is _UNSET:
            effective_output_dir = self.output_dir
        elif output_dir is None:
            effective_output_dir = None
        else:
            effective_output_dir = Path(output_dir)

        run_id = run_id or uuid.uuid4().hex
        workdir = self.workdir_base / f"sipnet_{run_id}"

        # Validate output_dir before any I/O so errors are immediate and clear.
        if effective_output_dir is not None:
            self._check_output_dir(effective_output_dir, workdir)
            effective_output_dir.mkdir(parents=True, exist_ok=True)

        workdir.mkdir(parents=True, exist_ok=True)

        try:
            write_param_file(parameters, flags, workdir / "sipnet.param")
            self._stage_clim_file(climate, workdir / "sipnet.clim")

            if events is not None:
                events.to_file(workdir / "events.in")
                events_flag = "1"
            else:
                events_flag = "0"

            (workdir / "sipnet.in").write_text(f"fileName = sipnet\nEVENTS = {events_flag}\n")

            proc = subprocess.run(
                [str(self.binary_path)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            provenance = RunProvenance(
                preset=self.preset,
                binary_path=self.binary_path,
                run_id=run_id,
                workdir=workdir,
                returncode=proc.returncode,
                success=(proc.returncode == 0),
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

            out_src = workdir / "sipnet.out"
            outputs = self._build_output(provenance, out_src, effective_output_dir, run_id)

        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

        return SIPNETResult(
            outputs=outputs,
            parameters=parameters,
            climate=climate,
            flags=flags,
            provenance=provenance,
            events=events,
        )

    def _build_output(
        self,
        provenance: RunProvenance,
        out_src: Path,
        effective_output_dir: Path | None,
        run_id: str,
    ) -> SIPNETOutput:
        """Copy or parse the output file and return an appropriate SIPNETOutput."""
        import shutil

        import pandas as pd

        from pysipnet.io.output_reader import read_output_file
        from pysipnet.output import SIPNETOutput

        if not (provenance.returncode == 0 and out_src.exists()):
            return SIPNETOutput.from_dataframe(pd.DataFrame())

        if effective_output_dir is not None:
            dest = effective_output_dir / f"sipnet_{run_id}.out"
            shutil.copy2(out_src, dest)
            return SIPNETOutput.from_path(dest)

        return SIPNETOutput.from_dataframe(read_output_file(out_src))
