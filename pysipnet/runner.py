"""SIPNET binary runner.

The :class:`SIPNETRunner` manages subprocess execution of the SIPNET binary.
Each call to :meth:`~SIPNETRunner.run` is fully isolated: inputs are written
to a fresh temporary directory, SIPNET is executed there, and the outputs are
returned as a :class:`~pysipnet.result.SIPNETResult`.

This design has a key property: **runs are stateless and share no resources**,
making it trivial to parallelize them with any executor (``concurrent.futures``,
Dask, Parsl, Ray, etc.)::

    from concurrent.futures import ProcessPoolExecutor
    from pysipnet.runner import SIPNETRunner

    runner = SIPNETRunner(flags=ModelFlags.standard())

    def run_one(config_dict):
        from pysipnet.parameters.model import SIPNETParameters
        from pysipnet.climate import ClimateDrivers
        import pandas as pd
        params  = SIPNETParameters.model_validate(config_dict["params"])
        climate = ClimateDrivers.from_dataframe(pd.DataFrame(config_dict["climate"]))
        return runner.run(params, climate).outputs.pandas.to_dict()

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_one, ensemble_configs))

The SIPNET binary
-----------------
There is a single SIPNET binary. Every model option is chosen at run time and
written into ``sipnet.in``, so one binary serves every configuration. Where it
comes from is :mod:`pysipnet.build`'s business: by default the runner takes the
first binary in that module's search order — ``$PYSIPNET_BINARY``, a binary
bundled in the wheel, ``.sipnet_cache/`` in a checkout, the per-user cache —
and ``SIPNETRunner(binary=...)`` names one outright. Install one with ``pysipnet install-sipnet``.

Before the first run, the runner asks the binary for its version and refuses
one built from a different SIPNET than this pySIPNET pins, so a stale or
mismatched binary fails loudly rather than producing another model's output.
``verify_binary=False`` switches that off.

Output persistence
------------------
By default the output is parsed eagerly and the working directory is deleted.
Set ``output_dir`` to copy ``sipnet.out`` to a stable location before cleanup
and return a file-backed :class:`~pysipnet.output.SIPNETOutput` instead::

    runner = SIPNETRunner(
        flags=ModelFlags.standard(),
        output_dir=Path("ensemble_out"),
    )
    results = [runner.run(params_i, climate, run_id=f"m{i}") for i in range(1000)]
    # No DataFrames in memory yet.
    nee = pd.concat([r.outputs.select(["nee"], format="pandas") for r in results])

Each run writes ``sipnet_<run_id>.out`` inside ``output_dir``. Two runs sharing
a ``run_id`` would name the same file, so the second is refused rather than
allowed to change what the first one's lazily-read result answers with; pass
``overwrite=True`` when that is what you want.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import uuid
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pysipnet.build import (
    BINARY_NAME,
    missing_binary_message,
    verify_binary_matches_pin,
)
from pysipnet.build import (
    binary_path as _default_binary_path,
)
from pysipnet.parameters.model import ModelFlags

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.events import EventSequence
    from pysipnet.output import SIPNETOutput
    from pysipnet.parameters.model import SIPNETParameters
    from pysipnet.result import RunProvenance, SIPNETResult

# Sentinel used to distinguish "not passed" from None in output_dir overrides.
_UNSET = object()


class ClimateStaging(StrEnum):
    """How the runner stages the climate file into each run's working directory.

    +----------+--------------------------------------------------------------+
    | Value    | Behavior                                                    |
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


class SIPNETRunError(RuntimeError):
    """SIPNET exited non-zero, or produced no output file.

    Carries everything needed to diagnose the run without re-running it. The
    binary writes the actual reason to stdout or stderr — a missing parameter,
    an unreadable climate file — so those are the first place to look.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        workdir: Path,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.workdir = workdir


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_run_id(run_id: str) -> str:
    """Return *run_id* if it is safe to use in a path, else raise.

    The run id becomes a directory name under ``workdir_base``, and that
    directory is deleted when the run finishes. Without this check a run id
    containing ``..`` walks out of ``workdir_base``, so SIPNET's inputs are
    written into some unrelated directory and the cleanup then removes it.

    Ensemble run ids are often built from site or member identifiers read out
    of data files, so this is reachable without anyone doing anything strange.
    """
    if not _SAFE_RUN_ID.match(run_id):
        raise ValueError(
            f"Invalid run_id {run_id!r}. A run id becomes a directory name, so it "
            "may contain only letters, digits, dot, underscore and hyphen, and must "
            "start with a letter or digit. Path separators and '..' are refused "
            "because the run directory is deleted afterwards."
        )
    return run_id


def _render_sipnet_in(flags: ModelFlags, *, events_enabled: bool) -> str:
    """Build the contents of the ``sipnet.in`` config file for one run.

    SIPNET reads its run configuration from this file. Everything pySIPNET
    depends on is written explicitly, including settings that match SIPNET's
    own defaults, so a saved run keeps its meaning even if a future SIPNET
    changes one of those defaults.

    Settings pySIPNET does not rely on — ``QUIET``, ``EVENTS_PREFIX``,
    ``DUMP_CONFIG``, ``INPUT_FILE`` — are left out and take SIPNET's defaults.

    Parameters
    ----------
    flags:
        Model options for this run.
    events_enabled:
        Whether an ``events.in`` file was written alongside this config. When
        no events are supplied this is turned off explicitly, so a stale
        ``events.in`` left in the working directory cannot be picked up.

    Returns
    -------
    str
        File contents, one ``KEY = VALUE`` pair per line.
    """
    settings: dict[str, object] = {
        # Look for sipnet.param and sipnet.clim, and write sipnet.out.
        "FILE_NAME": "sipnet",
        # Always emit the column-name header row: the output reader matches
        # columns by name, so results stay parseable even when a flag change
        # alters which columns SIPNET writes.
        "PRINT_HEADER": 1,
        # Written rather than assumed: if a future SIPNET defaulted this off,
        # there would be no output file to read and the failure would look
        # like a missing file rather than a configuration change.
        "DO_MAIN_OUTPUT": 1,
        # We parse the combined output, so the per-variable files are just
        # extra work and extra files in the working directory.
        #
        # Note the plural. SIPNET builds its config keys from the C *field*
        # name (doSingleOutputs), not from the label it prints for the setting
        # (DO_SINGLE_OUTPUT), and the two disagree here. The singular form —
        # the one SIPNET's own docs give — is silently ignored.
        "DO_SINGLE_OUTPUTS": 0,
        "EVENTS": int(events_enabled),
    }
    settings.update(flags.to_config_keys())

    lines = ["! SIPNET run configuration - generated by pySIPNET"]
    lines += [f"{key} = {value}" for key, value in settings.items()]
    return "\n".join(lines) + "\n"


class SIPNETRunner:
    """Execute a single SIPNET run in an isolated working directory.

    Parameters
    ----------
    flags:
        Which optional SIPNET processes to switch on for every run made by
        this runner. Defaults to :meth:`ModelFlags.standard`. The flags also
        decide which parameters must be present, so a mismatch between these
        and the parameters passed to :meth:`run` is reported before SIPNET is
        invoked.
    output_dir:
        Directory where output files are copied after each run.  When set,
        ``sipnet.out`` is copied to ``<output_dir>/sipnet_<run_id>.out``
        before the working directory is deleted, and the returned
        :class:`~pysipnet.result.SIPNETResult` holds a file-backed
        :class:`~pysipnet.output.SIPNETOutput` pointing at the persistent
        copy.  When ``None`` (default), the output is parsed eagerly into
        memory and no file is retained.  Can be overridden per-call via the
        ``output_dir`` argument to :meth:`run`.
    overwrite:
        Whether a run may replace an output file left by an earlier run with
        the same ``run_id``.  ``False`` (default) refuses; see :meth:`run`.
    binary:
        Path of the SIPNET binary to run, resolved to an absolute path.
        Default: the first binary found in
        :func:`pysipnet.build.binary_candidates` order, looked up afresh on
        each run so a binary installed after the runner was created is used.
    cache_dir:
        Directory holding a binary named ``sipnet``; the same as passing
        ``binary=cache_dir / "sipnet"``. Ignored when *binary* is given.
    verify_binary:
        Ask the binary for its version before the first run and refuse one
        not built from the pinned SIPNET tag (default ``True``). See
        :func:`pysipnet.build.verify_binary_matches_pin`.
    climate_staging:
        How file-backed climate instances are staged into the working
        directory.  See :class:`ClimateStaging`.
    workdir_base:
        Parent directory for per-run working directories.  Defaults to the
        system temp directory.  Each run gets a freshly created subdirectory
        with a unique name, so two runs never share one even when they share a
        ``run_id``.
    keep_workdir:
        If ``True``, do not delete the working directory after the run.
        Useful for debugging.  Default is ``False``.
    timeout:
        Maximum wall-clock time (seconds) allowed for a single SIPNET run.
        Raises :class:`subprocess.TimeoutExpired` if exceeded.
    """

    def __init__(
        self,
        flags: ModelFlags | None = None,
        *,
        output_dir: Path | str | None = None,
        overwrite: bool = False,
        climate_staging: ClimateStaging = ClimateStaging.COPY,
        binary: Path | str | None = None,
        cache_dir: Path | str | None = None,
        verify_binary: bool = True,
        workdir_base: Path | str | None = None,
        keep_workdir: bool = False,
        timeout: float = 300.0,
    ) -> None:
        self.flags = flags if flags is not None else ModelFlags.standard()
        # Resolved, so the path a file-backed output stores keeps meaning if the
        # process changes directory — which an ensemble scheduler may do per task.
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self.overwrite = overwrite
        self.climate_staging = climate_staging
        if binary is None and cache_dir is not None:
            binary = Path(cache_dir) / BINARY_NAME
        # Resolved, as output_dir is: the binary is executed with the run's
        # working directory as cwd, where a relative path would mean something
        # else, and --version on a bare name would search PATH instead.
        self._binary = Path(binary).resolve() if binary is not None else None
        self.verify_binary = verify_binary
        self.workdir_base = Path(workdir_base) if workdir_base else Path(tempfile.gettempdir())
        self.keep_workdir = keep_workdir
        self.timeout = timeout

    @property
    def binary_path(self) -> Path:
        """Path of the SIPNET binary this runner will execute.

        An explicit *binary* wins; otherwise the search in
        :mod:`pysipnet.build` is repeated on each access, so a binary installed
        after the runner was created is picked up. :meth:`run` reads it once
        per run and uses that one path for the check, the execution and the
        provenance record, so all three describe the same file.
        """
        if self._binary is not None:
            return self._binary
        return _default_binary_path()

    def _check_binary(self) -> Path:
        """Return the binary to run, after checking it exists and is the pinned SIPNET."""
        path = self.binary_path
        if not path.is_file():
            if self._binary is None:
                raise FileNotFoundError(missing_binary_message())
            raise FileNotFoundError(
                f"SIPNET binary not found at {path}. Install one with "
                "'pysipnet install-sipnet', or pass the path of an existing binary."
            )
        if self.verify_binary:
            verify_binary_matches_pin(path)
        return path

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

    @staticmethod
    def _output_path(output_dir: Path, run_id: str) -> Path:
        """Where this run's output file is kept. The one place that name is formed."""
        return output_dir / f"sipnet_{run_id}.out"

    @staticmethod
    def _occupied_message(dest: Path, run_id: str) -> str:
        """Why replacing an output file is refused.

        The output file is named from the run id, so two runs sharing an id
        name the same file — and unlike the working directory, this name is one
        users are told to predict, so it cannot be given a random suffix.
        Replacing it does not just lose the earlier file: a file-backed
        SIPNETOutput reads lazily, so the first result still holds the path and
        would answer with the second run's numbers, having reported success.
        Wrong numbers rather than an error, which is why this is refused rather
        than warned about.
        """
        return (
            f"{dest} already exists: another run used run_id={run_id!r} with this "
            "output_dir. Replacing it would silently change the numbers that run's "
            "result reads, because a file-backed output is read lazily. Give this run a "
            "distinct run_id, or pass overwrite=True if the earlier output is finished "
            "with."
        )

    def _check_output_path(self, dest: Path, run_id: str, overwrite: bool) -> None:
        """Refuse an occupied output path before the run, so a doomed run costs nothing.

        This is the courtesy check; :meth:`_publish_output` is the one that
        decides, because only it can do so atomically. A file deleted between
        the two — or created by a concurrent run — is caught there.
        """
        if overwrite or not (dest.exists() or dest.is_symlink()):
            return
        raise FileExistsError(self._occupied_message(dest, run_id))

    def _publish_output(self, out_src: Path, dest: Path, run_id: str, overwrite: bool) -> None:
        """Copy the output into place, claiming the name atomically unless overwriting.

        The pre-run check cannot settle this on its own: between it and here sits
        the entire model run, and two concurrent runs sharing an id would both
        pass it and then both copy, leaving one result reading the other's
        numbers. ``O_EXCL`` makes the claim and the creation the same operation,
        so the loser is told rather than quietly overwritten. Nothing is left
        behind by a run that fails, because a failed run never reaches here.
        """
        import os
        import shutil

        if overwrite:
            shutil.copy2(out_src, dest)
            return
        try:
            handle = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise FileExistsError(self._occupied_message(dest, run_id)) from None
        with os.fdopen(handle, "wb") as destination, out_src.open("rb") as source:
            shutil.copyfileobj(source, destination)
        shutil.copystat(out_src, dest)

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
        parameters: SIPNETParameters,
        climate: ClimateDrivers,
        *,
        run_id: str | None = None,
        events: EventSequence | None = None,
        output_dir: Path | str | None | object = _UNSET,
        overwrite: bool | None = None,
        check: bool = True,
    ) -> SIPNETResult:
        """Execute SIPNET and return a parsed result.

        Each call writes inputs to a fresh directory, runs the binary there,
        and packages the output.  The working directory is deleted on
        completion unless ``keep_workdir=True``.

        Working directory
        -----------------
        The working directory is ``<workdir_base>/sipnet_<run_id>_<random>/``,
        created by :func:`tempfile.mkdtemp`.  The random suffix means every run
        gets its own directory even when two runs share a ``run_id``, so
        concurrent runs cannot overwrite each other's files.  The name is not
        predictable in advance; read it from ``provenance.workdir``.

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
        overwrite:
            Whether this run may replace ``<output_dir>/sipnet_<run_id>.out``
            if it already exists.  ``None`` (default) takes the runner-level
            ``overwrite``,
            which is ``False``: a second run with the same ``run_id`` is
            refused before it starts, because the earlier run's result reads
            that file lazily and would silently answer with this run's numbers.
            Pass ``True`` when the earlier output is finished with — a loop
            that reruns one member under a fixed id, say.  Runs with distinct
            ids get distinct files, and the default ``run_id`` is a fresh UUID.

            The refusal is based on the file being present, so deleting it
            releases the name — and with it the protection for any result still
            reading from that path.

        Returns
        -------
        SIPNETResult
            Contains the output (file-backed if ``output_dir`` is set, otherwise
            in-memory), run provenance, and process metadata.

        Raises
        ------
        ValueError
            If *output_dir* is inside the run's working directory.
        FileExistsError
            If this run's output file already exists and *overwrite* is false.
        FileNotFoundError
            If the SIPNET binary cannot be found.
        pysipnet.build.BinaryVersionError
            If *verify_binary* is on and the binary was built from a
            different SIPNET than this pySIPNET pins.
        subprocess.TimeoutExpired
            If the run exceeds *timeout* seconds.
        """
        import shutil

        from pysipnet.io.param_io import write_param_file
        from pysipnet.result import RunProvenance, SIPNETResult

        binary = self._check_binary()
        flags = self.flags

        # Resolve effective output_dir (per-call overrides runner-level default).
        effective_output_dir: Path | None
        if output_dir is _UNSET:
            effective_output_dir = self.output_dir
        elif output_dir is None:
            effective_output_dir = None
        else:
            effective_output_dir = Path(cast("str | Path", output_dir)).resolve()

        run_id = _check_run_id(run_id) if run_id else uuid.uuid4().hex

        # The run id labels the run; it does not name the directory. Deriving
        # the path from it would put two concurrent runs that share an id into
        # the same directory under a shared temp dir, and because the run
        # succeeds by reading whatever sipnet.out it finds, the result is wrong
        # numbers rather than an error. mkdtemp guarantees a fresh directory,
        # keeping the id in the prefix so it is still recognizable while
        # debugging.
        self.workdir_base.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix=f"sipnet_{run_id}_", dir=self.workdir_base))

        effective_overwrite = self.overwrite if overwrite is None else overwrite

        # Validate output_dir before any I/O so errors are immediate and clear,
        # and before the binary runs so a refused run costs nothing but the
        # working directory, which is removed again on the way out.
        if effective_output_dir is not None:
            try:
                self._check_output_dir(effective_output_dir, workdir)
                effective_output_dir.mkdir(parents=True, exist_ok=True)
                self._check_output_path(
                    self._output_path(effective_output_dir, run_id), run_id, effective_overwrite
                )
            except Exception:
                shutil.rmtree(workdir, ignore_errors=True)
                raise

        try:
            write_param_file(parameters, flags, workdir / "sipnet.param")
            self._stage_clim_file(climate, workdir / "sipnet.clim")

            if events is not None:
                events.to_file(workdir / "events.in")
                events_enabled = True
            else:
                events_enabled = False

            (workdir / "sipnet.in").write_text(
                _render_sipnet_in(flags, events_enabled=events_enabled)
            )

            proc = subprocess.run(
                [str(binary)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            provenance = RunProvenance(
                flags=flags,
                binary_path=binary,
                run_id=run_id,
                workdir=workdir,
                returncode=proc.returncode,
                success=(proc.returncode == 0),
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

            out_src = workdir / "sipnet.out"
            if check and not (provenance.returncode == 0 and out_src.exists()):
                # Returning an empty frame here would defer the failure to
                # whatever the caller does next — typically a column lookup,
                # which raises KeyError a long way from the cause, with SIPNET's
                # own explanation stranded on the provenance object. In an
                # ensemble the empties are collected silently.
                reason = (
                    f"SIPNET exited with code {provenance.returncode}"
                    if provenance.returncode != 0
                    else "SIPNET exited cleanly but wrote no output file"
                )
                raise SIPNETRunError(
                    f"{reason} (run_id={run_id!r}).\n"
                    f"stderr:\n{proc.stderr or '(empty)'}\n"
                    f"stdout:\n{proc.stdout or '(empty)'}\n"
                    "Pass check=False to get a result object instead of this error.",
                    returncode=provenance.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    workdir=workdir,
                )
            outputs = self._build_output(
                provenance,
                out_src,
                effective_output_dir,
                run_id,
                climate,
                effective_overwrite,
            )

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
        climate: ClimateDrivers,
        overwrite: bool,
    ) -> SIPNETOutput:
        """Copy or parse the output file and return an appropriate SIPNETOutput.

        The timestep lengths come from the climate drivers; SIPNET does not
        write them, and the Dataset would otherwise have to reconstruct the
        interval each row covers from the timestamps.  They are handed over as a
        callable so a file-backed climate is not read just to build a result
        nobody has asked for the Dataset of.

        The flags travel with the output so that selecting a variable this run
        wrote as constant zero is refused rather than silently answered, and the
        run id so that a Dataset saved to disk says which run produced it.
        """
        import numpy as np
        import pandas as pd

        from pysipnet.io.output_reader import read_output_file
        from pysipnet.output import SIPNETOutput

        if not (provenance.returncode == 0 and out_src.exists()):
            return SIPNETOutput.from_dataframe(pd.DataFrame())

        def step_length() -> np.ndarray:
            return climate.pandas["time_step_length"].to_numpy()

        if effective_output_dir is not None:
            dest = self._output_path(effective_output_dir, run_id)
            self._publish_output(out_src, dest, run_id, overwrite)
            return SIPNETOutput.from_path(
                dest,
                time_step_length=step_length,
                flags=provenance.flags,
                run_id=run_id,
            )

        return SIPNETOutput.from_dataframe(
            read_output_file(out_src),
            time_step_length=step_length,
            flags=provenance.flags,
            run_id=run_id,
        )
