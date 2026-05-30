"""RunConfig — serialisable run specification for reproducible SIPNET workflows."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.events import EventSequence
    from pysipnet.parameters.v1 import SIPNETParametersV1
    from pysipnet.result import SIPNETResult
    from pysipnet.runner import ModelPreset

_MODE_COPY = "copy"
_MODE_REFERENCE = "reference"


@dataclass
class RunConfig:
    """Serialisable specification for a single SIPNET run.

    A ``RunConfig`` captures everything needed to reproduce a model run:
    the binary preset, the full parameter set, the climate forcing, and any
    management events.  Use :meth:`save` to write it to disk and
    :meth:`load` to reconstruct it.

    ``RunConfig`` is the same object whether used for a single run or as the
    *shared context* in an ensemble or iterative workflow — the distinction
    lies only in how the loaded config is subsequently used.

    Parameters
    ----------
    preset:
        Which compiled binary preset to use.
    params:
        Full parameter set for the run.
    climate:
        Meteorological forcing.
    events:
        Optional management event sequence.

    Examples
    --------
    Save and reload a run configuration::

        config = RunConfig(preset=ModelPreset.STANDARD, params=params, climate=climate)
        config.save("my_run/")

        config2 = RunConfig.load("my_run/")
        runner  = SIPNETRunner(preset=config2.preset)
        result  = runner.run(config2.params, config2.climate)

    Promote an exploratory result to a saved configuration::

        result = model()
        config = RunConfig.from_result(result)
        config.save("interesting_run/")
    """

    preset: ModelPreset
    params: SIPNETParametersV1
    climate: ClimateDrivers
    events: EventSequence | None = None

    def save(
        self,
        path: str | Path,
        *,
        reference_only: bool = False,
    ) -> Path:
        """Write this configuration to a directory.

        The directory is created if it does not exist.  Existing files are
        overwritten without warning, so calling ``save`` on a pre-existing
        directory updates the config in place.

        Directory layout
        ----------------
        .. code-block:: text

            <path>/
            ├── config.json   # preset, params, climate mode, metadata
            ├── sipnet.clim   # present only when reference_only=False (default)
            └── events.in     # present only when events were supplied

        Parameters
        ----------
        path:
            Directory to write into.
        reference_only:
            When ``False`` (default), the climate data is written as
            ``sipnet.clim`` inside the directory, producing a fully
            self-contained archive.  When ``True``, no climate file is
            written; ``config.json`` instead records the absolute path and
            SHA-256 hash of the source file for integrity verification on
            load.  Use ``reference_only=True`` when the climate file is large
            and shared across many configs (e.g. ensemble or iterative
            workflows).  Requires a file-backed
            :class:`~pysipnet.climate.ClimateDrivers` instance (created via
            :meth:`~pysipnet.climate.ClimateDrivers.from_path`).

        Returns
        -------
        Path
            The resolved config directory path.

        Raises
        ------
        ValueError
            If ``reference_only=True`` is requested but the climate instance
            has no source path (i.e. was created via
            :meth:`~pysipnet.climate.ClimateDrivers.from_dataframe`).
        """
        from pysipnet.io.clim_io import write_clim_file
        from pysipnet.version import PYSIPNET_VERSION, SIPNET_PINNED_COMMIT

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if reference_only:
            if self.climate.source_path is None:
                raise ValueError(
                    "reference_only=True requires a file-backed ClimateDrivers "
                    "(created via ClimateDrivers.from_path()). This instance has no "
                    "source path. Use reference_only=False (the default) to write "
                    "the climate data into the config directory."
                )
            clim_path = self.climate.source_path.resolve()
            climate_meta = {
                "mode": _MODE_REFERENCE,
                "path": str(clim_path),
                "sha256": _sha256(clim_path),
            }
        else:
            write_clim_file(self.climate, path / "sipnet.clim")
            climate_meta = {"mode": _MODE_COPY}

        has_events = self.events is not None and len(self.events) > 0
        if has_events:
            self.events.to_file(path / "events.in")

        config_data = {
            "preset": self.preset.value,
            "params": self.params.model_dump(mode="json"),
            "climate": climate_meta,
            "has_events": has_events,
            "sipnet_commit": SIPNET_PINNED_COMMIT,
            "pysipnet_version": PYSIPNET_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (path / "config.json").write_text(json.dumps(config_data, indent=2))

        return path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Reconstruct a :class:`RunConfig` from a directory written by :meth:`save`.

        For configs saved with ``reference_only=False`` (the default), the
        climate is loaded lazily via
        :meth:`~pysipnet.climate.ClimateDrivers.from_path` — no data is read
        from disk until :attr:`~pysipnet.climate.ClimateDrivers.data` is first
        accessed.

        Parameters
        ----------
        path:
            Directory containing a ``config.json`` file.

        Returns
        -------
        RunConfig

        Raises
        ------
        FileNotFoundError
            If ``config.json`` is absent, or if a ``reference_only`` config
            points to a climate file that no longer exists at the recorded path.

        Warns
        -----
        UserWarning
            If a ``reference_only`` climate file has a different SHA-256 digest
            from the one recorded at save time, indicating the file has changed.
        """
        from pysipnet.climate import ClimateDrivers
        from pysipnet.events import EventSequence
        from pysipnet.parameters.v1 import SIPNETParametersV1
        from pysipnet.runner import ModelPreset

        path = Path(path)
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config.json found in {path}. "
                "Pass the directory written by RunConfig.save()."
            )
        data = json.loads(config_path.read_text())

        preset = ModelPreset(data["preset"])
        params = SIPNETParametersV1.model_validate(data["params"])

        clim_meta = data["climate"]
        if clim_meta["mode"] == _MODE_COPY:
            climate = ClimateDrivers.from_path(path / "sipnet.clim")
        else:
            clim_path = Path(clim_meta["path"])
            if not clim_path.exists():
                raise FileNotFoundError(
                    f"Referenced climate file no longer exists: {clim_path}\n"
                    f"(path recorded in {config_path})"
                )
            stored_hash = clim_meta.get("sha256")
            if stored_hash:
                actual_hash = _sha256(clim_path)
                if actual_hash != stored_hash:
                    warnings.warn(
                        f"Referenced climate file has changed since this RunConfig was "
                        f"saved. Stored SHA-256: {stored_hash[:16]}..., current "
                        f"SHA-256: {actual_hash[:16]}...\nFile: {clim_path}",
                        stacklevel=2,
                    )
            climate = ClimateDrivers.from_path(clim_path)

        events = None
        if data.get("has_events"):
            events = EventSequence.from_file(path / "events.in")

        return cls(preset=preset, params=params, climate=climate, events=events)

    @classmethod
    def from_result(cls, result: SIPNETResult) -> RunConfig:
        """Create a :class:`RunConfig` from a completed :class:`~pysipnet.result.SIPNETResult`.

        Useful for promoting an exploratory run to a saved, reusable
        configuration without having to reconstruct the inputs manually.

        Parameters
        ----------
        result:
            A :class:`~pysipnet.result.SIPNETResult` returned by
            :meth:`~pysipnet.runner.SIPNETRunner.run`.
        """
        return cls(
            preset=result.provenance.preset,
            params=result.parameters,
            climate=result.climate,
            events=result.events,
        )


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
