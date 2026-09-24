"""Shared test helpers.

Kept separate from ``conftest.py`` (which holds pytest fixtures) so these can be
imported as plain functions from any test module.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pysipnet.version import SIPNET_NUMERIC_VERSION, SIPNET_PINNED_TAG

if TYPE_CHECKING:
    import pandas as pd

    from pysipnet.climate import ClimateDrivers
    from pysipnet.parameters.model import SIPNETParameters
    from pysipnet.runner import SIPNETRunError

PINNED_VERSION_LINE = f"SIPNET version {SIPNET_NUMERIC_VERSION} ({SIPNET_PINNED_TAG})"
"""What the pinned SIPNET answers to ``--version``."""

WORKER_TIMEOUT_SECONDS = 120.0
"""How long a test waits on a worker process before failing instead of hanging."""


def make_run_error(returncode: int = 3, cls: type[SIPNETRunError] | None = None) -> SIPNETRunError:
    """A ``SIPNETRunError`` (or subclass *cls*) with attributes derived from *returncode*."""
    from pysipnet.runner import SIPNETRunError

    return (cls or SIPNETRunError)(
        f"SIPNET exited with code {returncode}",
        returncode=returncode,
        stdout="out",
        stderr="err",
        workdir=Path(f"/scratch/run-{returncode}"),
    )


# The functions below run in spawned worker processes, which find them by
# module name. They live here rather than in a test module because a test
# module's name is only importable under pytest's default import mode.


def raise_run_error(returncode: int) -> None:
    raise make_run_error(returncode)


def return_value(value: int) -> int:
    return value


def empty_climate() -> ClimateDrivers:
    """A climate with every column and no rows, which SIPNET refuses to run."""
    import pandas as pd

    from pysipnet.climate import CLIMATE_COLUMNS, ClimateDrivers

    return ClimateDrivers.from_dataframe(
        pd.DataFrame({name: pd.Series(dtype=float) for name in CLIMATE_COLUMNS})
    )


def run_with_no_climate_rows(params: SIPNETParameters) -> None:
    from pysipnet.parameters.model import ModelFlags
    from pysipnet.runner import SIPNETRunner

    SIPNETRunner(flags=ModelFlags.standard()).run(params, empty_climate())


def fake_sipnet_binary(path: Path, version_line: str = PINNED_VERSION_LINE) -> Path:
    """A shell script standing in for SIPNET that answers ``--version`` with *version_line*.

    For tests of what pySIPNET does with the answer — the search order, the
    pin check, the CLI — where running the real model would prove nothing more
    and would need a compiled binary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "{version_line}"\n')
    path.chmod(0o755)
    return path


def params_from_sipnet_file(path: Path):
    """Reconstruct a ``SIPNETParameters`` from a SIPNET ``.param`` file.

    Inverse of :func:`pysipnet.io.param_io.write_param_file` across the
    :data:`~pysipnet.io.param_io.PYTHON_TO_SIPNET` mapping.  Names not in the
    mapping (obsolete placeholders, out-of-scope submodel params) are ignored,
    exactly as the writer omits them.

    This is a temporary test-side stand-in for the production reader tracked in
    https://github.com/TARPS-group/pySIPNET/issues/19; fold callers over to the
    library API once it lands.
    """
    from pysipnet.io.param_io import PYTHON_TO_SIPNET, read_param_file
    from pysipnet.parameters.model import (
        AllocationParams,
        InitialConditions,
        LeafPhysiologyParams,
        PhenologyParams,
        PhotosynthesisParams,
        RespirationParams,
        SIPNETParameters,
        WaterParams,
    )

    group_classes: dict[str, type] = {
        "initial_conditions": InitialConditions,
        "photosynthesis": PhotosynthesisParams,
        "phenology": PhenologyParams,
        "respiration": RespirationParams,
        "allocation": AllocationParams,
        "water": WaterParams,
        "leaf": LeafPhysiologyParams,
    }

    flat = read_param_file(path)
    groups: dict[str, dict[str, float]] = {group: {} for group in group_classes}
    for python_path, sipnet_name in PYTHON_TO_SIPNET.items():
        if sipnet_name in flat:
            group, field = python_path.split(".", 1)
            groups[group][field] = flat[sipnet_name]
    kwargs = {group: cls(**groups[group]) for group, cls in group_classes.items()}
    return SIPNETParameters(**kwargs)


@dataclass(frozen=True)
class BareRun:
    """What running the SIPNET binary by hand produced."""

    returncode: int
    log: str
    """stdout and stderr together; SIPNET writes its own errors to stdout."""
    output: pd.DataFrame | None
    """The parsed ``sipnet.out``, or ``None`` when the run failed."""


def run_sipnet_directly(binary: Path, param_path: Path, clim_path: Path) -> BareRun:
    """Run the bare SIPNET binary on these inputs, as a user would by hand.

    Copies the inputs into a clean directory, writes a minimal ``sipnet.in``,
    runs the binary there and parses ``sipnet.out`` with the standard reader.
    No pySIPNET writer or runner is involved, which is the point: tests compare
    what the wrapper does with what this does.
    """
    from pysipnet.io.output_reader import read_output_file

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "sipnet.param").write_bytes(param_path.read_bytes())
        (workdir / "sipnet.clim").write_bytes(clim_path.read_bytes())
        (workdir / "sipnet.in").write_text("fileName = sipnet\nEVENTS = 0\n")
        proc = subprocess.run(
            [str(binary)], cwd=workdir, capture_output=True, text=True, timeout=300
        )
        output = read_output_file(workdir / "sipnet.out") if proc.returncode == 0 else None
        return BareRun(returncode=proc.returncode, log=proc.stdout + proc.stderr, output=output)
