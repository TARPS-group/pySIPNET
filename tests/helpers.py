"""Shared test helpers.

Kept separate from ``conftest.py`` (which holds pytest fixtures) so these can be
imported as plain functions from any test module.
"""

from __future__ import annotations

from pathlib import Path

from pysipnet.version import SIPNET_NUMERIC_VERSION, SIPNET_PINNED_TAG

PINNED_VERSION_LINE = f"SIPNET version {SIPNET_NUMERIC_VERSION} ({SIPNET_PINNED_TAG})"
"""What the pinned SIPNET answers to ``--version``."""


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
