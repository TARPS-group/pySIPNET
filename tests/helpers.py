"""Shared test helpers.

Kept separate from ``conftest.py`` (which holds pytest fixtures) so these can be
imported as plain functions from any test module.
"""

from __future__ import annotations

from pathlib import Path


def params_from_sipnet_file(path: Path):
    """Reconstruct a ``SIPNETParametersV1`` from a SIPNET ``.param`` file.

    Inverse of :func:`pysipnet.io.param_io.write_param_file` across the
    :data:`~pysipnet.io.param_io.PYTHON_TO_SIPNET` mapping.  Names not in the
    mapping (obsolete placeholders, out-of-scope submodel params) are ignored,
    exactly as the writer omits them.

    This is a temporary test-side stand-in for the production reader tracked in
    https://github.com/TARPS-group/pySIPNET/issues/19; fold callers over to the
    library API once it lands.
    """
    from pysipnet.io.param_io import PYTHON_TO_SIPNET, read_param_file
    from pysipnet.parameters.v1 import (
        AllocationParams,
        InitialConditions,
        LeafPhysiologyParams,
        PhenologyParams,
        PhotosynthesisParams,
        RespirationParams,
        SIPNETParametersV1,
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
    return SIPNETParametersV1(**kwargs)
