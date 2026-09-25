"""SIPNET parameter models, organized by model version.

The top-level imports expose the v1 API directly for convenience::

    from pysipnet.parameters import SIPNETParameters, ModelFlags
"""

from pysipnet.parameters.model import (
    SIPNET_PARAMS_BY_GROUP,
    AllocationParams,
    InitialConditions,
    LeafPhysiologyParams,
    ModelFlags,
    PhenologyParams,
    PhotosynthesisParams,
    RespirationParams,
    SIPNETParameters,
    WaterParams,
    parameter_dataarray,
)

__all__ = [
    "AllocationParams",
    "InitialConditions",
    "LeafPhysiologyParams",
    "ModelFlags",
    "PhenologyParams",
    "PhotosynthesisParams",
    "RespirationParams",
    "SIPNET_PARAMS_BY_GROUP",
    "SIPNETParameters",
    "WaterParams",
    "parameter_dataarray",
]
