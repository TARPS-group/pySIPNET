"""pySIPNET: a clean Python interface to the SIPNET land surface model.

Quickstart::

    from pysipnet import (
        SIPNETRunner, ModelPreset, SIPNETModel,
        SIPNETParametersV1, ModelFlagsV1,
        ClimateDrivers,
    )

    params  = SIPNETParametersV1(...)
    climate = ClimateDrivers.from_file("site.clim", version="v1")
    runner  = SIPNETRunner(preset=ModelPreset.STANDARD)
    model   = SIPNETModel(runner, base_params=params, base_climate=climate)

    result = model()                    # baseline run
    result = model(a_max=120.0)         # single parameter override

With agronomic events::

    from pysipnet import EventSequence, IrrigationEvent, IrrigationMethod

    events = EventSequence(events=[
        IrrigationEvent(year=2020, day=150, amount=5.0,
                        method=IrrigationMethod.SOIL),
    ])
    result = model(events=events)
"""

# Runner
# Build utilities
from pysipnet.build import build_preset, ensure_binary

# Config (reproducible workflows)
from pysipnet.config import ClimateArchiveMode, RunConfig

# Climate
from pysipnet.climate import ClimateDrivers

# Events
from pysipnet.events import (
    EventSequence,
    FertilizationEvent,
    HarvestEvent,
    IrrigationEvent,
    IrrigationMethod,
    PlantingEvent,
    TillageEvent,
)

# Model (high-level interface)
from pysipnet.model import SIPNETModel

# Parameters (top-level groups available via pysipnet.parameters)
from pysipnet.parameters.v1 import SIPNET_PARAMS_BY_GROUP, ModelFlagsV1, SIPNETParametersV1

# Results
from pysipnet.result import RunProvenance, SIPNETResult
from pysipnet.runner import ClimateStaging, ModelPreset, SIPNETRunner

# Version
from pysipnet.version import PYSIPNET_VERSION, SIPNET_PINNED_COMMIT, SIPNET_TARGET_VERSION

__version__ = PYSIPNET_VERSION

__all__ = [
    # Model (high-level interface)
    "SIPNETModel",
    # Runner
    "SIPNETRunner",
    "ModelPreset",
    "ClimateStaging",
    # Config (reproducible workflows)
    "RunConfig",
    "ClimateArchiveMode",
    # Results
    "SIPNETResult",
    "RunProvenance",
    # Parameters
    "SIPNETParametersV1",
    "ModelFlagsV1",
    "SIPNET_PARAMS_BY_GROUP",
    # Climate
    "ClimateDrivers",
    # Events
    "EventSequence",
    "HarvestEvent",
    "IrrigationEvent",
    "IrrigationMethod",
    "FertilizationEvent",
    "PlantingEvent",
    "TillageEvent",
    # Build
    "build_preset",
    "ensure_binary",
    # Version
    "PYSIPNET_VERSION",
    "SIPNET_PINNED_COMMIT",
    "SIPNET_TARGET_VERSION",
]
