"""pySIPNET: a clean Python interface to the SIPNET land surface model.

Quickstart::

    from pysipnet import (
        SIPNETRunner, ModelPreset,
        SIPNETParametersV1, ModelFlagsV1,
        ClimateDrivers,
    )

    params  = SIPNETParametersV1(...)
    climate = ClimateDrivers.from_file("site.clim", version="v1")
    runner  = SIPNETRunner(preset=ModelPreset.STANDARD)
    result  = runner.run(params, climate)

    print(result.outputs[["nee", "gpp"]].describe())

With agronomic events::

    from pysipnet import EventSequence, IrrigationEvent, IrrigationMethod

    events = EventSequence(events=[
        IrrigationEvent(year=2020, day=150, amount=5.0,
                        method=IrrigationMethod.SOIL),
    ])
    result = runner.run(params, climate, events=events)
"""

# Runner
# Build utilities
from pysipnet.build import build_preset, ensure_binary

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

# Parameters (top-level groups available via pysipnet.parameters)
from pysipnet.parameters.v1 import ModelFlagsV1, SIPNETParametersV1

# Results
from pysipnet.result import RunProvenance, SIPNETResult
from pysipnet.runner import ModelPreset, SIPNETRunner

# Version
from pysipnet.version import PYSIPNET_VERSION, SIPNET_PINNED_COMMIT, SIPNET_TARGET_VERSION

__version__ = PYSIPNET_VERSION

__all__ = [
    # Runner
    "SIPNETRunner",
    "ModelPreset",
    # Results
    "SIPNETResult",
    "RunProvenance",
    # Parameters
    "SIPNETParametersV1",
    "ModelFlagsV1",
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
