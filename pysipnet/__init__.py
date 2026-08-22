"""pySIPNET: a clean Python interface to the SIPNET land surface model.

Quickstart::

    from pysipnet import (
        SIPNETRunner, SIPNETModel,
        SIPNETParameters, ModelFlags,
        ClimateDrivers,
    )

    params  = SIPNETParameters(...)
    climate = ClimateDrivers.from_file("site.clim", version="v1")
    runner  = SIPNETRunner(flags=ModelFlags.standard())
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
from pysipnet.build import build_sipnet, ensure_binary, sipnet_version

# Climate
from pysipnet.climate import ClimateDrivers

# Config (reproducible workflows)
from pysipnet.config import RunConfig

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

# Results
from pysipnet.output import SIPNETOutput

# Parameters (top-level groups available via pysipnet.parameters)
from pysipnet.parameters.model import SIPNET_PARAMS_BY_GROUP, ModelFlags, SIPNETParameters
from pysipnet.result import RunProvenance, SIPNETResult
from pysipnet.runner import ClimateStaging, SIPNETRunner

# Version
from pysipnet.version import PYSIPNET_VERSION, SIPNET_PINNED_COMMIT, SIPNET_TARGET_VERSION

__version__ = PYSIPNET_VERSION

__all__ = [
    # Model (high-level interface)
    "SIPNETModel",
    # Runner
    "SIPNETRunner",
    "ClimateStaging",
    # Config (reproducible workflows)
    "RunConfig",
    # Results
    "SIPNETResult",
    "SIPNETOutput",
    "RunProvenance",
    # Parameters
    "SIPNETParameters",
    "ModelFlags",
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
    "build_sipnet",
    "ensure_binary",
    "sipnet_version",
    # Version
    "PYSIPNET_VERSION",
    "SIPNET_PINNED_COMMIT",
    "SIPNET_TARGET_VERSION",
]
