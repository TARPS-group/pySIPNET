"""pySIPNET: a clean Python interface to the SIPNET land surface model.

Quickstart::

    from pysipnet import (
        SIPNETRunner, SIPNETModel,
        SIPNETParameters, ModelFlags,
        ClimateDrivers,
    )

    params  = SIPNETParameters(...)
    climate = ClimateDrivers.from_file("site.clim")
    runner  = SIPNETRunner(flags=ModelFlags.standard())
    model   = SIPNETModel(runner, base_params=params, base_climate=climate)

    result = model()                    # baseline run
    result = model(max_photosynthesis_rate=120.0)         # single parameter override

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
from pysipnet.build import (
    BinaryVersionError,
    BuildError,
    DownloadError,
    build_sipnet,
    download_sipnet,
    ensure_binary,
    install_sipnet,
    sipnet_version,
)

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

# Bundled reference data
from pysipnet.io.reference import (
    NiwotReferenceFiles,
    niwot_reference_climate,
    niwot_reference_files,
    niwot_reference_output,
)

# Model (high-level interface)
from pysipnet.model import SIPNETModel

# Results
from pysipnet.output import SIPNETOutput

# Parameters (top-level groups available via pysipnet.parameters)
from pysipnet.parameters.model import (
    PARAMETER_SPECS,
    SIPNET_PARAMS_BY_GROUP,
    ModelFlags,
    SIPNETParameters,
    resolve_parameter_name,
)
from pysipnet.resample import (
    check_frequency,
    check_not_upsampling,
    check_resampling_method,
    drop_padding,
    resample,
    resampled_attributes,
)
from pysipnet.result import RunProvenance, SIPNETResult
from pysipnet.runner import ClimateStaging, SIPNETRunError, SIPNETRunner
from pysipnet.variables import (
    CLIMATE_VARIABLES,
    OUTPUT_VARIABLES,
    check_variable_is_written,
    resolve_climate_variable,
    resolve_output_variable,
    variable_kind,
)

# Version
from pysipnet.version import (
    PYSIPNET_VERSION,
    SIPNET_NUMERIC_VERSION,
    SIPNET_PINNED_COMMIT,
    SIPNET_PINNED_TAG,
)

__version__ = PYSIPNET_VERSION

__all__ = [
    # Model (high-level interface)
    "SIPNETModel",
    # Runner
    "SIPNETRunner",
    "SIPNETRunError",
    "ClimateStaging",
    # Config (reproducible workflows)
    "RunConfig",
    # Results
    "SIPNETResult",
    "SIPNETOutput",
    "RunProvenance",
    "check_frequency",
    "check_not_upsampling",
    "check_resampling_method",
    "drop_padding",
    "resample",
    "resampled_attributes",
    # Parameters
    "SIPNETParameters",
    "ModelFlags",
    "SIPNET_PARAMS_BY_GROUP",
    "PARAMETER_SPECS",
    "resolve_parameter_name",
    # Variable registries
    "OUTPUT_VARIABLES",
    "CLIMATE_VARIABLES",
    "resolve_output_variable",
    "resolve_climate_variable",
    "variable_kind",
    "check_variable_is_written",
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
    # Bundled reference data
    "NiwotReferenceFiles",
    "niwot_reference_files",
    "niwot_reference_climate",
    "niwot_reference_output",
    # Build
    "install_sipnet",
    "build_sipnet",
    "download_sipnet",
    "ensure_binary",
    "sipnet_version",
    "BinaryVersionError",
    "BuildError",
    "DownloadError",
    # Version
    "PYSIPNET_VERSION",
    "SIPNET_PINNED_COMMIT",
    "SIPNET_PINNED_TAG",
    "SIPNET_NUMERIC_VERSION",
]
