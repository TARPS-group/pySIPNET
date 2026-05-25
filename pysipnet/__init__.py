"""pySIPNET: a clean Python interface to the SIPNET land surface model.

Quickstart::

    from pysipnet import SIPNETRunner, ModelPreset
    from pysipnet.parameters.v1 import SIPNETParametersV1, ModelFlagsV1
    from pysipnet.climate import ClimateDrivers

    params  = SIPNETParametersV1(...)
    climate = ClimateDrivers.from_file("site.clim", version="v1")
    runner  = SIPNETRunner(preset=ModelPreset.STANDARD)
    result  = runner.run(params, climate)

    print(result.timeseries[["nee", "gpp"]].describe())
"""

from pysipnet.version import PYSIPNET_VERSION, SIPNET_PINNED_COMMIT, SIPNET_TARGET_VERSION

__version__ = PYSIPNET_VERSION
__all__ = [
    "PYSIPNET_VERSION",
    "SIPNET_PINNED_COMMIT",
    "SIPNET_TARGET_VERSION",
]
