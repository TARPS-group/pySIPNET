# pySIPNET

A Python interface to [SIPNET](https://github.com/PecanProject/sipnet) — the Simplified Photosynthesis and Evapotranspiration Model.

## What is SIPNET?

SIPNET is a lightweight process-based terrestrial carbon and water flux model.  It runs at a single site, is driven by meteorological forcing data, and produces estimates of NEE, GPP, ecosystem respiration, ET, and carbon pool dynamics.

## What is pySIPNET?

pySIPNET provides:

- **Typed, hierarchical parameter models** — every parameter carries its units, domain, and a description.
- **Validated climate drivers** — the forcing data structure catches format errors before SIPNET does.
- **Isolated run execution** — each run gets a fresh working directory; runs never share state.
- **Clean output** — columns are named for what they are (`net_ecosystem_exchange`, not `nee`), carry units and a stated start-of-step or end-of-step time reference, and are available as a DataFrame or an xarray Dataset.

SIPNET is often run within the [PEcAn](https://github.com/pecanproject) ecosystem, but pySIPNET is **independent of PEcAn**.  It makes no assumptions about PEcAn conventions, file layouts, or data formats.

## What pySIPNET is NOT

pySIPNET does not include an ensemble runner.  The single-run interface is designed to be composed by external tools suited to the target compute environment.

## Quick example

```python
from pysipnet import ClimateDrivers, ModelFlags, SIPNETParameters, SIPNETRunner
from pysipnet.parameters import (
    InitialConditions, PhotosynthesisParams, PhenologyParams,
    RespirationParams, AllocationParams, WaterParams, LeafPhysiologyParams,
)

params = SIPNETParameters(
    initial_conditions=InitialConditions(
        total_wood_carbon=9600.0, leaf_area_index=4.2, soil_carbon=16000.0,
        soil_wetness_fraction=0.5, fine_root_fraction=0.2, coarse_root_fraction=0.2,
    ),
    photosynthesis=PhotosynthesisParams(
        max_photosynthesis_rate=8.3, daily_mean_photosynthesis_fraction=0.76, foliar_respiration_fraction=0.1,
        min_photosynthesis_temperature=2.0, optimum_photosynthesis_temperature=24.0,
        vapor_pressure_deficit_slope=0.05, vapor_pressure_deficit_exponent=2.0,
        half_saturation_light=17.0, light_extinction_coefficient=0.5,
    ),
    # ... and the five remaining groups
)

climate = ClimateDrivers.from_file("data/my_site.clim")
runner  = SIPNETRunner(flags=ModelFlags.standard())
result  = runner.run(params, climate)

print(result.outputs["nee"])   # net_ecosystem_exchange, with units attached
```

`data/my_site.clim` stands in for your own climate file.  For something that
runs as written, the [Quickstart](user-guide/quickstart.md) uses the year of
Niwot Ridge forcing this repository ships.

## Getting started

See [Installation](installation.md) for setup instructions, including building the SIPNET binary.
