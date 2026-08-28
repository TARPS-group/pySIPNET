# pySIPNET

A Python interface to [SIPNET](https://github.com/PecanProject/sipnet) — the Simplified Photosynthesis and Evapotranspiration Model.

## What is SIPNET?

SIPNET is a lightweight process-based terrestrial carbon and water flux model.  It runs at a single site, is driven by meteorological forcing data, and produces estimates of NEE, GPP, ecosystem respiration, ET, and carbon pool dynamics.

## What is pySIPNET?

pySIPNET provides:

- **Typed, hierarchical parameter models** — every parameter carries its units, domain, and a description.
- **Validated climate drivers** — the forcing data structure catches format errors before SIPNET does.
- **Isolated run execution** — each run gets a fresh working directory; runs never share state.
- **Clean output** — SIPNET output is parsed into a labelled DataFrame.

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
        plant_wood=30000, lai=0.0, soil=10000,
        soil_water_frac=0.5, fine_root_frac=0.05, coarse_root_frac=0.15,
    ),
    photosynthesis=PhotosynthesisParams(
        a_max=112.0, a_max_frac=0.76, base_fol_resp_frac=0.1,
        psn_t_min=2.0, psn_t_opt=24.0,
        d_vpd_slope=0.05, d_vpd_exp=1.0,
        half_sat_par=300.0, attenuation=0.5,
    ),
    # ... and the five remaining groups
)

climate = ClimateDrivers.from_file("data/era5_site1.clim", n_columns=14)
runner  = SIPNETRunner(flags=ModelFlags.standard())
result  = runner.run(params, climate)

print(result.nee().describe())
```

## Getting started

See [Installation](installation.md) for setup instructions, including building the SIPNET binary.
