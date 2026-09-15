# Quickstart

This guide walks through a minimal end-to-end SIPNET run.  For a complete
reference — all parameter groups, runner options, output inspection — see
[Running a Model](running-a-model.md).

## Prerequisites

Complete [installation](../installation.md) first, including building the
SIPNET binary (`make sipnet`).

## 1. Load climate data

`data/era5_site1.clim` below is a stand-in for your own file — pySIPNET ships
no climate data at that path.  For something you can run immediately, the
repository includes one year of Niwot Ridge forcing at
`docs/examples/data/niwot_1999_daily.clim`.

```python
from pysipnet import ClimateDrivers

climate = ClimateDrivers.from_file("data/era5_site1.clim", n_columns=14)
print(climate)
# ClimateDrivers(n_columns=14, timesteps=29200, range=2012-001 to 2023-365)
```

## 2. Define parameters

Parameters are organized into domain-specific groups.  Pydantic validates
every value at construction time — invalid values raise `ValidationError`
immediately.

```python
from pysipnet import SIPNETParameters
from pysipnet.parameters import (
    InitialConditions, PhotosynthesisParams, PhenologyParams,
    RespirationParams, AllocationParams, WaterParams, LeafPhysiologyParams,
)

params = SIPNETParameters(
    initial_conditions=InitialConditions(
        total_wood_carbon=30000.0, leaf_area_index=0.0, soil_carbon=10000.0,
        soil_wetness_fraction=0.5, fine_root_fraction=0.05, coarse_root_fraction=0.15,
    ),
    photosynthesis=PhotosynthesisParams(
        max_photosynthesis_rate=112.0, daily_mean_photosynthesis_fraction=0.76, foliar_respiration_fraction=0.1,
        min_photosynthesis_temperature=2.0, optimum_photosynthesis_temperature=24.0,
        vapor_pressure_deficit_slope=0.05, vapor_pressure_deficit_exponent=1.0,
        half_saturation_light=300.0, light_extinction_coefficient=0.5,
    ),
    phenology=PhenologyParams(
        leaf_off_day=270.0, leaf_on_growing_degree_days=100.0,
        leaf_on_growth=50.0, leaf_off_fall_fraction=0.95,
        leaf_allocation=0.25, leaf_turnover_rate=1.0, leaf_on_reallocation_fraction=0.2,
    ),
    respiration=RespirationParams(
        base_wood_respiration_rate=0.02, wood_respiration_q10=2.0, growth_respiration_fraction=0.0,
        frozen_soil_foliar_respiration_factor=0.5, frozen_soil_threshold=-1.0,
        base_fine_root_respiration_rate=0.5, base_coarse_root_respiration_rate=0.1,
        fine_root_respiration_q10=2.0, coarse_root_respiration_q10=2.0,
        base_soil_respiration_rate=0.06, soil_respiration_q10=2.0, soil_respiration_moisture_exponent=1.5,
    ),
    allocation=AllocationParams(
        fine_root_allocation=0.35, wood_allocation=0.30,
        fine_root_turnover_rate=1.0, coarse_root_turnover_rate=0.1,
        wood_turnover_rate=0.02,
    ),
    water=WaterParams(
        water_removal_fraction=0.1, frozen_soil_water_fraction=0.1, water_use_efficiency=10.0,
        soil_water_holding_capacity=12.0,
        interception_evaporation_fraction=0.1, fast_flow_fraction=0.1,
        snow_melt_rate=0.15, aerodynamic_resistance_constant=100.0, soil_resistance_intercept=3.0, soil_resistance_slope=2.0,
    ),
    leaf=LeafPhysiologyParams(leaf_carbon_per_area=32.0, leaf_carbon_fraction=0.45),
)
```

!!! note "These values are illustrative"
    The numbers above are plausible placeholders chosen to show the structure,
    not parameters calibrated for any real site.  Run them against real
    forcing and you will get a physically meaningless answer.  For a worked
    example with parameters matched to its site, see the
    [MCMC calibration notebook](../examples/mcmc_calibration.ipynb), which uses
    the Niwot Ridge data included in the repository.

## 3. Run SIPNET

```python
from pysipnet import SIPNETRunner, ModelFlags

runner = SIPNETRunner(flags=ModelFlags.standard())
result = runner.run(params, climate)

print(result.provenance.success)   # True
print(result.outputs.variable("nee").sum())   # annual NEE (g C m⁻²)
print(result.outputs.variable("gpp").sum())   # annual GPP
```

## 4. Inspect the result

```python
ts = result.outputs.data   # pandas DataFrame, one row per timestep
print(ts.columns.tolist())
# ['year', 'day_of_year', 'hour_of_day', 'wood_carbon', ...,
#  'net_ecosystem_exchange', 'gross_primary_production', ...]

import matplotlib.pyplot as plt
ts.plot(x="day_of_year", y=["net_ecosystem_exchange", "gross_primary_production"])
plt.show()

# Or the xarray view, which knows its units:
result.outputs["nee"].plot()
```

Every column's meaning, units and SIPNET name are listed on the
[Output variables](../reference/output-variables.md) page.

## Next steps

- [Running a Model](running-a-model.md) — parameter overrides, `SIPNETModel`,
  result inspection, and parameter metadata.
- [File I/O](file-io.md) — lazy output loading, climate staging, keeping files
  on disk.
- [Common Workflows](workflows.md) — end-to-end patterns from interactive
  exploration to full ensemble archival.
