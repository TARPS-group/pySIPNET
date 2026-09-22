# Quickstart

This guide walks through a minimal end-to-end SIPNET run.  For a complete
reference — all parameter groups, runner options, output inspection — see
[Running a Model](running-a-model.md).

## Prerequisites

Complete [installation](../installation.md) first, including building the
SIPNET binary (`make sipnet`).

## 1. Load climate data

This guide runs on data the repository ships: one year of daily Niwot Ridge
(US-NR1) forcing, aggregated from SIPNET's own example driver.  Paths are
relative to the repository root; point `from_file` at your own `.clim` file to
run a different site.

```python
from pysipnet import ClimateDrivers

climate = ClimateDrivers.from_file("docs/examples/data/niwot_1999_daily.clim", n_columns=14)
print(climate)
# ClimateDrivers(n_columns=14, timesteps=365, range=1999-001 to 1999-365)
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
        total_wood_carbon=9600.0, leaf_area_index=4.2, soil_carbon=16000.0,
        soil_wetness_fraction=0.5, snow_water_equivalent=0.0,
        fine_root_fraction=0.2, coarse_root_fraction=0.2,
    ),
    photosynthesis=PhotosynthesisParams(
        max_photosynthesis_rate=8.3, daily_mean_photosynthesis_fraction=0.76,
        foliar_respiration_fraction=0.1,
        min_photosynthesis_temperature=2.0, optimum_photosynthesis_temperature=24.0,
        vapor_pressure_deficit_slope=0.05, vapor_pressure_deficit_exponent=2.0,
        half_saturation_light=17.0, light_extinction_coefficient=0.5,
    ),
    phenology=PhenologyParams(
        leaf_off_day=285.0, leaf_on_growing_degree_days=500.0,
        leaf_on_growth=0.0, leaf_off_fall_fraction=0.0,
        leaf_allocation=0.2, leaf_turnover_rate=0.13, leaf_on_reallocation_fraction=0.2,
    ),
    respiration=RespirationParams(
        base_wood_respiration_rate=0.006, wood_respiration_q10=2.0, growth_respiration_fraction=0.2,
        frozen_soil_foliar_respiration_factor=0.0, frozen_soil_threshold=0.0,
        base_fine_root_respiration_rate=0.09, base_coarse_root_respiration_rate=0.006,
        fine_root_respiration_q10=2.6, coarse_root_respiration_q10=2.6,
        base_soil_respiration_rate=0.06, soil_respiration_q10=2.9,
        soil_respiration_moisture_exponent=1.0,
    ),
    allocation=AllocationParams(
        fine_root_allocation=0.4, wood_allocation=0.2,
        fine_root_turnover_rate=0.137, coarse_root_turnover_rate=0.056,
        wood_turnover_rate=0.014,
    ),
    water=WaterParams(
        water_removal_fraction=0.088, frozen_soil_water_fraction=0.0, water_use_efficiency=10.9,
        soil_water_holding_capacity=12.0,
        interception_evaporation_fraction=0.1, fast_flow_fraction=0.1,
        snow_melt_rate=0.15, aerodynamic_resistance_constant=36.5,
        soil_resistance_intercept=8.2, soil_resistance_slope=4.3,
    ),
    leaf=LeafPhysiologyParams(leaf_carbon_per_area=270.0, leaf_carbon_fraction=0.45),
)
```

!!! note "These values are nominal, not calibrated"
    The numbers above are SIPNET's own nominal Niwot Ridge values, taken from
    the parameter file the model authors ship (bundled with the package as
    `pysipnet/data/niwot/sipnet.param`; see
    [Bundled reference data](file-io.md#bundled-reference-data)), so they match the climate
    loaded in step 1.  They are a sensible starting point, not a fit: nothing
    here was calibrated against observations for 1999, and the annual totals
    below should be read as "the model ran", not as an estimate of this site's
    carbon balance.  For what calibration involves, see the
    [MCMC calibration notebook](../examples/mcmc_calibration.ipynb), which
    fits two of these parameters to the same year.

## 3. Run SIPNET

```python
from pysipnet import SIPNETRunner, ModelFlags

runner = SIPNETRunner(flags=ModelFlags.standard())
result = runner.run(params, climate)

print(result.provenance.success)              # True
print(float(result.outputs["nee"].sum()))    # 558.009 — annual NEE (g C m⁻², +ve = source)
print(float(result.outputs["gpp"].sum()))    # 686.052 — annual GPP
```

## 4. Inspect the result

```python
ts = result.outputs.pandas   # pandas DataFrame, one row per timestep
print(ts.columns.tolist())
# ['year', 'day_of_year', 'hour_of_day', 'wood_carbon', 'leaf_carbon', ...,
#  'net_ecosystem_exchange', 'gross_primary_production', ...]  # 35 columns

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
