# Quickstart

This guide walks through a minimal end-to-end SIPNET run.  For a complete
reference — all parameter groups, runner options, output inspection — see
[Running a Model](running-a-model.md).

## Prerequisites

Complete [installation](../installation.md) first, including building the
SIPNET binary (`make sipnet`).

## 1. Load climate data

```python
from pysipnet import ClimateDrivers

climate = ClimateDrivers.from_file("data/era5_site1.clim", version="v1")
print(climate)
# ClimateDrivers(version='v1', timesteps=29200, range=2012-001 to 2023-365)
```

## 2. Define parameters

Parameters are organised into domain-specific groups.  Pydantic validates
every value at construction time — invalid values raise `ValidationError`
immediately.

```python
from pysipnet import SIPNETParametersV1
from pysipnet.parameters import (
    InitialConditions, PhotosynthesisParams, PhenologyParams,
    RespirationParams, AllocationParams, WaterParams, LeafPhysiologyParams,
)

params = SIPNETParametersV1(
    initial_conditions=InitialConditions(
        plant_wood=30000.0, lai=0.0, soil=10000.0,
        soil_water_frac=0.5, fine_root_frac=0.05, coarse_root_frac=0.15,
    ),
    photosynthesis=PhotosynthesisParams(
        a_max=112.0, a_max_frac=0.76, base_fol_resp_frac=0.1,
        psn_t_min=2.0, psn_t_opt=24.0,
        d_vpd_slope=0.05, d_vpd_exp=1.0,
        half_sat_par=300.0, attenuation=0.5,
    ),
    phenology=PhenologyParams(
        leaf_off_day=270.0, gdd_leaf_on=100.0,
        leaf_growth=50.0, frac_leaf_fall=0.95,
        leaf_allocation=0.25, leaf_turnover_rate=1.0,
    ),
    respiration=RespirationParams(
        base_veg_resp=0.02, veg_resp_q10=2.0, growth_resp_frac=0.0,
        frozen_soil_fol_r_eff=0.5, frozen_soil_threshold=-1.0,
        base_fine_root_resp=0.5, base_coarse_root_resp=0.1,
        fine_root_q10=2.0, coarse_root_q10=2.0,
        base_soil_resp=0.06, soil_resp_q10=2.0, soil_resp_moist_effect=1.5,
    ),
    allocation=AllocationParams(
        fine_root_allocation=0.35, wood_allocation=0.30,
        fine_root_turnover_rate=1.0, coarse_root_turnover_rate=0.1,
        wood_turnover_rate=0.02,
    ),
    water=WaterParams(
        water_remove_frac=0.1, frozen_soil_eff=0.1, wue_const=10.0,
        soil_whc=12.0, litter_whc=5.0,
        immed_evap_frac=0.1, fast_flow_frac=0.1,
        snow_melt=0.15, rd_const=100.0, r_soil_const1=3.0, r_soil_const2=2.0,
    ),
    leaf=LeafPhysiologyParams(leaf_c_sp_wt=32.0, c_frac_leaf=0.45),
)
```

## 3. Run SIPNET

```python
from pysipnet import SIPNETRunner, ModelPreset

runner = SIPNETRunner(preset=ModelPreset.STANDARD)
result = runner.run(params, climate)

print(result.provenance.success)   # True
print(result.nee().sum())          # annual NEE  (g C m⁻²)
print(result.gpp().sum())          # annual GPP
```

## 4. Inspect the result

```python
ts = result.outputs.data   # pandas DataFrame, one row per timestep
print(ts.columns.tolist())
# ['year', 'day', 'time', 'plant_wood_c', ..., 'nee', 'gpp', ...]

import matplotlib.pyplot as plt
ts.plot(x="day", y=["nee", "gpp"])
plt.show()
```

## Next steps

- [Running a Model](running-a-model.md) — parameter overrides, `SIPNETModel`,
  result inspection, and parameter metadata.
- [File I/O](file-io.md) — lazy output loading, climate staging, keeping files
  on disk.
- [Common Workflows](workflows.md) — end-to-end patterns from interactive
  exploration to full ensemble archival.
