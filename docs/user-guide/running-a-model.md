# Running a Model

This guide walks through a complete SIPNET run: loading climate data, defining
parameters, executing the model, and working with the results.

Before you start, complete [installation](../installation.md), including
building the SIPNET binary with `make sipnet`.

---

## What a run requires

A SIPNET run takes three inputs and produces one result:

| Object | Class | What it represents |
|:-------|:------|:-------------------|
| Parameters | `SIPNETParametersV1` | All model parameters and initial conditions |
| Climate | `ClimateDrivers` | Meteorological forcing timeseries |
| Runner | `SIPNETRunner` | Manages binary selection and subprocess execution |

---

## Loading climate data

Climate forcing is stored in a SIPNET `.clim` file — one row per timestep,
14 columns of meteorological variables.

```python
from pysipnet import ClimateDrivers

climate = ClimateDrivers.from_file("data/era5_site1.clim", version="v1")
print(climate)
# ClimateDrivers(version='v1', timesteps=29200, range=2012-001 to 2023-365)
```

`ClimateDrivers` validates the file on load: it checks that every row is
complete, that timesteps are monotonically increasing, and that VPD and wind
speed are positive. A failed validation raises `ValueError` with a message
that identifies the offending row.

---

## Defining parameters

Parameters are organised into seven domain-specific groups.  Each group is a
Pydantic model; the top-level `SIPNETParametersV1` composes them all.

```python
from pysipnet import (
    SIPNETParametersV1,
)
from pysipnet.parameters import (
    InitialConditions,
    PhotosynthesisParams,
    PhenologyParams,
    RespirationParams,
    AllocationParams,
    WaterParams,
    LeafPhysiologyParams,
)

params = SIPNETParametersV1(
    initial_conditions=InitialConditions(
        plant_wood=30000.0,   # g C m⁻² — initial aboveground + root C
        lai=0.0,              # m² m⁻² — leaf area index at t=0
        soil=10000.0,         # g C m⁻² — initial soil C pool
        soil_water_frac=0.5,  # fraction of water holding capacity
        fine_root_frac=0.05,  # fraction of plant_wood allocated to fine roots
        coarse_root_frac=0.15,
    ),
    photosynthesis=PhotosynthesisParams(
        a_max=112.0,             # nmol CO₂ g⁻¹ leaf s⁻¹
        a_max_frac=0.76,
        base_fol_resp_frac=0.1,
        psn_t_min=2.0,           # °C — minimum temperature for net PSN
        psn_t_opt=24.0,          # °C — optimum temperature
        d_vpd_slope=0.05,        # kPa⁻¹
        d_vpd_exp=1.0,
        half_sat_par=300.0,      # mol photons m⁻² day⁻¹
        attenuation=0.5,
    ),
    phenology=PhenologyParams(
        leaf_off_day=270.0,       # day of year
        gdd_leaf_on=100.0,        # °C·day — GDD threshold for leaf-out
        leaf_growth=50.0,         # g C m⁻²
        frac_leaf_fall=0.95,
        leaf_allocation=0.25,     # fraction of NPP to leaves
        leaf_turnover_rate=1.0,   # year⁻¹
    ),
    respiration=RespirationParams(
        base_veg_resp=0.02,           # year⁻¹ (SIPNET divides by 365 internally)
        veg_resp_q10=2.0,
        growth_resp_frac=0.0,
        frozen_soil_fol_r_eff=0.5,
        frozen_soil_threshold=-1.0,   # °C
        base_fine_root_resp=0.5,      # year⁻¹
        base_coarse_root_resp=0.1,    # year⁻¹
        fine_root_q10=2.0,
        coarse_root_q10=2.0,
        base_soil_resp=0.06,          # year⁻¹
        soil_resp_q10=2.0,
        soil_resp_moist_effect=1.5,
    ),
    allocation=AllocationParams(
        fine_root_allocation=0.35,
        wood_allocation=0.30,
        fine_root_turnover_rate=1.0,   # year⁻¹
        coarse_root_turnover_rate=0.1,
        wood_turnover_rate=0.02,
    ),
    water=WaterParams(
        water_remove_frac=0.1,
        frozen_soil_eff=0.1,
        wue_const=10.0,
        soil_whc=12.0,        # cm — soil water holding capacity
        litter_whc=5.0,       # cm — litter water holding capacity
        immed_evap_frac=0.1,
        fast_flow_frac=0.1,
        snow_melt=0.15,       # cm °C⁻¹ day⁻¹ — required for SNOW=1
        rd_const=100.0,
        r_soil_const1=3.0,
        r_soil_const2=2.0,
    ),
    leaf=LeafPhysiologyParams(
        leaf_c_sp_wt=32.0,    # g C m⁻² leaf
        c_frac_leaf=0.45,
    ),
)
```

### Validation

Pydantic validates every field immediately when you construct the model.
Out-of-range values, missing required fields, and constraint violations all
raise `ValidationError` before any file is written or binary called:

```python
from pydantic import ValidationError

try:
    bad = PhotosynthesisParams(a_max=-1.0, ...)  # negative — not allowed
except ValidationError as exc:
    print(exc)
    # 1 validation error for PhotosynthesisParams
    # a_max
    #   Input should be greater than 0 [type=greater_than, ...]
```

The allocation groups enforce a cross-model constraint:
`leaf_allocation + fine_root_allocation + wood_allocation` must be strictly
less than 1, because SIPNET derives coarse-root allocation as the residual.
Violating this raises `ValidationError` on `SIPNETParametersV1`.

### Flag-dependent parameters

Some parameters are only used when the corresponding compile-time flag is
active in the binary.  The `STANDARD` preset enables `SNOW=1`, `GDD=1`, and
`WATER_HRESP=1`.  With this preset:

- `water.snow_melt` is required (because `SNOW=1`).
- `phenology.gdd_leaf_on` is required (because `GDD=1`).
- `phenology.leaf_on_day` is ignored.

Call `params.validate_for_flags(preset.flags)` to check for flag-parameter
mismatches before running:

```python
from pysipnet import ModelPreset

ModelPreset.STANDARD.flags  # returns ModelFlagsV1(snow=True, gdd=True, ...)
params.validate_for_flags(ModelPreset.STANDARD.flags)  # raises ValueError if misconfigured
```

---

## Running SIPNET

`SIPNETRunner` wraps the subprocess call.  You select a binary preset at
construction time and pass parameters and climate to `run()`:

```python
from pysipnet import SIPNETRunner, ModelPreset

runner = SIPNETRunner(preset=ModelPreset.STANDARD)
result = runner.run(params, climate)
```

Each call to `run()` is fully isolated: inputs are written to a fresh
temporary directory, the binary executes there, and the directory is cleaned
up on completion.  This makes it safe to call `run()` concurrently across
many threads or processes.

### Available presets

| Preset | Active flags |
|:-------|:-------------|
| `ModelPreset.STANDARD` | SNOW=1, GDD=1, WATER_HRESP=1 |
| `ModelPreset.FOREST` | standard + LITTER_POOL=1 |

Use `ModelPreset.FOREST` for sites with a distinct litter carbon layer (e.g.
boreal or deciduous forest). The `FOREST` preset requires
`respiration.litter_breakdown_rate` and `respiration.frac_litter_respired`.

### Keeping the working directory

By default the temporary working directory is deleted after the run.  Set
`keep_workdir=True` on the runner to preserve it for inspection:

```python
runner = SIPNETRunner(preset=ModelPreset.STANDARD, keep_workdir=True)
result = runner.run(params, climate)
print(result.provenance.workdir)   # path to the preserved directory
```

---

## Inspecting the result

`runner.run()` returns a `SIPNETResult`.  The most important attribute is
`outputs` — a pandas DataFrame with one row per model timestep:

```python
print(result.outputs.columns.tolist())
# ['year', 'day', 'time',
#  'plant_wood_c', 'plant_leaf_c', 'wood_creation',
#  'soil_c', 'coarse_root_c', 'fine_root_c', 'litter_c',
#  'soil_water', 'soil_wetness_frac', 'snow',
#  'npp', 'nee', 'cum_nee', 'gpp',
#  'r_aboveground', 'r_soil', 'r_root', 'ra', 'rh', 'rtot',
#  'evapotranspiration', 'transpiration', 'f_par']

print(result.outputs.head())
```

### Common output variables

| Column | Units | Description |
|:-------|:------|:------------|
| `nee` | g C m⁻² per timestep | Net ecosystem exchange (positive = to atmosphere) |
| `gpp` | g C m⁻² per timestep | Gross primary production |
| `npp` | g C m⁻² per timestep | Net primary production |
| `ra` | g C m⁻² per timestep | Total autotrophic respiration |
| `rh` | g C m⁻² per timestep | Heterotrophic respiration |
| `evapotranspiration` | cm per timestep | Evapotranspiration |
| `plant_wood_c` | g C m⁻² | Wood + root C pool |
| `soil_c` | g C m⁻² | Soil C pool |

### Convenience accessors

Three common variables have dedicated accessor methods that return
`pd.Series`:

```python
result.nee()   # net ecosystem exchange timeseries
result.gpp()   # gross primary production timeseries
result.et()    # evapotranspiration timeseries
```

### Annual summaries

Aggregate daily outputs to annual totals using standard pandas operations:

```python
annual = (
    result.outputs
    .groupby("year")[["nee", "gpp", "evapotranspiration"]]
    .sum()
)
print(annual)
```

### xarray output

If you have the `xarray` extra installed (`pip install pysipnet[xarray]`),
convert to a Dataset with `year`, `day`, and `time` as coordinates:

```python
ds = result.to_xarray()
ds["nee"].plot()   # requires matplotlib
```

### Checking for success

```python
print(result.provenance.success)   # True if returncode == 0
print(result.provenance.returncode)
print(result.provenance.stderr)    # SIPNET's stderr output, if any
```

---

## Serialising a run configuration

A complete run configuration round-trips through plain Python dicts — useful
for logging, diffing, and reconstructing parameter sets in other processes:

```python
import json

config = {
    "params": params.model_dump(),
    "preset": ModelPreset.STANDARD.value,
}
json_str = json.dumps(config)

# Reconstruct in another process or from a saved file
config2  = json.loads(json_str)
params2  = SIPNETParametersV1.model_validate(config2["params"])
preset2  = ModelPreset(config2["preset"])
```

---

## Querying parameter metadata

Each parameter carries domain and unit information.
`get_parameter_specs()` returns this as a flat dict, keyed by dotted path:

```python
from pysipnet.parameters.base import get_parameter_specs, ParameterDomain

specs = get_parameter_specs(SIPNETParametersV1)
# {"photosynthesis.a_max": ParameterSpec(unit="nmol / (g * s)", domain=POSITIVE, ...), ...}

# Parameters that require a log transform for unconstrained optimisation
log_params = {k for k, s in specs.items() if s.domain == ParameterDomain.POSITIVE}

# Parameters specified as per-year rates in the .param file
annual_rates = {k for k, s in specs.items() if s.per_year}
```
