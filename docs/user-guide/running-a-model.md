# Running a Model

This guide walks through running SIPNET in Python: loading climate data,
defining parameters, executing the model, and working with the results.

Before you start, complete [installation](../installation.md), including
building the SIPNET binary with `make sipnet`.

---

## Two interfaces

pySIPNET provides two interfaces for running SIPNET.

**`SIPNETModel`** is the recommended entry point for most users.  You
construct it once with a baseline parameter set and climate, then call it
as a function — optionally supplying parameter or climate overrides for
each run:

```python
model = SIPNETModel(runner, base_params=params, base_climate=climate)

result        = model()                # baseline run
result_tuned  = model(a_max=120.0)     # single parameter override
result_site_b = model(climate=other)   # different climate drivers
```

**`SIPNETRunner`** is the lower-level subprocess manager that `SIPNETModel`
uses internally.  You can call it directly when you need control over
execution details — keeping the working directory for debugging, setting a
custom timeout, or passing an explicit run identifier:

```python
result = runner.run(params, climate)   # direct call
```

Both return the same `SIPNETResult`.  The rest of this guide covers how to
use both, starting with `SIPNETModel`.

---

## Prepare your inputs

A SIPNET run requires two inputs: climate drivers and a parameter set.

### Climate data

Climate forcing is stored in a SIPNET `.clim` file — one row per timestep,
14 columns of meteorological variables.

```python
from pysipnet import ClimateDrivers

climate = ClimateDrivers.from_file("data/era5_site1.clim", version="v1")
print(climate)
# ClimateDrivers(version='v1', timesteps=29200, range=2012-001 to 2023-365)
```

`ClimateDrivers` validates the file on load: every row must be complete,
timesteps must be monotonically increasing, and VPD and wind speed must be
positive.  A failed validation raises `ValueError` with a message that
identifies the offending row.

### Parameters

Parameters are grouped into seven domain-specific sub-models that compose
into a single `SIPNETParametersV1`.  Pydantic validates every field at
construction time — out-of-range values and missing required fields raise
`ValidationError` immediately.

```python
from pysipnet import SIPNETParametersV1
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
        plant_wood=30000.0,    # g C m⁻² — initial aboveground + root C
        lai=0.0,               # m² m⁻² — leaf area index at t=0
        soil=10000.0,          # g C m⁻² — initial soil C pool
        soil_water_frac=0.5,   # fraction of water holding capacity
        fine_root_frac=0.05,
        coarse_root_frac=0.15,
    ),
    photosynthesis=PhotosynthesisParams(
        a_max=112.0,           # nmol CO₂ g⁻¹ leaf s⁻¹
        a_max_frac=0.76,
        base_fol_resp_frac=0.1,
        psn_t_min=2.0,         # °C
        psn_t_opt=24.0,        # °C
        d_vpd_slope=0.05,
        d_vpd_exp=1.0,
        half_sat_par=300.0,    # mol photons m⁻² day⁻¹
        attenuation=0.5,
    ),
    phenology=PhenologyParams(
        leaf_off_day=270.0,
        gdd_leaf_on=100.0,     # °C·day — required for GDD=1 preset
        leaf_growth=50.0,      # g C m⁻²
        frac_leaf_fall=0.95,
        leaf_allocation=0.25,
        leaf_turnover_rate=1.0,  # year⁻¹
    ),
    respiration=RespirationParams(
        base_veg_resp=0.02,        # year⁻¹ (SIPNET divides by 365 internally)
        veg_resp_q10=2.0,
        growth_resp_frac=0.0,
        frozen_soil_fol_r_eff=0.5,
        frozen_soil_threshold=-1.0,
        base_fine_root_resp=0.5,   # year⁻¹
        base_coarse_root_resp=0.1,
        fine_root_q10=2.0,
        coarse_root_q10=2.0,
        base_soil_resp=0.06,       # year⁻¹
        soil_resp_q10=2.0,
        soil_resp_moist_effect=1.5,
    ),
    allocation=AllocationParams(
        fine_root_allocation=0.35,
        wood_allocation=0.30,
        fine_root_turnover_rate=1.0,
        coarse_root_turnover_rate=0.1,
        wood_turnover_rate=0.02,
    ),
    water=WaterParams(
        water_remove_frac=0.1,
        frozen_soil_eff=0.1,
        wue_const=10.0,
        soil_whc=12.0,     # cm — soil water holding capacity
        litter_whc=5.0,    # cm — litter water holding capacity
        immed_evap_frac=0.1,
        fast_flow_frac=0.1,
        snow_melt=0.15,    # cm °C⁻¹ day⁻¹ — required for SNOW=1 preset
        rd_const=100.0,
        r_soil_const1=3.0,
        r_soil_const2=2.0,
    ),
    leaf=LeafPhysiologyParams(
        leaf_c_sp_wt=32.0,   # g C m⁻² leaf
        c_frac_leaf=0.45,
    ),
)
```

#### Flag-dependent parameters

The `STANDARD` preset enables `SNOW=1`, `GDD=1`, and `WATER_HRESP=1`.  This
means `water.snow_melt` and `phenology.gdd_leaf_on` are required.  Call
`validate_for_flags` to catch mismatches before running:

```python
from pysipnet import ModelPreset

params.validate_for_flags(ModelPreset.STANDARD.flags)
# raises ValueError listing any missing flag-required parameters
```

---

## Running with SIPNETModel

`SIPNETModel` is constructed from a `SIPNETRunner` plus a baseline parameter
set and (optionally) a default climate.  The runner determines the binary
preset; `SIPNETModel` handles the override logic.

```python
from pysipnet import SIPNETRunner, ModelPreset, SIPNETModel

runner = SIPNETRunner(preset=ModelPreset.STANDARD)
model  = SIPNETModel(runner, base_params=params, base_climate=climate)
```

### Baseline run

Call `model()` with no arguments to run the baseline:

```python
result = model()
print(result.outputs[["nee", "gpp"]].sum())
```

### Parameter overrides

Pass any SIPNET v1 parameter name as a keyword argument to override its
value for that run.  All other parameters stay at their baseline values.
The override is applied, Pydantic-validated, and discarded — `model.base_params`
is never mutated.

```python
result_high_psn = model(a_max=140.0)
result_warm     = model(psn_t_opt=28.0)
result_combined = model(a_max=140.0, psn_t_opt=28.0)
```

Unrecognised parameter names raise `ValueError` immediately.  Invalid values
(e.g., a negative `a_max`) raise `ValidationError` before the binary is
called.

### Climate and event overrides

Pass `climate=` to replace the climate for a specific run, or `events=` to
supply a management event sequence:

```python
result_site_b = model(climate=other_climate)
result_with_events = model(events=event_sequence)
result_full   = model(a_max=120.0, climate=other_climate, events=event_sequence)
```

### Sensitivity exploration

`SIPNETModel` makes it easy to explore parameter sensitivity interactively:

```python
import pandas as pd

rows = []
for a_max in [80.0, 100.0, 112.0, 130.0, 150.0]:
    r = model(a_max=a_max)
    rows.append({"a_max": a_max, "annual_gpp": r.gpp().sum()})

pd.DataFrame(rows)
```

---

## Inspecting the result

Both `SIPNETModel` and `SIPNETRunner.run()` return a `SIPNETResult`.

### The outputs DataFrame

`result.outputs` is a pandas DataFrame with one row per model timestep:

```python
print(result.outputs.columns.tolist())
# ['year', 'day', 'time',
#  'plant_wood_c', 'plant_leaf_c', 'wood_creation',
#  'soil_c', 'coarse_root_c', 'fine_root_c', 'litter_c',
#  'soil_water', 'soil_wetness_frac', 'snow',
#  'npp', 'nee', 'cum_nee', 'gpp',
#  'r_aboveground', 'r_soil', 'r_root', 'ra', 'rh', 'rtot',
#  'evapotranspiration', 'transpiration', 'f_par']
```

Key variables:

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

```python
result.nee()   # pd.Series — net ecosystem exchange
result.gpp()   # pd.Series — gross primary production
result.et()    # pd.Series — evapotranspiration
```

### Annual summaries

```python
annual = (
    result.outputs
    .groupby("year")[["nee", "gpp", "evapotranspiration"]]
    .sum()
)
```

### xarray output

With the `xarray` extra installed, convert to a Dataset with `year`, `day`,
and `time` as coordinates:

```python
ds = result.to_xarray()
```

---

## Using SIPNETRunner directly

You will rarely need `SIPNETRunner.run()` directly — `SIPNETModel` covers
most cases more cleanly.  The situations where direct access is useful:

- **Inspecting the working directory.** Set `keep_workdir=True` on the runner
  to preserve the temp directory after the run, then examine `sipnet.param`,
  `sipnet.clim`, and `sipnet.out` directly.
- **Custom timeout.** The default timeout is 300 s per run; set a different
  value at construction time.
- **Explicit run IDs.** Useful for correlating working directories with runs
  in your own logging.

```python
runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    keep_workdir=True,    # preserve temp dir for inspection
    timeout=600.0,        # seconds
)

result = runner.run(params, climate, run_id="my_baseline")

print(result.provenance.workdir)    # path to the preserved directory
print(result.provenance.success)    # True if returncode == 0
print(result.provenance.stderr)     # SIPNET's stderr, if any
```

`SIPNETRunner.run()` is stateless: the same runner instance can be used for
any number of runs with different parameters and climate drivers.

### Binary presets

| Preset | Active flags |
|:-------|:-------------|
| `ModelPreset.STANDARD` | SNOW=1, GDD=1, WATER_HRESP=1 |
| `ModelPreset.FOREST` | standard + LITTER_POOL=1 |

Use `FOREST` for sites with a distinct litter carbon layer.  It additionally
requires `respiration.litter_breakdown_rate` and `respiration.frac_litter_respired`.

---

## Serialising a run configuration

`SIPNETParametersV1` round-trips through plain Python dicts, making it easy
to log, diff, or reconstruct parameter sets:

```python
import json

config = {
    "params": params.model_dump(),
    "preset": ModelPreset.STANDARD.value,
}
json_str = json.dumps(config)

# Reconstruct in another process or from a saved file
config2 = json.loads(json_str)
params2 = SIPNETParametersV1.model_validate(config2["params"])
preset2 = ModelPreset(config2["preset"])
```

---

## Querying parameter metadata

### SIPNET_PARAM_GROUPS

`SIPNET_PARAM_GROUPS` maps each group name to the list of parameter names in
that group.  It is useful for discovering available parameters and for
building calibration tooling:

```python
from pysipnet import SIPNET_PARAM_GROUPS

# What parameters are in the photosynthesis group?
SIPNET_PARAM_GROUPS["photosynthesis"]
# ['a_max', 'a_max_frac', 'base_fol_resp_frac', 'psn_t_min', 'psn_t_opt',
#  'd_vpd_slope', 'd_vpd_exp', 'half_sat_par', 'attenuation']

# All groups
list(SIPNET_PARAM_GROUPS.keys())
# ['initial_conditions', 'photosynthesis', 'phenology', 'respiration',
#  'allocation', 'water', 'leaf']

# Total parameter count
sum(len(ps) for ps in SIPNET_PARAM_GROUPS.values())  # 61
```

### get_parameter_specs

For calibration and DA workflows, `get_parameter_specs` returns the full
`ParameterSpec` for each parameter — including unit, mathematical domain, and
whether the value is a per-year rate:

```python
from pysipnet.parameters.base import get_parameter_specs, ParameterDomain

specs = get_parameter_specs(SIPNETParametersV1)
# {"photosynthesis.a_max": ParameterSpec(unit="nmol / (g * s)", domain=POSITIVE, ...), ...}

# Parameters requiring a log bijector for unconstrained optimisation
log_params = {k for k, s in specs.items() if s.domain == ParameterDomain.POSITIVE}

# Per-year rate parameters
annual_rates = {k for k, s in specs.items() if s.per_year}
```
