# Running a Model

This guide walks through running SIPNET in Python: loading climate data,
defining parameters, executing the model, and working with the results.

Before you start, complete [installation](../installation.md), including
building the SIPNET binary with `make sipnet`.

---

## Two interfaces

pySIPNET provides two interfaces for running SIPNET.

**`SIPNETRunner`** is the subprocess manager that executes the SIPNET binary.
Each call writes inputs to a fresh working directory, runs the binary there,
and returns a `SIPNETResult`.  You can call it directly or use it as the
foundation for `SIPNETModel`.

**`SIPNETModel`** wraps a runner and a baseline parameter set.  You call it as
a function — optionally supplying parameter or climate overrides for each run
— which makes it the preferred entry point for most workflows:

```python
runner = SIPNETRunner(flags=ModelFlags.standard())
model  = SIPNETModel(runner, base_params=params, base_climate=climate)

result        = model()                # baseline run
result_tuned  = model(a_max=120.0)     # single parameter override
result_site_b = model(climate=other)   # different climate drivers
```

Both return the same `SIPNETResult`.  The rest of this guide covers both,
starting with `SIPNETRunner` since `SIPNETModel` builds on top of it.

---

## Prepare your inputs

A SIPNET run requires two inputs: climate drivers and a parameter set.

### Climate data

Climate forcing is stored in a SIPNET `.clim` file — one row per timestep,
12 meteorological values per row.

```python
from pysipnet import ClimateDrivers

climate = ClimateDrivers.from_file("data/era5_site1.clim", n_columns=14)
print(climate)
# ClimateDrivers(n_columns=14, timesteps=29200, range=2012-001 to 2023-365)
```

`ClimateDrivers.from_file` loads the data into memory.  For ensemble
workflows with pre-existing files, `ClimateDrivers.from_path` creates a
lightweight file reference without loading the data — see [File I/O](file-io.md).

### Parameters

Parameters are grouped into seven domain-specific sub-models that compose
into a single `SIPNETParameters`.  Pydantic validates every field at
construction time — out-of-range values and missing required fields raise
`ValidationError` immediately.

```python
from pysipnet import SIPNETParameters
from pysipnet.parameters import (
    InitialConditions,
    PhotosynthesisParams,
    PhenologyParams,
    RespirationParams,
    AllocationParams,
    WaterParams,
    LeafPhysiologyParams,
)

params = SIPNETParameters(
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
        gdd_leaf_on=100.0,     # °C·day — required when the gdd flag is on
        leaf_growth=50.0,      # g C m⁻²
        frac_leaf_fall=0.95,
        leaf_allocation=0.25,
        leaf_turnover_rate=1.0,  # year⁻¹
        leaf_on_realloc_frac=0.2,  # cap on wood C drawn at leaf-out
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
        immed_evap_frac=0.1,
        fast_flow_frac=0.1,
        snow_melt=0.15,    # cm °C⁻¹ day⁻¹ — required when the snow flag is on
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

`ModelFlags.standard()` turns on snow, degree-day leaf-out, and moisture-sensitive soil respiration.  This
means `water.snow_melt` and `phenology.gdd_leaf_on` are required.  Call
`validate_for_flags` to catch mismatches before running:

```python
from pysipnet import ModelFlags

params.validate_for_flags(ModelFlags.standard())
# raises ValueError listing any missing flag-required parameters
```

---

## Running with SIPNETRunner

`SIPNETRunner` is the direct interface to the binary.  Construct it once and
reuse it across any number of runs — the runner holds no per-run state.

```python
from pysipnet import SIPNETRunner, ModelFlags

runner = SIPNETRunner(flags=ModelFlags.standard())
result = runner.run(params, climate)

print(result.provenance.success)   # True if returncode == 0
```

A failed run raises `SIPNETRunError`, carrying SIPNET's own stdout and stderr —
that is where the reason lives. Pass `check=False` to get a result object
instead, which is what you want in an ensemble where one member failing should
not stop the rest:

```python
result = runner.run(params, climate, check=False)
if not result.provenance.success:
    print(result.provenance.stderr)
print(result.provenance.stderr)    # SIPNET's stderr output, if any
```

### Key runner parameters

| Parameter | Default | Purpose |
|:----------|:--------|:--------|
| `flags` | optional | Model options; defaults to `ModelFlags.standard()` |
| `timeout` | 300 s | Maximum wall-clock time per run |
| `run_id` | UUID hex | Identifier used in the working directory name |
| `output_dir` | `None` | Copy `sipnet.out` here before workdir cleanup (lazy loading) |
| `keep_workdir` | `False` | Suppress working directory cleanup for debugging |

```python
runner = SIPNETRunner(
    flags=ModelFlags.standard(),
    timeout=600.0,
)

result = runner.run(params, climate, run_id="my_baseline")

print(result.provenance.workdir)    # path to the (deleted) working directory
print(result.provenance.run_id)     # "my_baseline"
```

For I/O options — keeping files on disk, lazy output loading, climate staging
— see [File I/O](file-io.md) and [Common Workflows](workflows.md).

### Named flag sets

| Preset | Active flags |
|:-------|:-------------|
| `ModelFlags.standard()` | snow, degree-day leaf-out, moisture-sensitive soil respiration |
| `ModelFlags.forest()` | as above, plus a separate litter carbon pool |

Use `ModelFlags.forest()` for sites with a distinct litter carbon layer.  It additionally
requires `respiration.litter_breakdown_rate` and `respiration.frac_litter_respired`.

---

## Running with SIPNETModel

`SIPNETModel` wraps a `SIPNETRunner` and a baseline parameter set.  Each call
applies overrides on top of the baseline and delegates the actual execution to
the runner.

```python
from pysipnet import SIPNETRunner, ModelFlags, SIPNETModel

runner = SIPNETRunner(flags=ModelFlags.standard())
model  = SIPNETModel(runner, base_params=params, base_climate=climate)
```

### Baseline run

Call `model()` with no arguments to run the baseline:

```python
result = model()
print(result.outputs.data[["nee", "gpp"]].sum())
```

### Parameter overrides

Pass any SIPNET parameter name as a keyword argument to override its value
for that run.  All other parameters stay at their baseline values.  The
override is applied, Pydantic-validated, and discarded — `model.base_params`
is never mutated.

```python
result_high_psn = model(a_max=140.0)
result_warm     = model(psn_t_opt=28.0)
result_combined = model(a_max=140.0, psn_t_opt=28.0)
```

Unrecognised parameter names raise `ValueError` immediately.  Invalid values
(e.g., a negative `a_max`) raise `ValidationError` before the binary is called.

### Climate and event overrides

Pass `climate=` to replace the climate for a specific run, or `events=` to
supply a management event sequence:

```python
result_site_b      = model(climate=other_climate)
result_with_events = model(events=event_sequence)
result_full        = model(a_max=120.0, climate=other_climate, events=event_sequence)
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

`result.outputs` is a `SIPNETOutput` object.  Access the full DataFrame via
`.data`:

```python
print(result.outputs.data.columns.tolist())
# 35 columns:
#  'year', 'day', 'time', 'plant_wood_c',
#  'plant_leaf_c', 'wood_creation', 'soil_c', 'coarse_root_c',
#  'fine_root_c', 'litter_c', 'soil_water', 'soil_wetness_frac',
#  'snow', 'npp', 'nee', 'cum_nee',
#  'gpp', 'r_aboveground', 'r_soil', 'r_root',
#  'ra', 'rh', 'rtot', 'evapotranspiration',
#  'transpiration', 'mineral_n', 'soil_organic_n', 'litter_n',
#  'plant_storage_n', 'n2o', 'n_leaching', 'n_fixation',
#  'n_uptake', 'ch4', 'npp_storage'
```

Every column is always present. A process that is switched off writes zeros
rather than omitting its column, so the nitrogen and methane columns are there
but empty unless those processes are on. SIPNET checks its own carbon and
nitrogen closure but reports the result as a log warning rather than an
output column, so a failed check appears in `result.provenance.stderr`.

Key variables:

| Column | Units | Description |
|:-------|:------|:------------|
| `nee` | g C m⁻² per timestep | Net ecosystem exchange (positive = to atmosphere) |
| `gpp` | g C m⁻² per timestep | Gross primary production |
| `npp` | g C m⁻² per timestep | Net primary production |
| `ra` | g C m⁻² per timestep | Total autotrophic respiration |
| `rh` | g C m⁻² per timestep | Heterotrophic respiration |
| `evapotranspiration` | cm per timestep | Evapotranspiration |
| `plant_wood_c` | g C m⁻² | Aboveground wood C; roots are separate columns |
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
    result.outputs.data
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

## Querying parameter metadata

### SIPNET_PARAMS_BY_GROUP

`SIPNET_PARAMS_BY_GROUP` maps each group name to the list of parameter names in
that group.  It is useful for discovering available parameters and for
building calibration tooling:

```python
from pysipnet import SIPNET_PARAMS_BY_GROUP

# What parameters are in the photosynthesis group?
SIPNET_PARAMS_BY_GROUP["photosynthesis"]
# ['a_max', 'a_max_frac', 'base_fol_resp_frac', 'psn_t_min', 'psn_t_opt',
#  'd_vpd_slope', 'd_vpd_exp', 'half_sat_par', 'attenuation']

# All groups
list(SIPNET_PARAMS_BY_GROUP.keys())
# ['initial_conditions', 'photosynthesis', 'phenology', 'respiration',
#  'allocation', 'water', 'leaf']

# Total parameter count
sum(len(ps) for ps in SIPNET_PARAMS_BY_GROUP.values())  # 58
```

### get_parameter_specs

For calibration and DA workflows, `get_parameter_specs` returns the full
`ParameterSpec` for each parameter — including unit, mathematical domain, and
whether the value is a per-year rate:

```python
from pysipnet.parameters.base import get_parameter_specs, ParameterDomain

specs = get_parameter_specs(SIPNETParameters)
# {"photosynthesis.a_max": ParameterSpec(unit="nmol / (g * s)", domain=POSITIVE, ...), ...}

# Parameters requiring a log bijector for unconstrained optimisation
log_params = {k for k, s in specs.items() if s.domain == ParameterDomain.POSITIVE}

# Per-year rate parameters
annual_rates = {k for k, s in specs.items() if s.per_year}
```
