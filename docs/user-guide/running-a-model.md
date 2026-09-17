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
result_tuned  = model(max_photosynthesis_rate=120.0)     # single parameter override
result_site_b = model(climate=other)   # different climate drivers
```

Both return the same `SIPNETResult`.  The rest of this guide covers both,
starting with `SIPNETRunner` since `SIPNETModel` builds on top of it.

---

## Prepare your inputs

A SIPNET run requires two inputs: climate drivers and a parameter set.

### Climate data

Climate forcing is stored in a SIPNET `.clim` file — one row per timestep.
The current layout has 12 columns: four identifying the timestep (SIPNET's
`year day time length`, which become `year`, `day_of_year`, `hour_of_day`,
`time_step_length` in Python) and eight meteorological values. Every column's
name, units and the conversion SIPNET applies on read are on the
[Climate drivers](../reference/climate-drivers.md) page.

```python
from pysipnet import ClimateDrivers

climate = ClimateDrivers.from_file("docs/examples/data/niwot_1999_daily.clim", n_columns=14)
print(climate)
# ClimateDrivers(n_columns=14, timesteps=365, range=1999-001 to 1999-365)
```

That file — one year of daily Niwot Ridge forcing — ships with the repository,
so every example on this page runs as written from the repository root.

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
        total_wood_carbon=9600.0,     # g C m⁻² — initial aboveground + root C
        leaf_area_index=4.2,          # m² m⁻² — leaf area index at t=0
        soil_carbon=16000.0,          # g C m⁻² — initial soil C pool
        soil_wetness_fraction=0.5,    # fraction of water holding capacity
        snow_water_equivalent=0.0,    # cm water equivalent
        fine_root_fraction=0.2,
        coarse_root_fraction=0.2,
    ),
    photosynthesis=PhotosynthesisParams(
        max_photosynthesis_rate=8.3,             # nmol CO₂ g⁻¹ leaf s⁻¹
        daily_mean_photosynthesis_fraction=0.76,
        foliar_respiration_fraction=0.1,
        min_photosynthesis_temperature=2.0,      # °C
        optimum_photosynthesis_temperature=24.0,  # °C
        vapor_pressure_deficit_slope=0.05,
        vapor_pressure_deficit_exponent=2.0,
        half_saturation_light=17.0,   # mol photons m⁻² day⁻¹
        light_extinction_coefficient=0.5,
    ),
    phenology=PhenologyParams(
        leaf_off_day=285.0,
        leaf_on_growing_degree_days=500.0,  # °C·day — required when the gdd flag is on
        leaf_on_growth=0.0,       # g C m⁻²
        leaf_off_fall_fraction=0.0,
        leaf_allocation=0.2,
        leaf_turnover_rate=0.13,  # year⁻¹
        leaf_on_reallocation_fraction=0.2,  # cap on wood C drawn at leaf-out
    ),
    respiration=RespirationParams(
        base_wood_respiration_rate=0.006,      # year⁻¹ (SIPNET divides by 365 internally)
        wood_respiration_q10=2.0,
        growth_respiration_fraction=0.2,
        frozen_soil_foliar_respiration_factor=0.0,
        frozen_soil_threshold=0.0,
        base_fine_root_respiration_rate=0.09,  # year⁻¹
        base_coarse_root_respiration_rate=0.006,
        fine_root_respiration_q10=2.6,
        coarse_root_respiration_q10=2.6,
        base_soil_respiration_rate=0.06,       # year⁻¹
        soil_respiration_q10=2.9,
        soil_respiration_moisture_exponent=1.0,
    ),
    allocation=AllocationParams(
        fine_root_allocation=0.4,
        wood_allocation=0.2,
        fine_root_turnover_rate=0.137,
        coarse_root_turnover_rate=0.056,
        wood_turnover_rate=0.014,
    ),
    water=WaterParams(
        water_removal_fraction=0.088,
        frozen_soil_water_fraction=0.0,
        water_use_efficiency=10.9,
        soil_water_holding_capacity=12.0,     # cm — soil water holding capacity
        interception_evaporation_fraction=0.1,
        fast_flow_fraction=0.1,
        snow_melt_rate=0.15,    # cm °C⁻¹ day⁻¹ — required when the snow flag is on
        aerodynamic_resistance_constant=36.5,
        soil_resistance_intercept=8.2,
        soil_resistance_slope=4.3,
    ),
    leaf=LeafPhysiologyParams(
        leaf_carbon_per_area=270.0,  # g C m⁻² leaf
        leaf_carbon_fraction=0.45,
    ),
)
```

These are SIPNET's own nominal Niwot Ridge values, from the parameter file the
model authors ship (kept here as `tests/fixtures/niwot_reference/sipnet.param`),
so they go with the climate loaded above.  They are a starting point, not a fit
to any particular year.

#### Flag-dependent parameters

`ModelFlags.standard()` turns on snow, degree-day leaf-out, and moisture-sensitive soil respiration.  This
means `water.snow_melt_rate` and `phenology.leaf_on_growing_degree_days` are required.  Call
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
    print(result.provenance.stderr)   # SIPNET's own explanation
```

### Key runner options

Set on the `SIPNETRunner`, applying to every run it performs:

| Option | Default | Purpose |
|:-------|:--------|:--------|
| `flags` | `ModelFlags.standard()` | Model options |
| `timeout` | 300 s | Maximum wall-clock time per run |
| `output_dir` | `None` | Copy `sipnet.out` here before workdir cleanup (lazy loading) |
| `keep_workdir` | `False` | Suppress working directory cleanup for debugging |

Passed to `run()`, applying to that call only:

| Argument | Default | Purpose |
|:---------|:--------|:--------|
| `run_id` | UUID hex | Identifier used in the working directory name |
| `events` | `None` | Management events for this run |
| `output_dir` | runner's value | Overrides the runner-level setting |
| `check` | `True` | Raise `SIPNETRunError` if SIPNET exits non-zero |

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

### The default flag set

`ModelFlags.standard()` turns on snow, degree-day leaf-out and moisture-sensitive
soil respiration, and leaves every other process off.  That is not a pySIPNET
choice: it is exactly what SIPNET compiles in as its own defaults
(`CREATE_INT_CONTEXT` in `src/common/context.c` registers `gdd`, `snow` and
`waterHResp` as on), so it is what the binary does when `sipnet.in` sets no
flags.  `ModelFlags()` gives the same thing without the label.

Any other combination is built directly:

```python
flags = ModelFlags(litter_pool=True, name="niwot")
```

Adding `litter_pool` gives plant litter its own carbon pool instead of routing it
straight into soil carbon, and additionally requires
`respiration.litter_breakdown_rate` and `respiration.litter_respired_fraction`.

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
print(result.outputs.pandas[["net_ecosystem_exchange", "gross_primary_production"]].sum())
```

### Parameter overrides

Pass any parameter field name as a keyword argument to override its value
for that run. Field names follow the same convention as output columns
(`max_photosynthesis_rate`, not `a_max`); the full list is on the
[Parameters](../reference/parameters.md) page. An old or SIPNET-style name raises
`ValueError` naming the current field, and `pysipnet.resolve_parameter_name("aMax")`
returns it directly.  All other parameters stay at their baseline values.  The
override is applied, Pydantic-validated, and discarded — `model.base_params`
is never mutated.

```python
result_high_psn = model(max_photosynthesis_rate=140.0)
result_warm     = model(optimum_photosynthesis_temperature=28.0)
result_combined = model(max_photosynthesis_rate=140.0, optimum_photosynthesis_temperature=28.0)
```

Unrecognized parameter names raise `ValueError` immediately.  Invalid values
(e.g., a negative `max_photosynthesis_rate`) raise `ValidationError` before the binary is called.

### Climate and event overrides

Pass `climate=` to replace the climate for a specific run, or `events=` to
supply a management event sequence:

```python
result_site_b      = model(climate=other_climate)
result_with_events = model(events=event_sequence)
result_full        = model(max_photosynthesis_rate=120.0, climate=other_climate, events=event_sequence)
```

### Sensitivity exploration

`SIPNETModel` makes it easy to explore parameter sensitivity interactively:

```python
import pandas as pd

rows = []
for max_photosynthesis_rate in [80.0, 100.0, 112.0, 130.0, 150.0]:
    r = model(max_photosynthesis_rate=max_photosynthesis_rate)
    rows.append({"max_photosynthesis_rate": max_photosynthesis_rate, "annual_gpp": r.outputs.variable("gpp").sum()})

pd.DataFrame(rows)
```

---

## Inspecting the result

Both `SIPNETModel` and `SIPNETRunner.run()` return a `SIPNETResult`.

### Output variables

`result.outputs` is a `SIPNETOutput`. Its columns are named for what they are,
not for what SIPNET calls them: `net_ecosystem_exchange` rather than `nee`,
`soil_respiration` rather than `rSoil`. The full list, with units, meaning and
the SIPNET column each one comes from, is on the
[Output variables](../reference/output-variables.md) page; the same information
is available in code from `pysipnet.variables`.

```python
from pysipnet.variables import resolve_output_variable

spec = resolve_output_variable("nee")      # aliases resolve to the full spec
spec.name           # 'net_ecosystem_exchange'
spec.units          # 'g m-2'  (UDUNITS syntax; the substance is kept separate, see below)
spec.constituent    # 'C'
spec.formatted_units()  # 'g C m⁻²'
spec.axis_label()   # 'Net ecosystem exchange (g C m⁻²)'
spec.time_reference # 'total over the timestep'
```

Unit strings use UDUNITS syntax (`"g m-2"`, `"cm d-1"`, `"1"` for dimensionless) with
the substance in a separate `constituent` field, because a `C` inside a unit string
would be read as coulombs by units libraries. `formatted_units()` and `axis_label()`
put it back for display. See [Design](../design.md) for the convention.

Every column is always present. A process that is switched off writes zeros
rather than omitting its column, so the nitrogen and methane columns are there
but zero unless those processes are on (`requires_flag` on the spec says which).
SIPNET checks its own carbon and nitrogen closure but reports the result as a
log warning rather than an output column, so a failed check appears in
`result.provenance.stderr`.

!!! note "Start of step versus end of step"
    SIPNET labels each row with the **start** of its timestep. Pools
    (`wood_carbon`, `soil_water`, ...) are the values at the **end** of that
    step, and fluxes (`net_ecosystem_exchange`, `evapotranspiration`, ...) are
    totals **over** it. One column, `transpiration_rate`, is a per-day rate
    rather than a total. The `kind` and `time_reference` fields of each
    variable spell this out, and the xarray view below carries them as
    attributes.

### Three views of the same output

```python
df = result.outputs.pandas                  # pandas DataFrame, one row per timestep
ds = result.outputs.xarray                  # xarray Dataset, one `time` dimension
nee = result.outputs["nee"]                 # one variable as a DataArray, by name or alias
nee_series = result.outputs.variable("nee") # ... or as a pandas Series
```

The Dataset is the representation to use when metadata matters or when
results will be combined across runs:

```python
ds["net_ecosystem_exchange"].attrs
# {'units': 'g m-2', 'long_name': 'Net ecosystem exchange',
#  'kind': 'timestep_total', 'time_reference': 'total over the timestep',
#  'cell_methods': 'time: sum', 'constituent': 'C',
#  'sign_convention': 'positive is a flux from the ecosystem to the atmosphere', ...}

ds["time"]              # datetime64, start of each timestep
ds["time_step_end"]     # datetime64, end of each timestep
ds["time_step_length"]  # timedelta64

ds.to_netcdf("run.nc")  # self-describing on disk
```

### Annual summaries

Fluxes sum; pools average. The registry records the right rule for each
variable as `spec.aggregation`:

```python
annual = (
    result.outputs.pandas
    .groupby("year")[["net_ecosystem_exchange", "gross_primary_production", "evapotranspiration"]]
    .sum()
)
```

## Querying parameter metadata

### SIPNET_PARAMS_BY_GROUP

`SIPNET_PARAMS_BY_GROUP` maps each group name to the list of parameter names in
that group.  It is useful for discovering available parameters and for
building calibration tooling:

```python
from pysipnet import SIPNET_PARAMS_BY_GROUP

# What parameters are in the photosynthesis group?
SIPNET_PARAMS_BY_GROUP["photosynthesis"]
# ['max_photosynthesis_rate', 'daily_mean_photosynthesis_fraction', 'foliar_respiration_fraction', 'min_photosynthesis_temperature', 'optimum_photosynthesis_temperature',
#  'vapor_pressure_deficit_slope', 'vapor_pressure_deficit_exponent', 'half_saturation_light', 'light_extinction_coefficient']

# All groups
list(SIPNET_PARAMS_BY_GROUP.keys())
# ['initial_conditions', 'photosynthesis', 'phenology', 'respiration',
#  'allocation', 'water', 'leaf']

# Total parameter count
sum(len(ps) for ps in SIPNET_PARAMS_BY_GROUP.values())  # 58
```

### get_parameter_specs

For calibration and DA workflows, `get_parameter_specs` returns the full
`ParameterSpec` for each parameter — including units, mathematical domain, and
whether the value is a per-year rate. The same dict is precomputed as
`pysipnet.PARAMETER_SPECS`, and the [Parameters](../reference/parameters.md) page is
generated from it:

```python
from pysipnet.parameters.base import get_parameter_specs, ParameterDomain

specs = get_parameter_specs(SIPNETParameters)
# {"photosynthesis.max_photosynthesis_rate": ParameterSpec(sipnet_name="aMax", units="nmol g-1 s-1", constituent="CO2", domain=POSITIVE, ...), ...}

# Parameters requiring a log bijector for unconstrained optimization
log_params = {k for k, s in specs.items() if s.domain == ParameterDomain.POSITIVE}

# Per-year rate parameters
annual_rates = {k for k, s in specs.items() if s.per_year}
```
