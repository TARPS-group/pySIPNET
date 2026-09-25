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

climate = ClimateDrivers.from_file("docs/examples/data/niwot_1999_daily.clim")
print(climate)
# ClimateDrivers(n_columns=14, timesteps=365, range=1999-001 to 1999-365)
```

That file — one year of daily Niwot Ridge forcing — ships with the repository,
so every example on this page runs as written from the repository root.

`ClimateDrivers.from_file` loads the data into memory.  For ensemble
workflows with pre-existing files, `ClimateDrivers.from_path` creates a
lightweight file reference without loading the data, which also defers
validating it until the first read — see [File I/O](file-io.md).

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
model authors ship (bundled as `pysipnet/data/niwot/sipnet.param`, see
[Bundled reference data](file-io.md#bundled-reference-data)),
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
    rows.append({"max_photosynthesis_rate": max_photosynthesis_rate, "annual_gpp": float(r.outputs["gpp"].sum())})

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

### Converting units

Three functions in `pysipnet.units` translate a value from one
`(units, constituent)` pair to another, for comparing model output with data
reported in different units. Pint does the
prefixes and dimensions; pySIPNET adds the chemistry, since a gram of carbon
becomes a mole only through carbon's molar mass.

- `convert_dataarray_units(array, to_units=..., to_constituent=...)` converts
  an xarray `DataArray`, reading the units it is in from `array.attrs["units"]`
  and `array.attrs["constituent"]`. Every output and climate `DataArray`
  carries both, so there is nothing to misstate.
- `convert_units(values, units=..., constituent=..., to_units=..., to_constituent=...)`
  converts unlabeled values (a number, a NumPy array, a pandas object), with
  the caller stating the units they are in.
- `conversion_factor(...)` takes the same four strings and returns the number.

In all three, `to_constituent` defaults to the source constituent, so a change
of units alone names the substance once; pass `""` for none.

```python
from pysipnet.units import conversion_factor, convert_dataarray_units, convert_units

convert_dataarray_units(result.outputs["wood_carbon"], to_units="Mg ha-1")   # × 0.01

convert_units(obs_values, units="g m-2", constituent="C", to_units="Mg ha-1")

# NEE is a total per step; divide by the step length in days first, then:
conversion_factor(units="g m-2 d-1", constituent="C",
                  to_units="umol m-2 s-1", to_constituent="CO2")   # 0.9636228519
conversion_factor(units="cm", constituent="H2O", to_units="kg m-2") # 10.0
```

The rules, in order:

1. Same dimension and same constituent (or none on either side), with the
   first unit measuring the same kind of thing on both sides: the Pint factor.
2. Mass to amount or back for one constituent, through `MOLAR_MASS`
   (g mol⁻¹: C 12.011, N 14.007, H2O 18.015, CO2 44.009, CH4 16.043,
   N2O 44.013). A depth or volume of water to a mass or back, through
   `DENSITY` (H2O 1000 kg m⁻³).
3. A change of constituent only for a pair in `ATOMS_PER_MOLECULE`: one C per
   CO2, one C per CH4, two N per N2O. The ratio applies on an amount basis, so
   `g m-2` of C to `g m-2` of CO2 is 44.009 / 12.011, not 1.
4. Everything else raises a `ValueError` naming both unit strings and both
   constituents: mismatched dimensions, a constituent on one side only, an
   unknown constituent, a substance inside a unit string (`"g C m-2"`), a
   constituent on a temperature, or an offset temperature conversion such as
   `degC` to `K`, which is not a multiplication.

For rules 2 and 3 the constituent qualifies the **first** unit in the string,
where `formatted_units()` prints it: `"nmol g-1 s-1"` of CO2 is nanomoles of
CO2 per gram of leaf, so it converts to `"ug g-1 s-1"` of C. That first unit
must be an amount, a mass, or for water a depth or volume, so `"Pa"` or
`"W m-2"` with a constituent is refused rather than converted through a molar
mass.

The same reading is why rule 1 asks about the first unit. `"umol mol-1"` and
`"ug g-1"` are both dimensionless, so Pint alone would convert a mole fraction
of CO2 to a mass fraction by a factor of 1. The first unit changes from an
amount to a mass, so that conversion needs a molar mass for the numerator and
another for the denominator (air), and it is refused. Water content by mass
(`"kg kg-1"`) and by volume (`"m3 m-3"`) are refused for the same reason: the
denominator would need the soil's bulk density. A volume per area and a depth
(`"m3 m-2"` and `"mm"`) are geometry and convert freely, and a ratio to a pure
number (`"ug g-1"` to `"1"`) keeps Pint's factor.

`photons` (the constituent of PAR) is counted in moles and has no molar mass,
so it converts between amounts only.

`convert_dataarray_units` relabels its result: `units` and `constituent`
become the target's, and `output_decimals` and SIPNET's internal-conversion
attributes, which describe the original units, are dropped. Coordinates and
the name are unchanged, and the input is not modified. It refuses an array with
no `units` attribute, and a `Dataset`, whose variables do not share a unit;
convert `ds[name]`. `convert_units` refuses any xarray object and any pandas
object whose `attrs` has a `units` entry, because both keep their `attrs`
through the multiplication and the result would still claim the old units.
Factors are cached, so converting inside a calibration loop costs one
multiplication.

### Combining variables

xarray drops attributes in arithmetic, so `nee / days` comes back with no
`units`, and `convert_dataarray_units` has nothing to read. The functions in
`pysipnet.arithmetic` do the arithmetic and write the `units`, `constituent`
and `kind` that are true of the result. An NEE rate, from a total per step to
µmol of CO2 per square meter per second:

```python
from pysipnet.arithmetic import divide_with_units, multiply_with_units, step_length
from pysipnet.units import convert_dataarray_units

nee = result.outputs["nee"]            # 'g m-2' of C, kind 'timestep_total'
days = step_length(nee)                # the time_step_length coordinate, in 'd'
rate = divide_with_units(nee, days)    # 'g m-2 d-1' of C, kind 'daily_rate'
flux = convert_dataarray_units(rate, to_units="umol m-2 s-1", to_constituent="CO2")
# flux == rate × 0.9636228519, labeled 'umol m-2 s-1' of CO2
```

And leaf area index, from the leaf carbon pool and the parameter that relates
the two:

```python
import xarray as xr
from pysipnet.parameters.model import PARAMETER_SPECS

spec = PARAMETER_SPECS["leaf.leaf_carbon_per_area"]     # 'g m-2' of C
per_area = xr.DataArray(params.leaf.leaf_carbon_per_area, name="leaf_carbon_per_area",
                        attrs={"units": spec.units, "constituent": spec.constituent})
lai = divide_with_units(result.outputs["leaf_carbon"], per_area)
# '1', no constituent (C over C cancels), kind 'timestep_end_state'
```

An operand is a `DataArray` with a `units` attribute, or a plain number
(dimensionless). The unit and constituent rules are `pysipnet.units`' own, and
`product_units`, `quotient_units`, `sum_units` and `difference_units` apply
them to bare `(units, constituent)` pairs:

```python
from pysipnet.units import product_units
product_units(units="d-1", other_units="g m-2", other_constituent="C")
# ('g m-2 d-1', 'C')
```

The rules:

- **Values** are plain xarray arithmetic, broadcasting as usual (a `(site,)`
  parameter against a `(site, time)` stack gives per-site results). Index
  coordinates must match exactly: two different time axes are refused rather
  than cut to the labels they share, and so is a coordinate such as
  `time_step_start` that both operands carry with different values, which
  xarray would otherwise drop.
- **Units** combine symbol by symbol, and a symbol whose exponent reaches zero
  drops out; nothing is rescaled, so `cm` over `m` is `"cm m-1"`. The operand
  that carries the constituent comes first, since the constituent qualifies
  the first unit: `multiply_with_units(per_day, nee)` is `"g m-2 d-1"`, not
  `"d-1 g m-2"`. A combination that would cancel or change that first unit
  (`g m-2` of C divided by a mass in `g`) is refused, since the result would
  no longer say what quantity of carbon it measures.
- **Constituent**: at most one per product; in a quotient the numerator's is
  kept, and the same one on both sides cancels. A constituent only in the
  denominator, or two different ones, is refused.
- **Kind**: at most one operand of a product or quotient may have a kind, and
  not the denominator. A `timestep_total` divided by a time is a
  `daily_rate`, and a `daily_rate` times a time is a `timestep_total`
  (whatever the time unit; the units say which). Every other change of time
  dimension is refused: a pool times a turnover rate is a flux, which SIPNET
  reports itself.
- `add_with_units` and `subtract_with_units` require the same units,
  constituent and kind. A bare `degC` temperature does not multiply or add,
  and the difference of two is in `K`; degree-days in `degC d` combine like
  any other unit, since Pint already reads `degC` there as a difference.

The result carries `units`, `constituent`, `kind` with its `time_reference`
and `cell_methods`, the `sign_convention` if it is still true (not after
multiplying by a negative number, and never for a difference), and a
`derivation` naming the operands. It keeps the model variable's time
coordinates, so `resample` still works on it. Attributes describing the
source rather than the result (`description`, `sipnet_name`,
`output_decimals`) are dropped, the name is `None`, and the operands are not
modified.

Every column is always present. A process that is switched off writes zeros
rather than omitting its column, so the nitrogen and methane columns are there
but zero unless those processes are on (`requires_flag` on the spec says which).
Because a column of zeros is indistinguishable from a real result once it
reaches a likelihood, **selecting such a variable by name raises** — `result.outputs["litter_carbon"]`
on a run without `litter_pool` tells you which flag to turn on rather than
handing back the zeros. `.pandas` and `.xarray` still contain the column, being
a faithful view of the file.
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
    attributes. The xarray `time` coordinate is therefore the **end** of the
    step, the one instant at which a pool is the value "at `time`" and a flux
    is the total "over the bounds", as the Climate and Forecast (CF)
    `cell_methods` attributes say; `time_step_start` is the row's label.

!!! note "SIPNET has no time zone; the drivers declare the clock"
    SIPNET computes no solar geometry and never interprets its labels: it
    echoes each climate row's `year`, `day` and `time` into the matching
    output row and integrates on `time_step_length`. So the times are on
    whatever clock your climate drivers use. Say which when you build them —
    `ClimateDrivers.from_file(path, time_zone="UTC")`, or a fixed offset such
    as `"UTC-07:00"` for local standard time — and the declaration is recorded
    on the `time` coordinate of both the climate and the output Datasets. It
    is metadata only: nothing is converted, and undeclared is the default.

    A run's output takes its time axis from those drivers, row for row, so
    `result.outputs.xarray` and `climate.xarray` share one axis exactly.
    (SIPNET prints `hour_of_day` rounded to 0.01 h; the `.pandas` view keeps
    those printed values.)

### Five views of the same output

```python
df  = result.outputs.pandas          # pandas DataFrame, one row per timestep
ds  = result.outputs.xarray          # xarray Dataset, one `time` dimension
nee = result.outputs["nee"]          # one variable as a DataArray, by name or alias
both = result.outputs[["nee", "gpp"]]  # several variables as a Dataset

result.outputs.select(["nee", "gpp"], format="pandas")  # the same selection in pandas
```

A DataArray still converts to whatever you need: `nee.to_series()` for pandas,
`nee.to_numpy()` for a bare array.

The Dataset is the representation to use when metadata matters or when
results will be combined across runs:

```python
ds["net_ecosystem_exchange"].attrs
# {'units': 'g m-2', 'long_name': 'Net ecosystem exchange',
#  'kind': 'timestep_total', 'time_reference': 'total over the timestep',
#  'cell_methods': 'time: sum', 'constituent': 'C',
#  'sign_convention': 'positive is a flux from the ecosystem to the atmosphere', ...}

ds["time"]              # datetime64, END of each timestep (CF standard_name "time")
ds["time_step_start"]   # datetime64, start of each timestep, as the drivers label the row
ds["time_step_length"]  # timedelta64, the length declared to SIPNET
ds["time_bounds"]       # (time, bounds) — the interval [time_step_start, time]

ds.attrs["run_id"]                   # which run produced this
ds.attrs["time_step_length_source"]  # measured from the drivers, or inferred
ds.attrs["time_axis_source"]         # the drivers, or SIPNET's printed labels
ds.attrs["time_zone"]                # the drivers' declared clock, or "undeclared"

ds.to_netcdf("run.nc")  # self-describing on disk; needs a netCDF backend
                        # (`pip install h5netcdf`), which pySIPNET does not require
```

`time_bounds` is the Climate and Forecast conventions' way of saying which
interval each value covers, which is what you need in order to decide how
measurements over some other interval line up with model steps. It adds a second dimension,
`bounds`, so `ds.sizes` reads `{'time': 365, 'bounds': 2}`; use `result.outputs.pandas`
when you want a flat table. The Dataset declares `Conventions = "CF-1.11"`.
Writing it needs a netCDF backend that stores 64-bit integers (`h5netcdf` or
`netCDF4`); the netCDF3 backend built into scipy cannot hold nanosecond times.

### Daily, monthly and annual values

Use [`resample`][pysipnet.resample.resample] to combine steps into coarser
ones. It takes no default method, because the right one is not a property of
the variable: daily soil water can be the value at the end of the day or the
mean over it, and those are different quantities. What *is* a property of the
variable is which methods make sense at all, and `resample` refuses one that
does not, saying why:

```python
from pysipnet import resample

daily = resample(result.outputs[["nee", "gpp"]], "1D", how="sum")

annual = resample(
    result.outputs[["wood_carbon", "nee", "soil_wetness_fraction"]],
    "YS",
    how={
        "wood_carbon": "last",             # the pool at the end of the year
        "net_ecosystem_exchange": "sum",   # the annual total
        "soil_wetness_fraction": "mean",   # weighted by step length
    },
)

resample(result.outputs[["wood_carbon"]], "1D", how="sum")
# ValueError: Cannot resample 'wood_carbon' with 'sum': it is a pool reported at
# the end of the timestep (kind 'timestep_end_state'), and a pool is not additive
# across steps ... Valid for this kind: 'last' gives the pool at the end of the
# coarser step; 'mean' gives the time-weighted mean of the pool over the coarser
# step, which makes it a timestep_mean rather than a state.
```

| Kind | Valid methods | What you get |
|:-----|:--------------|:-------------|
| `timestep_total` (fluxes) | `sum` | the total over the coarser step |
| `timestep_end_state` (pools) | `last`, `mean` | the pool at the end, or its time-weighted mean (now a `timestep_mean`) |
| `daily_rate`, `timestep_mean` | `mean` | the mean weighted by `time_step_length` |
| `cumulative` | `last` | the running total so far |

Means are weighted by `time_step_length` because SIPNET steps need not be
equal: the Niwot record alternates day and night steps of 0.29 and 0.63 days,
and a plain mean would be biased toward the short ones. The result keeps the
same layout, with `time` at the end of each coarser step and `kind`,
`time_reference` and `cell_methods` rewritten to describe what each variable
now is. A step ending exactly at midnight belongs to the day that ended, so a
daily record resamples to itself.

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
