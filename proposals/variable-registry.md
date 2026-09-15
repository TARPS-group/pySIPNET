# Variable registry for pySIPNET: analysis and proposal

*Status: proposal, 2026-09-15. Everything marked "source" below was read from the
SIPNET submodule at the pinned commit (`v2.2.0-alpha.1`), not from SIPNET's docs.*

## 1. Problem statement

A pySIPNET user currently gets a DataFrame with 35 columns named things like
`r_soil`, `npp_storage`, `soil_wetness_frac` and `transpiration`, and nothing in
the package says what any of them is, what unit it is in, or whether the number
is a pool at an instant, a total over the timestep, a mean over the timestep, or
a running total since the start of the run. The only place that information
exists is SIPNET's C source. SIPNET's own output documentation is incomplete and
in two places wrong (§3.4), so sending users there would not fix it either.

The parameter side is better: every field carries a `ParameterSpec` with a Pint
unit string, a description and a domain. But the parameter names themselves
(`a_max`, `psn_t_min`, `d_vpd_slope`, `soil_whc`) are abbreviations, the unit
strings cannot express "grams of carbon" (§5.2), and there is no link between an
initial condition and the state variable it initialises.

This document proposes one registry that describes every variable pySIPNET
exposes (outputs first, then parameters and climate drivers), a naming
convention, a units convention, and a corrected xarray representation.

## 2. What SIPNET actually writes (source-verified)

### 2.1 Where the numbers come from

`runModelOutput()` in `src/sipnet/sipnet.c` loops over climate records and, for
each one, calls `updateState()` and then `outputState()`. The order matters:

- **`year`, `day`, `time` are the start of the timestep.** They are copied
  straight from the climate record (`climate->year`, `->day`, `->time`), and
  SIPNET's input docs define them as the start of the interval. `time` is
  fractional hours after midnight.
- **Pools are end-of-step values.** `outputState()` runs after
  `updatePoolsAndBalance()`, so every `envi.*` field printed is the state at
  start + length.
- **Fluxes are integrals over the step.** `updateTrackers()` multiplies each
  per-day flux by `climate->length` (days), so a tracker is in g C m⁻² (or cm)
  accumulated over the interval, not a rate.
- **One column is a rate, not an integral.** `fluxestranspiration` prints
  `fluxes.transpiration` directly, and the `Fluxes` struct is documented as
  "fluxes as per-day rates" with transpiration in cm water day⁻¹. It is the only
  flux column that is not multiplied by `length`. pySIPNET currently renames it
  to `transpiration`, sitting next to `evapotranspiration` (cm per step), which
  invites the user to add or compare them. This is the most misleading name we
  have today.
- **One column is a mean.** `soilWetnessFrac` is
  `(oldSoilWater + envi.soilWater) / (2 × soilWHC)`, the two-point linear mean
  of start and end wetness.
- **One column is cumulative.** `cumNEE` prints `trackers.totNee`, which is
  never reset. It is serialised in restart checkpoints
  (`restart.c`, `"trackers.totNee"`), so a restarted run continues the total.
- **Two columns are not what their headers say.**
  - `plantWoodC` prints `getTotalWoodC()` = `envi.plantWoodC +
    envi.plantCAccountingDelta`, i.e. structural wood plus the NPP storage-lag
    term that SIPNET's model-structure docs call C_wood,storage.
  - `nppStorage` prints `envi.plantCAccountingDelta`, a *state* variable (the
    storage term itself, which can be negative), not a flux.
- **`n2o` is total volatilisation.** `trackers.n2o = fluxes.nVolatilization ×
  length`, and `nVolatilization = nVolatilizationFrac × minN × f(T) × f(W)`.
  Nothing partitions it into N₂O versus other gases. It is in g N.
- **`ch4` is carbon mass.** `(soilMethane + litterMethane) × length`, g C m⁻².
- **Timestep length is not in the output.** Converting any per-step integral to
  a rate needs `length` from the climate file. That is available in
  `SIPNETResult.climate` but not from a bare `.out` file.
- **Fixed print precision.** Each column has a hard-coded `%w.pf` format
  (§2.2). NEE, GPP and all respiration terms are printed to 3 decimals, so a
  half-hourly NEE of 0.05 g C m⁻² carries a 1% quantisation floor. ET gets 8
  decimals; N pools 4; `n2o` 6. This is a real noise floor for data
  assimilation and belongs in the registry.

### 2.2 Column-by-column

Kinds: **coord** (timestamp), **state** (pool at end of step), **sum** (integral
over step), **mean** (two-point mean over step), **rate** (per-day rate),
**cumulative** (running total since run or restart start). "Flag" is the
`ModelFlags` field without which the column is written but is a constant zero.

| # | SIPNET header | Source expression | Kind | Units | Decimals | Flag | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `year` | `climate->year` | coord | year | int | | start of step |
| 2 | `day` | `climate->day` | coord | day of year | int | | 1 = 1 Jan; start of step |
| 3 | `time` | `climate->time` | coord | h | 2 | | hours after midnight; start of step |
| 4 | `plantWoodC` | `envi.plantWoodC + envi.plantCAccountingDelta` | state | g C m⁻² | 2 | | total wood incl. storage term; roots are separate pools |
| 5 | `plantLeafC` | `envi.plantLeafC` | state | g C m⁻² | 2 | | LAI = this / `leafCSpWt` |
| 6 | `woodCreation` | `fluxes.woodCreation × length` | sum | g C m⁻² | 2 | | carbon allocated to wood this step |
| 7 | `soil` | `envi.soilC` | state | g C m⁻² | 2 | | single soil organic C pool |
| 8 | `coarseRootC` | `envi.coarseRootC` | state | g C m⁻² | 2 | | |
| 9 | `fineRootC` | `envi.fineRootC` | state | g C m⁻² | 2 | | |
| 10 | `litter` | `envi.litterC` | state | g C m⁻² | 2 | `litter_pool` | |
| 11 | `soilWater` | `envi.soilWater` | state | cm | 3 | | plant-available soil water |
| 12 | `soilWetnessFrac` | `(W_start + W_end) / (2 soilWHC)` | mean | 1 | 3 | | diagnostic only; internal functions use instantaneous ratio |
| 13 | `snow` | `envi.snow` | state | cm water equiv. | 2 | `snow` | |
| 14 | `npp` | `gpp − ra` | sum | g C m⁻² | 3 | | |
| 15 | `nee` | `−(npp − rh)` = `rtot − gpp` | sum | g C m⁻² | 3 | | **positive = to atmosphere** |
| 16 | `cumNEE` | `trackers.totNee` | cumulative | g C m⁻² | 3 | | since run start (continues across restart) |
| 17 | `gpp` | `fluxes.photosynthesis × length` | sum | g C m⁻² | 3 | | |
| 18 | `rAboveground` | `fluxes.rVeg × length` | sum | g C m⁻² | 3 | | foliar + wood maintenance, + growth resp. if `growth_resp` |
| 19 | `rSoil` | `rRoot + rh` | sum | g C m⁻² | 3 | | root + heterotrophic; **SIPNET's docs call this R_H, which is wrong** |
| 20 | `rRoot` | `(rCoarseRoot + rFineRoot) × length` | sum | g C m⁻² | 3 | | |
| 21 | `ra` | `rRoot + rAboveground` | sum | g C m⁻² | 3 | | autotrophic |
| 22 | `rh` | `(rLitter + rSoil) × length` | sum | g C m⁻² | 3 | | heterotrophic (litter + soil) |
| 23 | `rtot` | `ra + rh` | sum | g C m⁻² | 3 | | ecosystem respiration |
| 24 | `evapotranspiration` | `(transp + immedEvap + evap + sublim + eventEvap) × length` | sum | cm | 8 | | includes irrigation canopy evaporation |
| 25 | `fluxestranspiration` | `fluxes.transpiration` | **rate** | cm day⁻¹ | 4 | | **not × length**; the only rate column |
| 26 | `minN` | `envi.minN` | state | g N m⁻² | 4 | `nitrogen_cycle` | |
| 27 | `soilOrgN` | `envi.soilOrgN` | state | g N m⁻² | 4 | `nitrogen_cycle` | |
| 28 | `litterN` | `envi.litterN` | state | g N m⁻² | 4 | `nitrogen_cycle` | |
| 29 | `plantStorageN` | `envi.plantStorageN` | state | g N m⁻² | 4 | `nitrogen_cycle` | filled by leaf-off resorption |
| 30 | `n2o` | `fluxes.nVolatilization × length` | sum | g N m⁻² | 6 | `nitrogen_cycle` | total volatilised mineral N, not speciated |
| 31 | `nLeaching` | `fluxes.nLeaching × length` | sum | g N m⁻² | 4 | `nitrogen_cycle` | |
| 32 | `nFixation` | `fluxes.nFixation × length` | sum | g N m⁻² | 4 | `nitrogen_cycle` | |
| 33 | `nUptake` | `fluxes.nUptake × length` | sum | g N m⁻² | 4 | `nitrogen_cycle` | |
| 34 | `ch4` | `(soilMethane + litterMethane) × length` | sum | g C m⁻² | 4 | `anaerobic` | methane as carbon |
| 35 | `nppStorage` | `envi.plantCAccountingDelta` | state | g C m⁻² | 4 | | storage-lag term; can be negative; component of #4 |

Note the N trackers are only assigned inside `if (ctx.nitrogenCycle)`, so they
are exactly zero otherwise, not merely small.

### 2.3 Other outputs, for completeness

- `DO_SINGLE_OUTPUTS` writes four one-column files named `NEE`, `NEE_cum`,
  `GPP`, `GPP_cum` (`setupOutputItems()`). pySIPNET sets this to 0. The
  registry should record these as aliases of columns 15, 16, 17 and (for
  `GPP_cum`) a variable we do not otherwise receive, so they can be read if
  anyone turns it on.
- `events.out` records pool deltas per event with its own `eventLeafC=…`
  vocabulary. Out of scope here; SIPNET's docs say `sipnet.out` is
  authoritative.
- Restart checkpoints and the `--debug-log` files are not user-facing outputs.

## 3. What is wrong or missing today

### 3.1 In pySIPNET

1. **No metadata on outputs.** `SIPNET_TO_PYTHON_OUTPUT` in
   `pysipnet/io/output_reader.py` is a bare name map. Units, kind, sign
   convention and flag dependence live only in a module docstring and in
   `CLAUDE.md`.
2. **Misleading names.** `transpiration` (a rate next to a total),
   `npp_storage` (a state that sounds like a flux), `plant_wood_c` (includes
   the storage term), `r_soil` (not heterotrophic), `n2o` (not speciated).
3. **Abbreviations everywhere.** `nee`, `gpp`, `ra`, `rh`, `rtot`, `r_root`,
   `soil_wetness_frac`, `cum_nee`, `mineral_n`.
4. **`SIPNETResult.to_xarray()` builds the wrong shape.** It sets a
   `(year, day, time)` MultiIndex and calls `Dataset.from_dataframe`, which
   expands to a dense 3-D grid. On the golden fixture (60 rows) that is
   `{year: 1, day: 30, time: 4}` = 120 cells with 50% NaN. On a real run the
   `day` axis is not a time axis (day 100 of 2019 and of 2020 share an index),
   leap years and partial first/last years produce NaN holes, and irregular
   steps break it entirely.
5. **Plot labels are hand-typed** in `pysipnet/viz.py` (`_FLUX_COLS`,
   `_POOL_COLS`) and duplicate what a registry would hold.
6. **Parameter unit strings cannot say "carbon".** `"g / m**2"` plus a
   separate `constituent="C"` is a workaround for the fact that Pint parses
   `"g C / m**2"` as gram·coulomb per square metre without error (verified:
   `[mass]·[current]·[time]/[length]²`). `validate_unit_string` would accept
   that silently. Only 12 of ~60 parameter fields set `constituent`.
7. **Parameter names follow SIPNET's abbreviations**, not the convention the
   project wants: `a_max`, `psn_t_min`, `psn_t_opt`, `d_vpd_slope`,
   `half_sat_par`, `soil_whc`, `wue_const`, `rd_const`, `leaf_c_sp_wt`,
   `c_frac_leaf`, `frozen_soil_fol_r_eff`, `lai`.
8. **No initial-condition ↔ state link.** `initial_conditions.plant_wood`
   initialises the `plantWoodC` pool; `lai` initialises `plantLeafC` *via*
   `leafCSpWt`; `soil_water_frac` initialises `soilWater` *via* `soilWHC`. A DA
   workflow that restarts from a posterior state needs exactly this mapping, and
   the two indirect ones need the conversion spelled out.

### 3.2 Consumers that a rename touches

Grep for the current snake_case names finds ~80 uses in: `pysipnet/{runner,
result, output, model, viz}.py`, `pysipnet/io/output_reader.py`,
`tests/{test_io, test_viz, test_integration}.py`, the golden fixture header
(`tests/fixtures/golden/niwot_standard.out.csv`), `README.md`, and six pages
under `docs/`. Nothing outside the package depends on them yet (0.1.0.dev0).

### 3.3 Nothing to fix on the parameter I/O side

`PYTHON_TO_SIPNET` and `tests/test_param_name_mapping.py` already implement the
"independent statement of the mapping" pattern this proposal wants for outputs.
Reuse it.

### 3.4 Errors in SIPNET's own documentation (worth an upstream issue)

Found while cross-checking `sipnet/docs/user-guide/model-outputs.md` against the
source:

- `rSoil` is labelled "Heterotrophic respiration" with symbol R_H. Source:
  `rSoil = rRoot + rh`.
- The table omits `plantStorageN` and `nppStorage`, both of which
  `outputHeader()` writes; the example header is also missing them.
- `n2o` is described as "Nitrous oxide production"; source is total mineral-N
  volatilisation.
- `evapotranspiration` definition omits `eventEvap`.
- `plantWoodC` is described as "Woody plant carbon" without mentioning that the
  printed value includes the storage term (the model-structure page does say
  C_wood,total = C_wood + C_wood,storage).
- `state.h` still comments `plantWoodC` as "above-ground + roots", which
  predates the coarse-root split.

## 4. Proposal

### 4.1 Naming convention

- Lower-case words separated by underscores, no acronyms, no truncations.
  Regex: `^[a-z][a-z0-9]*(_[a-z0-9]+)*$`, enforced by a test together with a
  deny-list of tokens (`frac`, `resp`, `temp`, `psn`, `whc`, `vpd`, `par`,
  `lai`, `nee`, `gpp`, `npp`, `et`, `wue`, `q10`? see below).
- Substance is part of the name when the unit alone is ambiguous:
  `leaf_carbon`, `litter_nitrogen`, `snow_water_equivalent`.
- Chemical formulas and Q₁₀ are the deliberate exceptions to "no acronyms":
  `methane_production` rather than `ch4`, but `..._q10` stays because "Q10" is
  the name of the quantity, not an abbreviation of one. Recommend allowing
  `q10` and nothing else.
- Short forms (`nee`, `gpp`, `lai`) survive as **aliases**, resolvable in
  lookups and accepted by `load(variables=[...])`, never as column names.

Proposed output names (mapping is the registry's job; this is the content):

| SIPNET | Proposed name | Short label | Long label |
|---|---|---|---|
| `year` | `year` | | Year (start of step) |
| `day` | `day_of_year` | DOY | Day of year (start of step) |
| `time` | `hour_of_day` | | Hour of day (start of step) |
| `plantWoodC` | `wood_carbon` | Wood C | Wood carbon (incl. storage) |
| `plantLeafC` | `leaf_carbon` | Leaf C | Leaf carbon |
| `woodCreation` | `wood_growth` | | Carbon allocated to wood |
| `soil` | `soil_carbon` | Soil C | Soil organic carbon |
| `coarseRootC` | `coarse_root_carbon` | Coarse root C | Coarse root carbon |
| `fineRootC` | `fine_root_carbon` | Fine root C | Fine root carbon |
| `litter` | `litter_carbon` | Litter C | Litter carbon |
| `soilWater` | `soil_water` | | Plant-available soil water |
| `soilWetnessFrac` | `soil_wetness_fraction` | | Soil water / water holding capacity |
| `snow` | `snow_water_equivalent` | SWE | Snow water equivalent |
| `npp` | `net_primary_production` | NPP | Net primary production |
| `nee` | `net_ecosystem_exchange` | NEE | Net ecosystem exchange (+ to atmosphere) |
| `cumNEE` | `cumulative_net_ecosystem_exchange` | Σ NEE | Cumulative NEE since run start |
| `gpp` | `gross_primary_production` | GPP | Gross primary production |
| `rAboveground` | `above_ground_respiration` | R_above | Above-ground autotrophic respiration |
| `rSoil` | `soil_respiration` | R_soil | Soil respiration (root + heterotrophic) |
| `rRoot` | `root_respiration` | R_root | Root respiration |
| `ra` | `autotrophic_respiration` | R_a | Autotrophic respiration |
| `rh` | `heterotrophic_respiration` | R_h | Heterotrophic respiration |
| `rtot` | `ecosystem_respiration` | R_eco | Total ecosystem respiration |
| `evapotranspiration` | `evapotranspiration` | ET | Evapotranspiration |
| `fluxestranspiration` | `transpiration_rate` | | Transpiration rate |
| `minN` | `mineral_nitrogen` | | Soil mineral nitrogen |
| `soilOrgN` | `soil_organic_nitrogen` | | Soil organic nitrogen |
| `litterN` | `litter_nitrogen` | | Litter nitrogen |
| `plantStorageN` | `plant_nitrogen_storage` | | Plant stored nitrogen |
| `n2o` | `nitrogen_volatilization` | | Mineral N lost to volatilisation |
| `nLeaching` | `nitrogen_leaching` | | Mineral N lost to leaching |
| `nFixation` | `nitrogen_fixation` | | Nitrogen fixation |
| `nUptake` | `nitrogen_uptake` | | Plant nitrogen uptake |
| `ch4` | `methane_production` | CH₄ | Methane production (as C) |
| `nppStorage` | `wood_storage_carbon` | | Wood storage-lag carbon |

Two names need a decision from you:

- **`wood_carbon` vs `above_ground_wood_carbon`.** SIPNET never calls this pool
  above-ground; its docs say "total wood carbon". Roots are separate pools, so
  functionally it is above-ground woody biomass, which is also how PEcAn maps it
  (`AbvGrndWood`). Recommendation: `wood_carbon`, with "above-ground" in the
  description, because the column also contains the storage-lag term and
  "above_ground_wood_carbon" would over-promise a physical meaning.
- **`nitrogen_volatilization` vs `n2o_emission`.** Recommendation: the former.
  SIPNET's header says n2o but computes total volatilisation; naming it N₂O
  would repeat SIPNET's error under our name.

### 4.2 The registry

A pure data module, `pysipnet/variables.py` (data layer, no I/O), holding one
frozen dataclass per variable and a few lookup helpers.

```python
class VariableKind(StrEnum):
    COORDINATE = "coordinate"
    STATE = "state"            # pool at end of step
    FLUX = "flux"              # integral over step
    RATE = "rate"              # per-day rate valid for the step
    MEAN = "mean"              # mean over step
    CUMULATIVE = "cumulative"  # running total since run start

class Aggregation(StrEnum):    # how to aggregate over time, and CF cell_methods
    SUM = "sum"                # flux totals -> "time: sum"
    MEAN = "mean"              # states, means, rates -> "time: mean"
    LAST = "last"              # cumulative -> take last value in window
    POINT = "point"            # end-of-step state -> "time: point"

@dataclass(frozen=True)
class VariableSpec:
    name: str                  # pysipnet name, convention-checked
    sipnet_name: str           # exact header token SIPNET writes
    kind: VariableKind
    aggregation: Aggregation   # default rule for resampling/annual totals
    units: str                 # UDUNITS/CF string, e.g. "g m-2", "cm d-1", "1"
    constituent: str           # "C", "N", "H2O", "" (never inside `units`)
    description: str           # 1-3 sentences, includes the source formula
    long_label: str            # "Net ecosystem exchange"
    short_label: str           # "NEE"; falls back to long_label
    aliases: tuple[str, ...]   # ("nee",) etc.; also old pySIPNET names
    requires_flag: str | None  # ModelFlags field, e.g. "nitrogen_cycle"
    output_decimals: int | None  # printf precision -> quantisation floor
    sign_convention: str = ""  # "positive = flux to atmosphere"
    cf_standard_name: str | None = None
    group: str = ""            # "carbon_pools", "carbon_fluxes", "water", ...
```

Helpers: `OUTPUT_VARIABLES: tuple[VariableSpec, ...]` in SIPNET column order;
`by_name`, `by_sipnet_name`, `resolve(name_or_alias) -> VariableSpec`;
`axis_label(name, style="unicode"|"latex"|"html")` producing e.g.
`Net ecosystem exchange (g C m⁻²)`; `cf_attrs(name) -> dict` producing
`{"units", "long_name", "cell_methods", "comment", "standard_name"?}`;
`to_records() -> list[dict]` for JSON/YAML/markdown export.

Why Python declarations rather than a YAML file: they are type-checked by mypy,
importable without file I/O at import time (a stated convention), and
`to_records()` gives you the serialisable form on demand. The docs table is
generated from the registry by a `mkdocs-gen-files` script so it cannot drift.

Why `constituent` stays separate from `units`: see §5.

`cf_standard_name` is optional and should only be filled after checking the
current CF table; candidates exist for GPP, NPP, LAI, soil/litter/wood carbon
content, evapotranspiration and transpiration, but several of our columns
(storage term, soil respiration as root+heterotrophic, volatilisation) have no
exact CF name and must stay `None` rather than be approximated.

### 4.3 How the registry is used

- **`output_reader.read_output_file`** renames via `by_sipnet_name`; an
  unmapped header token keeps its SIPNET name *and emits a warning*, mirroring
  the "Unknown param" contract on the input side. `columns=` accepts names or
  aliases.
- **`SIPNETOutput`** gains `variables` (the specs for the columns present),
  `__getitem__(name_or_alias)`, and the xarray path in §6.
- **`viz.py`** drops `_FLUX_COLS`/`_POOL_COLS` and reads labels from the
  registry.
- **`SIPNETResult.nee()/gpp()/et()`** become thin wrappers over aliases or are
  removed in favour of `result.outputs["nee"]`.
- **Serialisation:** `RunConfig` and `RunProvenance` are unaffected; the
  registry version travels implicitly with the package version.

### 4.4 Tests (each one guards a silent failure)

1. **Header contract.** Run the pinned binary on the Niwot fixture and assert
   the header tokens equal `[v.sipnet_name for v in OUTPUT_VARIABLES]` exactly,
   in order. Catches a column added, dropped or renamed upstream. Same spirit as
   `test_param_file_contract.py`.
2. **Independent mapping statement.** A hand-written `EXPECTED_OUTPUT_NAMES`
   table in a test file, as `test_param_name_mapping.py` does for parameters.
3. **Convention.** Every `name` matches the regex and contains no deny-listed
   token; every alias is unique across names and aliases; every spec has a
   non-empty description, long label and units.
4. **Units parse.** Every `units` string parses in the pySIPNET unit registry
   (§5) and never contains a constituent token (`C`, `N`, `H2O`).
5. **Precision.** For each column, assert the fixture values have no more
   decimals than `output_decimals`. Cheap, and it pins the printf contract.
6. **Golden.** Regenerate `niwot_standard.out.csv` with the new header via a
   one-off script that renames columns only; assert values are bit-identical to
   the current file before committing.
7. **Flag zeros.** For each `requires_flag`, a run without the flag yields an
   all-zero column (already partly covered in `test_integration.py` for
   `litter_c`).

### 4.5 Extending the same registry to parameters and climate

Parameters already have `ParameterSpec`. Proposed changes, in a later phase:

- Add `long_label`, `short_label`, `aliases`, `initializes: str | None`
  (output variable name), `initializes_via: str` (formula text, e.g.
  `"leaf_carbon = lai × leaf_carbon_per_area"`), `sipnet_name` (currently only
  in `PYTHON_TO_SIPNET`), and switch `unit` to the same UDUNITS syntax.
- Rename fields to the convention. Illustrative subset; the full table is the
  work item, and each rename must be mirrored in `PYTHON_TO_SIPNET` and
  `test_param_name_mapping.py`:

  | Current | SIPNET | Proposed |
  |---|---|---|
  | `initial_conditions.plant_wood` | `plantWoodInit` | `initial_conditions.wood_carbon` |
  | `initial_conditions.lai` | `laiInit` | `initial_conditions.leaf_area_index` |
  | `initial_conditions.soil_water_frac` | `soilWFracInit` | `initial_conditions.soil_wetness_fraction` |
  | `photosynthesis.a_max` | `aMax` | `photosynthesis.max_photosynthesis_rate` |
  | `photosynthesis.psn_t_min` | `psnTMin` | `photosynthesis.min_photosynthesis_temperature` |
  | `photosynthesis.psn_t_opt` | `psnTOpt` | `photosynthesis.optimum_photosynthesis_temperature` |
  | `photosynthesis.d_vpd_slope` | `dVpdSlope` | `photosynthesis.vapour_pressure_deficit_slope` |
  | `photosynthesis.half_sat_par` | `halfSatPar` | `photosynthesis.half_saturation_light` |
  | `water.soil_whc` | `soilWHC` | `water.soil_water_holding_capacity` |
  | `water.wue_const` | `wueConst` | `water.water_use_efficiency_constant` |
  | `leaf.leaf_c_sp_wt` | `leafCSpWt` | `leaf.leaf_carbon_per_area` |
  | `leaf.c_frac_leaf` | `cFracLeaf` | `leaf.leaf_carbon_fraction` |
  | `respiration.veg_resp_q10` | `vegRespQ10` | `respiration.vegetation_respiration_q10` |

- Climate drivers: `tair → air_temperature`, `tsoil → soil_temperature`,
  `par → photosynthetically_active_radiation`, `precip → precipitation`,
  `vpd → vapour_pressure_deficit`, `vpd_soil → soil_vapour_pressure_deficit`,
  `vpress → vapour_pressure`, `wspd → wind_speed`, `length →
  timestep_length`, with `time → hour_of_day`, `day → day_of_year` to match
  outputs. Same `VariableSpec` type, `kind=DRIVER`, plus a `file_units` field
  where SIPNET converts on read (precip mm→cm, VPD Pa→kPa, PAR total→per day).

Doing outputs first, alone, is deliberate: it is self-contained, it has the
worst current state, and it lets the naming and units conventions be exercised
before touching the parameter model that `RunConfig` serialises.

## 5. Units (your question 2)

### 5.1 Recommendation

Store units as **strings in UDUNITS-2 / CF syntax**, physical units only, with
the substance in a separate `constituent` field and the temporal meaning in
`aggregation`/`cell_methods`. Validate every string at import by parsing it
with a pySIPNET-owned Pint `UnitRegistry` configured with a UDUNITS
preprocessor. Generate display labels from the parsed unit, never by hand.

Examples: `"g m-2"`, `"g m-2 d-1"`, `"cm"`, `"cm d-1"`, `"1"`, `"degC"`,
`"K d"`, `"m2 m-2"`, `"nmol g-1 s-1"`, `"mol m-2 d-1"`.

### 5.2 Why not the current Pint-expression syntax

- Pint parses `"g C / m**2"` as gram·coulomb per square metre and
  `"g N m-2"` as gram·newton·metre, with no error. The constituent can never go
  in the string, so the string is not self-describing in the way a reader
  expects, and the validator gives false confidence.
- `"g / m**2"` is not what netCDF, xarray, cf_xarray, MetPy, PEcAn or any CF
  tool writes or reads. `"g m-2"` is.
- Pint's own syntax also failed on `"einstein / m**2"` (undefined) and
  `"m2 m-2"`, both of which appear in this project's docs.

Verified in this environment (pint 0.25.3): a registry with a one-line
preprocessor turning `m-2` into `m**-2`, plus `define("einstein = mole")`,
parses `"g m-2 d-1"`, `"einstein m-2"`, `"kg m-2 s-1"` and `"cm d-1"` to the
expected dimensionalities. That is enough to validate the whole registry
without adding a dependency. If the project later wants full UDUNITS
compatibility (`%`, `since` time units, etc.), `cf_xarray.units` provides an
already-configured registry and `pint-xarray` puts quantities on DataArrays;
both are optional add-ons, not requirements.

### 5.3 What goes where

| Concern | Field | Example |
|---|---|---|
| physical dimension | `units` | `"g m-2"` |
| what substance | `constituent` | `"C"` |
| over what interval | `aggregation` / CF `cell_methods` | `"time: sum"` |
| how to print | derived | `g C m⁻²` (unicode), `g\,C\,m^{-2}` (LaTeX) |

"per timestep" is not a unit and must not appear in `units`; it is exactly what
`kind=FLUX` + `cell_methods="time: sum"` + the `time_bounds` coordinate says.
This is also how CF and PEcAn treat it (PEcAn converts to `kg C m-2 s-1` by
dividing by the step length; we can offer that as a `to_rate()` helper once
`timestep_length` is on the dataset, §6).

### 5.4 Migration of `ParameterSpec.unit`

Mechanical: `"g / m**2"` → `"g m-2"`, `"1 / year"` → `"yr-1"` (or `"a-1"`;
recommend `"yr-1"`, Pint knows `year`), `"K * day"` → `"K d"`,
`"nmol / (g * s)"` → `"nmol g-1 s-1"`, `"mol / (m**2 * day)"` →
`"mol m-2 d-1"`, `"m**2 / m**2"` → `"m2 m-2"`. Fill `constituent` on every
carbon, nitrogen and water field, not the current 12.

## 6. xarray versus DataFrame (your question 1)

### 6.1 Recommendation

**Yes, make an xarray `Dataset` the canonical, metadata-carrying
representation, and keep the DataFrame as a cheap view.** Concretely:

- `SIPNETOutput.dataset` (lazy, cached, like `.data` today) returns a Dataset
  with **one dimension, `time`**, a `datetime64` coordinate built from
  `year`/`day`/`time` (start of step), auxiliary coordinates `year`,
  `day_of_year`, `hour_of_day`, and every variable carrying `attrs` from
  `cf_attrs(name)`. When the climate is available (always, from
  `SIPNETResult`), add `timestep_length` (days) as a variable and a
  `time_bounds` coordinate `[start, start + length]` per CF.
- `SIPNETOutput.data` keeps returning the flat DataFrame with the new column
  names. It is what people reach for interactively and what `groupby("year")`
  examples in the docs use.
- `load(variables=[...])` keeps its read-only-these-columns semantics and grows
  an `as_xarray=True` option; file-backed instances still read nothing until
  asked, so the lazy ensemble pattern in `runner.py`'s docstring is unchanged.
- `SIPNETResult.to_xarray()` is replaced by `result.outputs.dataset`; the
  current implementation is removed rather than fixed, because its shape is
  wrong (§3.1 item 4).

Reasons this is the right container, in order of weight:

1. **Metadata travels with the data.** `DataArray.attrs` is where units, long
   name and cell methods live in every CF tool; `DataFrame.attrs` exists but is
   dropped by most operations and by every serialiser. The registry is only
   useful if the values carry it.
2. **Ensembles stack.** `xr.concat(datasets, dim="member")` and a `site`
   dimension are the natural shape for the DA workflows this package exists
   for; a list of DataFrames needs a MultiIndex and re-invents this. The
   ensemble runner is out of scope, but the single-run object should be the
   thing it stacks.
3. **Serialisation.** `to_netcdf` / `to_zarr` with CF attrs gives a
   self-describing on-disk format for free; `CSV` of a DataFrame loses units.
4. **Time handling.** A real `datetime64` axis fixes the leap-year/partial-year
   problem and makes `resample("1YE").sum()` honour `cell_methods`
   (we can provide `aggregate(freq)` that applies the registry rule per
   variable: sum for fluxes, mean for states, last for cumulative).

Costs, and why they are acceptable: xarray becomes a required dependency
(currently optional). It is pure Python and its dependencies (numpy, pandas,
packaging) are already required, so the install footprint change is small.
Reading a `.out` file into a Dataset costs one extra `from_dataframe` per run,
which is negligible against the subprocess. If you would rather keep xarray
optional, the fallback is `dataset` raising a clear `ImportError` and `data`
working everywhere; the registry itself has no xarray dependency either way.

### 6.2 One subtlety to document

The time coordinate marks the **start** of each step (SIPNET's labelling), but
pools are values at the **end**. In CF terms fluxes are `time: sum` over
`time_bounds`, and states are `time: point` valid at the *upper* bound. The
registry records this per variable, the Dataset carries `time_bounds`, and the
user guide says it in one sentence. The alternative, shifting the coordinate to
the end of the step, would make the output disagree with the climate file and
with SIPNET's own files; not recommended.

## 7. Implementation plan

Phase 1, outputs (self-contained; one PR or two):

1. `pysipnet/variables.py` with `VariableSpec`, the 35 output specs, lookup
   helpers, `cf_attrs`, `axis_label`, `to_records`.
2. `pysipnet/units.py`: the Pint registry with UDUNITS preprocessor, `einstein`
   definition, `parse_units`, `format_units(style)`.
3. `output_reader.py` renames via the registry, warns on unknown headers,
   accepts aliases in `columns=`.
4. `SIPNETOutput`: `variables`, `__getitem__`, `dataset`, `load(as_xarray=)`;
   remove `SIPNETResult.to_xarray`; make `nee()/gpp()/et()` alias lookups or
   drop them.
5. `viz.py` labels from the registry.
6. Tests in §4.4; golden fixture header regenerated with a value-identity check.
7. Docs: generated "Output variables" reference page from the registry; update
   the six user-guide pages and README; note the start-of-step/end-of-step
   convention and the `transpiration_rate` distinction.
8. Optional: open an upstream issue on the documentation errors in §3.4.

Phase 2, parameters: UDUNITS unit strings, `constituent` on every field,
labels, `initializes` links, renames per §4.5 with `PYTHON_TO_SIPNET` and its
independent test updated in the same commit. This changes `RunConfig` key
names; since the package is pre-alpha the recommendation is a clean break
with a loud `ValidationError` naming the new field for each old key, not a
compatibility shim.

Phase 3, climate drivers: same treatment, plus `file_units` for the columns
SIPNET converts on read.

## 8. Decisions needed from you

1. `wood_carbon` or `above_ground_wood_carbon` (recommend the former).
2. `nitrogen_volatilization` or `n2o_emission` (recommend the former).
3. Keep short aliases (`nee`, `gpp`, `lai`, …) resolvable, or forbid them
   entirely (recommend keep as aliases, never as column names).
4. xarray required (recommended) or optional with `dataset` gated.
5. Units syntax: UDUNITS/CF strings (recommended) or stay with Pint
   expressions.
6. Do parameter renames in this effort (Phase 2) or defer.
