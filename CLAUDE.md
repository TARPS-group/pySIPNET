# pySIPNET — Developer Context

## What is this project?

**pySIPNET** is a clean, well-documented Python interface to [SIPNET](https://github.com/PecanProject/sipnet) — the Simplified Photosynthesis and Evapotranspiration Model, a lightweight process-based C model for coupled carbon, water, nitrogen, and greenhouse-gas dynamics at a single site.

## What this project is NOT

There is an existing R interface to SIPNET inside [PEcAn](https://github.com/PecanProject/pecan/tree/develop/models/sipnet). **We are not replicating that interface.** The PEcAn interface is complex because it must conform to PEcAn's internal conventions and data standards. It is also poorly documented and not cleanly organized. pySIPNET is:

- **Completely independent of PEcAn** — no dependency on PEcAn conventions, file structures, or data formats.
- **Lean and purpose-built** — minimal, focused, well-documented.
- **Designed for ensemble and DA workflows** — the single-run interface is architected with ensemble runs (parameter calibration, data assimilation) in mind from the start.

## SIPNET Version Target

**pySIPNET v0.x targets a specific pinned SIPNET commit, not a release tag.**

The submodule is pinned to `e4abf14f2445133c785b756025a2e39e60c7760f` (2025-07-14) on
`PecanProject/sipnet` `master`. `git describe` reports `v1.3.0-82-ge4abf14`: it is 82
commits *after* the `v1.3.0` tag and 20 commits *before* `v2.0.0`. Do not call this
"v1" — a large amount of v2-bound work had already landed on `master` by this commit.

### Why this exact commit

The next commit on `master` (`0bac82b`, "SIP78 Convert switches to run time options
part 3") **deletes `src/sipnet/modelStructures.h`** and moves the model-structure
switches into the runtime `Context` system. Our whole build strategy —
`patches/apply_flags_patch.py`, the `-DSNOW=1 -DGDD=1 …` preset flags, and the
`sipnet_standard` / `sipnet_forest` cached binaries — depends on those switches being
compile-time `#define`s. `e4abf14` is the newest commit where that is still true.

The commit after that (`c2962f2`) removes the `soilWetness` climate column, and later
pre-2.0.0 commits remove the obsolete params (`0d7a9b9`, SIP124) and output columns
(SIP145). So the pin is also the last point where the input/output formats our parsers
target are all still intact.

### What the pin actually looks like

Relative to the `v1.3.0` release, the following had **already landed** by `e4abf14`:

- **Multi-site support is gone** (`ed448dc`). The `.clim` `loc` column is auto-detected
  and ignored with a warning; a file with *differing* loc values is a hard error.
- **The `estimate` executable and its MCMC/observation-file inputs are gone**
  (`8e50299`). SIPNET reads no observation data.
- **Runtime config exists** (`262fbf0`, `4086960`). I/O options come from `sipnet.in`
  and CLI flags via `src/common/context.c`, with precedence
  `DEFAULT < INPUT_FILE < COMMAND_LINE < CALCULATED`.
- **`./sipnet --version` works** and prints `2.0.0` (`src/sipnet/version.h` already
  reads `NUMERIC_VERSION "2.0.0"`; the CITATION.cff still says 1.3.0).

So configuration is a **hybrid** at this pin: I/O and run options are runtime; model
structure (`GDD`, `SNOW`, `LITTER_POOL`, `WATER_HRESP`, `GROWTH_RESP`, `LEAF_WATER`,
`SOIL_PHENOL`, `HEADER`) is compile-time.

### Pinning mechanics

SIPNET is managed as a **git submodule** pinned to a specific commit hash. This approach:
- Is fully reproducible — anyone cloning pySIPNET gets the exact SIPNET source used.
- Makes version bumps an explicit, reviewable git change (updating the submodule pointer).
- Allows multiple SIPNET versions to coexist if needed in the future.

The submodule lives at `sipnet/` in the repo root. A `Makefile` (or CMake target) compiles
it; the resulting binary is stored at `sipnet/sipnet` (gitignored). Never commit the binary.

When we move to a tagged v2 release, we will add a `SIPNETVersion` abstraction that selects
input/output parsers and parameter schemas per version without changing the public API. Note
that v2.0.0 removes `soilWetness`, the obsolete params, and several output columns, and makes
model structure a runtime option — so the version adapter must cover build strategy, not just
file parsing.

## Project Goals

### Design Principles

1. **Hierarchical, named data structures.** All inputs are organized into logical groups (photosynthesis, respiration, water, phenology, initial conditions, etc.) using named fields — not positional. Code that consumes these structures should never break because a column order changed.

2. **No ambiguity about units, formats, or required fields.** Every parameter carries its units in its definition (docstring + Pydantic field metadata). Required vs. optional fields are enforced by the data model, not discovered at runtime when SIPNET crashes.

3. **Fully serializable.** A complete model run specification (parameters + climate + flags) must be representable as a plain dict/YAML/JSON with no hidden state, so it can be:
   - diffed against other runs
   - stored as experimental metadata
   - passed to ensemble runners without modification

4. **Clean separation of concerns:**
   - **Data layer**: parameter containers, climate drivers, events — pure Python data structures, no I/O
   - **I/O layer**: readers and writers that translate between Python objects and SIPNET file formats
   - **Run layer**: manages the binary, working directory, and subprocess execution
   - **Output layer**: parses SIPNET output into typed Python/pandas/xarray objects

5. **Version-forward design.** Version-specific logic (file format differences, available parameters) is isolated behind version adapters so the public API stays stable when v2 support is added.

### Primary Use Cases (in priority order)

1. **Single-site, single-run** — user specifies parameters, climate, flags; runs SIPNET; gets back a result object.
2. **Multi-site ensemble** — many sites, each with an ensemble of initial conditions and/or parameters. Ensemble running is **not** part of this package (see below).
3. **Parameter calibration / data assimilation** — iterative runs where parameters are perturbed. The run interface must be fast to invoke programmatically and must impose no per-run overhead from Python.

## Ensemble Runner (Out of Scope for this Package)

The single-run interface is designed to be composed by an external ensemble runner. pySIPNET will **not** contain an ensemble runner. Candidate tools for the ensemble layer:

- **[Hydra](https://hydra.cc/)** — excellent for structured config sweeps (grid, random, Ax/Optuna). Integrates well with Python dataclasses/OmegaConf. The pySIPNET config objects map naturally to Hydra structured configs.
- **[Dask](https://dask.org/)** / **[Ray](https://ray.io/)** — for distributing independent runs across cores or cluster nodes with minimal code.
- **[Parsl](https://parsl-project.org/)** — designed for scientific workflows on HPC clusters; supports futures-based parallelism across local, SLURM, PBS, etc.

The recommendation is: use Hydra for parameter sweep specification and Parsl or Dask for execution. pySIPNET's job is to make each single run a clean, stateless function call (`run(config) -> result`).

## SIPNET Inputs — Complete Inventory

Verified by grepping every `fopen` / `openFile` / `access` / `getenv` / `stdin` in `src/`.
There are exactly **four** read paths, and **no** environment variables or stdin input:

| File | Opened at | Required |
|---|---|---|
| `sipnet.in` (overridable via `-i` / `--input_file`) | `frontend.c` `readInputFile` | **Yes** — `openFile` hard-exits if absent |
| `<FILENAME>.param` | `sipnet.c` `readParamData` | **Yes** |
| `<FILENAME>.clim` | `sipnet.c` `readClimData` | **Yes** |
| `events.in` (fixed name, `EVENT_IN_FILE`) | `events.c` `readEventData` | **No** — guarded by `access(..., F_OK)` |

There is **no phenology / LAI / NDVI / fPAR data input** and **no observation-data input**
at any point in this repo's history. See gotcha 8.

The compile-time model-structure flags are effectively a fifth input: they change both the
model equations and the required parameter set, so they must be part of any serialized run
spec. `ModelPreset` covers this.

### `sipnet.in` (run configuration)

Parsed into the global `Context` (`src/common/context.c`). Separators are ` \t=:`, comment
character `!`, and the string `none` means the empty string. Keys are normalised by
`nameToKey` — non-alphanumerics stripped, lowercased — so `DO_MAIN_OUTPUT`, `do-main-output`
and `domainoutput` are the same key. Unknown keys produce a warning and are ignored.

| Key | Default |
|---|---|
| `FILENAME` | none — **fatal if unset** |
| `DO_MAIN_OUTPUT` | 1 |
| `EVENTS` | 1 |
| `PRINT_HEADER` | 1 |
| `DO_SINGLE_OUTPUT` | 0 |
| `DUMP_CONFIG` | 0 |
| `QUIET` | 0 |
| `INPUT_FILE` | `sipnet.in` |
| `RUNTYPE` | obsolete — any value other than `standard` is a **fatal error** |

Precedence is `DEFAULT < INPUT_FILE < COMMAND_LINE < CALCULATED`, so CLI flags override the
file. Flags take a `--no_` prefix to force off (e.g. `--no_print_header`).

`PARAM_FILE`, `CLIM_FILE`, `OUT_FILE` and `OUT_CONFIG_FILE` exist as context entries but are
unconditionally overwritten as `FILENAME + ".param"/".clim"/".out"/".config"` at
`CTX_CALCULATED` precedence — **setting them in `sipnet.in` has no effect.**

### `.param` file

Two+ columns: `parameter_name  value [changeable min max sigma...]`. Only first two columns are used; extras produce a warning (`"extra columns in .param file are being ignored"`) and are otherwise accepted (legacy). Comment character: `!`. Order-independent. Name lookup is **case-insensitive** (`strcasecmp` in `locateParam`).

Hard errors in `readModelParams`: a value of `*` (the old spatially-varying marker), and any
parameter specified **twice**. An unrecognised parameter name is only a warning.

**Important unit gotcha:** Several parameters are specified as **per-year rates** in the param file but converted to per-day internally. These are: `baseVegResp`, `baseSoilResp`, `litterBreakdownRate`, `woodTurnoverRate`, `leafTurnoverRate`, `fineRootTurnoverRate`, `coarseRootTurnoverRate`, `baseFineRootResp`, `baseCoarseRootResp`. The Python interface must be explicit about this: users should specify these in per-year units, and the writer will pass them through unchanged.

### `.clim` file (13 core columns, optional leading `loc`)

`NUM_CLIM_FILE_COLS` is **13** (`src/sipnet/sipnet.c`). No header, no comment character
(`readClimData` never calls `stripComment`). Whitespace/tab-delimited, one row per timestep.

A leading `loc` column is **auto-detected, not assumed**: `readClimData` runs `countFields`
on the first line, and if it finds 14 it logs `"ignoring location column"` and switches to a
14-field `sscanf` format. Both 13- and 14-column files are accepted.

| Col (13) | Col (14) | Name | File units | Internal conversion |
|---|---|---|---|---|
| — | 1 | loc | integer | detected, warned about, ignored |
| 1 | 2 | year | integer | — |
| 2 | 3 | day | integer (1 = Jan 1) | — |
| 3 | 4 | time | fractional hours (12.0 = noon) | — |
| 4 | 5 | length | **days**; if negative, parsed as −seconds | `length / -86400` |
| 5 | 6 | tair | °C | — |
| 6 | 7 | tsoil | °C | — |
| 7 | 8 | par | Einstein m⁻², summed over the timestep | `× 1/length` → E m⁻² day⁻¹ |
| 8 | 9 | precip | mm | `× 0.1` → cm |
| 9 | 10 | vpd | Pa | `× 0.001` → kPa, clamped to ≥ `TINY` (1e-6) |
| 10 | 11 | vpdSoil | Pa | `× 0.001` → kPa |
| 11 | 12 | vPress | Pa | `× 0.001` → kPa |
| 12 | 13 | wspd | m s⁻¹ | clamped to ≥ `TINY` |
| 13 | 14 | soilWetness | fraction | **read and stored, never used by the model** |

Format gotchas (all verified in `readClimData`):

- **Multi-location files are a hard error.** If any row's loc differs from the first row's,
  SIPNET exits with "multiple locations not supported". Multi-site support was removed in
  `ed448dc`, before our pin.
- **`length < 0` means seconds, not days.** Undocumented outside the source.
- **The first line is read with `getline`, the rest with `fscanf`.** A *leading* blank line
  is fatal; interior blank lines are actually tolerated (`fscanf` skips whitespace), despite
  the source comment claiming otherwise. We still forbid both.
- **`soilWetness` is required-but-inert.** It must be present to satisfy the column count,
  but nothing reads `climate->soilWetness`. The `soilWetnessFrac` in the output is a computed
  tracker (`envi.soilWater / params.soilWHC`), unrelated to this column.

SIPNET itself performs **no** validation beyond the field count and the loc check — no NaN
check, no monotonicity check, no range check. Our strict pre-write validation (no NaN, no
blank lines, vpd > 0, wspd > 0, monotonically increasing timesteps) is doing real work.

The sample climate file `data/era5_site1.clim` is in this format.

### `events.in` (optional)

Fixed filename, read from the working directory only when `EVENTS` is on. A **missing file
is harmless** — `readEventData` guards with `access(..., F_OK)` and logs
`"No event file found, assuming no events"`.

Format: `year day <type> <type-specific params...>`, one event per line, and records
**must be in ascending time order** (otherwise a hard error). Arities are fixed
(`src/sipnet/events.h`):

| Keyword | Params |
|---|---|
| `fert` | orgN, orgC, minN |
| `harv` | fractionRemovedAbove, fractionRemovedBelow, fractionTransferredAbove, fractionTransferredBelow |
| `irrig` | amountAdded, method (0 = canopy, 1 = soil, 2 = flood — declared but **not supported**) |
| `plant` | leafC, woodC, fineRootC, coarseRootC |
| `till` | fractionLitterTransferred, somDecompModifier, litterDecompModifier |

An unknown event keyword or a wrong param count is a hard error. Events produce an
`events.out` file alongside the main output.

### SIPNET Output

Output has a "Notes: ..." comment line before the column header. Detect: if `line[0].startswith("Notes")`, skip one line before the header. Key output variables include NEE, GPP, NPP, Ra, Rh, ET, soil/litter/wood C pools, soil water.

## SIPNET Parameters — Full Grouped List

Parameters are grouped as they appear in the Python data model. All initial conditions are also in the `.param` file (SIPNET makes no distinction).

The authoritative source is the `initializeOneModelParam` block in
`src/sipnet/sipnet.c` (`readParamData`). Its third argument is the required flag:
`1` = always required, `0` = optional, `OBSOLETE_PARAM` (-1) = see gotcha 7, and a
macro expression = **required only under that compile-time flag**.

### Initial Conditions
`plantWoodInit` (g C m⁻²), `laiInit` (m² m⁻²), `litterInit` (g C m⁻²), `soilInit` (g C m⁻²), `litterWFracInit` (fraction of litterWHC), `soilWFracInit` (fraction of WHC), `snowInit` (cm water equiv.), `fineRootFrac` (fraction), `coarseRootFrac` (fraction)

### Photosynthesis
`aMax` (nmol CO₂ g⁻¹ leaf s⁻¹), `aMaxFrac`, `baseFolRespFrac`, `psnTMin` (°C), `psnTOpt` (°C), `dVpdSlope` (kPa⁻¹), `dVpdExp`, `halfSatPar` (Einstein m⁻² ground day⁻¹), `attenuation`

### Phenology
`leafOnDay` (DOY), `gddLeafOn` (°C·day), `soilTempLeafOn` (°C), `leafOffDay` (DOY), `leafGrowth` (g C m⁻²), `fracLeafFall`, `leafAllocation`, `leafTurnoverRate` (year⁻¹)

The three leaf-on parameters are **mutually exclusive and flag-selected**, not all required:
`leafOnDay` is registered `!((GDD) || (SOIL_PHENOL))`, `gddLeafOn` is `GDD`, and
`soilTempLeafOn` is `SOIL_PHENOL`. Both our presets build with `-DGDD=1`, so
**`gddLeafOn` is required and `leafOnDay` is optional**. `leafOffDay` is always required
(leaf-off is always fixed-DOY). See gotcha 8.

### Leaf
`leafCSpWt` (g C m⁻² leaf), `cFracLeaf` — both always required. `leafCSpWt` converts
`laiInit` to the initial leaf C pool (`envi.plantLeafC = laiInit × leafCSpWt`) and relates
to SLA as `leafCSpWt = cFracLeaf / SLA`.

### Autotrophic Respiration
`baseVegResp` (year⁻¹), `vegRespQ10`, `growthRespFrac`, `frozenSoilFolREff`, `frozenSoilThreshold` (°C), `baseFineRootResp` (year⁻¹), `baseCoarseRootResp` (year⁻¹), `fineRootQ10`, `coarseRootQ10`

`growthRespFrac` is required only under `GROWTH_RESP` (0 in both presets → optional).

### Soil Respiration
`baseSoilResp` (year⁻¹), `soilRespQ10`, `soilRespMoistEffect`, `litterBreakdownRate` (year⁻¹), `fracLitterRespired`

`soilRespMoistEffect` is required under `WATER_HRESP` (1 in both presets).
`litterBreakdownRate` and `fracLitterRespired` are required under `LITTER_POOL`
(0 in `sipnet_standard`, 1 in `sipnet_forest`).

### Allocation
`fineRootAllocation`, `woodAllocation`, `fineRootExudation`, `coarseRootExudation`, `fineRootTurnoverRate` (year⁻¹), `coarseRootTurnoverRate` (year⁻¹), `woodTurnoverRate` (year⁻¹)

### Water
`waterRemoveFrac` (day⁻¹), `frozenSoilEff`, `wueConst`, `soilWHC` (cm), `litterWHC` (cm), `immedEvapFrac`, `fastFlowFrac`, `snowMelt` (cm °C⁻¹ day⁻¹), `rdConst`, `rSoilConst1`, `rSoilConst2`, `leafPoolDepth`

`snowMelt` is required under `SNOW` (1 in both presets). `leafPoolDepth` is required
under `LEAF_WATER` (0 in both presets → optional).

### Registered but unreachable at our pin
`qualityLeaf`, `qualityWood` (`SOIL_QUALITY`), `efficiency`, `maxIngestionRate`,
`microbeInit` (`SOIL_QUALITY || MICROBES`), `halfSatIngestion`, `baseMicrobeResp`,
`microbeQ10`, `microbePulseEff` (`MICROBES`). `SOIL_MULTIPOOL`, `SOIL_QUALITY` and
`MICROBES` are hard `#define`s in `sipnet.c` (not `#ifndef`-guarded by our patch), so
they cannot be switched on with `-D` and these params are always optional.

### Required-count summary

- **52 parameters are unconditionally required** (required flag = `1`).
- `sipnet_standard` adds `gddLeafOn`, `snowMelt`, `soilRespMoistEffect` → **55 required**,
  plus the 9 obsolete placeholders = **64 lines minimum**.
- `sipnet_forest` adds `litterBreakdownRate` and `fracLitterRespired` → **57 required**,
  plus 9 = **66 lines minimum**.

Note: `fAnoxia` is a v2-only parameter and is **not** in our Python model or param file.

## Key Technical Gotchas

1. **`psnTMax` and `coarseRootAllocation` are derived**, not specified. `psnTMax = 2×psnTOpt − psnTMin`. `coarseRootAllocation = 1 − leafAllocation − fineRootAllocation − woodAllocation`.

2. **PAR units scale with timestep.** Climate file PAR is total Einsteins m⁻² for the whole timestep interval. When converting from instantaneous flux measurements, multiply by `length` (in days × 86400 seconds/day).

3. **SIPNET expects files in the current working directory.** The runner writes all inputs to a fresh temp dir per run and executes the binary there. The `sipnet.in` config file uses `fileName = sipnet` so SIPNET looks for `sipnet.param`, `sipnet.clim`, and writes `sipnet.out`.

4. **Compile-time model-structure flags.** At our pin, `GROWTH_RESP`, `WATER_HRESP`, `LEAF_WATER`, `SNOW`, `GDD`, `SOIL_PHENOL`, `LITTER_POOL` and `HEADER` are `#ifndef`-guarded (by `patches/apply_flags_patch.py`) and settable with `-D`. `SOIL_MULTIPOOL`, `SOIL_QUALITY`, `MICROBES` and `NUMBER_SOIL_CARBON_POOLS` are hard `#define`s that **cannot** be overridden without editing source. Named binary presets (`sipnet_standard`, `sipnet_forest`) are stored in `.sipnet_cache/` and built with `make sipnet-standard` / `make sipnet-forest`. Because these change the required parameter set, the preset is part of the run spec, not a build detail.

5. **No missing climate values.** Climate validation must be strict: every row must be complete, timesteps must be monotonically increasing, and the start/end dates must bracket the intended simulation period.

6. **Events file.** SIPNET defaults to `EVENTS=1` (looks for `events.in`). The runner writes `EVENTS = 0` in `sipnet.in` to suppress this for basic runs. Note that this is belt-and-braces: a *missing* `events.in` is already harmless (`access` guard, `logInfo` only), so `EVENTS = 0` mainly guards against a stale file in the working directory. When events are used, the file must be in the working directory.

7. **`OBSOLETE_PARAM` bug.** Nine parameters are declared `OBSOLETE_PARAM = -1` in the SIPNET source, meaning they are read from the file but never used. However, SIPNET's `checkAllRead()` tests `if (param->isRequired)` which is truthy for -1, so SIPNET **errors out if these params are absent** from the file. The pySIPNET writer automatically appends these as backward-compatibility placeholders (`_OBSOLETE_DEFAULTS` in `param_io.py`); they are completely hidden from the user-facing API. The nine params are: `baseSoilRespCold`, `soilRespQ10Cold`, `coldSoilThreshold`, `E0`, `T0`, `litWaterDrainRate`, `totNitrogen`, `microbeNC`, `m_ballBerry`.

8. **Phenology is parameter-driven — there is no phenology data input.** `pastLeafGrowth()` selects one of three leaf-on triggers at compile time: `climate->gdd >= gddLeafOn` under `GDD`, `climate->tsoil >= soilTempLeafOn` under `SOIL_PHENOL`, or `currTime >= leafOnDay` otherwise. Growing degree days are accumulated inside `readClimData` from `tair × length` and reset on year change — **never read from a file**. Leaf-off is always `leafOffDay`. `laiInit` only sets the initial leaf C pool (`envi.plantLeafC = laiInit × leafCSpWt`); LAI is prognostic thereafter. "Phenology data" enters SIPNET only by two indirect routes, **both of which resolve to the `.param` file before the binary runs** (verified against PEcAn source, not inferred):

    - `PEcAn.data.remote::extract_phenology_MODIS()` pulls MODIS **MCD12Q2** `MidGreenup` / `MidGreendown` bands, QA-filters them, and writes a CSV (`year, site_id, lat, lon, leafonday, leafoffday, leafon_qa, leafoff_qa`). `PEcAn.SIPNET::write.config.SIPNET()` then reads it via `settings$run$inputs$leaf_phenology$path` and **overwrites `leafOnDay` / `leafOffDay` in the param table** (fallback chain: matching year → site mean across years → param-file default; applied only `if (leafOffDay > leafOnDay)`).
    - `gddLeafOn` is set from a PFT **trait** named `GDD`, i.e. the calibration/prior route — *not* from remote sensing.

    Upstream of our pin this changes: see gotcha 9.

9. **Upstream SIPNET *does* accept prescribed phenology — via `events.in`, not a new file.** PR #326 (landed after `v2.1.0`; present in `v2.2.0-alpha.1` and `origin/master`, absent from `v2.0.0`/`v2.1.0`) adds `leafon` and `leafoff` event types with **zero** parameters, so a line is just `year day leafon`. They are real input keywords (`eventStringToType` in `events.c`), and they are **mutually exclusive with every calculated mechanism**: `checkForCalculatedLeafEvents()` exits with `EXIT_CODE_BAD_PARAMETER_VALUE` if `ctx.gdd || ctx.soilPhenol || params.leafOnDay > 0 || params.leafOffDay > 0`. The event applies the *same* flux the calculated trigger would (`params.leafGrowth / climLen` for leaf-on, `envi.plantLeafC * params.fracLeafFall` for leaf-off), so events prescribe **timing** while magnitude still comes from parameters. PEcAn already emits them (`write.events.SIPNET.R`) and, when it sees them, zeroes `leafOnDay`/`leafOffDay`/`gddLeafOn` and ignores the `leaf_phenology` CSV. Note the upstream docs (`docs/user-guide/model-inputs.md`) still list event types as only `plant`/`harv`/`till`/`fert`/`irrig` — the code is ahead of the docs.

10. **`leafOnDay = 0` means "disabled" upstream but NOT at our pin.** At `origin/master`, `pastLeafGrowth`/`pastLeafFall` are gated on `params.leafOnDay > 0` / `params.leafOffDay > 0`, so 0 disables the trigger — that is how PEcAn turns off internal scheduling. **At our pin there is no such gate**: `pastLeafFall` returns `(day + time/24) >= params.leafOffDay` unconditionally, so `leafOffDay = 0` triggers leaf fall on the first timestep of every year rather than disabling it. Never port a PEcAn-style zeroed param set to our pinned binary.

## File Structure (planned)

```
pySIPNET/
├── sipnet/                    # git submodule — SIPNET source, pinned to commit e4abf14
├── pysipnet/
│   ├── __init__.py
│   ├── version.py             # SIPNET version constants and detection
│   ├── parameters/
│   │   ├── __init__.py
│   │   ├── v1.py              # Pydantic models for the pinned-commit parameter set
│   │   │                      #   ("v1" is a legacy filename, not the SIPNET version)
│   │   └── base.py            # Abstract base for version-agnostic parameter access
│   ├── climate.py             # Climate driver data structure and validators
│   ├── events.py              # Management events data structure
│   ├── io/
│   │   ├── __init__.py
│   │   ├── param_writer.py    # Writes .param file
│   │   ├── param_reader.py    # Reads .param file
│   │   ├── clim_writer.py     # Writes .clim file
│   │   ├── clim_reader.py     # Reads .clim file
│   │   ├── output_reader.py   # Reads .out file
│   │   └── events_io.py       # Reads/writes events.in
│   ├── runner.py              # SIPNETRunner — subprocess management
│   ├── result.py              # SIPNETResult — output container
│   └── build.py               # Compiles SIPNET from submodule source
├── tests/
│   └── ...
├── data/                      # (gitignored) sample data
├── docs/
├── CLAUDE.md                  # This file
├── README.md
├── pyproject.toml
└── Makefile                   # sipnet build targets
```

## Development Conventions

- **Python ≥ 3.11**
- **Pydantic v2** for all data models (parameter validation, units enforcement)
- **pandas** for climate time series and output; **xarray** as an optional output format
- **NumPy** for numerical operations
- **No comments unless the WHY is non-obvious.** Well-named identifiers are preferred.
- **Type hints everywhere.**
- All file I/O is in the `pysipnet/io/` subpackage. The rest of the package never touches the filesystem directly.
- Tests use real SIPNET binaries where possible (integration tests), not mocks.
