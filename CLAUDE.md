# pySIPNET — Developer Context

## What is this project?

**pySIPNET** is a clean, well-documented Python interface to [SIPNET](https://github.com/PecanProject/sipnet) — the Simplified Photosynthesis and Evapotranspiration Model, a lightweight process-based C model for coupled carbon, water, nitrogen, and greenhouse-gas dynamics at a single site.

## What this project is NOT

There is an existing R interface to SIPNET inside [PEcAn](https://github.com/PecanProject/pecan/tree/develop/models/sipnet). **We are not replicating that interface.** The PEcAn interface is complex because it must conform to PEcAn's internal conventions and data standards. It is also poorly documented and not cleanly organized. pySIPNET is:

- **Completely independent of PEcAn** — no dependency on PEcAn conventions, file structures, or data formats.
- **Lean and purpose-built** — minimal, focused, well-documented.
- **Designed for ensemble and DA workflows** — the single-run interface is architected with ensemble runs (parameter calibration, data assimilation) in mind from the start.

## SIPNET Version Target

**pySIPNET v0.x targets SIPNET v1.** The [documentation site](https://pecanproject.github.io/sipnet/) reflects v2, which is still changing rapidly. We deliberately avoid v2 for now.

### v1 vs v2 key differences

| Aspect | v1 | v2 (target for future) |
|---|---|---|
| Mode flags | Compile-time `#define` switches | Runtime CLI flags / `sipnet.in` |
| Multi-site | Yes (location column in `.clim`) | No (single-site only) |
| `.clim` format | 14 columns (loc + 12 core + soilWetness) | 12 columns |
| `.param` format | 5+ columns (name, value, changeable, min, max, sigma…) | 2 columns (name, value) |
| Output format | Includes `loc`, `litterWater`, `fPAR`, `microbeC` columns | Restructured columns |
| Versioning | No runtime `--version` flag | `./sipnet --version` prints semver |

### How we pin to v1

SIPNET is managed as a **git submodule** pinned to a specific commit hash. This approach:
- Is fully reproducible — anyone cloning pySIPNET gets the exact SIPNET source used.
- Makes version bumps an explicit, reviewable git change (updating the submodule pointer).
- Allows multiple SIPNET versions to coexist if needed in the future.

The submodule lives at `sipnet/` in the repo root. A `Makefile` (or CMake target) compiles it; the resulting binary is stored at `sipnet/sipnet` (gitignored). Never commit the binary.

When v2 stabilizes, we will add a `SIPNETVersion` abstraction that selects input/output parsers and parameter schemas per version without changing the public API.

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

## SIPNET Input File Formats (v1)

### `.param` file

Two+ columns: `parameter_name  value [changeable min max sigma...]`. Only first two columns are used; extras are silently accepted (legacy). Comment character: `!`. Order-independent.

**Important unit gotcha:** Several parameters are specified as **per-year rates** in the param file but converted to per-day internally. These are: `baseVegResp`, `baseSoilResp`, `litterBreakdownRate`, `woodTurnoverRate`, `leafTurnoverRate`, `fineRootTurnoverRate`, `coarseRootTurnoverRate`, `baseFineRootResp`, `baseCoarseRootResp`. The Python interface must be explicit about this: users should specify these in per-year units, and the writer will pass them through unchanged.

### `.clim` file (v1 format, 14 columns)

No header. Space-delimited. One row per timestep.

| Col | Name | Units |
|---|---|---|
| 1 | loc | integer (ignored) |
| 2 | year | integer |
| 3 | day | integer (1 = Jan 1) |
| 4 | time | fractional hours |
| 5 | length | days (timestep duration) |
| 6 | tair | °C |
| 7 | tsoil | °C |
| 8 | par | Einstein m⁻² (total over timestep) |
| 9 | precip | mm |
| 10 | vpd | Pa |
| 11 | vpdSoil | Pa |
| 12 | vPress | Pa |
| 13 | wspd | m s⁻¹ |
| 14 | soilWetness | (ignored) |

Constraints: no NaN, no blank lines, vpd > 0, wspd > 0 (SIPNET clamps these internally, but we validate before writing).

The sample climate file `data/era5_site1.clim` is in this v1 format.

### SIPNET Output (v1)

v1 output has a "Notes: ..." comment line before the column header. Detect: if `line[0].startswith("Notes")`, skip one line before the header. Key output variables include NEE, GPP, NPP, Ra, Rh, ET, soil/litter/wood C pools, soil water.

## SIPNET Parameters — Full Grouped List

Parameters are grouped as they appear in the Python data model. All initial conditions are also in the `.param` file (SIPNET makes no distinction).

### Initial Conditions
`plantWoodInit` (g C m⁻²), `laiInit` (m² m⁻²), `litterInit` (g C m⁻²), `soilInit` (g C m⁻²), `soilWFracInit` (fraction of WHC), `snowInit` (cm water equiv.), `fineRootFrac` (fraction), `coarseRootFrac` (fraction)

### Photosynthesis
`aMax` (nmol CO₂ g⁻¹ leaf s⁻¹), `aMaxFrac`, `baseFolRespFrac`, `psnTMin` (°C), `psnTOpt` (°C), `dVpdSlope` (kPa⁻¹), `dVpdExp`, `halfSatPar` (Einstein m⁻² ground day⁻¹), `attenuation`

### Phenology
`leafOnDay` (DOY), `gddLeafOn` (°C·day), `soilTempLeafOn` (°C), `leafOffDay` (DOY), `leafGrowth` (g C m⁻²), `fracLeafFall`, `leafAllocation`, `leafTurnoverRate` (year⁻¹)

### Autotrophic Respiration
`baseVegResp` (year⁻¹), `vegRespQ10`, `growthRespFrac`, `frozenSoilFolREff`, `frozenSoilThreshold` (°C), `baseFineRootResp` (year⁻¹), `baseCoarseRootResp` (year⁻¹), `fineRootQ10`, `coarseRootQ10`

### Soil Respiration
`baseSoilResp` (year⁻¹), `soilRespQ10`, `soilRespMoistEffect`, `litterBreakdownRate` (year⁻¹), `fracLitterRespired`

### Allocation
`fineRootAllocation`, `woodAllocation`, `fineRootTurnoverRate` (year⁻¹), `coarseRootTurnoverRate` (year⁻¹), `woodTurnoverRate` (year⁻¹)

### Water
`waterRemoveFrac` (day⁻¹), `frozenSoilEff`, `wueConst`, `soilWHC` (cm), `litterWHC` (cm), `immedEvapFrac`, `fastFlowFrac`, `snowMelt` (cm °C⁻¹ day⁻¹), `rdConst`, `rSoilConst1`, `rSoilConst2`, `leafPoolDepth`

Note: `fAnoxia` is a v2-only parameter and is **not** in the v1 Python model or param file.

## Key Technical Gotchas

1. **`psnTMax` and `coarseRootAllocation` are derived**, not specified. `psnTMax = 2×psnTOpt − psnTMin`. `coarseRootAllocation = 1 − leafAllocation − fineRootAllocation − woodAllocation`.

2. **PAR units scale with timestep.** Climate file PAR is total Einsteins m⁻² for the whole timestep interval. When converting from instantaneous flux measurements, multiply by `length` (in days × 86400 seconds/day).

3. **SIPNET expects files in the current working directory.** The runner writes all inputs to a fresh temp dir per run and executes the binary there. The `sipnet.in` config file uses `fileName = sipnet` so SIPNET looks for `sipnet.param`, `sipnet.clim`, and writes `sipnet.out`.

4. **v1 compile-time flags.** In v1, features like `SNOW`, `LITTER_POOL`, etc. are `#define` switches set at compile time. Named binary presets (`sipnet_standard`, `sipnet_forest`) are stored in `.sipnet_cache/` and built with `make sipnet-standard` / `make sipnet-forest`.

5. **No missing climate values.** Climate validation must be strict: every row must be complete, timesteps must be monotonically increasing, and the start/end dates must bracket the intended simulation period.

6. **Events file.** SIPNET defaults to `EVENTS=1` (looks for `events.in`). The runner writes `EVENTS = 0` in `sipnet.in` to suppress this for basic runs. When events are supported, the events file must be in the working directory.

7. **`OBSOLETE_PARAM` bug in SIPNET v1.** Nine parameters are declared `OBSOLETE_PARAM = -1` in the SIPNET source, meaning they are read from the file but never used. However, SIPNET's `checkAllRead()` tests `if (param->isRequired)` which is truthy for -1, so SIPNET **errors out if these params are absent** from the file. The pySIPNET writer automatically appends these as backward-compatibility placeholders (`_OBSOLETE_DEFAULTS` in `param_io.py`); they are completely hidden from the user-facing API. The nine params are: `baseSoilRespCold`, `soilRespQ10Cold`, `coldSoilThreshold`, `E0`, `T0`, `litWaterDrainRate`, `totNitrogen`, `microbeNC`, `m_ballBerry`.

## File Structure (planned)

```
pySIPNET/
├── sipnet/                    # git submodule — SIPNET source, pinned to v1 tag
├── pysipnet/
│   ├── __init__.py
│   ├── version.py             # SIPNET version constants and detection
│   ├── parameters/
│   │   ├── __init__.py
│   │   ├── v1.py              # Pydantic models for all v1 parameters
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
