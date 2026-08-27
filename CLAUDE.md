# pySIPNET — Developer Context

## What is this project?

**pySIPNET** is a clean, well-documented Python interface to [SIPNET](https://github.com/PecanProject/sipnet) — the Simplified Photosynthesis and Evapotranspiration Model, a lightweight process-based C model for coupled carbon, water, nitrogen, and greenhouse-gas dynamics at a single site.

## What this project is NOT

There is an existing R interface to SIPNET inside [PEcAn](https://github.com/PecanProject/pecan/tree/develop/models/sipnet). **We are not replicating that interface.** The PEcAn interface is complex because it must conform to PEcAn's internal conventions and data standards. It is also poorly documented and not cleanly organized. pySIPNET is:

- **Completely independent of PEcAn** — no dependency on PEcAn conventions, file structures, or data formats.
- **Lean and purpose-built** — minimal, focused, well-documented.
- **Designed for ensemble and DA workflows** — the single-run interface is architected with ensemble runs (parameter calibration, data assimilation) in mind from the start.

## SIPNET Version Target

**pySIPNET pins SIPNET to the `v2.2.0-alpha.1` pre-release** (commit
`41fa853e7131f542c52fcc0f4e3ea76892b52eda`), recorded in
`pysipnet/version.py` and in the `sipnet/` submodule.

The SIPNET developers created this tag so that everyone's analyses pin to the
same version, which is why it is preferred over a bare commit even though
newer commits exist.

Pinning by commit rather than branch means anyone who clones the repo compiles
the same model source, and a version change is an explicit, reviewable edit.

### Why this tag rather than a newer commit

Three candidates were on the table: `v2.2.0-alpha.1` (this one), `bc96ae1`, and
`d2fc7a2`. They form a strict linear chain — the tag is the **oldest** of the
three, and `d2fc7a2` the newest, which is the opposite of how they were first
described to us (the `-0400`/`-0700` author timestamps mislead; check ancestry).

The differences do not matter to pySIPNET:

- tag → `bc96ae1` adds `checkMineralNLimitation()` in `limitations.c`, a real
  model change but inside `if (ctx.nitrogenCycle)`, which `ModelFlags` refuses.
- `bc96ae1` → `d2fc7a2` touches only `tools/sipnet_view.py`, which we do not use.

So the tag was chosen for the property a bare commit cannot have: it is a name
everyone can pin to and cite. If the group wants the tip, ask upstream to tag
it rather than pinning a loose commit.

### The trap: a pre-release tag can move

A tag is not immutable. If upstream re-tags `v2.2.0-alpha.1`,
`tests/test_build.py` fails, because it asserts the submodule sits at
`SIPNET_PINNED_COMMIT` — a commit, which cannot move. That is the intended
behaviour: loud, not silent.

### The trap: the numeric version lags the tag

`version.h` at this tag still declares `NUMERIC_VERSION "2.1.0"`, so:

    $ sipnet --version
    SIPNET version 2.1.0 (v2.2.0-alpha.1)

The numeric version therefore **cannot identify the pin** — a v2.1.0 binary
reports the same number. The parenthesised part is `git describe --tags`,
injected by SIPNET's Makefile as `GIT_HASH`, and that is what
`sipnet_build_tag()` extracts and `test_build.py` checks. `SIPNET_NUMERIC_VERSION`
is recorded for reporting only. Do not be tempted to check it instead; a test
that does would pass against the wrong release.

### Consequences worth remembering

- **Model options are runtime, not compile-time.** One binary, built by
  `make sipnet` with no `-D` flags. Options travel in the generated
  `sipnet.in`. There is no source patch and no `patches/` directory.
- **Parameters calibrated before this pin are not transferable.** v2.1.0
  restructured carbon allocation, splitting `plantWoodC` from a storage-lag
  term. On the Niwot record this moves `plantWoodC` by 5.0% and cumulative NEE
  by 1.8%. Upstream says as much: the change "will likely require
  recalibration of SIPNET params."
- **Upstream ships binaries** for `linux-x86_64` and `macos-arm64` from v2.0.0
  on, so local compilation is optional on those platforms.
  `pysipnet.build.download_sipnet()` (or `make sipnet-download`) fetches one.
  The archive SHA-256 is pinned in `SIPNET_RELEASE_ASSETS` and checked before
  extraction, archive members are inspected so a hostile path cannot escape the
  destination, and the installed binary is asked for its version afterwards.
  A failure at any of those steps leaves nothing installed. Refresh the digests
  whenever `SIPNET_PINNED_COMMIT` moves; a stale one fails loudly, which is what
  you want. `pytest -m network` checks them against the published release.

### Pinning mechanics

The submodule lives at `sipnet/`. `make sipnet` compiles it and copies the
result to `.sipnet_cache/sipnet` (gitignored). Never commit the binary.
`tests/test_build.py` asserts that the submodule is at
`SIPNET_PINNED_COMMIT` and that the compiled binary reports
`SIPNET_TARGET_VERSION`, so a stale binary or half-finished bump fails loudly
rather than producing quietly wrong output.

When moving to a newer SIPNET, expect to touch: the required-parameter set,
the output column list, the `sipnet.in` keys, and the golden fixtures. Bump
`SIPNET_PINNED_COMMIT` and `SIPNET_TARGET_VERSION` together, and rebuild with
`make sipnet` — `build_sipnet()` trusts an existing binary unless passed
`force=True`.

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

Verified by grepping every `fopen` / `openFile` / `access` / `getenv` / `stdin`
in `sipnet/src/`. There are **five** read paths and no environment-variable or
stdin input:

| File | Read by | Required |
|---|---|---|
| `sipnet.in` (or `-i <path>`) | `frontend.c` `readInputFile` | **Yes** — hard exit if absent |
| `<FILE_NAME>.param` | `sipnet.c` `readParamData` | **Yes** |
| `<FILE_NAME>.clim` | `sipnet.c` `readClimData` | **Yes** |
| `<EVENTS_PREFIX>.in` (default `events.in`) | `events.c` `readEventData` | No — `access()` guarded |
| `RESTART_IN` | `restart.c` | No — only when set |

There is **no phenology, LAI, NDVI or observation-data input**. See gotcha 8.

Restart checkpoints (`RESTART_IN` / `RESTART_OUT`) are new at this pin and
serialise `envi`, `trackers`, `phenologyTrackers` and `event_trackers` for
segmented runs. pySIPNET does not use them yet; they are the obvious route to
efficient sequential data assimilation.

### `sipnet.in` (run configuration)

Parsed into a global `Context` (`sipnet/src/common/context.c`). Separators are
` \t=:`, comment character `!`. Keys are normalised by `nameToKey` —
non-alphanumerics stripped, lowercased — so `FILE_NAME`, `file-name` and
`fileName` are the same key. Unknown keys produce a log line and are
**ignored**, which is why `tests/test_sipnet_in.py` reads SIPNET's output back
and fails on `"ignoring input file parameter"`.

pySIPNET generates this file in `_render_sipnet_in()` (`pysipnet/runner.py`) and
writes **every** key explicitly, including ones matching SIPNET's defaults, so a
saved run does not change meaning if a future default changes.

| Key | Default | Notes |
|---|---|---|
| `FILE_NAME` | `sipnet` | prefix for `.param`, `.clim`, `.out` |
| `EVENTS` | 1 | we always set it explicitly |
| `PRINT_HEADER` | 1 | we always set 1; the output reader matches columns by name |
| `DO_MAIN_OUTPUT` | 1 | |
| `DO_SINGLE_OUTPUTS` | 0 | **plural** — see below |
| `DUMP_CONFIG` | 0 | writes SIPNET's resolved config; used by our tests |
| `QUIET` | 0 | |
| `INPUT_FILE` | `sipnet.in` | |
| `EVENTS_PREFIX` | `events` | |
| `RESTART_IN` / `RESTART_OUT` | unset | |
| **Model flags** | see below | `ModelFlags.to_config_keys()` renders these |

Model flags: `GDD` (1), `SNOW` (1), `WATER_HRESP` (1), `GROWTH_RESP` (0),
`LEAF_WATER` (0), `LITTER_POOL` (0), `SOIL_PHENOL` (0), `NITROGEN_CYCLE` (0),
`ANAEROBIC` (0), `FLOODING` (0).

`validateContext()` rejects three combinations, mirrored in `ModelFlags`:
`GDD` with `SOIL_PHENOL`; `ANAEROBIC` without `WATER_HRESP`; `NITROGEN_CYCLE`
without both `LITTER_POOL` and `ANAEROBIC`.

**`DO_SINGLE_OUTPUTS` must be plural.** SIPNET derives each config key from
the C *field* name via `nameToKey`, not from the label it prints for the
setting. `CREATE_INT_CONTEXT(doSingleOutputs, "DO_SINGLE_OUTPUT", ...)` gives
the field `doSingleOutputs` → key `dosingleoutputs`, while the printed label
and SIPNET's own docs say `DO_SINGLE_OUTPUT` → `dosingleoutput`. The singular
form is silently ignored. `tests/test_sipnet_in.py` caught this the moment the
key was added; it is the reason that test reads SIPNET's log back.

Precedence is `DEFAULT < INPUT_FILE < COMMAND_LINE < CALCULATED`.
`PARAM_FILE` / `CLIM_FILE` / `OUT_FILE` exist as keys but are overwritten from
`FILE_NAME` at `CTX_CALCULATED`, so **setting them has no effect**.

### `.param` file

Two+ columns: `parameter_name  value [ignored...]`. Extra columns produce a
warning and are otherwise accepted. Comment character `!`. Order-independent.
Name lookup is case-insensitive (`strcasecmp` in `locateParam`).

Hard errors: a value of `*` (the old spatially-varying marker), and any
parameter given twice. An **unrecognised name is only a warning**, which is why
`tests/test_param_file_contract.py` runs the binary and fails on any
`"Unknown param"` line — a renamed parameter would otherwise silently stop
having any effect.

The nine obsolete placeholder parameters that older SIPNET required but ignored
are gone at this pin, and so is the `_OBSOLETE_DEFAULTS` workaround that wrote
them.

**Important unit gotcha:** several parameters are given as **per-year** rates in
the file and converted to per-day internally: `baseVegResp`, `baseSoilResp`,
`litterBreakdownRate`, `woodTurnoverRate`, `leafTurnoverRate`,
`fineRootTurnoverRate`, `coarseRootTurnoverRate`, `baseFineRootResp`,
`baseCoarseRootResp`. Always specify per-year; the writer passes them through
unchanged.

### `.clim` file (12 columns, or 14 for older files)

`NUM_CLIM_FILE_COLS` is **12**, with `NUM_CLIM_FILE_COLS_LEGACY` = 14. SIPNET
counts the columns on the first line to decide which layout it has; anything
other than 12 or 14 is a hard error. **A 13-column file is rejected** at this
pin, though the previous pin accepted it.

No header, no comment character. Whitespace-delimited, one row per timestep.

| Col (12) | Col (14) | Name | File units | Internal conversion |
|---|---|---|---|---|
| — | 1 | loc | integer | ignored, with a log line |
| 1 | 2 | year | integer | — |
| 2 | 3 | day | integer (1 = Jan 1) | — |
| 3 | 4 | time | fractional hours | — |
| 4 | 5 | length | **days**; negative means −seconds | `length / -86400` |
| 5 | 6 | tair | °C | — |
| 6 | 7 | tsoil | °C | — |
| 7 | 8 | par | Einstein m⁻², summed over the step | `× 1/length` |
| 8 | 9 | precip | mm | `× 0.1` → cm |
| 9 | 10 | vpd | Pa | `× 0.001` → kPa, clamped ≥ 1e-6 |
| 10 | 11 | vpdSoil | Pa | `× 0.001` → kPa |
| 11 | 12 | vPress | Pa | `× 0.001` → kPa |
| 12 | 13 | wspd | m s⁻¹ | clamped ≥ 1e-6 |
| — | 14 | soilWetness | fraction | ignored |

**pySIPNET's writer still emits 14 columns**, which SIPNET accepts via the
legacy path with an informational log line. Switching to 12 is optional
cleanup; `pysipnet/io/clim_io.py` already supports both. In that module the
layouts are named by column count (`_N_COLS_12/13/14`, `_write_14_column`), and
the public discriminator is `n_columns: Literal[12, 14]` — never "v1"/"v2",
since one SIPNET version reads both.

Other format details, all still true:

- **Multi-location files are a hard error** if any row's loc differs from the
  first row's.
- **`length < 0` means seconds.** Undocumented outside the source.
- **First line via `getline`, rest via `fscanf`.** A leading blank line is
  fatal; interior blank lines are tolerated. We forbid both.

SIPNET performs no validation beyond the column count and the loc check — no
NaN, monotonicity or range checks — so our strict pre-write validation is doing
real work.

### `events.in` (optional)

Read from the working directory only when `EVENTS` is on. The name comes
from `EVENTS_PREFIX` (default `events`, giving `events.in` and `events.out`);
pySIPNET leaves that at the default. A **missing file
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
| `till` | tillageEffect |

An unknown event keyword is a hard error. A wrong param **count** is not, in
one direction: `sscanf` stops once it has filled its arguments, so a line with
*too many* values is accepted and the surplus discarded without a word. That is
how a tillage bug survived — pySIPNET wrote the three parameters older SIPNET
took, v2.1.0 reads one, and the litter fraction was silently applied as the
decomposition boost. `pysipnet/events.py` now checks the count exactly on read,
and `tests/test_events_contract.py` asserts every arity against
`NUM_*_PARAMS` in `sipnet/src/sipnet/events.h` so it cannot drift again.

Events produce an `events.out` file alongside the main output, recording what
SIPNET actually applied — which is what makes this contract testable.

### SIPNET Output

35 columns, header row present (we always set `PRINT_HEADER = 1`). **The
`Notes:` preamble line that older SIPNET wrote above the header is gone at this
pin.** `pysipnet/io/output_reader.py` detects the header by content rather than
by looking for that line — a first field that does not parse as a number means
the line is a header — so all three layouts read: header-only, `Notes:`+header,
and no header.

Columns are always all present; a switched-off process writes zeros rather than
omitting its column. New at this pin: `woodCreation`, `nppStorage`, the
nitrogen group (`minN`, `soilOrgN`, `litterN`, `n2o`, `nLeaching`, `nFixation`,
`nUptake`), `ch4`, and `plantStorageN`. Gone relative to v2.1.0: `bcdeltaC` and `bcdeltaN`.

SIPNET audits its own carbon and nitrogen closure, but reports the result as a
log warning from `checkBalance()` rather than as output columns — v2.1.0 wrote
`bcdeltaC`/`bcdeltaN` for this and no longer does.
`tests/test_integration.py::TestMassBalance` reads the log.

## SIPNET Parameters — Full Grouped List

Parameters are grouped as they appear in the Python data model. All initial conditions are also in the `.param` file (SIPNET makes no distinction).

The authoritative source is the `initializeOneModelParam` block in
`src/sipnet/sipnet.c` (`readParamData`). Its third argument is the required flag:
`1` = always required, `0` = optional, and a `ctx.*` expression = **required
only when that runtime flag is on**. No `OBSOLETE_PARAM` entries remain (gotcha 7).

### Initial Conditions
`plantWoodInit` (g C m⁻²), `laiInit` (m² m⁻²), `litterInit` (g C m⁻²), `soilInit` (g C m⁻²), `soilWFracInit` (fraction of WHC), `snowInit` (cm water equiv.), `fineRootFrac` (fraction), `coarseRootFrac` (fraction)

### Photosynthesis
`aMax` (nmol CO₂ g⁻¹ leaf s⁻¹), `aMaxFrac`, `baseFolRespFrac`, `psnTMin` (°C), `psnTOpt` (°C), `dVpdSlope` (kPa⁻¹), `dVpdExp`, `halfSatPar` (Einstein m⁻² ground day⁻¹), `attenuation`

### Phenology
`leafOnDay` (DOY), `gddLeafOn` (°C·day), `soilTempLeafOn` (°C), `leafOffDay` (DOY), `leafGrowth` (g C m⁻²), `fracLeafFall`, `leafAllocation`, `leafTurnoverRate` (year⁻¹)

The three leaf-on parameters are **mutually exclusive and flag-selected**, not all required:
`leafOnDay` is registered `!((ctx.gdd) || (ctx.soilPhenol))`, `gddLeafOn` is
`ctx.gdd`, and `soilTempLeafOn` is `ctx.soilPhenol`. `gdd` is on by default, so **`gddLeafOn` is required and `leafOnDay` is optional**. `leafOffDay` is always required
(leaf-off is always fixed-DOY). See gotcha 8.

### Leaf
`leafCSpWt` (g C m⁻² leaf), `cFracLeaf` — both always required. `leafCSpWt` converts
`laiInit` to the initial leaf C pool (`envi.plantLeafC = laiInit × leafCSpWt`) and relates
to SLA as `leafCSpWt = cFracLeaf / SLA`.

### Autotrophic Respiration
`baseVegResp` (year⁻¹), `vegRespQ10`, `growthRespFrac`, `frozenSoilFolREff`, `frozenSoilThreshold` (°C), `baseFineRootResp` (year⁻¹), `baseCoarseRootResp` (year⁻¹), `fineRootQ10`, `coarseRootQ10`

`growthRespFrac` is required only when `growth_resp` is on (off by default).

### Soil Respiration
`baseSoilResp` (year⁻¹), `soilRespQ10`, `soilRespMoistEffect`, `litterBreakdownRate` (year⁻¹), `fracLitterRespired`

`soilRespMoistEffect` is required when `water_hresp` is on (the default).
`litterBreakdownRate` and `fracLitterRespired` are required when `litter_pool`
is on — off in `ModelFlags.standard()`, on in `ModelFlags.forest()`.

### Allocation
`fineRootAllocation`, `woodAllocation`, `fineRootTurnoverRate` (year⁻¹), `coarseRootTurnoverRate` (year⁻¹), `woodTurnoverRate` (year⁻¹)

### Water
`waterRemoveFrac` (day⁻¹), `frozenSoilEff`, `wueConst`, `soilWHC` (cm), `immedEvapFrac`, `fastFlowFrac`, `snowMelt` (cm °C⁻¹ day⁻¹), `rdConst`, `rSoilConst1`, `rSoilConst2`, `leafPoolDepth`

`snowMelt` is required when `snow` is on (the default). `leafPoolDepth` is required
when `leaf_water` is on (off by default).

### Optional nitrogen, methane and flooding parameters

Registered at this pin but required only under their flag, and **not yet in the
Python model** because the flags default off:

- `NITROGEN_CYCLE`: `mineralNInit`, `soilOrgNInit`, `litterOrgNInit`,
  `nVolatilizationFrac`, `nLeachingFrac`, `leafCN`, `woodCN`, `fineRootCN`,
  `kCN`, `nFixationFracMax`, `halfNFixationMax`
- `ANAEROBIC`: `anaerobicDecompRate`, `anaerobicTransExp`, `soilMethaneRate`,
  `litterMethaneRate`
- `ANAEROBIC` **or** `NITROGEN_CYCLE`: `fAnoxia` — registered
  `ctx.anaerobic || ctx.nitrogenCycle`, so either flag alone demands it
- `FLOODING`: `waterDrainFrac`

**These three flags are refused by `ModelFlags`,** because accepting them
would produce a run that fails inside SIPNET with "Did not find required
parameter" — exactly what design principle 2 says the data model should
prevent. `UNSUPPORTED_FLAGS` in `pysipnet/parameters/model.py` holds the flag,
a plain-language description, and the parameters SIPNET would demand; the error
message reproduces all of it, so a caller learns what is missing rather than
merely that something is.

The refusal is checked *before* the SIPNET-mirroring restrictions, so
`nitrogen_cycle=True` reports "not supported yet" rather than sending the
caller to set `litter_pool` and `anaerobic` — advice that would not have
helped.

To enable one: model its parameters, mark them required under the flag in
`validate_for_flags`, and delete its entry from `UNSUPPORTED_FLAGS`. A test
asserts every name in that table is a parameter the pinned SIPNET actually
registers **and** is still absent from our model, so the table cannot go stale
in either direction.

Adding these is tracked in issue #26. Do it as its own opt-in group so the
default parameter set stays as small as it is now, and start with `flooding` —
one parameter, no flag dependencies, so it exercises the whole path with the
least in the way. Note `fAnoxia` **does** exist
at this pin, contradicting the earlier note that it was v2-only.

The `MICROBES` and `SOIL_QUALITY` processes were removed upstream, so their
parameters are gone entirely rather than merely unreachable.

### Required-count summary

Required-ness is expressed at runtime now: `initializeOneModelParam` takes a
`ctx.*` expression rather than a compile-time macro, so the same binary demands
different parameters depending on `sipnet.in`. `ModelFlags` mirrors this in
`SIPNETParameters.validate_for_flags`.

- **49 parameters are unconditionally required.**
- Default flags (`gdd`, `snow`, `water_hresp`) add `gddLeafOn`, `snowMelt` and
  `soilRespMoistEffect` → **52 required**. The writer emits every parameter
  that is not `None`, so the actual line count is 51 plus whichever optional
  ones you set.
- `litter_pool` adds `litterBreakdownRate` and `fracLitterRespired` → **54**.
- No obsolete placeholders. The previous pin needed 64 lines for the equivalent
  configuration.

## Key Technical Gotchas

1. **`psnTMax` and `coarseRootAllocation` are derived**, not specified. `psnTMax = 2×psnTOpt − psnTMin`. `coarseRootAllocation = 1 − leafAllocation − fineRootAllocation − woodAllocation`.

2. **PAR units scale with timestep.** Climate file PAR is total Einsteins m⁻² for the whole timestep interval. When converting from instantaneous flux measurements, multiply by `length` (in days × 86400 seconds/day).

3. **SIPNET expects files in the current working directory.** The runner writes all inputs to a fresh temp dir per run and executes the binary there. The generated `sipnet.in` sets `FILE_NAME = sipnet`, so SIPNET reads `sipnet.param` and `sipnet.clim` and writes `sipnet.out`.

4. **Model options are runtime, and they change which parameters are required.** All ten (`GDD`, `SNOW`, `WATER_HRESP`, `GROWTH_RESP`, `LEAF_WATER`, `LITTER_POOL`, `SOIL_PHENOL`, `NITROGEN_CYCLE`, `ANAEROBIC`, `FLOODING`) are set in `sipnet.in`. One binary, `make sipnet`, no `-D` flags, no source patch. Because they change the required parameter set, the flags are part of the run specification, not a build detail — which is why `ModelFlags` is serialised into `RunConfig` and `RunProvenance`. SIPNET rejects three combinations (`validateContext()`); `ModelFlags` rejects them first.

5. **No missing climate values.** Climate validation must be strict: every row must be complete, timesteps must be monotonically increasing, and the start/end dates must bracket the intended simulation period.

6. **Events file.** SIPNET defaults to `EVENTS=1` (looks for `events.in`). The runner writes `EVENTS = 0` in `sipnet.in` to suppress this for basic runs. Note that this is belt-and-braces: a *missing* `events.in` is already harmless (`access` guard, `logInfo` only), so `EVENTS = 0` mainly guards against a stale file in the working directory. When events are used, the file must be in the working directory.

7. **The `OBSOLETE_PARAM` workaround is gone.** Historical note, because it explains a chunk of deleted code. Older SIPNET declared nine parameters obsolete — read and then ignored — but its `checkAllRead()` tested `if (param->isRequired)`, which is truthy for the `-1` obsolete marker, so it **errored if they were absent**. pySIPNET appended fixed placeholders to every param file to satisfy that. SIPNET deleted the nine at v2.0.0, so `_OBSOLETE_DEFAULTS` is gone too. Writing them now would produce nine `"Unknown param"` log lines per run, which is harmless in itself but would bury a real unknown parameter; `tests/test_param_file_contract.py` asserts a clean run logs none at all.

8. **Phenology is parameter-driven — there is no phenology data input.** `pastLeafGrowth()` selects one of three leaf-on triggers from the runtime flags: `cumulativeGdd >= gddLeafOn` when `ctx.gdd`, `climate->tsoil >= soilTempLeafOn` when `ctx.soilPhenol`, or `currTime >= leafOnDay` otherwise. Growing degree days are accumulated inside `readClimData` from `tair × length` and reset on year change — **never read from a file**. Leaf-off is always `leafOffDay`. `laiInit` only sets the initial leaf C pool (`envi.plantLeafC = laiInit × leafCSpWt`); LAI is prognostic thereafter. "Phenology data" enters SIPNET only by two indirect routes, **both of which resolve to the `.param` file before the binary runs** (verified against PEcAn source, not inferred):

    - `PEcAn.data.remote::extract_phenology_MODIS()` pulls MODIS **MCD12Q2** `MidGreenup` / `MidGreendown` bands, QA-filters them, and writes a CSV (`year, site_id, lat, lon, leafonday, leafoffday, leafon_qa, leafoff_qa`). `PEcAn.SIPNET::write.config.SIPNET()` then reads it via `settings$run$inputs$leaf_phenology$path` and **overwrites `leafOnDay` / `leafOffDay` in the param table** (fallback chain: matching year → site mean across years → param-file default; applied only `if (leafOffDay > leafOnDay)`).
    - `gddLeafOn` is set from a PFT **trait** named `GDD`, i.e. the calibration/prior route — *not* from remote sensing.

    Upstream of our pin this changes: see gotcha 9.

9. **Prescribed phenology is available at this pin, and pySIPNET does not use it yet.** Via `events.in`, not a new file type. SIPNET accepts `leafon` and `leafoff` events with **zero** parameters, so a line is just `year day leafon`. They are real input keywords (`eventStringToType` in `events.c`). Two properties matter before wiring them up: they are **mutually exclusive with every calculated mechanism** — `checkForCalculatedLeafEvents()` exits with `EXIT_CODE_BAD_PARAMETER_VALUE` if `ctx.gdd || ctx.soilPhenol || params.leafOnDay > 0 || params.leafOffDay > 0` — and they prescribe **timing only**: the flux is the same `params.leafGrowth / climLen` the calculated trigger would apply, so `leafGrowth` and `fracLeafFall` remain fitted parameters either way. PEcAn already emits them (`write.events.SIPNET.R`) and, when it sees them, zeroes `leafOnDay`/`leafOffDay`/`gddLeafOn` and ignores its `leaf_phenology` CSV. Note upstream's own docs still list event types as only `plant`/`harv`/`till`/`fert`/`irrig` — read `eventStringToType`, not the table. Tracked in issue #25; `tests/test_events_contract.py` pins the set of unmodelled types so a fourth cannot appear unnoticed.

10. **`leafOnDay = 0` now means "disabled", and at our previous pin it did not.** `pastLeafGrowth` and `pastLeafFall` are gated on `params.leafOnDay > 0` / `params.leafOffDay > 0`, so zero switches the trigger off — which is how PEcAn disables internal scheduling when it supplies leaf events instead. This gating arrived with the prescribed-phenology work and is **absent from `v2.1.0`**, where `pastLeafFall` compared unconditionally and `leafOffDay = 0` fired leaf fall on the first timestep of every year. Mentioned because the previous pin behaved the other way, and a parameter set carried over from it will now behave differently.

11. **A new required parameter arrived with the leaf events: `leafOnReallocFrac`.** Leaf-out has to take carbon from somewhere, and this caps how much of `plantWoodC + coarseRootC` it may draw on. SIPNET scales the transfer down if demand exceeds `(plantWoodC + coarseRootC) × leafOnReallocFrac`. Required unconditionally, so every param file needs it; upstream's Niwot fixture uses `0.2`.

## File Structure

```
pySIPNET/
├── sipnet/                       # git submodule — SIPNET source, pinned to v2.2.0-alpha.1
├── Makefile                      # `make sipnet`, `make sipnet-download`
├── pysipnet/
│   ├── version.py                # pinned commit, target version, clim column counts
│   ├── build.py                  # compile or download, locate, verify the binary
│   ├── parameters/
│   │   ├── base.py               # ParameterSpec, param_field, domains (version-agnostic)
│   │   └── model.py              # ModelFlags and SIPNETParameters
│   ├── climate.py                # ClimateDrivers + validation
│   ├── events.py                 # management events (arity checked against SIPNET)
│   ├── io/
│   │   ├── param_io.py           # read/write .param
│   │   ├── clim_io.py            # read/write .clim (12- and 14-column layouts)
│   │   └── output_reader.py      # read .out, header detected by content
│   ├── runner.py                 # SIPNETRunner, _render_sipnet_in
│   ├── model.py                  # SIPNETModel — high-level callable interface
│   ├── config.py                 # RunConfig — a saveable run specification
│   ├── result.py                 # SIPNETResult, RunProvenance
│   ├── output.py                 # SIPNETOutput — lazy/eager output access
│   ├── ensemble.py               # helpers for driving many runs
│   └── viz.py                    # plotting
├── tests/
│   ├── test_sipnet_in.py         # the sipnet.in contract, incl. SIPNET's config dump
│   ├── test_param_file_contract.py  # the .param contract across flag combinations
│   ├── test_events_contract.py   # the events.in contract, incl. arities
│   ├── test_download.py          # prebuilt-binary download and its verification
│   ├── test_fidelity.py          # wrapper output == bare binary output
│   ├── test_golden.py            # frozen numeric baseline
│   ├── test_build.py             # binary/pin agreement
│   └── fixtures/
│       ├── niwot_reference/      # SIPNET-authored .param and .clim
│       └── golden/               # frozen output baseline
├── data/                         # (gitignored) sample data
├── docs/
└── CLAUDE.md                     # this file
```

Note there is no `patches/` directory and no per-option build targets; both
belonged to the pre-v2.0.0 compile-time-flag era.

### Test layers, and what each one would catch

Worth knowing which test to look at when something breaks:

- `test_build.py` — the binary matches the recorded pin. Catches a stale
  binary or a half-finished version bump.
- `test_sipnet_in.py` — SIPNET understood every config key we wrote, proven by
  reading its own resolved-config dump. Catches a key silently ignored.
- `test_param_file_contract.py` — SIPNET recognised every parameter name and
  found everything it required, across six flag combinations. Catches a
  renamed or dropped parameter.
- `test_fidelity.py` — driving SIPNET through the wrapper gives the same
  numbers as running the bare binary by hand. Catches distortion anywhere in
  the writers, runner or parser. Version-independent, since both sides run the
  same binary.
- `test_golden.py` — the numbers themselves match a checked-in baseline.
  Catches an unintended model change that the wrapper and binary would still
  agree about. Regenerate deliberately with `python -m tests.test_golden`.
- `test_integration.py` — end-to-end behaviour, including that flags visibly
  change results and that SIPNET's own mass-balance errors stay near zero.

The first three exist because the failure they catch is **silent**: SIPNET logs
an ignored key or unknown parameter and carries on, so the run succeeds and the
output looks plausible. Any new input we start writing should get the same
treatment.

## Development Conventions

- **Python ≥ 3.11**
- **Pydantic v2** for all data models (parameter validation, units enforcement)
- **pandas** for climate time series and output; **xarray** as an optional output format
- **NumPy** for numerical operations
- **No comments unless the WHY is non-obvious.** Well-named identifiers are preferred.
- **Type hints everywhere.**
- All file I/O is in the `pysipnet/io/` subpackage. The rest of the package never touches the filesystem directly.
- Tests use real SIPNET binaries where possible (integration tests), not mocks.
