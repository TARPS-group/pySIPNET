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

### Why a tag rather than a bare commit

Newer commits exist on `master`. None of them change anything pySIPNET uses:
the only model change since this tag is inside `if (ctx.nitrogenCycle)`, which
`ModelFlags` refuses, and the rest touches `tools/`, which we do not use.

A tag is a name everyone can pin to and cite; a loose commit is not. To move
closer to the tip, ask upstream to tag the commit rather than pinning it
directly.

### The trap: a pre-release tag can move

A tag is not immutable. If upstream re-tags `v2.2.0-alpha.1`,
`tests/test_build.py` fails, because it asserts the submodule sits at
`SIPNET_PINNED_COMMIT` — a commit, which cannot move. That is the intended
behavior: loud, not silent.

### The trap: the numeric version lags the tag

`version.h` at this tag still declares `NUMERIC_VERSION "2.1.0"`, so:

    $ sipnet --version
    SIPNET version 2.1.0 (v2.2.0-alpha.1)

The numeric version therefore **cannot identify the pin** — a v2.1.0 binary
reports the same number. The parenthesized part is `git describe --tags`,
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
  `pysipnet.build.download_sipnet()` (or `pysipnet install-sipnet`) fetches
  one. The archive SHA-256 is pinned in `SIPNET_RELEASE_ASSETS` and checked
  before extraction, archive members are inspected so a hostile path cannot
  escape the destination, and the installed binary is asked for its version
  afterwards. A failure at any of those steps leaves nothing installed.
  Refresh the digests whenever `SIPNET_PINNED_COMMIT` moves; a stale one fails
  loudly, which is what you want. `pytest -m network` checks them against the
  published release.
- **The published binaries only run on recent systems.** Read from the
  binaries, not upstream's docs: the macOS one declares `minos 26.0` in its
  `LC_BUILD_VERSION` load command, and the Linux one references `GLIBC_2.34`.
  Each requirement is recorded once, as the platform tag of the wheel that
  bundles the binary (`SIPNET_WHEEL_PLATFORM_TAGS`: `macosx_26_0_arm64`,
  `manylinux_2_34_x86_64`). pip installs a wheel only where its tag is
  accepted, and `prebuilt_unavailable_reason()` asks `packaging.tags.sys_tags()`
  the same question before downloading, so `install_sipnet()` compiles instead
  of fetching something dyld or ld.so would refuse — correct on musl and
  Rosetta without a hand-written branch. `pytest -m network` re-derives both
  tags from the archives.

### Pinning mechanics

The submodule lives at `sipnet/`. `make sipnet` compiles it and copies the
result to `.sipnet_cache/sipnet` (gitignored). Never commit the binary.
`tests/test_build.py` asserts that the submodule is at
`SIPNET_PINNED_COMMIT` and that the compiled binary reports
`SIPNET_PINNED_TAG`, so a stale binary or half-finished bump fails loudly
rather than producing quietly wrong output. The same check runs **at
runtime**: `SIPNETRunner._check_binary()` calls
`verify_binary_matches_pin()` before the first run (cached per file, one
`--version` call per process) and raises `BinaryVersionError` on a mismatch,
so a user who never runs the tests still cannot drive the wrong SIPNET.
`verify_binary=False` is the escape hatch. `run()` resolves the binary path
once and uses that one path for the check, the exec and the provenance, so
the three always describe the same file. `_pinned_version_or_raise()` is the
single statement of "the right binary", shared by the runtime check and by
every install route (which re-raise it as `DownloadError` or `BuildError`).

### Where the binary comes from

Installing the Python package does not install SIPNET, and nothing in the
library fetches one on its own. `pysipnet/build.py` owns the whole story:

- **Search order** (`binary_candidates()`, first existing wins):
  `$PYSIPNET_BINARY` (resolved to an absolute path) → `pysipnet/bin/sipnet`
  bundled in a platform wheel → `.sipnet_cache/<commit12>/sipnet` when
  `in_source_tree()` (repo `Makefile` and `sipnet/` beside `pyproject.toml`;
  true for editable installs) → the per-user cache,
  `platformdirs.user_cache_dir("pysipnet")/sipnet/<commit12>/sipnet`, root
  replaceable by `$PYSIPNET_CACHE_DIR`. **Both caches are named by the pinned
  commit** (`PINNED_CACHE_SUBDIR`; the Makefile reads the same 12 characters
  out of `version.py` with `sed`), so a pin bump lands in an empty directory
  on every route and `force` means only "replace the binary for this pin".
  The runtime check exists for `$PYSIPNET_BINARY`, the one place that cannot
  be keyed.
- **Acquisition** (`install_sipnet()`, the `pysipnet install-sipnet`
  command): an existing binary is returned only if it passes the pin check;
  otherwise download when `prebuilt_unavailable_reason()` is `None`, else
  compile. SIPNET's Makefile hard-codes `CC=gcc`, so `gcc` is the tool the
  pre-check asks for. `build_sipnet()` runs `make sipnet` in a checkout; outside one it
  fetches the pinned commit with git (by commit, not a GitHub tarball, whose
  bytes are generated on request and have changed before), verifies
  `rev-parse HEAD`, and runs SIPNET's Makefile with `GIT_HASH=<tag>` so the
  binary is stamped even from a shallow fetch. Both routes stage the binary
  beside the target, check its version, and `os.replace` it into place.
- **Bundled wheels**: `hatch_build.py` is a hatch build hook that, when
  `$PYSIPNET_BUNDLE_SIPNET=<platform key>`, force-includes the binary
  `pysipnet stage-bundle <key>` put at `pysipnet/bin/<key>/sipnet` as
  `pysipnet/bin/sipnet` and tags the wheel from `SIPNET_WHEEL_PLATFORM_TAGS`.
  Staging per platform means the tag and the payload are chosen by the same
  string and cannot disagree. It downloads nothing itself: the build env has
  only hatchling, so it loads `pysipnet/version.py` by path (that module must
  stay import-free, which `test_bundle_hook.py` asserts) and leaves
  verification to the staging step. `pysipnet/bin/` is gitignored and excluded from the pure wheel.
  `.github/workflows/wheels.yml` builds sdist, pure wheel and both platform
  wheels, smoke-tests each on a runner that can execute its binary, and on a
  `v*` tag attaches them to a **draft** release. No PyPI publishing yet.
- `SIPNETRunner(binary=...)` names a binary outright, resolved to an absolute
  path because the run's cwd is the temp workdir; `cache_dir=` is the older
  spelling and is normalized to `binary=<cache_dir>/sipnet` in `__init__`.

When moving to a newer SIPNET, expect to touch: the required-parameter set,
the output column list, the `sipnet.in` keys, and the golden fixtures. Bump
`SIPNET_PINNED_COMMIT`, `SIPNET_PINNED_TAG` and `SIPNET_NUMERIC_VERSION`
together, refresh `SIPNET_RELEASE_ASSETS`, and rebuild with
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

5. **Version-forward design.** Version-specific logic (file format differences, available parameters) is isolated behind version adapters so the public API stays stable when the pinned SIPNET version moves.

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
serialize `envi`, `trackers`, `phenologyTrackers` and `event_trackers` for
segmented runs. pySIPNET does not use them yet; they are the obvious route to
efficient sequential data assimilation.

### `sipnet.in` (run configuration)

Parsed into a global `Context` (`sipnet/src/common/context.c`). Separators are
` \t=:`, comment character `!`. Keys are normalized by `nameToKey` —
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
`ANAEROBIC` (0), `FLOODING` (0), `CARBON_SATURATION` (0).

`validateContext()` rejects four combinations, mirrored in `ModelFlags`:
`GDD` with `SOIL_PHENOL`; `ANAEROBIC` without `WATER_HRESP`; `NITROGEN_CYCLE`
without both `LITTER_POOL` and `ANAEROBIC`; `CARBON_SATURATION` without
`LITTER_POOL`.

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
parameter given twice. An **unrecognized name is only a warning**, which is why
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

The 14-column layout is the pre-v2.0.0 one: SIPNET v2.0.0 removed multi-site
runs and the soil-wetness mode (upstream #92, #127), which left the leading
`loc` and trailing `soilWetness` columns with nothing to do, and 12 columns
became the standard. Both carry the same 12 values;
`tests/test_clim_layout_contract.py` runs the binary on both and gets identical
output.

No header, no comment character. Whitespace-delimited, one row per timestep.

The table uses SIPNET's column names. The Python column names are the
registry names in `CLIMATE_VARIABLES` (`pysipnet/variables.py`):
`air_temperature` (tair), `soil_temperature` (tsoil),
`photosynthetically_active_radiation` (par), `precipitation` (precip),
`vapor_pressure_deficit` (vpd), `soil_vapor_pressure_deficit` (vpdSoil),
`vapor_pressure` (vPress), `wind_speed` (wspd), and the shared time columns
`year`, `day_of_year`, `hour_of_day`, `time_step_length`. Each spec records
the file units and SIPNET's internal conversion below as `units` /
`internal_units` / `internal_conversion`. `ClimateDrivers.from_dataframe`
accepts the old short names and SIPNET's names as aliases and renames them;
`ClimateDrivers.xarray` gives the same 1-D `time` layout as outputs. The
docs page `reference/climate-drivers.md` is generated from the registry.

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

**pySIPNET reads the layout from the file, exactly as SIPNET does, and never
asks the caller.** `detect_clim_layout()` in `pysipnet/io/clim_io.py` counts the
first line's fields and accepts 12 or 14; every reader (`read_clim_file`,
`peek_clim_file`, `from_file`, `from_path`, `ClimateDrivers(source_path=...)`)
uses it, and none takes an `n_columns`. It refuses what SIPNET refuses: 13 or
any other count, a leading blank line, and a 14-column file whose `loc`
changes between rows. `tests/test_clim_layout_contract.py` runs the binary on
each variant and asserts that pySIPNET accepts exactly what SIPNET accepts.

This replaced a caller-supplied `n_columns` on every reader, defaulting to 14,
which was the root cause of `RunConfig.load()` failing on a config saved with
12-column drivers (the layout was saved in the file but never asked of it), and
which let pySIPNET read a 13-column file that `from_path` would then stage
unchanged for SIPNET to refuse.

`ClimateDrivers.n_columns` is now only the layout the drivers are *written*
in: **12 for drivers built in memory** (the standard; pass `n_columns=14` for
the legacy layout), and **the file's own layout for drivers read from one**,
because the runner stages a file-backed climate by copying it unchanged and
`RunConfig` rewrites it in the same layout. A 14-column file's `loc` becomes
`ClimateDrivers.loc`. The layouts are named by column count (`_N_COLS_12`,
`_N_COLS_14`, `ClimLayout = Literal[12, 14]`), never "v1"/"v2", since one
SIPNET version reads both.

Other format details, all still true:

- **Multi-location files are a hard error** if any row's loc differs from the
  first row's.
- **`length < 0` means seconds.** Undocumented outside the source.
- **First line via `getline`, rest via `fscanf`.** A leading blank line is
  fatal; interior blank lines are tolerated. pySIPNET behaves the same (its
  reader refuses the first and skips the second), and its writer writes
  neither.

SIPNET performs no validation beyond the column count and the loc check — no
NaN, monotonicity or range checks — so our strict pre-write validation is doing
real work.

**SIPNET has no time zone, and never checks labels against lengths.**
`readClimData` reads every row into a linked list and never compares
`time + length` with the next row's `time`; all physics is integrated on
`length`. There is no solar geometry. `climate->time` is read only by
`outputState`, by the `leafOnDay`/`leafOffDay` comparisons in
`pastLeafGrowth`/`pastLeafFall`, by the restart boundary checks in
`restart.c`, and in log messages. So a row's labels are the start of its step
**on whatever clock the climate drivers use**, and it is the `ClimateDrivers`
that declare which: `time_zone="UTC"` or a fixed offset `"UTC±HH:MM"`,
undeclared by default, metadata only (nothing is converted). Named zones are
refused because daylight saving time would make the labels jump. The
declaration is written to the `time` attributes of `ClimateDrivers.xarray` and
of every output Dataset run on those drivers, and into `RunConfig`'s
`config.json`, since a `.clim` file cannot hold it.

`ClimateDrivers.validate()` checks labels against lengths
(`dataset.check_step_continuity`), because the files that prompted it — ERA5
drivers whose `time` column was `linspace(0, 24n - 1, 8n) % 24`, drifting
2.46 s per 3-hour step and 2 h by 31 December — ran through SIPNET silently:

- **overlap** (next row starts more than 60 s before this one ends): error;
- **drift** (a label more than 5 min from the previous gap's row plus the
  running sum of the declared lengths): error. A per-step tolerance loose
  enough for rounded lengths cannot see 2.46 s per step; the running sum does,
  after 122 steps. It also refuses hourly lengths written as `0.042`, which
  integrate 24.19 h of forcing per day;
- **gap** (next row starts more than 60 s after this one ends): a warning. It
  restarts the drift reconstruction and shows as a gap between `time_bounds`.

Niwot's rounded lengths (`0.292` for exactly 7 h) peak at 43.2 s per step and
100.8 s cumulative over 800 rows; `tests/test_time_axis.py` pins both.

**Validation runs exactly once per set of data, when it is loaded.** Every
path that puts data into a `ClimateDrivers` goes through `__init__(data=...)`,
which normalizes the columns (aliases renamed, extras dropped, copied) and runs
the checks: `from_dataframe`, `from_file`, the direct constructor, and a
`from_path` instance's first read of `.pandas`. Nothing downstream repeats
them — not the runner, not `climate.xarray`, not an output's axis — so the
frame behind `.pandas` must not be modified in place. `validate()` on
unloaded data just loads it; on loaded data it re-runs the checks, on request.
`head(n)` returns a prefix without re-checking, since every check holds for a
prefix of a checked record (the drift reconstruction starts from the same
row). **`from_path` defers validation along with the read**: the runner
copies or symlinks the file without reading it, so a file that fails the
checks still runs, and the failure surfaces on the first read, typically
`result.outputs["nee"]`. Call `climate.validate()` to check it up front.

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
*too many* values is accepted and the surplus discarded without a word. A
mismatched arity therefore applies the wrong quantity silently.
`pysipnet/events.py` checks the count exactly on read, and
`tests/test_events_contract.py` asserts every arity against `NUM_*_PARAMS` in
`sipnet/src/sipnet/events.h`.

Events produce an `events.out` file alongside the main output, recording what
SIPNET actually applied — which is what makes this contract testable.

### SIPNET Output

35 columns, header row present (we always set `PRINT_HEADER = 1`). **The
`Notes:` preamble line that older SIPNET wrote above the header is gone at this
pin.** `pysipnet/io/output_reader.py` detects the header by content rather than
by looking for that line — a first field that does not parse as a number means
the line is a header — so all three layouts read: header-only, `Notes:`+header,
and no header.

**What each column is lives in one place: `pysipnet/variables.py`.** Every
column has a `VariableSpec` with the pySIPNET name, the SIPNET header token,
its kind, units, constituent, description, labels, aliases, flag dependence
and printf precision. `tests/test_variables.py` runs the binary and asserts
its header equals the registry token-for-token, so an upstream column change
fails loudly. The docs page `reference/output-variables.md` is generated from
the registry at build time (`docs/gen_variable_tables.py`).

Facts read from `outputState()` / `updateTrackers()` that the registry encodes
and that SIPNET's own docs get wrong or omit:

- `year`/`day`/`time` are the **start** of the step on the climate drivers'
  clock — SIPNET has none of its own and echoes each climate row's labels, one
  output row per climate row (a restart does not change that: loading a
  checkpoint only validates, it never advances the climate list). `time` is
  printed `%5.2f`, i.e. rounded to 0.01 h. Pools (`envi.*`) are
  written **after** `updateState()`, so they are end-of-step values; trackers
  are `flux × length`, i.e. totals over the step.
- `fluxestranspiration` prints `fluxes.transpiration` **without** `× length`.
  It is a cm day⁻¹ rate, the only rate column → `transpiration_rate`.
- `plantWoodC` prints `getTotalWoodC()` = `plantWoodC + plantCAccountingDelta`
  → `wood_carbon`; `nppStorage` prints `plantCAccountingDelta`, a state that
  can be negative → `wood_storage_carbon`.
- `rSoil` = `rRoot + rh` (root **plus** heterotrophic) → `soil_respiration`.
  SIPNET's docs label it R_H; that is wrong.
- `n2o` = `nVolatilization × length`, total volatilized mineral N, g N →
  `nitrogen_volatilization`. `ch4` is g **C**.
- `soilWetnessFrac` is the two-point mean of start and end wetness.
- `cumNEE` (`totNee`) is never reset and is serialized in restart checkpoints.
- N trackers are assigned only inside `if (ctx.nitrogenCycle)`, so they are
  exactly zero otherwise.
- Every column has a fixed `%w.pf` precision: carbon fluxes 3 decimals, pools
  2, ET 8, N pools 4, `n2o` 6. Recorded as `output_decimals`.

Column names follow the convention **lower-case words, underscores, no
acronyms** (`net_ecosystem_exchange`, not `nee`). Short forms and the old
pySIPNET names are aliases that `resolve_output_variable()` and every
`SIPNETOutput` selection accept; they are never column names. The
time coordinates are `year`, `day_of_year`, `hour_of_day`.

Units are UDUNITS strings (`"g m-2"`, `"cm d-1"`, `"1"`) validated at import by
`pysipnet/units.py`; the substance goes in `constituent` (`"C"`, `"N"`,
`"H2O"`), never in the string, because Pint reads `g C` as gram·coulomb without
error.
`conversion_factor()` / `convert_units()` in the same module take the constituent
as an argument beside each unit string and add the chemistry Pint lacks:
`MOLAR_MASS` for mass↔amount, `DENSITY` (H2O only) for depth↔mass, and
`ATOMS_PER_MOLECULE` (C–CO2, C–CH4, N–N2O) for a change of constituent on an
amount basis. Anything else is refused. The constituent qualifies the
**first** unit token (where `format_units` prints it), which must be an amount,
a mass, or for H2O a depth or volume; whole-string dimensionality would find a
"mass" inside `Pa` or `W` and lose the one in `ug g-1`. The same first-unit
reading gates Pint's plain factor: if the first unit changes kind
(`_kind_change`: amount/mass/depth/volume, depth↔volume exempt as geometry),
the conversion bridges through the tables or is refused, even when the whole
dimensions match. Without that, `umol mol-1` CO2 → `ug g-1` (both
dimensionless) returned 1, and so did `kg kg-1` → `m3 m-3` water content. `photons` is in
`AMOUNT_ONLY_CONSTITUENTS`, and `tests/test_units.py` fails if a registry
declares a constituent the tables do not know. `_CONSTITUENT_TOKENS`, the
substances `validate_units` refuses inside a unit string, is derived from
`MOLAR_MASS`, so a new convertible substance is refused there automatically.
`conversion_factor` is `lru_cache`d, and in all three functions
`to_constituent=None` means "same as the source". There are two ways to apply
it, split because xarray and pandas both keep `attrs` through `values * factor`:
`convert_units(values, units=..., ...)` takes unlabeled values (number, NumPy,
pandas) and **refuses** any xarray object and any pandas object with a `units`
attr, since the result would still claim its old units;
`convert_dataarray_units(array, to_units=..., ...)` **reads** the source
`units`/`constituent` from `array.attrs` (no argument for them, so they cannot
be misstated), rewrites both on a copy, and drops `output_decimals` and the
`sipnet_internal_*` attributes. It refuses a `Dataset` and an array without
`units`. It was one `convert(values, units=...)` until the caller-supplied
`units` was found to be unchecked against the `DataArray`'s own: a wrong one
gave wrong numbers and then stamped the target units over the evidence.

**Arithmetic on labeled arrays lives in `pysipnet/arithmetic.py`, not
`units.py`,** because it reads `VariableKind` and `variables.py` imports
`units` at import time. `multiply_with_units` / `divide_with_units` /
`add_with_units` / `subtract_with_units` write the `units`, `constituent` and
`kind` true of the result, since plain xarray arithmetic drops them and
`convert_dataarray_units` would then have nothing to read. The operand with
the constituent leads the unit string, and its first unit must come through
unchanged (`g m-2` of C ÷ `g` is refused: `m-2` of C is not a quantity of
carbon). Kind changes only as `KIND_AFTER_TIME_POWER` in `variables.py` says
(total ÷ time → `daily_rate`, rate × time → total). Index coordinates must
align exactly, so two different time axes are refused rather than
intersected. `step_length()` makes the `time_step_length` coordinate an
operand; the coordinate itself is a timedelta with `kind=timestep_total` and
no `units`, so it cannot be one. `TIME_REFERENCE_FOR_KIND[DAILY_RATE]` says
"rate during the timestep" rather than "per-day", because a derived or
converted `daily_rate` may be per hour or per second.

`SIPNETOutput` exposes `.pandas` (DataFrame), `.xarray` (xarray Dataset, one
`time` dimension = step **end**), `["nee"]` (DataArray by name or alias),
`[["nee", "gpp"]]` (Dataset), and `.select(names, format="xarray"|"pandas")`
for the same selection in either library. xarray is a required dependency.

`select` is the one that reads only the columns named; `.pandas` / `.xarray`
read and cache all 35. `out[[...]]` is shorthand for `select(..., format="xarray")`,
and `format` is typed with `@overload` on the literals so a selection's static
type is the one library rather than a union. (The pandas half still reveals as
`Any`: the installed pandas ships no `py.typed`, so every `pd.DataFrame`
annotation in the package is `Any` to mypy. That is pandas, not the overload.) There is deliberately no
`dataset()` method — `out[[...]]` already is one, and `dataset` was retired as
a property name in `d7bd6a8` for saying nothing about which library it returns.

The `time` coordinate is the **end** of the step, not SIPNET's row label. It
is the one labeling under which every variable's CF `cell_methods` is literally
true: a pool is `time: point` at `time`, a total is `time: sum` over the bounds.
Under start labeling `time: point` would have claimed the pool was the
start-of-step value, which it is not. The Dataset states the interval each row
covers: `time_step_start`, `time_step_length` and a CF `time_bounds` variable
named by `time`'s `bounds` attribute, so `[time_step_start, time]` is
machine-readable — which is what an observation operator needs in order to
decide which steps an observation spans.

**The axis comes from the climate drivers, not from SIPNET's printed labels.**
Every output the runner returns carries its `ClimateDrivers`
(`SIPNETOutput(climate=...)`), and its axis is theirs row for row: their
starts, their lengths, their `time_zone`. The printed labels are rounded to
0.01 h, which put the output axis up to ±18 s off `climate.xarray` for the same
run and could not represent a 20-minute step at all. An output whose row count
differs from the drivers', or whose printed labels do not round from them, is
refused. `year`/`day_of_year`/`hour_of_day` coordinates carry the drivers'
unrounded values; `.pandas` keeps the printed ones, being a view of the file.
Without drivers (an output file re-opened with `from_path`/`from_dataframe`
and no `climate`) the axis falls back to the printed labels and the lengths
are inferred from them, so there is nothing to check labels against; the
Dataset's `time_axis_source` and `time_step_length_source` say so. The old
`time_step_length=` argument is gone: it was a third mode (printed starts,
supplied lengths) that the runner no longer used, and it needed its own looser
tolerance. Pass `climate=` instead — `climate.head(n)` for the first `n`
rows. `SIPNETOutput.time_step_length` remains, read from the climate.

`build_time_axis` takes lengths from exactly one place: a `ClimateDrivers`
(already checked, not re-checked), a bare frame's own `time_step_length`
column (checked there, since nothing else has), or inference from the labels
(consistent by construction).

A declared end within 60 s of the next row's start snaps to it, so
three-decimal `.clim` lengths do not put `time` seconds off the clock;
`time_step_length` itself is never adjusted. The snap is still needed for
that, but it no longer hides anything: it shares its tolerance with the
continuity check, which has already refused any overlap or drift it could
absorb. (It used to absorb the 2.46 s-per-step drift silently — though nothing
would have flagged the drift without it either, and the 2 h year-boundary
overlap in those files passed with no snap involved.)
Cumulative columns carry no `cell_methods` (their interval is the run so far,
which CF cannot express in that attribute) and `soil_wetness_fraction`'s says
it is a two-point mean. The Dataset declares `Conventions = "CF-1.11"`, `time`
has `standard_name`/`axis`, and no coordinate gets a `_FillValue` on encoding.

`pysipnet.resample.resample(ds, freq, how=...)` coarsens the axis. There is
**no default method**: `RESAMPLING_METHODS_FOR_KIND` in `variables.py` says
which of `sum`/`mean`/`last` are meaningful for each `VariableKind` (totals
sum; pools last or mean; rates and means mean; cumulatives last), `mean` is
weighted by `time_step_length` because Niwot's day/night steps differ in
length, and an invalid pairing raises with the reason and the valid menu.
`RESAMPLED_KIND` gives the result's kind (a pool averaged is a
`timestep_mean`), and the result's `kind`/`cell_methods`/`time_reference`
attributes are rewritten accordingly. The old `aggregation` attribute and
`Aggregation` enum are gone.

Step lengths come from the climate's `time_step_length`
column; when the output has no climate attached they are **inferred** from
consecutive timestamps (exact except for the last step, which repeats its
predecessor), and `time_step_length_source` in the Dataset's attributes says
which happened. `time_zone` is the drivers' declaration, or `"undeclared"`.
`run_id` and
`model_flags` (JSON) travel as attributes too, so an archived prediction says
which run produced it.

A selection never re-reads a column already in memory: `out["nee"]` then
`out["gpp"]` costs two partial reads rather than two full ones, and repeating
either costs no read. The one exception is `.pandas`/`.xarray` after a partial
read, which re-reads the file in full because only the file states the column
order; `variables` does the same, for the same reason. Variable-level column
selection is a *memory* optimization, not a speed one — `usecols` saves on the
order of a fifth of the parse time, since the tokenizer still scans every field.
`read_output_file` reads through the path rather than slurping the file into a
string, so peak memory during a parse is pandas' own rather than several times
the file size.

`build_time_axis` is the other thing worth knowing about the cost: it does not
depend on which variables were selected, so it is built once per output and
reused by every view. It uses integer `datetime64` arithmetic rather than a
string round-trip through `to_datetime`, which is more than an order of
magnitude faster and identical to the nanosecond. Years outside 1678-2261, and
non-finite values in any time column, are refused rather than wrapped or cast
to whatever the platform produces.

Selecting a variable whose `requires_flag` is off (e.g. `litter_carbon` without
`litter_pool`) **raises**: SIPNET writes it as constant zero, and a likelihood
would consume those zeros without complaint. `.pandas` and `.xarray` still
contain the column, being a faithful view of the file.

Columns present at other versions: `woodCreation`, `nppStorage`, the
nitrogen group, `ch4` and `plantStorageN` are new at this pin; `bcdeltaC` and
`bcdeltaN` (v2.1.0's mass-balance closure) are gone, mapped in
`LEGACY_OUTPUT_COLUMNS` so old files still read. SIPNET now reports closure as
a log warning from `checkBalance()`; `tests/test_integration.py::TestMassBalance`
reads the log.

## SIPNET Parameters — Full Grouped List

Parameters are grouped as they appear in the Python data model. All initial conditions are also in the `.param` file (SIPNET makes no distinction).

The lists below use **SIPNET's names**, because they describe the SIPNET
contract. The Python field names follow the same convention as output
variables (lower-case words, no acronyms): `aMax` is
`photosynthesis.max_photosynthesis_rate`, `soilWHC` is
`water.soil_water_holding_capacity`, `plantWoodInit` is
`initial_conditions.total_wood_carbon`. Each field's `ParameterSpec`
(`pysipnet/parameters/base.py`) records `sipnet_name`, UDUNITS `units`,
`constituent`, labels, `aliases` (the pre-convention pySIPNET names) and, for
initial conditions, `initializes` / `initializes_via` naming the output state
it sets. `PARAMETER_SPECS` in `pysipnet/parameters/model.py` is the flat
`{"group.field": spec}` view; `PYTHON_TO_SIPNET` in `param_io.py` is derived
from it, and `tests/test_param_name_mapping.py` restates the mapping by hand.
`resolve_parameter_name()` accepts a field name, an alias or a SIPNET name.
Parameter groups forbid unknown keys, so a parameter set saved under an old
name fails loudly on load. The docs page `reference/parameters.md` is
generated from the specs.

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
is on — off in `ModelFlags.standard()`, which is SIPNET's own default set.

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
least in the way. Note `fAnoxia` is registered
`ctx.anaerobic || ctx.nitrogenCycle`, so either flag alone requires it.

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
  that is not `None`, so a file has those 52 plus whichever optional ones you
  set.
- `litter_pool` adds `litterBreakdownRate` and `fracLitterRespired` → **54**.
- No obsolete placeholders. The previous pin needed 64 lines for the equivalent
  configuration.

## Key Technical Gotchas

1. **`psnTMax` and `coarseRootAllocation` are derived**, not specified. `psnTMax = 2×psnTOpt − psnTMin`. `coarseRootAllocation = 1 − leafAllocation − fineRootAllocation − woodAllocation`.

2. **PAR units scale with timestep.** Climate file PAR is total Einsteins m⁻² for the whole timestep interval. When converting from instantaneous flux measurements, multiply by `length` (in days × 86400 seconds/day).

3. **SIPNET expects files in the current working directory.** The runner writes all inputs to a fresh temp dir per run and executes the binary there. The generated `sipnet.in` sets `FILE_NAME = sipnet`, so SIPNET reads `sipnet.param` and `sipnet.clim` and writes `sipnet.out`.

4. **Model options are runtime, and they change which parameters are required.** All ten (`GDD`, `SNOW`, `WATER_HRESP`, `GROWTH_RESP`, `LEAF_WATER`, `LITTER_POOL`, `SOIL_PHENOL`, `NITROGEN_CYCLE`, `ANAEROBIC`, `FLOODING`) are set in `sipnet.in`. One binary, `make sipnet`, no `-D` flags, no source patch. Because they change the required parameter set, the flags are part of the run specification, not a build detail — which is why `ModelFlags` is serialized into `RunConfig` and `RunProvenance`. SIPNET rejects four combinations (`validateContext()`); `ModelFlags` rejects them first.

5. **No missing climate values.** Climate validation must be strict: every row must be complete, timesteps must be monotonically increasing, each label must agree with the previous row's start plus its length (no overlap, no drift; a gap warns), and the start/end dates must bracket the intended simulation period.

6. **Events file.** SIPNET defaults to `EVENTS=1` (looks for `events.in`). The runner writes `EVENTS = 0` in `sipnet.in` to suppress this for basic runs. Note that this is belt-and-braces: a *missing* `events.in` is already harmless (`access` guard, `logInfo` only), so `EVENTS = 0` mainly guards against a stale file in the working directory. When events are used, the file must be in the working directory.

7. **The `OBSOLETE_PARAM` workaround is gone.** Historical note, because it explains a chunk of deleted code. Older SIPNET declared nine parameters obsolete — read and then ignored — but its `checkAllRead()` tested `if (param->isRequired)`, which is truthy for the `-1` obsolete marker, so it **errored if they were absent**. pySIPNET appended fixed placeholders to every param file to satisfy that. SIPNET deleted the nine at v2.0.0, so `_OBSOLETE_DEFAULTS` is gone too. Writing them now would produce nine `"Unknown param"` log lines per run, which is harmless in itself but would bury a real unknown parameter; `tests/test_param_file_contract.py` asserts a clean run logs none at all.

8. **Phenology is parameter-driven — there is no phenology data input.** `pastLeafGrowth()` selects one of three leaf-on triggers from the runtime flags: `cumulativeGdd >= gddLeafOn` when `ctx.gdd`, `climate->tsoil >= soilTempLeafOn` when `ctx.soilPhenol`, or `currTime >= leafOnDay` otherwise. Growing degree days are accumulated inside `readClimData` from `tair × length` and reset on year change — **never read from a file**. Leaf-off is always `leafOffDay`. `laiInit` only sets the initial leaf C pool (`envi.plantLeafC = laiInit × leafCSpWt`); LAI is prognostic thereafter. "Phenology data" enters SIPNET only by two indirect routes, **both of which resolve to the `.param` file before the binary runs** (verified against PEcAn source, not inferred):

    - `PEcAn.data.remote::extract_phenology_MODIS()` pulls MODIS **MCD12Q2** `MidGreenup` / `MidGreendown` bands, QA-filters them, and writes a CSV (`year, site_id, lat, lon, leafonday, leafoffday, leafon_qa, leafoff_qa`). `PEcAn.SIPNET::write.config.SIPNET()` then reads it via `settings$run$inputs$leaf_phenology$path` and **overwrites `leafOnDay` / `leafOffDay` in the param table** (fallback chain: matching year → site mean across years → param-file default; applied only `if (leafOffDay > leafOnDay)`).
    - `gddLeafOn` is set from a PFT **trait** named `GDD`, i.e. the calibration/prior route — *not* from remote sensing.

    Upstream of our pin this changes: see gotcha 9.

9. **Prescribed phenology is available at this pin, and pySIPNET does not use it yet.** Via `events.in`, not a new file type. SIPNET accepts `leafon` and `leafoff` events with **zero** parameters, so a line is just `year day leafon`. They are real input keywords (`eventStringToType` in `events.c`). Two properties matter before wiring them up: they are **mutually exclusive with every calculated mechanism** — `checkForCalculatedLeafEvents()` exits with `EXIT_CODE_BAD_PARAMETER_VALUE` if `ctx.gdd || ctx.soilPhenol || params.leafOnDay > 0 || params.leafOffDay > 0` — and they prescribe **timing only**: the flux is the same `params.leafGrowth / climLen` the calculated trigger would apply, so `leafGrowth` and `fracLeafFall` remain fitted parameters either way. PEcAn already emits them (`write.events.SIPNET.R`) and, when it sees them, zeroes `leafOnDay`/`leafOffDay`/`gddLeafOn` and ignores its `leaf_phenology` CSV. Note upstream's own docs still list event types as only `plant`/`harv`/`till`/`fert`/`irrig` — read `eventStringToType`, not the table. Tracked in issue #25; `tests/test_events_contract.py` pins the set of unmodeled types so a fourth cannot appear unnoticed.

10. **`leafOnDay = 0` now means "disabled", and at our previous pin it did not.** `pastLeafGrowth` and `pastLeafFall` are gated on `params.leafOnDay > 0` / `params.leafOffDay > 0`, so zero switches the trigger off — which is how PEcAn disables internal scheduling when it supplies leaf events instead. This gating arrived with the prescribed-phenology work and is **absent from `v2.1.0`**, where `pastLeafFall` compared unconditionally and `leafOffDay = 0` fired leaf fall on the first timestep of every year. Mentioned because the previous pin behaved the other way, and a parameter set carried over from it will now behave differently.

11. **A new required parameter arrived with the leaf events: `leafOnReallocFrac`.** Leaf-out has to take carbon from somewhere, and this caps how much of `plantWoodC + coarseRootC` it may draw on. SIPNET scales the transfer down if demand exceeds `(plantWoodC + coarseRootC) × leafOnReallocFrac`. Required unconditionally, so every param file needs it; upstream's Niwot fixture uses `0.2`.

12. **The `SNOW` flag does not switch the snowpack off.** SIPNET's docs say
    `SNOW = 0` treats all precipitation as liquid, but at this pin nothing in
    `calcPrecip()`, `snowPack()` or `updateState()` reads `ctx.snow`; the only
    uses are the requiredness of `snowMelt` and the restart-checkpoint flag
    check. Precipitation below 0 °C falls as snow either way. SIPNET reads any
    registered parameter it finds whether or not it is required, and pySIPNET
    writes every non-`None` field, so with the flag off and `snowMelt` still
    supplied the run is identical to the flag being on; only when `snowMelt`
    is omitted does it default to zero and the snow never melt. Verified by
    running the binary both ways on identical climate: identical snow columns. `tests/test_integration.py::TestSnowFlag` pins this so an
    upstream fix shows up. Consequently the `snow` output column is **not**
    zero when the flag is off, unlike `litter`, the nitrogen group and `ch4`.

## File Structure

```
pySIPNET/
├── sipnet/                       # git submodule — SIPNET source, pinned to v2.2.0-alpha.1
├── Makefile                      # `make sipnet`, `make sipnet-download`
├── hatch_build.py                # build hook: bundle a staged binary into a platform-tagged wheel
├── RELEASING.md                  # maintainer notes: how to cut a release (tag → Wheels workflow → draft release)
├── pysipnet/
│   ├── version.py                # pinned commit, target version, release assets, wheel platform tags (import-free)
│   ├── build.py                  # find (search order), download, compile, verify the binary
│   ├── cli.py                    # `pysipnet install-sipnet | info | stage-bundle`
│   ├── parameters/
│   │   ├── base.py               # ParameterSpec, param_field, domains (version-agnostic)
│   │   └── model.py              # ModelFlags and SIPNETParameters
│   ├── variables.py              # the output-variable registry (names, units, kinds, labels)
│   ├── units.py                  # UDUNITS unit strings: Pint registry, validation, formatting, constituent-aware conversion
│   ├── arithmetic.py             # products, quotients, sums of labeled DataArrays; step_length()
│   ├── climate.py                # ClimateDrivers + validation
│   ├── dataset.py                # shared DataFrame → xarray builder (time = step end)
│   ├── resample.py               # explicit, kind-checked coarsening of the time axis
│   ├── events.py                 # management events (arity checked against SIPNET)
│   ├── io/
│   │   ├── param_io.py           # read/write .param
│   │   ├── clim_io.py            # read/write .clim; layout detected as SIPNET detects it
│   │   ├── output_reader.py      # read .out, header detected by content
│   │   └── reference.py          # locate and load the bundled Niwot data (importlib.resources)
│   ├── data/niwot/               # shipped in the wheel: SIPNET-authored .param/.clim, golden output, README
│   ├── runner.py                 # SIPNETRunner, _render_sipnet_in
│   ├── model.py                  # SIPNETModel — high-level callable interface
│   ├── config.py                 # RunConfig — a saveable run specification
│   ├── result.py                 # SIPNETResult, RunProvenance
│   ├── output.py                 # SIPNETOutput — DataFrame and xarray views, lazy or eager
│   ├── ensemble.py               # helpers for driving many runs
│   └── viz.py                    # plotting
├── tests/
│   ├── test_sipnet_in.py         # the sipnet.in contract, incl. SIPNET's config dump
│   ├── test_param_file_contract.py  # the .param contract across flag combinations
│   ├── test_events_contract.py   # the events.in contract, incl. arities
│   ├── test_clim_layout_contract.py  # pySIPNET reads exactly the .clim layouts SIPNET reads
│   ├── test_param_name_mapping.py   # the Python→SIPNET parameter map, stated by hand
│   ├── test_integration.py       # end-to-end behavior, flags, mass balance, snow flag
│   ├── test_variables.py         # the .out header contract and the registry's own rules
│   ├── test_units.py             # conversion factors, refusals, and every registry constituent convertible
│   ├── test_arithmetic.py        # units/constituent/kind of products and quotients, on Niwot output
│   ├── test_download.py          # prebuilt-binary download and its verification
│   ├── test_fidelity.py          # wrapper output == bare binary output
│   ├── test_time_axis.py         # axis from the drivers, label/length continuity, time_zone
│   ├── test_golden.py            # frozen numeric baseline
│   ├── test_reference.py         # bundled data ships in the wheel and matches the submodule
│   ├── test_bundle_hook.py       # platform wheels carry the binary and the right tag
│   ├── test_cli.py               # the pysipnet command
│   ├── test_exceptions.py        # every exception pickles and crosses a process boundary
│   └── test_build.py             # search order, compile fallback, runtime pin check
├── data/                         # (gitignored) sample data
├── docs/
└── CLAUDE.md                     # this file
```

Note there is no `patches/` directory and no per-option build targets; both
belonged to the pre-v2.0.0 compile-time-flag era.

### Bundled reference data

The Niwot fixtures live **inside the package**, at `pysipnet/data/niwot/`, not
under `tests/`, because a project that installs pySIPNET needs real SIPNET data
to test against and a wheel does not carry `tests/`. `pysipnet/io/reference.py`
locates them through `importlib.resources` and is the one place in the package
that reads package data; `niwot_reference_output()` returns the golden as a
`SIPNETOutput` paired with the climate's step lengths under standard flags.
pySIPNET's own tests read the same files, so there is one copy and no drift.
`sipnet.param` and `sipnet.clim` are SIPNET-authored and must stay byte-identical
to the submodule's smoke fixtures; never re-save them. The golden is shipped
too, so regenerating it is a visible change for consumers. hatchling includes
non-Python files under the listed package with no extra config, which
`tests/test_reference.py` proves by building the wheel rather than trusting it.

### Test layers, and what each one would catch

Worth knowing which test to look at when something breaks:

- `test_build.py` — the binary matches the recorded pin, the search order
  is what the docs say, and `verify_binary_matches_pin` refuses the wrong
  tag. Catches a stale binary, a half-finished version bump, or a candidate
  location silently outranking another.
- `test_bundle_hook.py` — a wheel built with `$PYSIPNET_BUNDLE_SIPNET` carries
  the staged binary, executable, under the platform tag from
  `SIPNET_PREBUILT_REQUIREMENTS`, and the pure wheel carries none. Under
  `-m network`, that the requirements table matches the published binaries.
- `test_sipnet_in.py` — SIPNET understood every config key we wrote, proven by
  reading its own resolved-config dump. Catches a key silently ignored.
- `test_param_file_contract.py` — SIPNET recognized every parameter name and
  found everything it required, across six flag combinations. Catches a
  renamed or dropped parameter.
- `test_clim_layout_contract.py` — pySIPNET accepts exactly the `.clim`
  layouts SIPNET accepts (12 and 14; not 13, 11, or a 14 whose site changes),
  for the reasons SIPNET gives, and the two accepted layouts give identical
  model output. Catches the layout detection drifting from SIPNET's.
- `test_variables.py` — the binary's output header equals the variable
  registry, token for token and in order. Catches a column added, dropped or
  renamed upstream, which the reader would otherwise pass through with only a
  warning.
- `test_fidelity.py` — driving SIPNET through the wrapper gives the same
  numbers as running the bare binary by hand. Catches distortion anywhere in
  the writers, runner or parser. Version-independent, since both sides run the
  same binary.
- `test_golden.py` — the numbers themselves match a checked-in baseline.
  Catches an unintended model change that the wrapper and binary would still
  agree about. Regenerate deliberately with `python -m tests.test_golden`.
- `test_reference.py` — the Niwot reference files are in the built wheel byte
  for byte, and the two inputs are still identical to the submodule's smoke
  fixtures. Catches a packaging change that drops them (reading the source
  tree would not) and any re-save or reformat of upstream-authored data.
- `test_time_axis.py` — a run's axis is its climate's exactly (20-minute
  steps, which SIPNET cannot print), a row-count or label mismatch is refused,
  drifted and overlapping drivers are refused while Niwot passes with margin,
  and `time_zone` survives the run, `RunConfig` and `resample`. Catches the
  axis silently reverting to SIPNET's rounded labels, and a tolerance change
  that would start accepting drift or refusing Niwot.
- `test_units.py` — every conversion factor, stated as arithmetic on the
  molar masses by hand, every refusal by message, round trips, relabeling of
  `attrs`, and that every constituent a registry declares is one the
  conversion tables know. Catches a wrong table entry, a wrong route through
  the tables, and a new constituent added without its molar mass.
- `test_integration.py` — end-to-end behavior, including that flags visibly
  change results and that SIPNET's own mass-balance errors stay near zero.

The first three and `test_variables.py` exist because the failure they catch is **silent**: SIPNET logs
an ignored key or unknown parameter and carries on, so the run succeeds and the
output looks plausible. Any new input we start writing should get the same
treatment.

## Development Conventions

- **Python ≥ 3.11**
- **Pydantic v2** for all data models (parameter validation, units enforcement)
- **pandas** for climate time series and output; **xarray** for the metadata-carrying `dataset` views (required dependency)
- **NumPy** for numerical operations
- **No comments unless the WHY is non-obvious.** Well-named identifiers are preferred.
- **Type hints everywhere.**
- All file I/O is in the `pysipnet/io/` subpackage. The rest of the package never touches the filesystem directly.
- Tests use real SIPNET binaries where possible (integration tests), not mocks.
- **Exceptions must pickle.** Ensemble runners send failures back from worker
  processes through pickle, and `BaseException` rebuilds from `cls(*self.args)`.
  An exception whose `__init__` takes more than the message needs a
  `__reduce__`, as `SIPNETRunError` has; without one it pickles and then fails
  to unpickle, which in a `ProcessPoolExecutor` breaks the pool.
  `tests/test_exceptions.py` round-trips every exception class in the package
  and fails when one is added without an example.
