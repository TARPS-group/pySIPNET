# SIPNET Version Contract

## Pinned commit

pySIPNET v0.x targets a single SIPNET commit:

```
e4abf14f2445133c785b756025a2e39e60c7760f
```

This is the last commit before PR #114 ("SIP78 Convert switches to run time options part 3") migrated SIPNET's compile-time feature flags to runtime CLI arguments, marking the boundary between what we call **v1** (compile-time flags, 14-column climate file) and **v2** (runtime flags, 12-column climate file).

No formal tag exists at this commit.  The nearest tag is `v1.3.0` (commit `8ff893e`), which predates it by several cleanup PRs.

## What changed between v1 and v2

### Climate file

| Format | Columns | Notes |
|:-------|:--------|:------|
| v1     | 14      | Col 1: location index (ignored). Col 14: soilWetness (ignored). |
| v2     | 12      | Location and soilWetness columns removed. |

The sample file `data/era5_site1.clim` is in v1 format.

### Parameter file

Both versions use a 2-column format (`name  value`) with `!` comments.  v1 files may have up to 6 columns (name, value, changeable, min, max, sigma); SIPNET silently ignores columns 3+.

### Compile-time flags

In v1, model features are controlled by `#define` constants in the C source.
Changing a feature requires recompilation.  In v2, all features became runtime
flags (CLI arguments and `sipnet.in`).

The pySIPNET build system patches v1 to allow flag overrides via `-D` compiler
arguments.  See [Installation](installation.md) for details.

**Remaining compile-time flags at the pinned commit:**

| Flag | Default | Controls |
|:-----|:--------|:---------|
| `SNOW` | 1 (on) | Snowpack tracking |
| `GDD` | 1 (on) | Growing degree-day phenology |
| `WATER_HRESP` | 1 (on) | Soil moisture effect on Rh |
| `GROWTH_RESP` | 0 (off) | Explicit growth respiration |
| `LEAF_WATER` | 0 (off) | Leaf water pool for sub-daily ET |
| `LITTER_POOL` | 0 (off) | Separate litter C pool |
| `SOIL_PHENOL` | 0 (off) | Soil-temperature phenology (⊕ with GDD) |
| `HEADER` | 0 (off) | Column header in output file |

pySIPNET always compiles with `HEADER=1` so the output parser can rely on the
header row.

### Output format

v1 output includes columns `fPAR`, `microbeC`, and `litterWater` that are
absent in v2.  When these features are compiled out (off), the columns output
zero.

## Extension to v2

When SIPNET v2 stabilises, the extension path is:

1. Add a new submodule pointer (or a parallel submodule path) for the v2 source.
2. Implement `pysipnet/io/clim_io_v2.py` with the 12-column reader/writer.
3. Implement `pysipnet/parameters/v2.py` with the updated parameter set.
4. Extend `ModelPreset` with v2 presets.
5. Route `ClimateDrivers(version="v2")` and `SIPNETParametersV2` through the new adapters.

The public API (`SIPNETRunner.run(params, climate)`) does not need to change.
