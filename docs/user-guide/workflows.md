# Common Workflows

This page shows complete, copy-pasteable examples for the most common ways to
use pySIPNET.  Each workflow is a different trade-off between simplicity,
memory use, and disk use.  Pick the one that fits your situation, or mix and
match — the options compose cleanly.

For the mechanics behind each option, see [File I/O](file-io.md).

---

## 1. Pure interactive

**Best for:** exploratory analysis, notebooks, quick sensitivity checks.

Everything lives in memory.  No files are created (beyond the transient
working directory that pySIPNET manages automatically).

```python
from pysipnet import SIPNETRunner, ModelPreset, SIPNETParametersV1, ClimateDrivers
import pandas as pd

# Load climate directly into memory
climate = ClimateDrivers.from_file("data/era5_site1.clim", version="v1")

# Run
runner = SIPNETRunner(preset=ModelPreset.STANDARD)
result = runner.run(params, climate)

# Work with results — all in memory
df = result.outputs.data
print(result.nee().sum())        # annual NEE
print(result.gpp().mean())       # mean GPP per timestep

# Quick parameter sensitivity
for a_max in [80.0, 100.0, 120.0, 140.0]:
    r = model(a_max=a_max)
    print(f"a_max={a_max}: NEE={r.nee().sum():.2f} g C m⁻²")
```

The `SIPNETModel` wrapper (see [Running a Model](running-a-model.md)) is
especially convenient here because it handles parameter overrides for you.

---

## 2. File-backed climate, outputs in memory

**Best for:** iterating over multiple pre-processed climate files without
reading them all up front; single-site or small multi-site runs where output
size is manageable.

```python
from pysipnet import SIPNETRunner, ModelPreset, ClimateDrivers, ClimateStaging

runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    climate_staging=ClimateStaging.COPY,   # or SYMLINK on Linux/macOS
)

site_files = [
    "data/harvard_forest.clim",
    "data/niwot_ridge.clim",
    "data/howland_forest.clim",
]

results = {}
for path in site_files:
    # Climate file is not read into Python memory — only staged to the workdir
    climate = ClimateDrivers.from_path(path, version="v1")
    result = runner.run(params, climate, run_id=Path(path).stem)

    # Output is eagerly parsed into memory (default behaviour)
    results[Path(path).stem] = result.outputs.data

# All outputs now in a dict of DataFrames — climate files never loaded into Python
annual_nee = {site: df["nee"].sum() for site, df in results.items()}
```

Use `ClimateStaging.SYMLINK` instead of `COPY` if the climate files are large
and you are on Linux/macOS — SIPNET will read the original file directly,
eliminating the copy.

---

## 3. Lazy outputs

**Best for:** ensemble runs where you want to defer loading outputs until
post-processing, or where memory is constrained.

```python
from pathlib import Path
from pysipnet import SIPNETRunner, ModelPreset

output_dir = Path("ensemble_outputs")

runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    output_dir=output_dir,        # each run copies sipnet.out here
)

# Run 100 members — no DataFrames created yet
results = [
    runner.run(params_i, climate, run_id=f"member_{i:04d}")
    for i, params_i in enumerate(param_samples)
]

# result.outputs is file-backed; source_path points at the persistent copy
print(results[0].outputs.source_path)
# PosixPath('ensemble_outputs/sipnet_member_0000.out')

# Load all outputs on demand — one at a time to keep memory low
import pandas as pd

nee_all = pd.concat(
    [r.outputs.data["nee"] for r in results],
    axis=1,
    keys=[r.provenance.run_id for r in results],
)
```

---

## 4. Lazy outputs with column selection

**Best for:** large ensembles where you only need a few output variables.
Column-selective loading avoids parsing columns you will never use.

```python
from pathlib import Path
import pandas as pd
from pysipnet import SIPNETRunner, ModelPreset

runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    output_dir=Path("ensemble_outputs"),
)

results = [
    runner.run(params_i, climate, run_id=f"m{i:04d}")
    for i, params_i in enumerate(param_samples)
]

# Read only NEE and GPP from each output file — year/day/time always included
frames = [
    r.outputs.load(columns=["nee", "gpp"]).assign(run_id=r.provenance.run_id)
    for r in results
]
combined = pd.concat(frames, ignore_index=True)
# combined has columns: year, day, time, nee, gpp, run_id
```

`load(columns=[...])` reads only the requested columns from disk each time it
is called — it does not cache the result.  This keeps peak memory at one
member's worth of data rather than the full ensemble.

---

## 5. Symlinked climate + lazy outputs

**Best for:** large ensembles on Linux/macOS where climate files are large
and shared across many members; maximum I/O throughput.

```python
from pathlib import Path
from pysipnet import SIPNETRunner, ModelPreset, ClimateDrivers, ClimateStaging

# One shared climate file for all members
climate = ClimateDrivers.from_path("data/era5_site1.clim", version="v1")

runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    climate_staging=ClimateStaging.SYMLINK,   # SIPNET reads original file — no copy
    output_dir=Path("ensemble_outputs"),       # outputs saved, not held in memory
)

results = [
    runner.run(params_i, climate, run_id=f"m{i:04d}")
    for i, params_i in enumerate(param_samples)
]
# I/O per run: zero bytes for climate (symlink), one file copy for output
```

!!! note "Symlink fallback"
    If `os.symlink` fails (e.g. on Windows or across filesystem boundaries),
    the runner falls back to `COPY` with a `UserWarning`.  In long ensemble
    runs only one warning is emitted regardless of how many members run.

---

## 6. Full file-oriented workflow

**Best for:** reproducible research pipelines where you want every input and
output file retained on disk; post-hoc inspection or re-analysis without
re-running the model.

```python
from pathlib import Path
from pysipnet import SIPNETRunner, ModelPreset, ClimateDrivers, ClimateStaging

run_dir = Path("runs/experiment_01")
run_dir.mkdir(parents=True, exist_ok=True)

runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    climate_staging=ClimateStaging.SYMLINK,
    output_dir=run_dir / "outputs",
    keep_workdir=True,           # also retain inputs in the working directory
    workdir_base=run_dir / "workdirs",
)

climate = ClimateDrivers.from_path("data/era5_site1.clim", version="v1")
result = runner.run(params, climate, run_id="baseline")

# After the run, the directory tree looks like:
# runs/experiment_01/
# ├── workdirs/
# │   └── sipnet_baseline/
# │       ├── sipnet.param
# │       ├── sipnet.clim   ← symlink to era5_site1.clim
# │       ├── sipnet.in
# │       └── sipnet.out
# └── outputs/
#     └── sipnet_baseline.out   ← persistent copy for lazy loading

# result.outputs is file-backed — nothing loaded yet
print(result.outputs.source_path)
# PosixPath('runs/experiment_01/outputs/sipnet_baseline.out')

# Load when needed
df = result.outputs.data
```

For even stronger reproducibility guarantees, combine this with
[`RunConfig.save()`](reproducible-runs.md) to record the full parameter set
and a SHA-256 hash of the climate file alongside the output.

---

## Choosing a workflow

| Situation | Climate | Output | Staging |
|:----------|:--------|:-------|:--------|
| Interactive / notebook | `from_file` or `from_dataframe` | Eager (default) | n/a |
| Pre-existing files, single run | `from_path` | Eager (default) | `COPY` or `SYMLINK` |
| Ensemble, moderate size | `from_path` | Lazy (`output_dir=`) | `COPY` |
| Ensemble, large files, Linux/macOS | `from_path` | Lazy (`output_dir=`) | `SYMLINK` |
| Need only select output columns | `from_path` | Lazy + `load(columns=)` | `SYMLINK` |
| Full archival / reproducible pipeline | `from_path` | Lazy + `keep_workdir=True` | `SYMLINK` |
