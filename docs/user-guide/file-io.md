# File I/O and Working Directories

This guide covers everything related to how pySIPNET interacts with the
filesystem: how run inputs are staged, where temporary files land, how to
preserve or inspect them, and how to avoid unnecessary I/O in ensemble
workflows.

---

## How a run uses the filesystem

Every call to `SIPNETRunner.run()` (and by extension `SIPNETModel.__call__`)
follows the same file lifecycle:

1. **Create a working directory** — a fresh subdirectory under `workdir_base`
   (default: the system temp directory).
2. **Stage inputs** — write `sipnet.param`, `sipnet.clim`, and (optionally)
   `events.in` into the working directory.
3. **Write `sipnet.in`** — the two-line config file that tells SIPNET where to
   find its inputs and whether to load an events file.
4. **Execute the binary** — SIPNET reads the staged files and writes
   `sipnet.out` in the same directory.
5. **Parse outputs** — `sipnet.out` is read back into a
   `SIPNETResult.outputs` DataFrame.
6. **Clean up** — the working directory is deleted unless `keep_workdir=True`.

This design is intentional: each run is self-contained and stateless.  Two
runs never share a working directory, so they can be parallelised safely with
any executor.

---

## Working directory location and naming

By default, working directories are placed in `tempfile.gettempdir()` (usually
`/tmp` on Linux/macOS or `C:\Users\...\AppData\Local\Temp` on Windows) and
named `sipnet_<run_id>` where `<run_id>` is a random UUID hex string.

You can control both:

```python
from pysipnet import SIPNETRunner, ModelPreset

runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    workdir_base="/scratch/my_runs",   # custom parent directory
)

result = runner.run(params, climate, run_id="baseline_2020")
# working directory: /scratch/my_runs/sipnet_baseline_2020/
```

Supplying an explicit `run_id` is useful when you want the directory name to
correspond to something meaningful in your own logging (e.g., a site name, an
ensemble member index, or a timestamp).

!!! note "run_id reuse"
    If the directory `sipnet_<run_id>` already exists, it is reused and
    its contents overwritten.  This is safe but means the previous run's
    files are not recoverable.  Use distinct `run_id` values across runs
    if you need to preserve them all.

---

## Preserving files after a run

Set `keep_workdir=True` on the runner to suppress cleanup:

```python
runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    keep_workdir=True,
)

result = runner.run(params, climate, run_id="debug_run")

# Inspect the preserved directory
import os
print(result.provenance.workdir)
for f in os.listdir(result.provenance.workdir):
    print(f)
# sipnet.param
# sipnet.clim
# sipnet.in
# sipnet.out
```

The preserved directory contains exactly the files SIPNET saw, so you can
reproduce the run manually:

```bash
cd /tmp/sipnet_debug_run
./path/to/sipnet_standard
```

`keep_workdir=True` applies to every run on that runner instance.  If you only
want to preserve a specific run for debugging, construct a one-off runner for
that call.

---

## Climate data: in-memory vs. file-backed

`ClimateDrivers` supports two construction modes.

### In-memory (default)

`from_dataframe` and `from_file` load all climate data into a pandas DataFrame
at construction time.  When the runner stages this for SIPNET, it serialises
the DataFrame to disk by writing a new `.clim` file in the working directory.

```python
# Full data in memory — good for interactive use and data manipulation
climate = ClimateDrivers.from_file("data/era5_site1.clim", version="v1")
climate.data        # DataFrame always available
climate.validate()  # full validation runs immediately
```

### File-backed (lazy)

`from_path` creates a lightweight reference to an existing `.clim` file
without reading it into memory.  The runner can stage it by copying or
linking the original file directly, skipping the read-then-write cycle
entirely.

```python
# No data loaded — good for ensemble workflows with pre-existing files
climate = ClimateDrivers.from_path("data/era5_site1.clim", version="v1")

print(climate.n_timesteps)  # available without loading data
print(climate.date_range)   # also available without loading data
```

Accessing `climate.data` on a file-backed instance triggers a full load and
caches the result — subsequent accesses are free.

!!! warning "Chronological ordering assumption"
    `from_path` validates the column count of the first and last rows and
    uses them to populate `n_timesteps` and `date_range`.  It **assumes the
    file is sorted chronologically** — this is not verified without loading
    the full file.

    If you are not certain about the ordering of a file you did not produce
    yourself, call `.validate()` after construction.  This triggers a full
    load and checks monotonicity:

    ```python
    climate = ClimateDrivers.from_path("unknown_source.clim")
    climate.validate()   # raises ValueError if not monotone
    ```

---

## Climate file staging modes

When `SIPNETRunner` stages a file-backed climate instance into the working
directory, it can either **copy** the file or create a **symlink**.  Set
`climate_staging` at runner construction time:

```python
from pysipnet import SIPNETRunner, ModelPreset, ClimateStaging

# Default: copy (safe on all platforms)
runner = SIPNETRunner(preset=ModelPreset.STANDARD)

# Symlink: zero I/O for large files (Linux/macOS only)
runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    climate_staging=ClimateStaging.SYMLINK,
)
```

| Mode | Behaviour | When to use |
|:-----|:----------|:------------|
| `COPY` (default) | `shutil.copy2` — physical copy of the file | All platforms; when source files may move or be modified during a run |
| `SYMLINK` | Symbolic link to the resolved absolute path | Linux/macOS; large files; source files are stable for the run duration |

### Symlink fallback

If `SYMLINK` is requested but `os.symlink` raises `OSError` (e.g., on Windows
without elevated privileges, or across filesystem boundaries), the runner
falls back to `COPY` and emits a `UserWarning`.  Python's warning system
deduplicates by call site, so in a long ensemble run you will see at most one
warning rather than one per run.

### Staging applies only to file-backed instances

`climate_staging` has no effect when climate data is in memory.  In-memory
instances are always serialised by the I/O layer regardless of this setting:

| `ClimateDrivers` type | `ClimateStaging.COPY` | `ClimateStaging.SYMLINK` |
|:----------------------|:----------------------|:-------------------------|
| In-memory (`from_dataframe`, `from_file`) | Write from DataFrame | Write from DataFrame |
| File-backed (`from_path`) | `shutil.copy2` | Symlink (fallback to copy) |

---

## Ensemble file I/O considerations

In large ensemble runs, file I/O can dominate wall time if not managed
carefully.  The following recommendations apply whether you are using PyEns,
Dask, Parsl, or a simple `ProcessPoolExecutor`.

### Use `from_path` for pre-existing climate files

If your climate files already exist on disk (e.g., pre-processed ERA5 or
FLUXNET data), use `from_path` instead of `from_file`:

```python
import numpy as np
from pyens import Axis
from pysipnet import ClimateDrivers
from pysipnet.ensemble import sipnet_site_fields

site_paths = [
    "data/harvard_forest.clim",
    "data/niwot_ridge.clim",
    "data/howland_forest.clim",
]

# Lightweight references — no data loaded yet
climates = [ClimateDrivers.from_path(p) for p in site_paths]

sites = Axis("site", labels=["hf", "nr", "hl"])
fields = sipnet_site_fields(sites, climates=climates, ...)
```

With `ClimateStaging.COPY` (default), each run copies the file once into its
working directory.  With `ClimateStaging.SYMLINK`, no data is ever moved — the
SIPNET binary reads directly from the original path.

### Avoid loading climate data on worker processes

When using multiprocessing backends, the model callable and all its arguments
are pickled and sent to worker processes.  In-memory `ClimateDrivers` objects
pickle their full DataFrame; file-backed instances pickle only a path.  For an
ensemble of 500 sites each with 30 years of daily data (~11 000 rows × 12
columns), the difference is roughly 5 MB vs. a few bytes per site.

```python
# Preferred: file-backed — only a path is pickled per site
climates = [ClimateDrivers.from_path(p) for p in site_paths]

# Avoid for large ensembles: entire DataFrame pickled per site
climates = [ClimateDrivers.from_file(p) for p in site_paths]
```

### workdir_base on fast local storage

If your project data lives on network-attached or high-latency storage, set
`workdir_base` to a fast local disk.  SIPNET reads and writes its working
directory files at high frequency relative to its compute time:

```python
runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    climate_staging=ClimateStaging.SYMLINK,   # source files stay on NFS
    workdir_base="/local/scratch",            # SIPNET I/O on fast local disk
)
```

### Disk space with keep_workdir

By default, working directories are deleted after each run.  The parsed
outputs are already in the `SIPNETResult.outputs` DataFrame at that point,
so in-memory data is not lost.  However, the raw `sipnet.out` file is gone.
This matters if you later want lower-level output than what you extracted
during the run — for example, if you summarised to annual totals and then
realised you needed the full sub-daily timeseries.

Each preserved working directory contains at minimum:

- `sipnet.param` — ~2–3 kB
- `sipnet.clim` — ~1 MB per year of daily data
- `sipnet.in` — trivial
- `sipnet.out` — ~500 kB per year of daily output

For a 500-run ensemble with `keep_workdir=True`, budget ~1 GB per year of
simulation.  Whether that trade-off is worthwhile depends on your workflow —
if you are confident the DataFrame captures everything you need, the default
delete-on-completion is fine; if you want the raw files as an archival record
or a fallback, set `keep_workdir=True`.

---

## Serialising run configurations

`SIPNETParametersV1` round-trips through plain Python dicts, making it
straightforward to log the full specification of every run:

```python
import json

config = {
    "preset": runner.preset.value,
    "params": params.model_dump(),
    "climate_path": str(climate.source_path),  # for file-backed instances
}
with open("run_log.json", "w") as f:
    json.dump(config, f, indent=2)
```

For in-memory climate data, serialise the DataFrame:

```python
config["climate_data"] = climate.data.to_dict(orient="records")
```

Note that a file-backed `ClimateDrivers` is not fully self-contained as a
serialised object — it is a reference to an external file.  Store the path
alongside a content hash (e.g., SHA-256 of the file) if you need long-term
reproducibility:

```python
import hashlib

def file_sha256(path):
    h = hashlib.sha256()
    h.update(open(path, "rb").read())
    return h.hexdigest()

config["climate_sha256"] = file_sha256(climate.source_path)
```

---

## Summary: choosing a construction method

| Situation | Recommended method |
|:----------|:------------------|
| Interactive exploration, data manipulation | `ClimateDrivers.from_file` |
| Constructing climate data programmatically | `ClimateDrivers.from_dataframe` |
| File already exists on disk, single run | `ClimateDrivers.from_path` |
| Large ensemble, files pre-exist on disk | `ClimateDrivers.from_path` + `ClimateStaging.SYMLINK` |
| Need to validate a file you did not produce | `ClimateDrivers.from_path` then `.validate()` |
