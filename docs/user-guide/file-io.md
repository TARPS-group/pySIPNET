# File I/O

This guide covers how pySIPNET interacts with the filesystem — both for
inputs (climate data) and outputs (model results).

For concrete end-to-end examples that combine these options, see
[Common Workflows](workflows.md).

---

## How a run uses the filesystem

Every call to `SIPNETRunner.run()` (and by extension `SIPNETModel.__call__`)
follows the same file lifecycle:

1. **Create a working directory** — a fresh subdirectory under `workdir_base`
   (default: the system temp directory).
2. **Stage inputs** — write `sipnet.param`, `sipnet.clim`, and (optionally)
   `events.in` into the working directory.
3. **Write `sipnet.in`** — the two-line config file that tells SIPNET where to
   find its inputs.
4. **Execute the binary** — SIPNET reads the staged files and writes
   `sipnet.out` in the same directory.
5. **Package outputs** — `sipnet.out` is either parsed immediately into memory
   or copied to a persistent location (see [Output I/O](#output-io)).
6. **Clean up** — the working directory is deleted unless `keep_workdir=True`.

This design is intentional: each run is self-contained and stateless.  Two
runs never share a working directory, so they can be parallelised safely with
any executor.

---

## Input I/O

### Working directory location and naming

By default, working directories are placed in `tempfile.gettempdir()` (usually
`/tmp` on Linux/macOS) and named `sipnet_<run_id>` where `<run_id>` is a
random UUID hex string.

You can control both:

```python
from pysipnet import SIPNETRunner, ModelPreset

runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    workdir_base="/scratch/my_runs",
)

result = runner.run(params, climate, run_id="baseline_2020")
# working directory: /scratch/my_runs/sipnet_baseline_2020/
```

!!! note "run_id reuse"
    If the directory `sipnet_<run_id>` already exists, it is reused and its
    contents overwritten.  Use distinct `run_id` values across runs if you
    need to preserve them all.

### Preserving the working directory

Set `keep_workdir=True` on the runner to suppress cleanup:

```python
runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    keep_workdir=True,
)

result = runner.run(params, climate, run_id="debug_run")

print(result.provenance.workdir)
# /tmp/sipnet_debug_run

import os
for f in os.listdir(result.provenance.workdir):
    print(f)
# sipnet.param  sipnet.clim  sipnet.in  sipnet.out
```

The preserved directory contains exactly the files SIPNET saw, so you can
reproduce the run manually:

```bash
cd /tmp/sipnet_debug_run
./path/to/sipnet_standard
```

### Climate data: in-memory vs. file-backed

`ClimateDrivers` supports two construction modes.

#### In-memory

`from_dataframe` and `from_file` load all climate data into a pandas DataFrame
at construction time.  The runner serialises this to disk as a new `.clim` file
in the working directory.

```python
# Full data in memory — good for interactive use and data manipulation
climate = ClimateDrivers.from_file("data/era5_site1.clim", version="v1")
climate.data        # DataFrame always available
climate.validate()  # full validation runs immediately
```

#### File-backed (lazy)

`from_path` creates a lightweight reference to an existing `.clim` file without
reading it into memory.  The runner can stage it by copying or linking the
original file directly, skipping the read-then-write cycle entirely.

```python
# No data loaded — good for ensemble workflows with pre-existing files
climate = ClimateDrivers.from_path("data/era5_site1.clim", version="v1")

print(climate.n_timesteps)  # available without loading data
print(climate.date_range)   # also available without loading data
```

Accessing `climate.data` triggers a full load and caches the result.

!!! warning "Chronological ordering assumption"
    `from_path` validates the column count of the first and last rows but
    **assumes the file is sorted chronologically**.  Call `.validate()` to
    perform a complete check (triggers a full load):

    ```python
    climate = ClimateDrivers.from_path("unknown_source.clim")
    climate.validate()   # raises ValueError if not monotone
    ```

### Climate file staging modes

When staging a file-backed climate instance, the runner can either **copy**
the file or create a **symlink**:

```python
from pysipnet import SIPNETRunner, ModelPreset, ClimateStaging

runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    climate_staging=ClimateStaging.SYMLINK,   # zero I/O for large files
)
```

| Mode | Behaviour | When to use |
|:-----|:----------|:------------|
| `COPY` (default) | `shutil.copy2` — physical copy | All platforms; when source files may move during a run |
| `SYMLINK` | Symbolic link to the resolved absolute path | Linux/macOS; large files; source files are stable for the run duration |

If `SYMLINK` is requested but `os.symlink` raises `OSError` (e.g. on Windows
or across filesystem boundaries), the runner falls back to `COPY` and emits a
`UserWarning`.

`climate_staging` has no effect on in-memory `ClimateDrivers` instances —
they are always serialised through the I/O layer regardless of this setting.

| `ClimateDrivers` type | `COPY` | `SYMLINK` |
|:----------------------|:-------|:----------|
| In-memory (`from_dataframe`, `from_file`) | Write from DataFrame | Write from DataFrame |
| File-backed (`from_path`) | `shutil.copy2` | Symlink (fallback to copy) |

---

## Output I/O

### Eager vs. lazy output

After each run, pySIPNET packages the SIPNET output (`sipnet.out`) as a
:class:`~pysipnet.output.SIPNETOutput` object stored in
`result.outputs`.  There are two modes:

**Eager (default):** the output file is parsed immediately and held in memory.
The working directory is then deleted.  No files are retained.

```python
runner = SIPNETRunner(preset=ModelPreset.STANDARD)
result = runner.run(params, climate)

# Data is already in memory:
df = result.outputs.data          # pandas DataFrame
nee_series = result.nee()         # convenience accessor
```

**Lazy (file-backed):** set `output_dir` on the runner.  Before the working
directory is deleted, `sipnet.out` is copied to
`<output_dir>/sipnet_<run_id>.out`.  The result holds a file-backed
`SIPNETOutput` — no DataFrame is created until you explicitly access the data.

```python
runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    output_dir=Path("run_outputs"),
)
result = runner.run(params, climate, run_id="baseline")

# No data in memory yet:
print(result.outputs.source_path)
# PosixPath('run_outputs/sipnet_baseline.out')

# Trigger load on demand:
df = result.outputs.data
```

### output_dir: runner-level and per-call

`output_dir` can be set at the runner level (applies to all runs) or
overridden per call:

```python
runner = SIPNETRunner(
    preset=ModelPreset.STANDARD,
    output_dir=Path("default_outputs"),   # runner-level default
)

# Uses runner default:
r1 = runner.run(params, climate, run_id="run_a")

# Overrides runner default for this call only:
r2 = runner.run(params, climate, run_id="run_b", output_dir=Path("special"))

# Suppresses output persistence entirely for this call:
r3 = runner.run(params, climate, output_dir=None)
```

!!! warning "output_dir must be outside the working directory"
    The working directory is deleted after each run, so `output_dir` must
    not be the same as, or a subdirectory of, the working directory.
    pySIPNET checks this **before** the binary runs and raises `ValueError`
    immediately if the paths overlap:

    ```python
    # workdir will be /tmp/sipnet_myrun/
    # This raises ValueError before anything runs:
    runner.run(params, climate, run_id="myrun",
               output_dir=Path("/tmp/sipnet_myrun/outputs"))
    ```

### Column-selective loading

For large ensemble outputs it is often wasteful to load every column.
`SIPNETOutput.load(columns=[...])` reads only the named columns from the file,
without caching the result:

```python
# Load just NEE and GPP — year/day/time are always included:
subset = result.outputs.load(columns=["nee", "gpp"])
# Returns a DataFrame with columns: year, day, time, nee, gpp
```

On a memory-backed instance, `load(columns=[...])` slices the in-memory
DataFrame — no file I/O occurs.

### Summary: choosing an output mode

| Situation | Recommended approach |
|:----------|:--------------------|
| Interactive exploration, single run | Default (no `output_dir`) — data in memory |
| Need the raw file for archival | `output_dir=` on runner or `keep_workdir=True` |
| Large ensemble, full outputs needed | `output_dir=` — lazy-load member by member |
| Large ensemble, only a few columns needed | `output_dir=` + `result.outputs.load(columns=[...])` |
| Debugging a failing run | `keep_workdir=True` — inspect all files in `provenance.workdir` |
