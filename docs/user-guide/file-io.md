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
3. **Write `sipnet.in`** — the run configuration file that tells SIPNET where to
   find its inputs.
4. **Execute the binary** — SIPNET reads the staged files and writes
   `sipnet.out` in the same directory.
5. **Package outputs** — `sipnet.out` is either parsed immediately into memory
   or copied to a persistent location (see [Output I/O](#output-io)).
6. **Clean up** — the working directory is deleted unless `keep_workdir=True`.

This design is intentional: each run is self-contained and stateless.  Two
runs never share a working directory, so they can be parallelized safely with
any executor.

---

## Input I/O

### Working directory location and naming

By default, working directories are placed in `tempfile.gettempdir()` (usually
`/tmp` on Linux/macOS).  Each is named `sipnet_<run_id>_<random>`, where
`<run_id>` defaults to a random UUID hex string and `<random>` is a suffix
added by `tempfile.mkdtemp`.

You can control the location and the `run_id` part of the name:

```python
from pysipnet import SIPNETRunner, ModelFlags

runner = SIPNETRunner(
    flags=ModelFlags.standard(),
    workdir_base="/scratch/my_runs",
)

result = runner.run(params, climate, run_id="baseline_2020")
# working directory: /scratch/my_runs/sipnet_baseline_2020_1l2wlzvt/
```

!!! note "Reusing a run_id is safe for the working directory"
    Every run gets a brand-new directory, because the name ends in a random
    suffix.  Two runs sharing a `run_id` never write to the same working
    directory, so concurrent runs cannot overwrite each other's inputs or read
    each other's output.

    The trade-off is that you cannot predict the directory name in advance.
    Read it from `result.provenance.workdir` instead of constructing it.

    The **output** file is a different matter: its name is predictable by
    design, so it cannot carry a random suffix, and a second run with the same
    `run_id` and `output_dir` is refused.  See
    [Eager vs. lazy output](#eager-vs-lazy-output) below.

### Climate data: in-memory vs. file-backed

`ClimateDrivers` supports two construction modes.

#### In-memory

`from_dataframe` and `from_file` load all climate data into a pandas DataFrame
at construction time.  The runner serializes this to disk as a new `.clim` file
in the working directory.

Columns are named as on the [Climate drivers](../reference/climate-drivers.md)
page (`air_temperature`, `vapor_pressure_deficit`, `time_step_length`, ...).
`from_dataframe` also accepts the short names pySIPNET used previously and
SIPNET's own column names (`tair`, `vpdSoil`, `length`) and renames them; the
stored columns are always the full names.

`data/my_site.clim` here and below stands in for your own climate file; the
[Quickstart](quickstart.md) runs on the Niwot Ridge year this repository ships.

```python
# Full data in memory — good for interactive use and data manipulation
climate = ClimateDrivers.from_file("data/my_site.clim", n_columns=14)
climate.pandas      # DataFrame always available
climate.xarray      # the same on an xarray `time` axis shared with outputs
```

Both validate the data as it is loaded (see
[Labels, lengths and the clock](#labels-lengths-and-the-clock)), and a
`ClimateDrivers` is never re-checked after that: not by the runner, not by
`climate.xarray`, not by a run's output. Treat `climate.pandas` as read-only.

#### File-backed (lazy)

`from_path` creates a lightweight reference to an existing `.clim` file without
reading it into memory.  The runner can stage it by copying or linking the
original file directly, skipping the read-then-write cycle entirely.

```python
# No data loaded — good for ensemble workflows with pre-existing files
climate = ClimateDrivers.from_path("data/my_site.clim", n_columns=14)

print(climate.n_timesteps)  # available without loading data
print(climate.date_range)   # also available without loading data
```

Accessing `climate.pandas` triggers a full load, validates the data, and
caches the result.

!!! warning "Validation is deferred along with the read"
    `from_path` checks only the column count of the first and last rows. The
    full validation — missing values, chronological order, labels against step
    lengths — runs when the file is first loaded, not when the object is
    created. The runner stages the file by copying or linking it without
    reading it, so **a file that fails validation still runs**, and the error
    appears later, the first time something reads the drivers: usually
    `result.outputs.xarray` or `result.outputs["nee"]`, which build their time
    axis from them. Until then `date_range` also assumes the file is sorted.

    To find out before spending a run (or an ensemble) on it, validate up
    front; it loads the file once, and later reads use the cached copy:

    ```python
    climate = ClimateDrivers.from_path("unknown_source.clim")
    climate.validate()   # loads and checks now; raises ValueError on a bad file
    ```

#### Labels, lengths and the clock

SIPNET integrates each row over its `time_step_length` and never checks that
the row's start plus that length is where the next row starts. pySIPNET checks
it whenever climate data is loaded (at construction for `from_file` and
`from_dataframe`, on first read for `from_path`):

- a row that starts more than a minute before the previous one ends is an
  **overlap**, and refused;
- labels that wander more than five minutes from the running sum of the
  lengths are a **drift**, and refused — a clock advancing 3.0007 h per
  3-hour step, or hourly lengths written as `0.042`;
- a row that starts more than a minute after the previous one ends is a
  **gap**, and only warned about.

For a file-backed climate that first read is often the output's Dataset,
which is where a bad `from_path` file is reported if it was never validated.

A `.clim` file cannot say which clock its labels are on, so tell
`ClimateDrivers`:

```python
climate = ClimateDrivers.from_file("data/my_site.clim", time_zone="UTC-07:00")
climate.xarray["time"].attrs["time_zone"]   # 'UTC-07:00'
```

Accepted values are `"UTC"` and fixed offsets `"UTC±HH:MM"`; a named zone
such as `"America/Denver"` is refused, because daylight saving time would make
the labels jump. The declaration is metadata only, travels to every output run
on these drivers, and is saved by `RunConfig`.

### Climate file staging modes

When staging a file-backed climate instance, the runner can either **copy**
the file or create a **symlink**:

```python
from pysipnet import SIPNETRunner, ModelFlags, ClimateStaging

runner = SIPNETRunner(
    flags=ModelFlags.standard(),
    climate_staging=ClimateStaging.SYMLINK,   # zero I/O for large files
)
```

| Mode | Behavior | When to use |
|:-----|:----------|:------------|
| `COPY` (default) | `shutil.copy2` — physical copy | All platforms; when source files may move during a run |
| `SYMLINK` | Symbolic link to the resolved absolute path | Linux/macOS; large files; source files are stable for the run duration |

If `SYMLINK` is requested but `os.symlink` raises `OSError` (e.g. on Windows
or across filesystem boundaries), the runner falls back to `COPY` and emits a
`UserWarning`.

`climate_staging` has no effect on in-memory `ClimateDrivers` instances —
they are always serialized through the I/O layer regardless of this setting.

| `ClimateDrivers` type | `COPY` | `SYMLINK` |
|:----------------------|:-------|:----------|
| In-memory (`from_dataframe`, `from_file`) | Write from DataFrame | Write from DataFrame |
| File-backed (`from_path`) | `shutil.copy2` | Symlink (fallback to copy) |

---

## Output I/O

### Eager vs. lazy output

After each run, pySIPNET packages the SIPNET output (`sipnet.out`) as a
`SIPNETOutput` object stored in `result.outputs`.  There are two modes:

**Eager (default):** the output file is parsed immediately and held in memory.
The working directory is then deleted.  If you also want the raw file retained
on disk, set `keep_workdir=True` (see [Keeping the working directory](#keeping-the-working-directory)).

```python
runner = SIPNETRunner(flags=ModelFlags.standard())
result = runner.run(params, climate)

# Data is already in memory:
df = result.outputs.pandas    # pandas DataFrame
ds = result.outputs.xarray    # xarray Dataset with units and descriptions
nee = result.outputs["nee"]       # one variable, by name or alias
```

**Lazy (file-backed):** set `output_dir` on the runner.  Before the working
directory is deleted, `sipnet.out` is copied to
`<output_dir>/sipnet_<run_id>.out`.  The result holds a file-backed
`SIPNETOutput` — no DataFrame is created until you explicitly access the data.

Because the name comes from the run id, a second run with the same `run_id` and
`output_dir` is **refused before it starts**, rather than replacing the file.
The earlier result reads that path lazily, so overwriting it would silently
change the numbers that result answers with, long after it reported success.
Give each run a distinct id — the default is a fresh UUID — or pass
`overwrite=True` when the earlier output is finished with.  The refusal is
based on the file being there, so deleting it releases the name again, along
with the protection for any result still pointing at it.

```python
# A loop that reruns one member under a fixed id, keeping one file on disk:
runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=Path("out"), overwrite=True)

# ... or decide per call, leaving the runner's default in place:
result = runner.run(params, climate, run_id="current", overwrite=True)
```

```python
runner = SIPNETRunner(
    flags=ModelFlags.standard(),
    output_dir=Path("run_outputs"),
)
result = runner.run(params, climate, run_id="baseline")

# No data in memory yet:
print(result.outputs.source_path)
# PosixPath('run_outputs/sipnet_baseline.out')

# Trigger load on demand:
df = result.outputs.pandas
```

The runner attaches the climate drivers to the `SIPNETOutput`, so
`result.outputs.xarray` is built on their time axis — their start times, step
lengths and declared `time_zone` — even when the output is read from disk
later. A file-backed climate is not read until a Dataset is asked for. To
rebuild the same axis for an output file you kept, pass its drivers:

```python
out = SIPNETOutput.from_path("run_outputs/sipnet_baseline.out", climate=climate)
```

Without `climate=`, the axis is rebuilt from the labels SIPNET printed, which
are rounded to 0.01 h; `ds.attrs["time_axis_source"]` says which you got.

### output_dir: runner-level and per-call

`output_dir` can be set at the runner level (applies to all runs) or
overridden per call:

```python
runner = SIPNETRunner(
    flags=ModelFlags.standard(),
    output_dir=Path("default_outputs"),   # runner-level default
)

# Uses runner default:
r1 = runner.run(params, climate, run_id="run_a")

# Overrides runner default for this call only:
r2 = runner.run(params, climate, run_id="run_b", output_dir=Path("special"))

# Suppresses output persistence entirely for this call:
r3 = runner.run(params, climate, output_dir=None)
```

!!! note "output_dir is always kept out of the working directory"
    The working directory is deleted after each run, so anything written
    inside it is lost.  pySIPNET checks, before the binary runs, that
    `output_dir` is not inside the run's working directory, and raises
    `ValueError` if it is.

    In practice you cannot trigger this: each run's directory name ends in a
    random suffix that is not known until the run starts, so a path you supply
    can never be inside it.  The check is there so the guarantee does not
    depend on that.

### `select`: the memory-efficient route

`.pandas` and `.xarray` give you everything: on a file-backed output they read
and cache all 35 columns. `select([...])` reads only the variables you name and
holds only those, which for a large ensemble is the difference between keeping
one column per member and keeping every member's full output.

```python
# Just NEE and GPP — year/day_of_year/hour_of_day are always included:
ds = result.outputs.select(["nee", "gpp"])                    # xarray Dataset (default)
df = result.outputs.select(["nee", "gpp"], format="pandas")   # ... or a DataFrame
# DataFrame columns: year, day_of_year, hour_of_day,
#                    net_ecosystem_exchange, gross_primary_production

ds  = result.outputs[["nee", "gpp"]]   # shorthand for the xarray default
nee = result.outputs["nee"]            # one variable, as a DataArray
```

Names or aliases both work (`"nee"`, `"NEE"`, `"net_ecosystem_exchange"`).

Measured on a half-hourly year, 35 columns, selecting NEE from a file-backed
output:

| | retained | columns held |
|:--|:--|:--|
| `select(["nee"], format="pandas")` | 0.60 MB | 4 |
| `pandas[["net_ecosystem_exchange"]]` | 4.94 MB | 35 |

It is **not** a speed optimization: the parser scans every field of every line
either way, so reading a few columns costs about four fifths of reading them
all. The saving is memory.

A selection never re-reads a column already in memory. Selecting NEE and then
GPP costs the same two reads as selecting both together, and asking for either
again costs none — so a likelihood over several output variables never re-reads
the file, however the request is spelled. The one exception is `.pandas` or
`.xarray` after a selective read: a whole-file view reads the file again,
because only the file states the order its columns belong in.

What stays in memory is the columns you have asked for, plus the three time
coordinates — not the whole output. (Peak memory during a read is another
matter: pandas needs room to parse.)

On a memory-backed instance no file I/O occurs at all: the selection slices the
DataFrame already in memory.

### Keeping the working directory

As an alternative to `output_dir`, you can tell the runner to skip workdir
cleanup entirely:

```python
runner = SIPNETRunner(
    flags=ModelFlags.standard(),
    keep_workdir=True,
)

result = runner.run(params, climate, run_id="debug_run")

print(result.provenance.workdir)
# /tmp/sipnet_debug_run_9fq2xk4b

import os
for f in os.listdir(result.provenance.workdir):
    print(f)
# sipnet.param  sipnet.clim  sipnet.in  sipnet.out
```

The preserved directory contains exactly the files SIPNET saw, so you can
reproduce the run manually:

```bash
cd /tmp/sipnet_debug_run_9fq2xk4b   # the path printed above
/path/to/.sipnet_cache/sipnet
```

`output_dir` and `keep_workdir=True` serve different purposes and can be used
together:

| | `output_dir` | `keep_workdir=True` |
|:--|:--|:--|
| What is kept | Only `sipnet.out`, copied to a named location | The entire working directory: param, clim, in, and out files |
| Primary use | Ensemble post-processing; lazy loading | Debugging; reproducibility audits |
| File naming | `sipnet_<run_id>.out` in your chosen directory | All files in `sipnet_<run_id>_<random>/` under `workdir_base` |
| Reusing a `run_id` | Refused unless `overwrite=True` | Always a fresh directory |

### Summary: choosing an output mode

| Situation | Recommended approach |
|:----------|:--------------------|
| Interactive exploration, single run | Default (no `output_dir`) — data in memory |
| Need the raw file for archival | `output_dir=` on runner or `keep_workdir=True` |
| Large ensemble, full outputs needed | `output_dir=` — lazy-load member by member |
| Large ensemble, only a few columns needed | `output_dir=` + `result.outputs.select([...])` |
| Debugging a failing run | `keep_workdir=True` — inspect all files in `provenance.workdir` |

## Bundled reference data

pySIPNET ships one real SIPNET input set and one real SIPNET output inside the
package, under `pysipnet/data/niwot/`, so that code depending on pySIPNET can
test against genuine model data with nothing but `pip install` — no source
checkout, no compiled binary. The files are the same ones pySIPNET's own tests
use; there is exactly one copy.

| File | What it is |
|:--|:--|
| `sipnet.param` | SIPNET's own Niwot Ridge parameters, byte-identical to the pinned submodule's smoke-test copy |
| `sipnet.clim` | The first 800 rows of the matching climate record (about one year from November 1998, sub-daily, 14-column layout) |
| `niwot_standard.out.csv` | pySIPNET's golden baseline: standard-flag output on the first 60 climate rows |

Three functions in `pysipnet.io.reference`, also exported from `pysipnet`,
give access to them:

```python
from pysipnet import niwot_reference_climate, niwot_reference_files, niwot_reference_output

output = niwot_reference_output()     # SIPNETOutput, 60 steps, no binary needed
nee = output["nee"]                   # DataArray on a time axis built from the
                                      # climate's own step lengths
climate = niwot_reference_climate()   # ClimateDrivers, 800 steps, in memory
paths = niwot_reference_files()       # paths.param, paths.clim, paths.output, paths.readme
```

The output loader returns the golden already paired with the climate's
`time_step_length` column and flagged `ModelFlags.standard()`, so its Dataset
carries the same time axis a live run would, and selecting a column SIPNET wrote
as constant zero is refused just as it is for a live run.

Two things to know before building on the golden. It is a narrow slice of the
model: one dormant-season month in which photosynthesis is almost inactive and
ten of the 35 columns are identically zero. And it changes whenever the
baseline is deliberately regenerated, for a SIPNET pin bump or an intended
wrapper change, so it is a fixed sample of the model's output, not a reference
solution. The `README.md` beside the files records the provenance in full.

There is not yet a reader that turns `sipnet.param` into a `SIPNETParameters`
(issue #19); `read_param_file(paths.param)` gives the flat
`{sipnet_name: value}` dictionary.
