# Reproducible Workflows

This guide covers how to record, save, and reload SIPNET run specifications
so that results can be reproduced exactly — whether for a single exploratory
run, a large ensemble, or an iterative workflow.

---

## Why reproducibility matters

A model run is defined by four things: the binary preset, the parameter set,
the climate forcing, and any management events.  In interactive work, these
inputs are live Python objects in memory.  When a session ends, they are
gone.  Reproducibility means being able to reconstruct those inputs precisely
and re-run the model to get the same output.

Beyond re-running, a saved specification is also a communication artefact:
it tells a future reader (or your future self) exactly what was fed into the
model to produce a given result.

---

## `RunConfig`: saving and loading a run specification

`RunConfig` is pySIPNET's first-class abstraction for a complete, reusable
run specification.  It holds the preset, parameters, climate, and (optionally)
events, and knows how to write itself to disk and reconstruct itself.

### Saving

```python
from pysipnet import RunConfig, ModelPreset

config = RunConfig(
    preset=ModelPreset.STANDARD,
    params=params,
    climate=climate,
)
config.save("my_run/")
```

`save` creates a directory with three possible files:

```
my_run/
├── config.json    # preset, parameters, climate mode, metadata
├── sipnet.clim    # climate data (default — see below)
└── events.in      # only written when events were provided
```

`config.json` records versioning metadata alongside the run specification:

```json
{
  "preset": "standard",
  "params": { ... },
  "climate": {"mode": "copy"},
  "has_events": false,
  "sipnet_commit": "e4abf14f...",
  "pysipnet_version": "0.1.0.dev0",
  "created_at": "2026-05-29T14:32:01+00:00"
}
```

`sipnet_commit` records the exact SIPNET C source revision.  If results ever
need to be verified or compared across time, this field tells you which binary
to build.

### Loading and re-running

```python
from pysipnet import RunConfig, SIPNETRunner

config = RunConfig.load("my_run/")
runner = SIPNETRunner(preset=config.preset)
result = runner.run(config.params, config.climate, events=config.events)
```

The loaded climate is file-backed and lazy: the data is not read from
`sipnet.clim` until the runner actually needs it.

### Promoting an exploratory result

After an interactive session you can promote any result to a saved
configuration without reconstructing the inputs manually:

```python
result = model()                       # exploratory run

config = RunConfig.from_result(result)
config.save("interesting_run/")
```

---

## Climate archiving: default vs `reference_only`

By default, `save` writes the full climate data into the config directory.
This produces a self-contained archive that can be moved, zipped, or shared
without any external dependencies.

For workflows where the climate file is large and shared across many configs
(ensemble or iterative runs — see below), writing a separate copy for each
config would waste disk space.  Passing `reference_only=True` instead stores
only the absolute path and a SHA-256 hash of the source file:

```python
from pysipnet import ClimateDrivers, ModelPreset, RunConfig

climate = ClimateDrivers.from_path("data/era5_site1.clim")
config = RunConfig(preset=ModelPreset.STANDARD, params=params, climate=climate)
config.save("context/", reference_only=True)
```

`reference_only=True` requires a file-backed `ClimateDrivers` instance
(created via `from_path`).  In-memory instances have no source path and can
only be saved without this flag.

On load, pySIPNET checks that the referenced file still exists and verifies
its SHA-256 hash.  If the hash has changed — indicating the file was modified
after the config was saved — a `UserWarning` is raised.  The config is still
loaded; the warning is informational.

---

## Fixed ensemble runs

A fixed ensemble specifies all members upfront before any runs happen.  A
typical case: 500 initial-condition members that share the same parameters and
climate but vary across a set of initial soil carbon and water values.

The efficient layout is **one shared `RunConfig` plus a compact override
manifest**:

```
my_ensemble/
├── context/              ← shared RunConfig (climate embedded once)
│   ├── config.json
│   └── sipnet.clim
└── overrides.csv         ← one row per member, one column per varying input
```

`overrides.csv` is plain data — pySIPNET does not own its format.  A
reasonable schema:

```
member_id, soil_init, soil_water_frac_init
0,         8000.0,    0.40
1,         8500.0,    0.42
...
```

Reproduce any member from this record:

```python
import pandas as pd
from pysipnet import RunConfig, SIPNETRunner, SIPNETModel

context = RunConfig.load("my_ensemble/context/")
overrides = pd.read_csv("my_ensemble/overrides.csv", index_col="member_id")

runner = SIPNETRunner(preset=context.preset)
model  = SIPNETModel(runner, base_params=context.params, base_climate=context.climate)

row = overrides.loc[42]
result = model(soil=row["soil_init"], soil_water_frac=row["soil_water_frac_init"])
```

### Using PyEns

If you are using PyEns, its `EnsembleSpec` is the structured override
manifest: it describes the ensemble axes and the parameter grids along them.
`EnsembleSpec` has its own `dump` and `load` methods for serialisation.
Save the shared context and the spec together so that the full specification
is self-contained:

```python
from pyens import Axis, EnsembleSpec, EnsembleRunner
from pyens.backends import LocalBackend
from pysipnet import ModelPreset, RunConfig, SIPNETModel, SIPNETRunner
from pysipnet.ensemble import sipnet_member_fields

# Define the shared context first — it is the source of truth for the run
context = RunConfig(preset=ModelPreset.STANDARD, params=base_params, climate=climate)
runner  = SIPNETRunner(preset=context.preset)
model   = SIPNETModel(runner, base_params=context.params, base_climate=context.climate)

# Build and run the ensemble
members = Axis("member", size=500)
fields  = sipnet_member_fields(
    members,
    soil=soil_samples,
    soil_water_frac=water_samples,
)
spec = EnsembleSpec(inputs={**fields})

ensemble_runner = EnsembleRunner(model, LocalBackend(n_workers=8))
results = ensemble_runner.run(spec)

# Archive — context was already defined; save it alongside the spec
context.save("my_ensemble/context/")
spec.dump("my_ensemble/spec.json")
```

Reload for replay or further analysis:

```python
context = RunConfig.load("my_ensemble/context/")
spec    = EnsembleSpec.load("my_ensemble/spec.json")
```

---

## Iterative Runs

Some workflows involve many sequential model evaluations driven by an
external algorithm, where the inputs to each evaluation depend on the results
of previous ones.  Examples include MCMC samplers, optimisation routines, and
data assimilation / state estimation algorithms such as ensemble Kalman
filters or particle filters.

### Separating responsibilities

pySIPNET's responsibility is to make each individual model evaluation
reproducible given its inputs.  This is already true by construction: runs are
stateless, and the same parameter set and climate always produce the same
output.

The external algorithm's responsibility is to record its own state — the
random seed, the step history, the acceptance decisions, the filter weights.
pySIPNET does not provide classes for this.

### The two-artefact pattern

For a typical iterative run, two artefacts together make the workflow fully
reproducible:

**1. A shared `RunConfig`** — written once at the start, capturing the preset,
the fixed (non-varying) parameters, and the climate.  Use `reference_only=True`
to avoid copying the climate file for every evaluation.

```python
from pysipnet import ClimateDrivers, ModelPreset, RunConfig, SIPNETModel, SIPNETRunner

climate = ClimateDrivers.from_path("data/era5_site1.clim")
context = RunConfig(preset=ModelPreset.STANDARD, params=base_params, climate=climate)
runner  = SIPNETRunner(preset=context.preset)
model   = SIPNETModel(runner, base_params=context.params, base_climate=context.climate)
context.save("experiment/context/", reference_only=True)
```

**2. A run log** — a table appended to during the run, with one row per model
evaluation.  The columns are whatever the algorithm needs: iteration index,
parameter values sampled, log-posterior or likelihood, acceptance flag, etc.

```python
import csv

log_path = open("experiment/run_log.csv", "w", newline="")
writer = csv.DictWriter(log_path, fieldnames=["iter", "a_max", "base_veg_resp", "log_lik"])
writer.writeheader()

for i, (a_max, bvr) in enumerate(sampler):
    result = model(a_max=a_max, base_veg_resp=bvr)
    log_lik = compute_log_likelihood(result)
    writer.writerow({"iter": i, "a_max": a_max, "base_veg_resp": bvr, "log_lik": log_lik})
```

### Replaying any evaluation

Given the two artefacts, replaying evaluation 42 is straightforward:

```python
import pandas as pd
from pysipnet import RunConfig, SIPNETModel, SIPNETRunner

context = RunConfig.load("experiment/context/")
log     = pd.read_csv("experiment/run_log.csv")

runner = SIPNETRunner(preset=context.preset)
model  = SIPNETModel(runner, base_params=context.params, base_climate=context.climate)

row    = log[log["iter"] == 42].iloc[0]
result = model(a_max=row["a_max"], base_veg_resp=row["base_veg_resp"])
```

### Saving outputs selectively

Storing full model output timeseries for every iteration is often infeasible.
`SIPNETResult.outputs` is an in-memory DataFrame — nothing is written to disk
unless you ask for it.  A practical pattern is to save only summary statistics
per iteration:

```python
writer.writerow({
    "iter": i,
    "a_max": a_max,
    "base_veg_resp": bvr,
    "annual_nee": result.nee().sum(),
    "annual_gpp": result.gpp().sum(),
    "log_lik": log_lik,
})
```

Full outputs can be recomputed on demand for any evaluation of interest by
replaying it as shown above.

### PyEns PartialSpec for iterative ensemble workflows

When each iteration of an external algorithm requires an *ensemble* of model
runs (e.g. an ensemble Kalman filter, where each iteration updates an ensemble
of state vectors), PyEns' `PartialSpec` is the appropriate tool.  It fixes the
structural parts of the spec (site layout, climate) while leaving parameter
fields open to be filled in per iteration.  See
[Ensemble Runs](ensemble-runs.md#using-partialspec-for-iterative-workflows)
for the `PartialSpec` pattern.

---

## Long-term archival

When reproducing results months or years later, the metadata fields in
`config.json` provide the anchor points:

| Field | What it tells you |
|:------|:------------------|
| `sipnet_commit` | Exact SIPNET C source revision.  Use `git checkout <hash>` in the `sipnet/` submodule, then `make sipnet` to rebuild the same binary. |
| `pysipnet_version` | pySIPNET version at save time.  Useful for identifying API or file-format differences if behaviour seems to differ. |
| `created_at` | ISO 8601 UTC timestamp of when the config was written. |

For `reference_only=True` configs, the `sha256` field in `config.json` lets
you verify that the climate file has not been modified since the config was
saved.  If you are archiving a complete experiment for long-term storage,
prefer the default (copy) mode so the climate data travels with the
specification.
