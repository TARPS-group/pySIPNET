# Ensemble Runs

This guide explains how to run SIPNET over many parameter and climate
configurations using [PyEns](https://github.com/andrewroberts/pyens).  It
assumes you are comfortable with the single-run interface covered in
[Running a Model](running-a-model.md).

!!! note "Optional dependency"
    Ensemble running requires the `pyens` package, which is not yet on PyPI.
    Install it from source:

    ```bash
    pip install git+https://github.com/arob5/PyEns.git
    ```

---

## The problem with loops

The most direct way to run many SIPNET configurations is a Python loop:

```python
results = []
for a_max in [80.0, 100.0, 120.0, 140.0]:
    p = params.model_copy(update={"photosynthesis": ...})  # tedious
    results.append(runner.run(p, climate))
```

This approach works for a handful of runs, but it leaves three problems
unsolved.

**Result association.** Each output must be manually paired with the
configuration that produced it.  With many parameters varying simultaneously,
tracking which output came from which combination becomes error-prone.

**Failure isolation.** If one run raises an exception, the loop stops.
Completed runs are lost unless you write extra error-handling code.

**Parallelism.** Switching from a serial loop to multiprocessing or an HPC
cluster requires rewriting the loop.

PyEns solves all three.  Because `SIPNETModel` is already a plain
``(**kwargs) → SIPNETResult`` callable, passing it to `EnsembleRunner`
requires no adapter or glue code.

---

## SIPNETModel as the PyEns entry point

`SIPNETModel` is a core part of pySIPNET — no PyEns import needed to
construct one.  You give it a runner and a *baseline* parameter set; each
call applies a dict of overrides on top of that baseline:

```python
from pysipnet import SIPNETRunner, ModelPreset, SIPNETModel

runner = SIPNETRunner(preset=ModelPreset.STANDARD)
model  = SIPNETModel(runner, base_params=params, base_climate=climate)
```

You can call it directly to verify it works before handing it to PyEns:

```python
result = model(a_max=112.0, base_veg_resp=0.02)
print(result.outputs[["nee", "gpp"]].sum())
```

Any SIPNET v1 parameter name can be passed as a keyword argument.  The
reserved names `climate` and `events` pass a `ClimateDrivers` or
`EventSequence` directly through to the runner.

---

## A simple parameter sweep

To sweep a single parameter, define a PyEns `Axis` for the sweep dimension
and a `Grid` of values, then run through `EnsembleRunner`:

```python
from pyens import Axis, EnsembleSpec, Grid, EnsembleRunner
from pyens.backends import SequentialBackend

a_max_values = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
ax   = Axis("a_max", size=len(a_max_values))
spec = EnsembleSpec(inputs={
    "a_max": Grid(a_max_values, along=ax),
})

ensemble_runner = EnsembleRunner(model, SequentialBackend())
result = ensemble_runner.run(spec)
```

`result` is an `EnsembleResult` — one `RunRecord` per combination, in the
order they were run:

```python
print(result.n_runs)    # 7
print(result.n_failed)  # 0 if all succeeded

for record in result:
    coord  = record.coordinate   # e.g. {"a_max": 3}  (integer axis index)
    output = record.output       # SIPNETResult
    print(coord, output.gpp().sum())
```

---

## Sweeping multiple parameters independently

Two `Grid` fields that reference **different** `Axis` instances are *crossed*:
the ensemble runs every combination (Cartesian product).

```python
a_max_ax = Axis("a_max",        size=5)
resp_ax  = Axis("base_veg_resp", size=4)

spec = EnsembleSpec(inputs={
    "a_max":         Grid([80.0, 100.0, 120.0, 140.0, 160.0], along=a_max_ax),
    "base_veg_resp": Grid([0.01, 0.02, 0.03, 0.04],            along=resp_ax),
})
# 5 × 4 = 20 runs
```

Two `Grid` fields that reference axes with the same name, size, and labels
are *aligned*: they co-vary and produce one run per axis position (zip
semantics).  In practice this means passing the same `Axis` object to both
`Grid` calls.  Use this when parameters or climate drivers naturally pair up
— for example, site climate and site initial conditions.

---

## Multi-site, multi-member ensembles

The helper functions `sipnet_site_fields` and `sipnet_member_fields` make it
easy to set up the most common ensemble pattern: multiple sites crossed with
multiple ensemble members.

```python
import numpy as np
from pyens import Axis, EnsembleSpec, Fixed, EnsembleRunner
from pyens.backends import LocalBackend
from pysipnet import SIPNETModel
from pysipnet.ensemble import sipnet_site_fields, sipnet_member_fields

# ── Data ──────────────────────────────────────────────────────────────────────
clim_hf = ClimateDrivers.from_file("data/harvard_forest.clim", version="v1")
clim_nr = ClimateDrivers.from_file("data/niwot_ridge.clim",    version="v1")

# ── Axes ──────────────────────────────────────────────────────────────────────
sites   = Axis("site",   labels=["harvard_forest", "niwot_ridge"])
members = Axis("member", size=50)

# ── Sampled parameters ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
a_max_samples = rng.uniform(80, 140, 50).tolist()

# ── Spec ──────────────────────────────────────────────────────────────────────
spec = EnsembleSpec(inputs={
    # Site-level: climate + IC vary per site, aligned on the sites axis
    **sipnet_site_fields(
        sites,
        climates=[clim_hf, clim_nr],
        plant_wood=[30000.0, 24000.0],    # initial wood C per site
        soil=[10000.0, 8500.0],           # initial soil C per site
    ),
    # Member-level: a_max varies per member, aligned on the members axis
    **sipnet_member_fields(
        members,
        a_max=a_max_samples,
    ),
})
# 2 sites × 50 members = 100 runs
print(spec.n_runs)  # 100
```

`sipnet_site_fields` returns a dict whose values are all `Grid` objects aligned
on `sites`.  `sipnet_member_fields` returns a dict aligned on `members`.
Because the two sets of fields use different axes, they are crossed — giving
a full Cartesian product of sites and members.

### Running in parallel

For large ensembles, use `LocalBackend` to distribute across CPU cores:

```python
model          = SIPNETModel(runner, base_params=params)
ensemble_runner = EnsembleRunner(model, LocalBackend(n_workers=8))
result          = ensemble_runner.run(spec)
```

!!! important "Pickling requirement"
    `LocalBackend` uses Python's `multiprocessing` module, which serialises
    the model callable and all field values with `pickle`.  `SIPNETModel` is
    picklable.  `ClimateDrivers` objects (which contain a pandas DataFrame)
    are also picklable.

    If you encounter a `PicklingError`, switch to `SequentialBackend` to
    reproduce the failure with a full traceback, fix the issue, then switch
    back.

---

## Working with EnsembleResult

`ensemble_runner.run()` returns an `EnsembleResult`.  Each record carries
the coordinate (which axis values produced this run) and the output (a
`SIPNETResult`, or an exception if the run failed).

### Checking for failures

```python
if result.n_failed > 0:
    for rec in result.failed:
        print(f"Failed at {rec.coordinate!r}: {rec.output!r}")
```

A failed run is one where the model callable raised an exception — for
example, a `ValidationError` from an invalid parameter combination, or a
`subprocess.CalledProcessError` if SIPNET itself exits non-zero.  Failed
runs are stored in position rather than dropped, so the indices of `outputs`
and `coordinates` always correspond.

### Extracting outputs

```python
# Annual NEE for every successful run
for rec in result.succeeded:
    site   = rec.coordinate.get("site", "?")
    member = rec.coordinate.get("member", "?")
    annual_nee = rec.output.nee().sum()
    print(f"site={site}, member={member}: NEE={annual_nee:.1f} g C m⁻²")
```

### Looking up a specific run

```python
record = result[{"site": "harvard_forest", "member": 12}]
record.output.gpp().plot()
```

### Collecting outputs into a DataFrame

```python
import pandas as pd

rows = []
for rec in result.succeeded:
    rows.append({
        **rec.coordinate,
        "annual_nee": rec.output.nee().sum(),
        "annual_gpp": rec.output.gpp().sum(),
    })
df = pd.DataFrame(rows)
print(df.groupby("site")[["annual_nee", "annual_gpp"]].mean())
```

---

## Describing the ensemble before running

Call `spec.describe()` to inspect the structure of the ensemble before
committing to a run:

```python
print(spec.describe())
# EnsembleSpec
#   Axes:
#     site   (2 labels): harvard_forest, niwot_ridge
#     member (50)
#   Fields:
#     climate    Grid along [site]
#     plant_wood Grid along [site]
#     soil       Grid along [site]
#     a_max      Grid along [member]
#   Total runs: 100
```

This is especially useful before submitting to an HPC cluster.

---

## Using PartialSpec for iterative workflows

`EnsembleSpec.freeze()` returns a `PartialSpec` — a callable with some fields
already fixed and others left open.  This is useful when the climate and site
structure are known upfront but the parameter values will be supplied by a
sampler in a loop:

```python
full_spec = EnsembleSpec(inputs={
    **sipnet_site_fields(sites, climates=[clim_hf, clim_nr]),
    "a_max": ...,        # to be filled in per iteration
})

param_map = full_spec.freeze(free=["a_max"])

for iteration in range(n_iterations):
    theta = sampler.next_sample()
    runnable = param_map(a_max=Grid(theta, along=members))
    result   = ensemble_runner.run(runnable)
    sampler.update(result)
```
