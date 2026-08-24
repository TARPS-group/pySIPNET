# Design

## Guiding principles

### 1. Hierarchical named data structures

All inputs are organised into domain-grouped Pydantic models:

```
SIPNETParameters
├── initial_conditions: InitialConditions
├── photosynthesis:     PhotosynthesisParams
├── phenology:          PhenologyParams
├── respiration:        RespirationParams
├── allocation:         AllocationParams
├── water:              WaterParams
└── leaf:               LeafPhysiologyParams
```

No code in pySIPNET reads parameters by column index or relies on dict key ordering.

### 2. Unambiguous units for every parameter

Every parameter field carries its units via a [`ParameterSpec`][pysipnet.parameters.base.ParameterSpec] embedded in Pydantic's `json_schema_extra`.  Unit strings follow [Pint](https://pint.readthedocs.io/) format and are validated at class-definition time:

| Quantity | `unit` string |
|:---------|:--------------|
| nmol CO₂ g⁻¹ leaf s⁻¹ | `"nmol / (g * s)"` |
| g C m⁻² | `"g / m**2"` |
| dimensionless | `"1"` |
| °C | `"degC"` |
| year⁻¹ | `"1 / year"` |
| Einstein m⁻² (total per step) | `"einstein / m**2"` |

When the physical unit does not fully capture the substance (e.g., "grams of *carbon*" vs. generic "grams"), the `constituent` field on `ParameterSpec` provides the qualifier (`"C"`, `"N"`, `"CO2 g-1 leaf"`).

### 3. Documented parameter domains

Every parameter has a [`ParameterDomain`][pysipnet.parameters.base.ParameterDomain] that encodes its mathematical support:

| Domain | Support | Bijector |
|:-------|:--------|:---------|
| `REAL` | (−∞, ∞) | identity |
| `POSITIVE` | (0, ∞) | log / softplus |
| `NON_NEGATIVE` | [0, ∞) | softplus |
| `UNIT_INTERVAL` | [0, 1] | logistic / sigmoid |
| `OPEN_UNIT_INTERVAL` | (0, 1) | logit |

Retrieve domains programmatically:

```python
from pysipnet.parameters.base import get_parameter_specs, ParameterDomain
from pysipnet.parameters.model import SIPNETParameters

specs = get_parameter_specs(SIPNETParameters)
log_params = [k for k, s in specs.items() if s.domain == ParameterDomain.POSITIVE]
```

This is useful, for example, to map parameters to an unconstrained domain for parameter estimation tasks. 

### 4. Fully serialisable run specification

A complete run specification (parameters + climate + flags) round-trips through plain dict/JSON:

```python
config_dict = {
    "params":  params.model_dump(),
    "climate": climate.data.to_dict(orient="list"),
    "flags":   flags.model_dump(),
}
params2  = SIPNETParameters.model_validate(config_dict["params"])
```

This enables:

- Storing run provenance as JSON alongside results
- Diffing parameter sets across ensemble members
- Passing configs to worker processes without complex pickling

### 5. Clean separation of concerns

```
Data layer    →    IO layer    →    Runner    →    Result
(Pydantic +        (.param,         (subprocess    (DataFrame)
 dataclass)        .clim, .out)      + workdir)
```

Nothing above the IO layer touches the filesystem.  The runner takes Python objects, the IO layer materialises them to disk, and the runner calls the binary.

### 6. Stateless, isolated runs

Each call to `SIPNETRunner.run()` writes to a fresh temporary directory.  Runs never share files.  This property is what makes parallelism trivial.

## Ensemble running (out of scope for this package)

pySIPNET is intentionally scoped to single runs.  The ensemble layer is separate.  Recommended tools:

**[Hydra](https://hydra.cc/)** — for structured sweep specification (grid, random, Ax/Optuna).  pySIPNET's Pydantic models map naturally to Hydra structured configs via `OmegaConf`.

**[Parsl](https://parsl-project.org/)** — for execution on HPC clusters (SLURM, PBS) and cloud.  A Parsl `python_app` wrapping `SIPNETRunner.run()` + `model_dump()` / `model_validate()` for serialisation is sufficient for most ensemble workflows.

**[Dask](https://dask.org/)** — for local multi-core or distributed cluster execution.  Simpler setup than Parsl; better for local development.

The key design property that enables all of these: `SIPNETRunner.run()` is a pure function from a serialisable config to a serialisable result.

## Model options

From SIPNET v2.0.0 onward, model options are chosen at run time rather than compiled in. pySIPNET therefore builds one binary and expresses the configuration as data: a `ModelFlags` instance, written into the `sipnet.in` file generated for each run.

`ModelFlags` carries two responsibilities:

- **Configuring SIPNET.** `to_config_keys()` renders the flags as the uppercase `sipnet.in` keys SIPNET reads. Every flag is written, including defaults, so a saved run does not depend on the binary's built-in defaults.
- **Deciding which parameters are required.** `SIPNETParameters.validate_for_flags()` uses the flags to check the parameter set before SIPNET is invoked, so a mismatch is a Python error naming the missing field rather than an exit code from the binary.

`ModelFlags` also rejects the three combinations SIPNET itself refuses (`gdd` with `soil_phenol`; `anaerobic` without `water_hresp`; `nitrogen_cycle` without both `litter_pool` and `anaerobic`), mirroring `validateContext()` in SIPNET's `src/common/context.c`.

Separately, three flags are refused because *pySIPNET* cannot serve them: `nitrogen_cycle`, `anaerobic` and `flooding`. SIPNET supports all three, but each requires parameters `SIPNETParameters` does not define, so a run would fail inside SIPNET with "Did not find required parameter" rather than in Python. `UNSUPPORTED_FLAGS` in `pysipnet/parameters/model.py` records what each one needs; the error message reproduces that list. Removing an entry from that table is what enables the flag once its parameters exist.

Earlier versions of pySIPNET compiled one binary per option combination and shipped a `ModelPreset` enum to select between them. Both are gone: arbitrary combinations are now free, so a closed set of presets would only get in the way. `ModelFlags.standard()` and `ModelFlags.forest()` remain as named starting points, and an optional `name` field carries that label into the run record.

## Per-year rate parameters

The following SIPNET parameters are specified as **per-year rates** in the
param file but converted to per-day internally (÷ 365):

- `respiration.base_veg_resp`
- `respiration.base_fine_root_resp`
- `respiration.base_coarse_root_resp`
- `respiration.base_soil_resp`
- `respiration.litter_breakdown_rate`
- `allocation.fine_root_turnover_rate`
- `allocation.coarse_root_turnover_rate`
- `allocation.wood_turnover_rate`
- `phenology.leaf_turnover_rate`

These fields have `per_year=True` in their `ParameterSpec`.  The Python interface always works in per-year units, matching SIPNET's param file convention.  Do not divide by 365 before passing values to pySIPNET.
