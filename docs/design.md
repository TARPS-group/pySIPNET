# Design

## Guiding principles

### 1. Hierarchical named data structures

All inputs are organized into domain-grouped Pydantic models:

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

### 2. Unambiguous names and units for every variable

Every output column is described by a [`VariableSpec`][pysipnet.variables.VariableSpec] in the registry `pysipnet.variables.OUTPUT_VARIABLES`: its pySIPNET name, the SIPNET column it comes from, whether it is a pool at the end of the step or a total over it, its units, a description, plot labels and aliases. The [Output variables](reference/output-variables.md) page is generated from that registry. Every parameter field carries the same kind of information via a [`ParameterSpec`][pysipnet.parameters.base.ParameterSpec] embedded in Pydantic's `json_schema_extra`; `pysipnet.PARAMETER_SPECS` is the flat `{"group.field": spec}` view, `resolve_parameter_name()` maps an old or SIPNET name to the current field, and the [Parameters](reference/parameters.md) and [Climate drivers](reference/climate-drivers.md) pages are generated from the specs.

Names are lower-case words joined by underscores with no acronyms or truncations (`net_ecosystem_exchange`, not `nee`); the familiar short forms are aliases that lookups accept.

Unit strings use UDUNITS syntax, the convention of netCDF and the Climate and Forecast metadata conventions, and are validated at import time by the Pint registry in `pysipnet.units`:

| Quantity | `units` string | `constituent` |
|:---------|:---------------|:--------------|
| g C m⁻² | `"g m-2"` | `"C"` |
| g N m⁻² d⁻¹ | `"g m-2 d-1"` | `"N"` |
| cm of water | `"cm"` | `"H2O"` |
| dimensionless | `"1"` | |
| °C | `"degC"` | |
| year⁻¹ | `"yr-1"` | |
| leaf area index | `"m2 m-2"` | |

The substance is deliberately kept out of the unit string. Pint parses `"g C m-2"` as gram·coulomb per square meter without complaint, so a qualifier inside the string would be a silent error rather than a caught one; `pysipnet.units.validate_units` refuses it. `format_units("g m-2", constituent="C")` puts it back for display as `g C m⁻²`.

A substance enters a conversion the same way, as an argument beside the unit string. `pysipnet.units.conversion_factor` takes a `units` and a `constituent` for each side and supplies the chemistry Pint cannot: mass and amount convert through `MOLAR_MASS`, a depth of water and a mass per area through `DENSITY`, and a change of constituent only for a pair in `ATOMS_PER_MOLECULE` (C–CO2, C–CH4, N–N2O), applied on an amount basis. Every other pairing is refused rather than guessed. See [Running a Model](user-guide/running-a-model.md#converting-units) for examples.

"Per timestep" is not a unit either. A flux integrated over the step is in `"g m-2"`, and the fact that it is a total over the step is the variable's *kind*, carried as the `time_reference` and `cell_methods` attributes on the xarray representation.

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

### 4. Fully serializable run specification

A complete run specification (parameters + climate + flags) round-trips through plain dict/JSON:

```python
config_dict = {
    "params":  params.model_dump(),
    "climate": climate.pandas.to_dict(orient="list"),
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
(Pydantic +        (.param,         (subprocess    (DataFrame and
 dataclass)        .clim, .out)      + workdir)     xarray Dataset)
```

Nothing above the IO layer touches the filesystem.  The runner takes Python objects, the IO layer materializes them to disk, and the runner calls the binary.

### 6. Stateless, isolated runs

Each call to `SIPNETRunner.run()` writes to a fresh temporary directory.  Runs never share files.  This property is what makes parallelism trivial.

## Ensemble running (out of scope for this package)

pySIPNET is intentionally scoped to single runs.  The ensemble layer is separate.  Recommended tools:

**[Hydra](https://hydra.cc/)** — for structured sweep specification (grid, random, Ax/Optuna).  pySIPNET's Pydantic models map naturally to Hydra structured configs via `OmegaConf`.

**[Parsl](https://parsl-project.org/)** — for execution on HPC clusters (SLURM, PBS) and cloud.  A Parsl `python_app` wrapping `SIPNETRunner.run()` + `model_dump()` / `model_validate()` for serialization is sufficient for most ensemble workflows.

**[Dask](https://dask.org/)** — for local multi-core or distributed cluster execution.  Simpler setup than Parsl; better for local development.

The key design property that enables all of these: `SIPNETRunner.run()` is a pure function from a serializable config to a serializable result.

## Model options

From SIPNET v2.0.0 onward, model options are chosen at run time rather than compiled in. pySIPNET therefore builds one binary and expresses the configuration as data: a `ModelFlags` instance, written into the `sipnet.in` file generated for each run.

`ModelFlags` carries two responsibilities:

- **Configuring SIPNET.** `to_config_keys()` renders the flags as the uppercase `sipnet.in` keys SIPNET reads. Every flag is written, including defaults, so a saved run does not depend on the binary's built-in defaults.
- **Deciding which parameters are required.** `SIPNETParameters.validate_for_flags()` uses the flags to check the parameter set before SIPNET is invoked, so a mismatch is a Python error naming the missing field rather than an exit code from the binary.

`ModelFlags` also rejects the four combinations SIPNET itself refuses, mirroring `validateContext()` in SIPNET's `src/common/context.c`:

- `gdd` with `soil_phenol`
- `anaerobic` without `water_hresp`
- `nitrogen_cycle` without both `litter_pool` and `anaerobic`
- `carbon_saturation` without `litter_pool`

Separately, four flags are refused because *pySIPNET* cannot serve them: `nitrogen_cycle`, `anaerobic`, `flooding` and `carbon_saturation`. SIPNET supports all four, but each requires parameters `SIPNETParameters` does not define, so a run would fail inside SIPNET with "Did not find required parameter" rather than in Python. `UNSUPPORTED_FLAGS` in `pysipnet/parameters/model.py` records what each one needs; the error message reproduces that list. Removing an entry from that table is what enables the flag once its parameters exist.

`ModelFlags.standard()` is a named starting point, not a closed set — any combination of flags is valid, subject to the restrictions above. It returns exactly the defaults SIPNET compiles in (`gdd`, `snow` and `water_hresp` on, the rest off; see the `CREATE_INT_CONTEXT` calls in `src/common/context.c`), so it is what the binary would do with no flags in `sipnet.in`. An optional `name` field carries a label into the run record without affecting the model.

## Per-year rate parameters

The following SIPNET parameters are specified as **per-year rates** in the
param file but converted to per-day internally (÷ 365):

- `respiration.base_wood_respiration_rate`
- `respiration.base_fine_root_respiration_rate`
- `respiration.base_coarse_root_respiration_rate`
- `respiration.base_soil_respiration_rate`
- `respiration.litter_breakdown_rate`
- `allocation.fine_root_turnover_rate`
- `allocation.coarse_root_turnover_rate`
- `allocation.wood_turnover_rate`
- `phenology.leaf_turnover_rate`

These fields have `per_year=True` in their `ParameterSpec`.  The Python interface always works in per-year units, matching SIPNET's param file convention.  Do not divide by 365 before passing values to pySIPNET.
