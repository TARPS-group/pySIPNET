"""Generate the variable and parameter reference pages from the registries at docs-build time.

Run by the ``gen-files`` mkdocs plugin (see ``mkdocs.yml``). Writing the table
from :data:`pysipnet.variables.OUTPUT_VARIABLES` means the documentation cannot
disagree with the code.
"""

from __future__ import annotations

import mkdocs_gen_files

from pysipnet.parameters.model import PARAMETER_SPECS
from pysipnet.variables import (
    CLIMATE_VARIABLES,
    OUTPUT_VARIABLES,
    RESAMPLING_METHODS_FOR_KIND,
)

lines = [
    "# Output variables",
    "",
    "Every column SIPNET writes, in file order, as described by",
    "[`pysipnet.variables.OUTPUT_VARIABLES`][pysipnet.variables]. This page is",
    "generated from the registry.",
    "",
    "**Time convention.** SIPNET labels each row with the *start* of its timestep",
    "(`year`, `day_of_year`, `hour_of_day`), on whatever clock the climate drivers",
    "use: SIPNET has no time zone, and prints `hour_of_day` rounded to 0.01 h. A pool",
    "is reported at the *end* of that step; a flux is the total *over* the step; the",
    "one rate column is a per-day rate during the step; a cumulative value runs from",
    "the start of the simulation to the end of the step. The `Time reference` column",
    "below says which applies, and the same text travels on every variable as the",
    "`time_reference` attribute of `result.outputs.xarray`, whose `time` coordinate is",
    "the *end* of the step so that the CF `cell_methods` attribute is literally true.",
    "`Resample` lists the methods",
    "[`resample`][pysipnet.resample.resample] accepts for the variable.",
    "",
    "**Precision.** SIPNET prints each column with a fixed number of decimals",
    "(`Decimals`). Treat it as a quantization floor when fitting to output.",
    "",
    "**Flags.** A variable with a `Requires flag` entry is written as constant zero",
    "unless that flag is on, so selecting it by name (`result.outputs[...]`) raises",
    "rather than returning the zeros. `result.outputs.pandas` still carries the column.",
    "",
    '**Aliases.** `result.outputs["nee"]` and `result.outputs.select(["nee"])` accept',
    "the aliases listed as well as the full names; column names are always the full names.",
    "",
    "| Name | SIPNET column | Time reference | Resample | Units | Decimals | Requires flag "
    "| Aliases | Description |",
    "|:-----|:--------------|:---------------|:---------|:------|:---------|:--------------|:--------|:------------|",
]
for spec in OUTPUT_VARIABLES:
    aliases = ", ".join(f"`{a}`" for a in spec.aliases) or ""
    decimals = "" if spec.output_decimals is None else str(spec.output_decimals)
    flag = f"`{spec.requires_flag}`" if spec.requires_flag else ""
    methods = ", ".join(f"`{m}`" for m in sorted(RESAMPLING_METHODS_FOR_KIND[spec.kind]))
    description = spec.description
    if spec.sign_convention:
        description += f" Sign: {spec.sign_convention}."
    lines.append(
        f"| `{spec.name}` | `{spec.sipnet_name}` | {spec.time_reference} | {methods} | "
        f"{spec.formatted_units()} | {decimals} | {flag} | {aliases} | {description} |"
    )

with mkdocs_gen_files.open("reference/output-variables.md", "w") as f:
    f.write("\n".join(lines) + "\n")


# ── Parameters ─────────────────────────────────────────────────────────────────

param_lines = [
    "# Parameters",
    "",
    "Every field of `SIPNETParameters`, grouped as in the model, as described by",
    "each field's [`ParameterSpec`][pysipnet.parameters.base.ParameterSpec]",
    "(`pysipnet.parameters.model.PARAMETER_SPECS`). This page is generated from",
    "those specs.",
    "",
    "**Names.** Field names follow the same convention as output variables: lower-case",
    "words, no acronyms. The `SIPNET name` column is what the `.param` file uses; the",
    "`Aliases` column lists the names pySIPNET used before this convention. Both are",
    "accepted by `resolve_parameter_name()` and reported in the error when passed to",
    "`SIPNETModel`, but only the current name is a field.",
    "",
    "**Per-year rates.** Parameters marked *per year* are read by SIPNET as annual rates",
    "and divided by 365 internally; specify them per year.",
    "",
    "**Initial conditions** set an output state variable at the start of the run; the",
    "`Initializes` column names it and, where the relation is not the identity, how.",
    "",
]
current_group = None
for path, spec in PARAMETER_SPECS.items():
    group, field = path.split(".", 1)
    if group != current_group:
        current_group = group
        param_lines += [
            f"## `{group}`",
            "",
            "| Field | SIPNET name | Units | Domain | Description | Aliases | Initializes |",
            "|:------|:------------|:------|:-------|:------------|:--------|:------------|",
        ]
    units = spec.formatted_units() + (" (per year)" if spec.per_year else "")
    aliases = ", ".join(f"`{a}`" for a in spec.aliases)
    initializes = ", ".join(f"`{v}`" for v in spec.initializes)
    if spec.initializes_via:
        initializes += f" via {spec.initializes_via}"
    param_lines.append(
        f"| `{field}` | `{spec.sipnet_name}` | {units} | {spec.domain.value} | "
        f"{spec.description} | {aliases} | {initializes} |"
    )
    if (
        path == list(PARAMETER_SPECS)[-1]
        or list(PARAMETER_SPECS)[list(PARAMETER_SPECS).index(path) + 1].split(".", 1)[0] != group
    ):
        param_lines.append("")

with mkdocs_gen_files.open("reference/parameters.md", "w") as f:
    f.write("\n".join(param_lines) + "\n")


# ── Climate drivers ───────────────────────────────────────────────────────────

clim_lines = [
    "# Climate drivers",
    "",
    "Every column of the `.clim` climate file, in file order, as described by",
    "[`pysipnet.variables.CLIMATE_VARIABLES`][pysipnet.variables]. This page is",
    "generated from the registry.",
    "",
    "**Units.** The `Units` column is what goes **in the file** and in a",
    "`ClimateDrivers` DataFrame. SIPNET converts some columns on read; the",
    "`SIPNET converts to` column records that, because SIPNET's own documentation",
    "quotes the converted units.",
    "",
    "**Time convention.** Rows are labeled with the *start* of the timestep, on",
    "whatever clock the drivers were written in. SIPNET has no time zone; the",
    'drivers declare it, as `ClimateDrivers(time_zone="UTC-07:00")` or similar, and',
    "the declaration travels on the `time` coordinate of every Dataset. Each row must",
    "start where the previous one ends (its start plus `timestep_length`):",
    "`ClimateDrivers.validate` refuses an overlap or a drift and warns about a gap.",
    "Means are over the step; `photosynthetically_active_radiation` and",
    "`precipitation` are totals over the step.",
    "",
    "**Aliases.** `ClimateDrivers.from_dataframe` accepts a DataFrame whose columns",
    "use these aliases (the short names pySIPNET used previously, or SIPNET's",
    "column names) and renames them; stored columns always use the full names.",
    "",
    "| Name | SIPNET column | Time reference | Units | SIPNET converts to | Aliases | Description |",
    "|:-----|:--------------|:---------------|:------|:-------------------|:--------|:------------|",
]
for spec in CLIMATE_VARIABLES:
    aliases = ", ".join(f"`{a}`" for a in spec.aliases)
    converts = ""
    if spec.internal_units or spec.internal_conversion:
        converts = spec.internal_conversion
        if spec.internal_units:
            converts = f"{spec.internal_units}: {converts}" if converts else spec.internal_units
    clim_lines.append(
        f"| `{spec.name}` | `{spec.sipnet_name}` | {spec.time_reference} | "
        f"{spec.formatted_units()} | {converts} | {aliases} | {spec.description} |"
    )

with mkdocs_gen_files.open("reference/climate-drivers.md", "w") as f:
    f.write("\n".join(clim_lines) + "\n")
