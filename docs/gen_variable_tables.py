"""Generate the output-variable reference page from the registry at docs-build time.

Run by the ``gen-files`` mkdocs plugin (see ``mkdocs.yml``). Writing the table
from :data:`pysipnet.variables.OUTPUT_VARIABLES` means the documentation cannot
disagree with the code.
"""

from __future__ import annotations

import mkdocs_gen_files

from pysipnet.variables import OUTPUT_VARIABLES, VariableKind

KIND_TEXT = {
    VariableKind.COORDINATE: "coordinate",
    VariableKind.STATE: "state (end of step)",
    VariableKind.FLUX: "flux (total over step)",
    VariableKind.RATE: "rate (per day)",
    VariableKind.MEAN: "mean over step",
    VariableKind.CUMULATIVE: "cumulative",
}

lines = [
    "# Output variables",
    "",
    "Every column SIPNET writes, in file order, as described by",
    "[`pysipnet.variables.OUTPUT_VARIABLES`][pysipnet.variables]. This page is",
    "generated from the registry.",
    "",
    "**Time convention.** Each row is labelled with the *start* of its timestep",
    "(`year`, `day_of_year`, `hour_of_day`). A *state* is the pool at the *end* of",
    "that step; a *flux* is the total *over* the step; the one *rate* column is a",
    "per-day rate during the step; a *cumulative* value runs from the start of the",
    "simulation to the end of the step. The `Kind` column below says which applies,",
    "and the same text travels on every variable as the `time_reference` attribute",
    "of `result.outputs.dataset`.",
    "",
    "**Precision.** SIPNET prints each column with a fixed number of decimals",
    "(`Decimals`). Treat it as a quantisation floor when fitting to output.",
    "",
    '**Aliases.** `result.outputs["nee"]` and `load(variables=["nee"])` accept the',
    "aliases listed as well as the full names; column names are always the full names.",
    "",
    "| Name | SIPNET column | Kind | Units | Decimals | Requires flag | Aliases | Description |",
    "|:-----|:--------------|:-----|:------|:---------|:--------------|:--------|:------------|",
]
for spec in OUTPUT_VARIABLES:
    aliases = ", ".join(f"`{a}`" for a in spec.aliases) or ""
    decimals = "" if spec.output_decimals is None else str(spec.output_decimals)
    flag = f"`{spec.requires_flag}`" if spec.requires_flag else ""
    description = spec.description
    if spec.sign_convention:
        description += f" Sign: {spec.sign_convention}."
    lines.append(
        f"| `{spec.name}` | `{spec.sipnet_name}` | {KIND_TEXT[spec.kind]} | "
        f"{spec.formatted_units()} | {decimals} | {flag} | {aliases} | {description} |"
    )

with mkdocs_gen_files.open("reference/output-variables.md", "w") as f:
    f.write("\n".join(lines) + "\n")
