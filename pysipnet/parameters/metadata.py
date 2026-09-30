"""A parameter set, or the parameter model itself, as one table for display.

:func:`parameter_metadata_table` is for reading and presenting parameters: a
report, a notebook, the generated documentation. It works nothing out for
itself. The columns are the fields of each
:class:`~pysipnet.parameters.base.ParameterSpec`, the values are
:meth:`~pysipnet.parameters.model.SIPNETParameters.flat_values`, and
requiredness is :meth:`~pysipnet.parameters.base.ParameterSpec.required_under`.

It returns a plain ``DataFrame`` rather than styled output, so the renderer
(``DataFrame.style``, ``great_tables``, LaTeX) and anything a caller wants to
mark, such as which parameters are calibrated, stay with the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from pysipnet.parameters.model import (
    _PARAMETER_PATHS,
    PARAMETER_SPECS,
    ModelFlags,
    SIPNETDocsSection,
    SIPNETParameters,
    resolve_parameter_name,
)
from pysipnet.units import UnitStyle

_SECTION_ORDER = {section.value: position for position, section in enumerate(SIPNETDocsSection)}


def parameter_metadata_table(
    parameters: SIPNETParameters | None = None,
    *,
    flags: ModelFlags | None = None,
    tags: Mapping[str, str] | None = None,
    units_style: UnitStyle = "unicode",
) -> pd.DataFrame:
    """One row per parameter, grouped by the section of SIPNET's documentation it is in.

    Parameters
    ----------
    parameters:
        A parameter set, whose values fill a ``value`` column. Only the
        parameters it sets are rows, since those are exactly what SIPNET
        receives. Omit it for every parameter of the model and no values.
    flags:
        Model flags, which add a ``required`` column: whether SIPNET requires
        the parameter under them. With *parameters* as well, the pair is
        checked first by
        :meth:`~pysipnet.parameters.model.SIPNETParameters.validate_for_flags`,
        so a table is never made for a configuration SIPNET would refuse.
        "Not required" does not mean "not used": SIPNET reads any parameter
        it is given.
    tags:
        ``{parameter name: tag}``, which adds a ``tag`` column, missing for a
        parameter not named. A name is anything
        :func:`~pysipnet.parameters.model.resolve_parameter_name` accepts, so
        a typo raises ``KeyError`` rather than silently tagging nothing.
        The tags are the caller's own vocabulary::

            parameter_metadata_table(params, tags=dict.fromkeys(calibrated, "calibrated"))

    units_style:
        How units are written: ``"unicode"``, ``"html"``, ``"latex"`` or
        ``"plain"``, as :meth:`~pysipnet.parameters.base.ParameterSpec.formatted_units`.

    Returns
    -------
    pandas.DataFrame
        Indexed by the dotted field path (``"photosynthesis.max_photosynthesis_rate"``),
        rows ordered by section and then as the model declares them, with
        columns ``section`` (an ordered categorical of
        :class:`~pysipnet.parameters.model.SIPNETDocsSection` values, in the
        order SIPNET's documentation gives them), ``name``, ``sipnet_name``,
        ``value`` (with *parameters*), ``units``, ``label``, ``description``,
        ``domain``, ``required_when``, ``required`` (with *flags*) and ``tag``
        (with *tags*).
    """
    if parameters is None:
        values: dict[str, float] | None = None
        paths = list(PARAMETER_SPECS)
    else:
        if flags is not None:
            parameters.validate_for_flags(flags)
        values = parameters.flat_values()
        paths = list(values)
    paths.sort(key=lambda path: _SECTION_ORDER[PARAMETER_SPECS[path].sipnet_docs_section])

    tag_by_path = _resolve_tags(tags, paths) if tags is not None else None

    rows: list[dict[str, Any]] = []
    for path in paths:
        spec = PARAMETER_SPECS[path]
        row: dict[str, Any] = {
            "section": spec.sipnet_docs_section,
            "name": path.split(".", 1)[1],
            "sipnet_name": spec.sipnet_name,
        }
        if values is not None:
            row["value"] = values[path]
        row |= {
            "units": spec.formatted_units(units_style),
            "label": spec.long_label,
            "description": spec.description,
            "domain": spec.domain.value,
            "required_when": spec.requirement_description(),
        }
        if flags is not None:
            row["required"] = spec.required_under(flags)
        if tag_by_path is not None:
            row["tag"] = tag_by_path.get(path)
        rows.append(row)

    table = pd.DataFrame(rows, index=pd.Index(paths, name="path"))
    table["section"] = pd.Categorical(
        table["section"], categories=[section.value for section in SIPNETDocsSection], ordered=True
    )
    if values is not None:
        table["value"] = table["value"].astype(float)
    return table


def _resolve_tags(tags: Mapping[str, str], paths: list[str]) -> dict[str, str]:
    """``{dotted path: tag}``, refusing a parameter tagged twice or not in the table."""
    tag_by_path: dict[str, str] = {}
    spelled: dict[str, str] = {}
    for name, tag in tags.items():
        path = _PARAMETER_PATHS[resolve_parameter_name(name)]
        if path in tag_by_path and tag_by_path[path] != tag:
            raise ValueError(
                f"{path} is tagged both {tag_by_path[path]!r} (as {spelled[path]!r}) "
                f"and {tag!r} (as {name!r})."
            )
        tag_by_path[path] = tag
        spelled[path] = name
    missing = sorted(set(tag_by_path) - set(paths))
    if missing:
        raise ValueError(
            "Tagged parameters that this parameter set does not set: " + ", ".join(missing) + "."
        )
    return tag_by_path
