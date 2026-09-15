"""Unit strings: the convention, the parser that validates them, and formatters.

Convention
----------
Units are stored as **UDUNITS-style strings**, the syntax used by netCDF, the
Climate and Forecast (CF) metadata conventions, xarray tooling and most
ecosystem-model output: symbols separated by spaces, exponents written as
plain signed integers.  Examples::

    "g m-2"          grams per square metre
    "g m-2 d-1"      grams per square metre per day
    "cm d-1"         centimetres per day
    "1"              dimensionless
    "degC"           degrees Celsius
    "K d"            kelvin-days (growing degree-days)
    "m2 m-2"         square metres per square metre (leaf area index)
    "nmol g-1 s-1"   nanomoles per gram per second

**The substance never goes in the unit string.**  Pint parses ``"g C m-2"``
as gram·coulomb per square metre and ``"g N m-2"`` as gram·newton·metre
without complaint, so a carbon or nitrogen qualifier inside the string is a
silent error rather than a caught one.  The qualifier lives in a separate
``constituent`` field (``"C"``, ``"N"``, ``"H2O"``) on the variable or
parameter spec, and :func:`format_units` puts it back for display.

**"per timestep" is not a unit.**  A flux integrated over the model step is in
``"g m-2"``; the fact that it is a total over the step is metadata carried by
the variable's kind (see :mod:`pysipnet.variables`).

Validation
----------
:data:`unit_registry` is a Pint registry with a preprocessor that turns
UDUNITS exponents (``m-2``) into Pint's (``m**-2``), plus an ``einstein``
definition (one mole of photons).  :func:`validate_units` parses a string
through it and additionally refuses the substance tokens above, so every unit
string in the package is checked at import time.
"""

from __future__ import annotations

import re
from typing import Literal

import pint

# 'm-2' -> 'm**-2', 'm2' -> 'm**2'.  Only digits directly following a letter
# are exponents; a bare '1' (dimensionless) has no letter before it.
_UDUNITS_EXPONENT = re.compile(r"(?<=[A-Za-z])(-?\d+)")

# Tokens that name a substance rather than a unit.  Pint accepts every one of
# them as something else (coulomb, newton, ...), so they must be refused here.
_CONSTITUENT_TOKENS: frozenset[str] = frozenset({"C", "N", "H2O", "CO2", "CH4", "N2O"})

# A symbol with an optional signed integer exponent ("m", "m-2", "kPa-1"), or the
# bare "1" of a dimensionless quantity.
_UDUNITS_TOKEN = re.compile(r"[A-Za-z]+(-?\d+)?|1")

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# Display symbols for tokens whose UDUNITS spelling is not what a reader wants
# to see on an axis.
_DISPLAY_SYMBOL: dict[str, str] = {"degC": "°C", "einstein": "E"}

UnitStyle = Literal["unicode", "latex", "html", "plain"]


def _udunits_to_pint(expression: str) -> str:
    return _UDUNITS_EXPONENT.sub(r"**\1", expression)


unit_registry = pint.UnitRegistry(preprocessors=[_udunits_to_pint])  # type: ignore[var-annotated]
unit_registry.define("einstein = mole = E")
"""Pint registry that understands the package's UDUNITS-style unit strings.

Use it for conversions::

    from pysipnet.units import unit_registry
    q = unit_registry.Quantity(values, "g m-2 d-1").to("kg m-2 s-1")
"""


def parse_units(units: str) -> pint.Unit:
    """Parse a UDUNITS-style string into a Pint unit, raising ``ValueError`` if invalid."""
    validate_units(units)
    return unit_registry.parse_units(units)


def validate_units(units: str) -> None:
    """Raise ``ValueError`` unless *units* is a valid, substance-free unit string."""
    if not units.strip():
        raise ValueError("Unit string is empty; use '1' for dimensionless quantities.")
    tokens = units.split()
    bad = [t for t in tokens if not _UDUNITS_TOKEN.fullmatch(t)]
    if bad:
        raise ValueError(
            f"Unit string {units!r} is not in UDUNITS syntax: {bad}. Write symbols separated "
            "by spaces with plain signed exponents, e.g. 'g m-2 d-1', or '1' for dimensionless."
        )
    offending = _CONSTITUENT_TOKENS & set(tokens)
    if offending:
        raise ValueError(
            f"Unit string {units!r} contains the substance token(s) {sorted(offending)}. "
            "Pint reads these as unrelated units (C is coulomb, N is newton). Put the "
            "substance in the 'constituent' field and keep the unit string physical."
        )
    try:
        unit_registry.parse_units(units)
    except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError, ValueError) as exc:
        raise ValueError(f"Unit string {units!r} is not a valid unit expression: {exc}") from exc


def format_units(units: str, *, constituent: str = "", style: UnitStyle = "unicode") -> str:
    """Render a unit string for display, with the constituent after the mass unit.

    ``format_units("g m-2 d-1", constituent="C")`` gives ``"g C m⁻² d⁻¹"`` in
    the default Unicode style, ``r"\\mathrm{g\\,C\\,m^{-2}\\,d^{-1}}"`` in LaTeX,
    ``"g C m<sup>-2</sup> d<sup>-1</sup>"`` in HTML and ``"g C m-2 d-1"`` plain.
    Dimensionless quantities render as ``"dimensionless"`` in every style.

    The rendering works token by token from the stored string, so the order
    the author wrote is preserved.
    """
    validate_units(units)
    tokens = units.split()
    if tokens == ["1"]:
        return "dimensionless"

    rendered: list[str] = []
    for i, token in enumerate(tokens):
        match = re.fullmatch(r"([A-Za-z]+)(-?\d+)?", token)
        if match is None:
            # Something Pint accepts but this simple formatter does not (e.g. a
            # parenthesised expression).  Fall back to the raw token.
            rendered.append(token)
            continue
        symbol, exponent = match.group(1), match.group(2)
        symbol = _DISPLAY_SYMBOL.get(symbol, symbol)
        rendered.append(_render_token(symbol, exponent, style))
        # The constituent qualifies the first (mass or amount) unit: "g C m-2".
        if i == 0 and constituent:
            rendered.append(constituent)

    if style == "latex":
        return r"\mathrm{" + r"\,".join(rendered) + "}"
    return " ".join(rendered)


def _render_token(symbol: str, exponent: str | None, style: UnitStyle) -> str:
    if exponent is None or exponent == "1":
        return symbol
    if style == "unicode":
        return symbol + exponent.translate(_SUPERSCRIPT)
    if style == "latex":
        return f"{symbol}^{{{exponent}}}"
    if style == "html":
        return f"{symbol}<sup>{exponent}</sup>"
    return f"{symbol}{exponent}"
