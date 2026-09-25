"""Unit strings: the convention, the parser that validates them, and formatters.

Convention
----------
Units are stored as **UDUNITS-style strings**, the syntax used by netCDF, the
Climate and Forecast (CF) metadata conventions, xarray tooling and most
ecosystem-model output: symbols separated by spaces, exponents written as
plain signed integers.  Examples::

    "g m-2"          grams per square meter
    "g m-2 d-1"      grams per square meter per day
    "cm d-1"         centimeters per day
    "1"              dimensionless
    "degC"           degrees Celsius
    "K d"            kelvin-days (growing degree-days)
    "m2 m-2"         square meters per square meter (leaf area index)
    "nmol g-1 s-1"   nanomoles per gram per second

**The substance never goes in the unit string.**  Pint parses ``"g C m-2"``
as gram·coulomb per square meter and ``"g N m-2"`` as gram·newton·meter
without complaint, so a carbon or nitrogen qualifier inside the string is a
silent error rather than a caught one.  The qualifier lives in a separate
``constituent`` field (``"C"``, ``"N"``, ``"H2O"``) on the variable or
parameter spec, and :func:`format_units` puts it back for display.

A substance enters a *conversion* the same way: as an argument beside the unit
string, never inside it.  :func:`conversion_factor` takes ``units`` and
``constituent`` for each side and does the chemistry Pint cannot, since grams
of carbon become moles only through carbon's molar mass::

    conversion_factor(units="g m-2 d-1", constituent="C",
                      to_units="umol m-2 s-1", to_constituent="CO2")  # 0.9636...

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

Conversion
----------
:func:`conversion_factor` returns the number that takes a value in one
``(units, constituent)`` pair to another.  Two functions apply it:

- :func:`convert_units` multiplies unlabeled values (a number, a NumPy array,
  a pandas object) by it, with the caller stating the units they are in.
- :func:`convert_dataarray_units` reads the units and constituent from an
  xarray ``DataArray``'s ``attrs``, so they cannot be misstated, and relabels
  the result.

The factor is a pure function of the four strings, so a test can pin it.  In
order, the rules are:

1. Same dimension, same constituent (or none on either side), and the first
   unit measures the same kind of thing on both sides: Pint's factor.
2. Mass to amount or back, one constituent: through :data:`MOLAR_MASS`.  A
   depth or volume of water to a mass or back: through :data:`DENSITY`.
3. A change of constituent: only for a pair in :data:`ATOMS_PER_MOLECULE`,
   applied on an amount basis and composed with rules 1 and 2.  Grams of C to
   grams of CO2 goes C mass → C amount → CO2 amount → CO2 mass.
4. Anything else raises ``ValueError`` naming both unit strings and both
   constituents: a dimension mismatch, a constituent on one side only, an
   unknown constituent, a substance token inside a unit string, a constituent
   on a temperature, or an offset temperature conversion, which is not a
   multiplication.

Rules 2 and 3 read the constituent as qualifying the **first** unit in the
string, the same position :func:`format_units` prints it in: ``"g m-2"`` of C
is grams of carbon per square meter, and ``"nmol g-1 s-1"`` of CO2 is
nanomoles of CO2 per gram of leaf.  That first unit must be an amount, a mass,
or (for water) a depth or volume.  Pint's dimensionality alone cannot say which
mass is the substance's: it would find one inside ``Pa`` or ``W`` and lose the
one in ``ug g-1`` to cancellation.

The same reading keeps rule 1 honest.  ``"umol mol-1"`` and ``"ug g-1"`` are
both dimensionless, so Pint converts a mole fraction to a mass fraction by a
factor of 1e-6 / 1e-6 = 1; but the first unit changes from an amount to a mass,
which takes a molar mass, so the conversion goes through rules 2 and 3 (with a
constituent) or is refused (without one).  Here it is refused either way, since
the denominator would need the molar mass of air.  Water content by mass
(``"kg kg-1"``) and by volume (``"m3 m-3"``) are refused for the same reason.
A depth and a volume (``"m3 m-2"`` and ``"m"``) are the same kind, being
geometry alone.

Combination
-----------
:func:`product_units`, :func:`quotient_units`, :func:`sum_units` and
:func:`difference_units` give the ``(units, constituent)`` of a product,
quotient, sum or difference of two quantities, so a derived quantity is still
one this module can convert::

    product_units(units="d-1", other_units="g m-2", other_constituent="C")
    # ("g m-2 d-1", "C")

The rules:

- Units combine symbol by symbol, adding exponents for a product and
  subtracting them for a quotient.  A symbol whose exponent reaches zero drops
  out, and an empty result is ``"1"``.  Prefixes are not merged: ``cm`` over
  ``m`` is ``"cm m-1"``, which converts to ``"1"`` on request.
- The operand that carries the constituent leads the string, because the
  constituent qualifies the first unit, and that first unit must come through
  unchanged: ``g m-2`` of C over ``g`` would leave ``m-2`` of C, which names no
  quantity of carbon, and is refused.
- A product may name at most one constituent.  A quotient keeps the
  numerator's, and the same constituent on both sides cancels (carbon over
  carbon is a ratio).  A constituent only in the denominator, or two different
  ones, is refused.
- A sum or difference needs the same unit string and constituent on both
  sides.
- A bare offset temperature (``degC``) does not multiply or add; the
  difference of two is a temperature difference, in ``K``.  Inside a compound
  unit Pint reads ``degC`` as a difference already, so degree-days in
  ``"degC d"`` combine like any other unit.

:mod:`pysipnet.arithmetic` applies these rules to ``DataArray`` objects,
reading the operands' units with :func:`read_dataarray_units`, and adds the
rules for a variable's ``kind``.  It is a separate module because the kinds live
in :mod:`pysipnet.variables`, which validates its unit strings with this module
at import.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Literal, overload

import numpy.typing as npt
import pandas as pd
import pint
import xarray as xr

# 'm-2' -> 'm**-2', 'm2' -> 'm**2'.  Only digits directly following a letter
# are exponents; a bare '1' (dimensionless) has no letter before it.
_UDUNITS_EXPONENT = re.compile(r"(?<=[A-Za-z])(-?\d+)")

MOLAR_MASS: Mapping[str, float] = MappingProxyType(
    {
        "C": 12.011,
        "N": 14.007,
        "H2O": 18.015,
        "CO2": 44.009,
        "CH4": 16.043,
        "N2O": 44.013,
    }
)
"""Molar mass of each constituent that has one, in g mol-1."""

DENSITY: Mapping[str, float] = MappingProxyType({"H2O": 1000.0})
"""Density in kg m-3, which lets a depth convert to a mass per area (1 cm of water is 10 kg m-2)."""

AMOUNT_ONLY_CONSTITUENTS: frozenset[str] = frozenset({"photons"})
"""Constituents counted in moles that have no molar mass."""

CONSTITUENTS: frozenset[str] = frozenset(MOLAR_MASS) | AMOUNT_ONLY_CONSTITUENTS
"""Every constituent a conversion accepts, besides ``""`` for none."""

ATOMS_PER_MOLECULE: Mapping[tuple[str, str], int] = MappingProxyType(
    {("C", "CO2"): 1, ("C", "CH4"): 1, ("N", "N2O"): 2}
)
"""``(element, molecule): atoms of the element in one molecule``, the only changes of
constituent a conversion makes."""

# Tokens that name a substance rather than a unit.  C, N, CO2 and CH4 parse as
# something else (coulomb, newton, ...), so they must be refused here; H2O and
# N2O fail the syntax check first.
_CONSTITUENT_TOKENS: frozenset[str] = frozenset(MOLAR_MASS)

# A symbol with an optional signed integer exponent ("m", "m-2", "kPa-1"), or the
# bare "1" of a dimensionless quantity.
_UDUNITS_TOKEN = re.compile(r"([A-Za-z]+)(-?\d+)?|1")

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# Display symbols for tokens whose UDUNITS spelling is not what a reader wants
# to see on an axis.
_DISPLAY_SYMBOL: dict[str, str] = {"degC": "°C", "einstein": "E"}

UnitStyle = Literal["unicode", "latex", "html", "plain"]

# Attributes that describe a value in its original units and would be false after
# a conversion: the printed precision, and SIPNET's conversion from the file units.
_UNIT_DEPENDENT_ATTRS = ("output_decimals", "sipnet_internal_units", "sipnet_internal_conversion")


def _udunits_to_pint(expression: str) -> str:
    return _UDUNITS_EXPONENT.sub(r"**\1", expression)


unit_registry = pint.UnitRegistry(preprocessors=[_udunits_to_pint])  # type: ignore[var-annotated]
"""Pint registry that understands the package's UDUNITS-style unit strings.

It converts between physical units only::

    from pysipnet.units import unit_registry
    q = unit_registry.Quantity(values, "g m-2 d-1").to("kg m-2 s-1")

For a quantity with a ``constituent``, use :func:`conversion_factor`,
:func:`convert_units` or :func:`convert_dataarray_units` instead, which apply
the constituent rules as well.
"""
unit_registry.define("einstein = mole = E")

_SubstanceKind = Literal["amount", "mass", "depth", "volume"]

# What a first unit can measure of a substance, by its dimensionality.
_SUBSTANCE_KINDS: dict[Any, _SubstanceKind] = {
    unit_registry.Quantity(1.0, "mol").dimensionality: "amount",
    unit_registry.Quantity(1.0, "g").dimensionality: "mass",
    unit_registry.Quantity(1.0, "m").dimensionality: "depth",
    unit_registry.Quantity(1.0, "m3").dimensionality: "volume",
}


def parse_units(units: str) -> pint.Unit:
    """Parse a UDUNITS-style string into a Pint unit, raising ``ValueError`` if invalid."""
    validate_units(units)
    return unit_registry.parse_units(units)


def validate_units(units: str) -> None:
    """Raise ``ValueError`` unless *units* is a valid, substance-free unit string.

    Raises ``TypeError`` if *units* is not a string at all.
    """
    if not isinstance(units, str):
        raise TypeError(f"A unit string must be a str, not {type(units).__name__}: {units!r}.")
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
    """Render a unit string for display, with the constituent after the first unit.

    ``format_units("g m-2 d-1", constituent="C")`` gives ``"g C m⁻² d⁻¹"`` in
    the default Unicode style, ``r"\\mathrm{g\\,C\\,m^{-2}\\,d^{-1}}"`` in LaTeX,
    ``"g C m<sup>-2</sup> d<sup>-1</sup>"`` in HTML and ``"g C m-2 d-1"`` plain.
    Dimensionless quantities render as ``"dimensionless"`` in every style.

    The rendering works token by token from the stored string, so the order
    the author wrote is preserved.
    """
    validate_units(units)
    if units.split() == ["1"]:
        return "dimensionless"

    rendered: list[str] = []
    for i, (symbol, exponent) in enumerate(_unit_powers(units)):
        rendered.append(_render_token(_DISPLAY_SYMBOL.get(symbol, symbol), exponent, style))
        # The constituent qualifies the first (mass or amount) unit: "g C m-2".
        if i == 0 and constituent:
            rendered.append(constituent)

    if style == "latex":
        return r"\mathrm{" + r"\,".join(rendered) + "}"
    return " ".join(rendered)


def _unit_powers(units: str) -> list[tuple[str, int]]:
    """A validated unit string as ``(symbol, exponent)`` pairs, in order; ``"1"`` gives none."""
    powers: list[tuple[str, int]] = []
    for token in units.split():
        match = _UDUNITS_TOKEN.fullmatch(token)
        if match is not None and match.group(1) is not None:
            powers.append((match.group(1), int(match.group(2) or 1)))
    return powers


def _render_token(symbol: str, exponent: int, style: UnitStyle) -> str:
    if exponent == 1:
        return symbol
    if style == "unicode":
        return symbol + str(exponent).translate(_SUPERSCRIPT)
    if style == "latex":
        return f"{symbol}^{{{exponent}}}"
    if style == "html":
        return f"{symbol}<sup>{exponent}</sup>"
    return f"{symbol}{exponent}"


@lru_cache(maxsize=256)
def conversion_factor(
    *, units: str, constituent: str = "", to_units: str, to_constituent: str | None = None
) -> float:
    """Return the factor taking *units* of *constituent* to *to_units* of *to_constituent*.

    ``conversion_factor(units="g m-2", constituent="C", to_units="g m-2",
    to_constituent="CO2")`` is ``44.009 / 12.011``.  *to_constituent*
    defaults to *constituent*, so a change of units alone names the substance
    once; ``""`` means no constituent.  See the module docstring for the
    rules; anything they do not cover raises ``ValueError``.  The result is
    cached, since it depends on nothing but the four strings.
    """
    if to_constituent is None:
        to_constituent = constituent
    try:
        return _conversion_factor(units, constituent, to_units, to_constituent)
    except ValueError as exc:
        raise ValueError(
            f"Cannot convert {units!r} ({_describe(constituent)}) to {to_units!r} "
            f"({_describe(to_constituent)}): {exc}"
        ) from None


# DataTree is in xarray from 2024.10; the dependency floor is older.
_XARRAY_TYPES: tuple[type, ...] = tuple(
    t
    for t in (xr.DataArray, xr.Dataset, xr.Variable, getattr(xr, "DataTree", None))
    if t is not None
)


@overload
def convert_units(
    values: float,
    *,
    units: str,
    constituent: str = "",
    to_units: str,
    to_constituent: str | None = None,
) -> float: ...
@overload
def convert_units(
    values: npt.NDArray[Any],
    *,
    units: str,
    constituent: str = "",
    to_units: str,
    to_constituent: str | None = None,
) -> npt.NDArray[Any]: ...
# The installed pandas has no type information, so mypy sees pd.Series as Any and
# this signature as unreachable.  It is kept for readers and for pandas-stubs.
@overload
def convert_units(  # type: ignore[overload-cannot-match]
    values: pd.Series | pd.DataFrame,
    *,
    units: str,
    constituent: str = "",
    to_units: str,
    to_constituent: str | None = None,
) -> pd.Series | pd.DataFrame: ...


def convert_units(
    values: float | npt.NDArray[Any] | pd.Series | pd.DataFrame,
    *,
    units: str,
    constituent: str = "",
    to_units: str,
    to_constituent: str | None = None,
) -> float | npt.NDArray[Any] | pd.Series | pd.DataFrame:
    """Convert unlabeled *values* from *units* of *constituent* to *to_units* of *to_constituent*.

    Returns ``values * conversion_factor(...)`` for the same keyword
    arguments, with the shape and type of *values*: a number, a NumPy array,
    or a pandas ``Series`` or ``DataFrame``.  NumPy's promotion rules apply, so
    an integer array comes back as float64 and a float32 array stays float32.
    The caller states the units the values are in; nothing about *values* is
    read or rewritten.
    *to_constituent* defaults to *constituent*.

    Refused with ``TypeError``: any xarray object, and a pandas object whose
    ``attrs`` has a ``"units"`` entry.  Both keep their ``attrs`` through the
    multiplication, so the result would still claim the old units.  Convert a
    ``DataArray`` with :func:`convert_dataarray_units`, which reads the units
    from its attributes and relabels the result.
    """
    if isinstance(values, _XARRAY_TYPES):
        raise TypeError(
            f"convert_units() takes unlabeled values, not an xarray {type(values).__name__}, "
            "whose units attribute would survive the conversion unchanged. Use "
            "convert_dataarray_units(), which reads the units from the attributes and "
            "relabels the result."
        )
    attrs = getattr(values, "attrs", None)
    if isinstance(attrs, dict) and "units" in attrs:
        raise TypeError(
            f"convert_units() takes unlabeled values, but this {type(values).__name__} has "
            f"attrs['units'] = {attrs['units']!r}, which would survive the conversion "
            "unchanged. Pass its .to_numpy(), or clear its attrs first."
        )
    factor = conversion_factor(
        units=units, constituent=constituent, to_units=to_units, to_constituent=to_constituent
    )
    return values * factor


def convert_dataarray_units(
    array: xr.DataArray, *, to_units: str, to_constituent: str | None = None
) -> xr.DataArray:
    """Convert *array* to *to_units* of *to_constituent*, reading its current units from ``attrs``.

    The source is ``array.attrs["units"]`` (required) and
    ``array.attrs.get("constituent", "")``, read by
    :func:`read_dataarray_units`.  Every pySIPNET output and climate
    ``DataArray`` carries both, so the units the values are in cannot be
    misstated.  *to_constituent* defaults to the
    source constituent; ``""`` means none.  Only the data is scaled:
    coordinates, dimensions and the name are unchanged.

    The result's attributes are *array*'s, with ``units`` set to *to_units*,
    ``constituent`` set to *to_constituent* (or removed if it is ``""``), and
    the attributes that describe the original units removed:
    ``output_decimals``, ``sipnet_internal_units`` and
    ``sipnet_internal_conversion``.  *array* itself is not modified.

    Raises ``TypeError`` for anything but a ``DataArray`` (for a ``Dataset``,
    convert ``ds[name]``), and ``ValueError`` if ``attrs`` has no ``units`` or
    the conversion is not one :func:`conversion_factor` makes.
    """
    if not isinstance(array, xr.DataArray):
        hint = (
            " A Dataset's variables have their own units; convert ds[name] instead."
            if isinstance(array, xr.Dataset)
            else " For unlabeled values, use convert_units()."
        )
        raise TypeError(
            f"convert_dataarray_units() takes an xarray DataArray, not {type(array).__name__}."
            + hint
        )
    units, constituent = read_dataarray_units(array)
    label = _array_label(array)
    if to_constituent is None:
        to_constituent = constituent
    try:
        factor = conversion_factor(
            units=units, constituent=constituent, to_units=to_units, to_constituent=to_constituent
        )
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from None
    result = array * factor
    relabeled = {k: v for k, v in array.attrs.items() if k not in _UNIT_DEPENDENT_ATTRS}
    relabeled["units"] = to_units
    relabeled.pop("constituent", None)
    if to_constituent:
        relabeled["constituent"] = to_constituent
    result.attrs = relabeled
    return result


def read_dataarray_units(array: xr.DataArray) -> tuple[str, str]:
    """The ``(units, constituent)`` a ``DataArray`` declares in its ``attrs``, validated.

    ``units`` is required and must pass :func:`validate_units`;
    ``constituent`` is optional, ``""`` when absent.  Raises ``ValueError``
    naming the array otherwise.
    """
    label = _array_label(array)
    units = array.attrs.get("units")
    if not isinstance(units, str):
        raise ValueError(
            f"{label} has no 'units' attribute, so the units of its values are unknown. Set "
            "array.attrs['units'], or work on array.values with convert_units()."
        )
    try:
        validate_units(units)
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from None
    constituent = array.attrs.get("constituent", "")
    if not isinstance(constituent, str):
        raise ValueError(
            f"{label} has a 'constituent' attribute of {constituent!r}; it must be a str, "
            "'' or absent for none."
        )
    return units, constituent


def _array_label(array: xr.DataArray) -> str:
    return f"DataArray {array.name!r}" if array.name is not None else "DataArray"


@lru_cache(maxsize=256)
def product_units(
    *, units: str, constituent: str = "", other_units: str, other_constituent: str = ""
) -> tuple[str, str]:
    """The ``(units, constituent)`` of *units* of *constituent* times *other_units* of theirs.

    See "Combination" in the module docstring for the rules; anything they
    refuse raises ``ValueError``.  ``product_units(units="d-1",
    other_units="g m-2", other_constituent="C")`` is ``("g m-2 d-1", "C")``:
    the operand with the constituent leads.
    """
    what = f"{_quantity(units, constituent)} times {_quantity(other_units, other_constituent)}"
    _refuse_offset_operands(what, units, other_units)
    if constituent and other_constituent:
        raise ValueError(
            f"Cannot multiply {what}: at most one factor of a product names a constituent. "
            "Divide instead if the result is a ratio."
        )
    if other_constituent:
        return _combined(what, other_units, other_constituent, units, +1), other_constituent
    return _combined(what, units, constituent, other_units, +1), constituent


@lru_cache(maxsize=256)
def quotient_units(
    *, units: str, constituent: str = "", divisor_units: str, divisor_constituent: str = ""
) -> tuple[str, str]:
    """The ``(units, constituent)`` of *units* of *constituent* over *divisor_units* of theirs.

    See "Combination" in the module docstring for the rules; anything they
    refuse raises ``ValueError``.  ``quotient_units(units="g m-2",
    constituent="C", divisor_units="g m-2", divisor_constituent="C")`` is
    ``("1", "")``: the same constituent cancels.
    """
    what = f"{_quantity(units, constituent)} over {_quantity(divisor_units, divisor_constituent)}"
    _refuse_offset_operands(what, units, divisor_units)
    if divisor_constituent and not constituent:
        raise ValueError(
            f"Cannot divide {what}: the result would be per unit of a substance the "
            "numerator does not measure."
        )
    if divisor_constituent and constituent != divisor_constituent:
        raise ValueError(
            f"Cannot divide {what}: convert one of them so both name the same constituent, "
            "and the quotient is a ratio."
        )
    kept = "" if divisor_constituent else constituent
    return _combined(what, units, kept, divisor_units, -1), kept


@lru_cache(maxsize=256)
def sum_units(
    *, units: str, constituent: str = "", other_units: str, other_constituent: str = ""
) -> tuple[str, str]:
    """The ``(units, constituent)`` of a sum: the operands' own, which must agree."""
    what = f"{_quantity(units, constituent)} plus {_quantity(other_units, other_constituent)}"
    _require_same_units(what, units, constituent, other_units, other_constituent)
    _refuse_offset_operands(what, units)
    return units, constituent


@lru_cache(maxsize=256)
def difference_units(
    *, units: str, constituent: str = "", other_units: str, other_constituent: str = ""
) -> tuple[str, str]:
    """The ``(units, constituent)`` of a difference: the operands' own, which must agree.

    The difference of two temperatures in ``degC`` is a temperature
    difference, ``("K", "")``.
    """
    what = f"{_quantity(units, constituent)} minus {_quantity(other_units, other_constituent)}"
    _require_same_units(what, units, constituent, other_units, other_constituent)
    if _is_offset(units):
        if _linear_map(units, "K")[0] != 1.0:
            raise ValueError(
                f"Cannot subtract {what}: a difference of {units!r} is a temperature "
                "difference with no UDUNITS name. Take differences in degC or K."
            )
        return "K", constituent
    return units, constituent


def _quantity(units: str, constituent: str) -> str:
    return f"{units!r} of {constituent}" if constituent else repr(units)


def _require_same_units(
    what: str, units: str, constituent: str, other_units: str, other_constituent: str
) -> None:
    validate_units(units)
    validate_units(other_units)
    if (units, constituent) != (other_units, other_constituent):
        raise ValueError(
            f"Cannot combine {what}: a sum or difference needs the same units and "
            "constituent on both sides. Convert one to the other's first."
        )


def _refuse_offset_operands(what: str, *units: str) -> None:
    for u in units:
        validate_units(u)
        if _is_offset(u):
            raise ValueError(
                f"Cannot combine {what}: {u!r} is an offset temperature scale, which does not "
                "multiply or add. Only the difference of two such temperatures is defined, "
                "and it is in K."
            )


def _combined(what: str, lead: str, constituent: str, other: str, sign: int) -> str:
    """*lead* times (``sign=+1``) or over (``sign=-1``) *other*, *lead*'s symbols first.

    With a constituent, *lead*'s first unit is the one it qualifies, so it
    must come through first and unchanged.
    """
    exponents: dict[str, int] = {}
    for symbol, exponent in _unit_powers(lead):
        exponents[symbol] = exponents.get(symbol, 0) + exponent
    for symbol, exponent in _unit_powers(other):
        exponents[symbol] = exponents.get(symbol, 0) + sign * exponent
    parts = [
        symbol if exponent == 1 else f"{symbol}{exponent}"
        for symbol, exponent in exponents.items()
        if exponent != 0
    ]
    combined = " ".join(parts) if parts else "1"
    if constituent and _unit_powers(combined)[:1] != _unit_powers(lead)[:1]:
        first = lead.split()[0]
        raise ValueError(
            f"Cannot combine {what}: the constituent {constituent!r} qualifies the first unit, "
            f"{first!r}, and the result {combined!r} changes it, so it would not say what "
            f"quantity of {constituent} it measures. Convert the other operand so that "
            f"{first!r} does not cancel or combine."
        )
    validate_units(combined)
    return combined


def _conversion_factor(units: str, constituent: str, to_units: str, to_constituent: str) -> float:
    validate_units(units)
    validate_units(to_units)
    for c in (constituent, to_constituent):
        if c and c not in CONSTITUENTS:
            raise ValueError(
                f"unknown constituent {c!r}; known constituents are {sorted(CONSTITUENTS)}."
            )
    if bool(constituent) != bool(to_constituent):
        raise ValueError(
            "a constituent is given on one side only; name the substance on both sides or neither."
        )

    source = unit_registry.Quantity(1.0, units)
    target = unit_registry.Quantity(1.0, to_units)

    if _is_offset(units) or _is_offset(to_units):
        if constituent:
            raise ValueError("a temperature has no constituent.")
        return _offset_scale_factor(units, to_units)

    same_dimensions = source.dimensionality == target.dimensionality
    kinds = _kind_change(units, to_units)
    if constituent == to_constituent and same_dimensions and kinds is None:
        return float(source.to(to_units).magnitude)
    if kinds is not None:
        mismatch = (
            f"the first unit changes from {_KIND_PHRASE[kinds[0]]} to {_KIND_PHRASE[kinds[1]]}"
        )
    else:
        mismatch = (
            f"the quantities have different dimensions ({source.dimensionality} and "
            f"{target.dimensionality})"
        )
    if not constituent:
        ratio = source.dimensionality / target.dimensionality
        mass_per_amount = unit_registry.Quantity(1.0, "g mol-1").dimensionality
        if set(kinds or ()) == {"amount", "mass"} or ratio in (
            mass_per_amount,
            1 / mass_per_amount,
        ):
            raise ValueError(
                "mass and amount convert through a molar mass, and a molar mass needs a "
                "substance; name the constituent on both sides."
            )
        if kinds is not None:
            needs = "a molar mass and a density" if "amount" in kinds else "a density"
            raise ValueError(
                f"{mismatch}, which takes {needs}, and that needs a substance; name the "
                "constituent on both sides."
            )
        raise ValueError(f"{mismatch} and no constituent to bridge them.")

    molecules = (
        1.0 if constituent == to_constituent else _molecule_ratio(constituent, to_constituent)
    )
    bridged = (
        source
        * _to_amount_basis(constituent, units)
        * molecules
        / _to_amount_basis(to_constituent, to_units)
    )
    try:
        return float(bridged.to(to_units).magnitude)
    except pint.errors.DimensionalityError:
        raise ValueError(
            f"{mismatch}, and restating both as amounts of the constituent does not reconcile them."
        ) from None


_KIND_PHRASE: dict[_SubstanceKind, str] = {
    "amount": "an amount",
    "mass": "a mass",
    "depth": "a depth",
    "volume": "a volume",
}


def _substance_kind(units: str) -> _SubstanceKind | None:
    """What the first unit of *units* measures, if it is a kind of quantity a substance has."""
    return _SUBSTANCE_KINDS.get(unit_registry.Quantity(1.0, units.split()[0]).dimensionality)


def _kind_change(units: str, to_units: str) -> tuple[_SubstanceKind, _SubstanceKind] | None:
    """The two kinds if the first unit changes between substance kinds, else ``None``.

    A depth and a volume do not count: one is the other per unit area, which is
    geometry and needs no property of the substance.
    """
    kinds = (_substance_kind(units), _substance_kind(to_units))
    if kinds[0] is None or kinds[1] is None or kinds[0] == kinds[1]:
        return None
    if {kinds[0], kinds[1]} == {"depth", "volume"}:
        return None
    return (kinds[0], kinds[1])


def _describe(constituent: str) -> str:
    return f"constituent {constituent!r}" if constituent else "no constituent"


def _is_offset(units: str) -> bool:
    return bool(unit_registry.Quantity(0.0, units).to_base_units().magnitude != 0)


def _linear_map(units: str, to_units: str) -> tuple[float, float]:
    """``(scale, offset)`` taking a value *v* in *units* to ``scale * v + offset`` in *to_units*."""
    zero = float(unit_registry.Quantity(0.0, units).to(to_units).magnitude)
    one = float(unit_registry.Quantity(1.0, units).to(to_units).magnitude)
    return one - zero, zero


def _offset_scale_factor(units: str, to_units: str) -> float:
    try:
        factor, zero = _linear_map(units, to_units)
    except pint.errors.DimensionalityError:
        raise ValueError("the quantities have different dimensions.") from None
    if zero != 0:
        raise ValueError(
            "an offset temperature scale (such as degC) converts by adding as well as "
            "multiplying, so it has no conversion factor. Convert temperature differences "
            "in K or with the same scale on both sides."
        )
    return float(factor)


def _to_amount_basis(constituent: str, units: str) -> pint.Quantity[Any]:
    """The multiplier that restates *units* of *constituent* as moles of it.

    The constituent qualifies the first unit in the string, as in
    :func:`format_units`: ``"g m-2"`` of C is grams *of carbon* per square
    meter, and ``"nmol g-1 s-1"`` of CO2 is nanomoles *of CO2* per gram of
    something else.  Reading the whole dimensionality instead would mistake
    the mass inside ``Pa`` or ``W`` for a mass of the substance, and lose the
    mass in ``ug g-1`` to cancellation.
    """
    kind = _substance_kind(units)
    if kind == "amount":
        return unit_registry.Quantity(1.0, "1")
    if constituent not in MOLAR_MASS:
        raise ValueError(f"{constituent!r} has no molar mass, so only its amount converts.")
    per_mole: pint.Quantity[Any] = 1 / unit_registry.Quantity(MOLAR_MASS[constituent], "g mol-1")
    if kind == "mass":
        return per_mole
    if constituent in DENSITY and kind in ("depth", "volume"):
        density = unit_registry.Quantity(DENSITY[constituent], "kg m-3")
        by_volume: pint.Quantity[Any] = density * per_mole
        return by_volume
    kinds = (
        "an amount, a mass, a depth or a volume"
        if constituent in DENSITY
        else "an amount or a mass"
    )
    raise ValueError(
        f"the constituent qualifies the first unit, and {units.split()[0]!r} in {units!r} is not "
        f"{kinds} of {constituent!r}."
    )


def _molecule_ratio(constituent: str, to_constituent: str) -> float:
    """Moles of *to_constituent* per mole of *constituent*."""
    if (constituent, to_constituent) in ATOMS_PER_MOLECULE:
        return 1 / ATOMS_PER_MOLECULE[(constituent, to_constituent)]
    if (to_constituent, constituent) in ATOMS_PER_MOLECULE:
        return float(ATOMS_PER_MOLECULE[(to_constituent, constituent)])
    pairs = ", ".join(f"{a}-{b}" for a, b in ATOMS_PER_MOLECULE)
    raise ValueError(
        f"there is no recorded atom ratio between {constituent!r} and {to_constituent!r}; "
        f"the pairs that convert are {pairs}."
    )
