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

1. Same dimension, same constituent (or none on either side): Pint's factor.
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
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Literal, overload

import numpy as np
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

# Tokens that name a substance rather than a unit.  Pint accepts every one of
# them as something else (coulomb, newton, ...), so they must be refused here.
_CONSTITUENT_TOKENS: frozenset[str] = frozenset(MOLAR_MASS)

# A symbol with an optional signed integer exponent ("m", "m-2", "kPa-1"), or the
# bare "1" of a dimensionless quantity.
_UDUNITS_TOKEN = re.compile(r"[A-Za-z]+(-?\d+)?|1")

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
unit_registry.define("einstein = mole = E")
"""Pint registry that understands the package's UDUNITS-style unit strings.

It converts between physical units only::

    from pysipnet.units import unit_registry
    q = unit_registry.Quantity(values, "g m-2 d-1").to("kg m-2 s-1")

For a quantity with a ``constituent``, use :func:`conversion_factor`,
:func:`convert_units` or :func:`convert_dataarray_units` instead, which apply
the constituent rules as well.
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
            # parenthesized expression).  Fall back to the raw token.
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
) -> npt.NDArray[np.float64]: ...
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
) -> float | npt.NDArray[np.float64] | pd.Series | pd.DataFrame:
    """Convert unlabeled *values* from *units* of *constituent* to *to_units* of *to_constituent*.

    Returns ``values * conversion_factor(...)`` for the same keyword
    arguments, with the shape and type of *values*: a number, a NumPy array,
    or a pandas ``Series`` or ``DataFrame``.  The caller states the units the
    values are in; nothing about *values* is read or rewritten.
    *to_constituent* defaults to *constituent*.

    Refused with ``TypeError``: any xarray object, and a pandas object whose
    ``attrs`` has a ``"units"`` entry.  Both keep their ``attrs`` through the
    multiplication, so the result would still claim the old units.  Convert a
    ``DataArray`` with :func:`convert_dataarray_units`, which reads the units
    from its attributes and relabels the result.
    """
    if isinstance(values, (xr.DataArray, xr.Dataset, xr.Variable)):
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
    ``array.attrs.get("constituent", "")``, the attributes every pySIPNET
    output, climate and parameter ``DataArray`` carries, so the units the
    values are in cannot be misstated.  *to_constituent* defaults to the
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
    label = f"DataArray {array.name!r}" if array.name is not None else "DataArray"
    units = array.attrs.get("units")
    if not isinstance(units, str):
        raise ValueError(
            f"{label} has no 'units' attribute, so its current units are unknown. Set "
            "array.attrs['units'], or convert array.values with convert_units()."
        )
    constituent = array.attrs.get("constituent", "")
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

    if constituent == to_constituent and source.dimensionality == target.dimensionality:
        return float(source.to(to_units).magnitude)
    mismatch = (
        f"the quantities have different dimensions ({source.dimensionality} and "
        f"{target.dimensionality})"
    )
    if not constituent:
        ratio = source.dimensionality / target.dimensionality
        mass_per_amount = unit_registry.Quantity(1.0, "g mol-1").dimensionality
        if ratio in (mass_per_amount, 1 / mass_per_amount):
            raise ValueError(
                "mass and amount convert through a molar mass, and a molar mass needs a "
                "substance; name the constituent on both sides."
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


def _describe(constituent: str) -> str:
    return f"constituent {constituent!r}" if constituent else "no constituent"


def _is_offset(units: str) -> bool:
    return bool(unit_registry.Quantity(0.0, units).to_base_units().magnitude != 0)


def _offset_scale_factor(units: str, to_units: str) -> float:
    try:
        factor = unit_registry.Quantity(1.0, units).to(to_units).magnitude
        zero = unit_registry.Quantity(0.0, units).to(to_units).magnitude
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
    first = units.split()[0]
    dimensions = unit_registry.Quantity(1.0, first).dimensionality
    if dimensions == unit_registry.Quantity(1.0, "mol").dimensionality:
        return unit_registry.Quantity(1.0, "1")
    if constituent not in MOLAR_MASS:
        raise ValueError(f"{constituent!r} has no molar mass, so only its amount converts.")
    per_mole: pint.Quantity[Any] = 1 / unit_registry.Quantity(MOLAR_MASS[constituent], "g mol-1")
    if dimensions == unit_registry.Quantity(1.0, "g").dimensionality:
        return per_mole
    length = unit_registry.Quantity(1.0, "m").dimensionality
    if constituent in DENSITY and dimensions in (length, length**3):
        density = unit_registry.Quantity(DENSITY[constituent], "kg m-3")
        by_volume: pint.Quantity[Any] = density * per_mole
        return by_volume
    kinds = (
        "an amount, a mass, a depth or a volume"
        if constituent in DENSITY
        else "an amount or a mass"
    )
    raise ValueError(
        f"the constituent qualifies the first unit, and {first!r} in {units!r} is not "
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
