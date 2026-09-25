"""Arithmetic on labeled ``DataArray`` objects, with a result that describes itself.

xarray drops attributes in arithmetic, so ``nee / days`` comes back with no
``units`` and :func:`~pysipnet.units.convert_dataarray_units` has nothing to
read.  The functions here do the arithmetic an observation operator needs and
write the ``units``, ``constituent`` and ``kind`` that are true of the result:

- :func:`multiply_with_units` and :func:`divide_with_units`, which combine
  units, constituent and kind by the rules below;
- :func:`add_with_units` and :func:`subtract_with_units`, which require the
  operands to agree;
- :func:`step_length`, a pySIPNET array's ``time_step_length`` as a float
  array with units, so that a per-step total divides into a rate.

Two uses motivate them.  Leaf area index is ``leaf_carbon`` over the
``leaf_carbon_per_area`` parameter, both ``g m-2`` of C, which is a
dimensionless pool.  An NEE rate is ``net_ecosystem_exchange`` over the step
length in days, ``g m-2 d-1`` of C, which converts to ``umol m-2 s-1`` of CO2::

    rate = divide_with_units(nee, step_length(nee))
    convert_dataarray_units(rate, to_units="umol m-2 s-1", to_constituent="CO2")

Operands
--------
An operand is a ``DataArray`` with a ``units`` attribute (and optionally
``constituent`` and ``kind``), or a plain real number, which is dimensionless
with no constituent and no kind.  At least one operand must be a
``DataArray``.  A parameter array carries ``units`` and ``constituent`` and no
``kind``; a model variable carries all three.  An operand in an offset unit
(``degC``) is refused, since a temperature on an offset scale does not
multiply; the one exception is the difference of two, which is in ``K``.

Values
------
Plain xarray arithmetic, broadcasting as usual, with nothing rescaled.  Index
coordinates must match exactly where both operands have them: two time axes
that differ are refused rather than silently cut to the labels they share.
Non-index coordinates of the operands carry through, so the result keeps a
model variable's ``time_step_start`` and ``time_step_length`` and can still be
passed to :func:`pysipnet.resample.resample`.

Units
-----
Combined symbol by symbol, adding exponents for a product and subtracting them
for a quotient.  A symbol whose exponent reaches zero drops out, and an empty
result is ``"1"``.  Prefixes are not merged: ``cm`` over ``m`` is
``"cm m-1"``, which :func:`~pysipnet.units.convert_dataarray_units` converts to
``"1"`` on request.  The operand that carries the constituent leads the string,
because a constituent qualifies the first unit (see :mod:`pysipnet.units`):
``per_day * nee`` is ``"g m-2 d-1"``, not ``"d-1 g m-2"``.  For the same reason
that first unit must survive unchanged: ``g m-2`` of C divided by ``g`` would
leave ``m-2`` of C, which names no quantity of carbon, and is refused.

Constituent
-----------
In a product, at most one operand names one and the result takes it.  In a
quotient, the numerator's is kept when the denominator has none, and cancels
when both name the same one: leaf carbon over leaf carbon per area is a ratio,
not carbon.  A denominator naming a constituent the numerator lacks is
refused, and so are two different constituents; convert one first.
:func:`add_with_units` and :func:`subtract_with_units` require the same one.

Kind
----
At most one operand of a product or quotient has a kind, and it may not be the
denominator: a state times a flux, or anything per pool, is not something
SIPNET reports.  The result keeps that kind unless the other operand's units
carry a time, in which case the kind changes as
:data:`~pysipnet.variables.KIND_AFTER_TIME_POWER` says: a ``timestep_total``
divided by a time (or multiplied by a per-time) is a ``daily_rate``, and a
``daily_rate`` multiplied by a time (or divided by a per-time) is a
``timestep_total``.  The kind is named ``daily_rate`` whatever the time unit;
the units say which.  Every other change of time dimension is refused: a pool
times a turnover rate is a flux SIPNET reports itself.  A time coordinate is
not a quantity and is refused as an operand.  :func:`add_with_units` and
:func:`subtract_with_units` require the same kind on both sides.

Result
------
``units``; ``constituent`` when there is one; and when there is a kind,
``kind`` with the ``time_reference`` and ``cell_methods`` pySIPNET gives it.
``sign_convention`` is kept when it is still true: in a product or quotient
when the other operand has no values below or at zero, in a sum when both
operands state the same one, and never in a difference.  ``long_name`` and
``derivation`` name the operation (``"net_ecosystem_exchange /
time_step_length"``), with an unnamed operand given by its own derivation in
parentheses.  Nothing describing a source rather than the result is carried:
not ``description``, ``sipnet_name``, ``output_decimals`` or SIPNET's
internal-conversion attributes.  The result's name is ``None``, and the
operands are not modified.

Refusals raise ``ValueError`` naming the operands and what would work, or
``TypeError`` for an operand that is neither a ``DataArray`` nor a real
number.  The algebra is deliberately not closed: a derivation the rules do not
cover sets the attributes itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr

from pysipnet.units import _is_offset, unit_registry, validate_units
from pysipnet.variables import (
    CELL_METHODS_FOR_KIND,
    KIND_AFTER_TIME_POWER,
    TIME_REFERENCE_FOR_KIND,
    VariableKind,
)

Operand = xr.DataArray | float
"""What each function takes on either side: a labeled ``DataArray`` or a plain number."""

StepLengthUnits = Literal["d", "h", "s"]

_STEP_LENGTH_COORDINATE = "time_step_length"
_NANOSECONDS_PER: dict[str, float] = {"d": 86_400e9, "h": 3_600e9, "s": 1e9}
_UNIT_TOKEN = re.compile(r"^([A-Za-z]+)(-?\d+)?$")


@dataclass(frozen=True)
class _Labeled:
    """One operand, with the attributes the rules read."""

    value: Any
    units: str
    constituent: str
    kind: VariableKind | None
    label: str
    sign_convention: str


def multiply_with_units(a: Operand, b: Operand) -> xr.DataArray:
    """``a * b``, labeled with the ``units``, ``constituent`` and ``kind`` true of it.

    See the module docstring for the rules.  ``per_day * nee`` and
    ``nee * per_day`` are both ``"g m-2 d-1"`` of C.
    """
    x, y = _operands(a, b, "multiply_with_units")
    _refuse_offset(x, y, "multiply")
    kind = _product_or_quotient_kind(x, y, "*")
    if x.constituent and y.constituent:
        raise ValueError(
            f"multiply_with_units(): {x.label} is a quantity of {x.constituent!r} and "
            f"{y.label} a quantity of {y.constituent!r}; at most one operand of a product "
            "names a constituent. Convert one of them to a quantity with no constituent, "
            "or divide instead if the result is a ratio."
        )
    lead, other = (y, x) if y.constituent else (x, y)
    units = _combine_units(lead, other, +1, "multiply_with_units")
    return _labeled_result(
        _apply(x, y, "*"),
        units=units,
        constituent=lead.constituent,
        kind=kind,
        sign_convention=_scaled_sign_convention(x, y),
        derivation=f"{x.label} * {y.label}",
    )


def divide_with_units(a: Operand, b: Operand) -> xr.DataArray:
    """``a / b``, labeled with the ``units``, ``constituent`` and ``kind`` true of it.

    See the module docstring for the rules.  ``nee / step_length(nee)`` is a
    ``daily_rate`` in ``"g m-2 d-1"`` of C.
    """
    x, y = _operands(a, b, "divide_with_units")
    _refuse_offset(x, y, "divide")
    kind = _product_or_quotient_kind(x, y, "/")
    constituent = _quotient_constituent(x, y)
    units = _combine_units(x, y, -1, "divide_with_units", keep_lead=bool(constituent))
    return _labeled_result(
        _apply(x, y, "/"),
        units=units,
        constituent=constituent,
        kind=kind,
        sign_convention=_scaled_sign_convention(x, y),
        derivation=f"{x.label} / {y.label}",
    )


def add_with_units(a: Operand, b: Operand) -> xr.DataArray:
    """``a + b``; the operands must agree in ``units``, ``constituent`` and ``kind``."""
    x, y = _operands(a, b, "add_with_units")
    _refuse_offset(x, y, "add")
    _require_agreement(x, y, "add_with_units")
    same_sign = x.sign_convention if x.sign_convention == y.sign_convention else ""
    return _labeled_result(
        _apply(x, y, "+"),
        units=x.units,
        constituent=x.constituent,
        kind=x.kind,
        sign_convention=same_sign,
        derivation=f"{x.label} + {y.label}",
    )


def subtract_with_units(a: Operand, b: Operand) -> xr.DataArray:
    """``a - b``; the operands must agree in ``units``, ``constituent`` and ``kind``.

    The difference of two temperatures in ``degC`` is a temperature difference
    and is labeled ``K``.  No ``sign_convention`` is kept: a positive
    difference of two upward fluxes is not an upward flux.
    """
    x, y = _operands(a, b, "subtract_with_units")
    _require_agreement(x, y, "subtract_with_units")
    units = x.units
    if _is_offset(units):
        if _kelvin_per_degree(units) != 1.0:
            _refuse_offset(x, y, "subtract")
        units = "K"
    return _labeled_result(
        _apply(x, y, "-"),
        units=units,
        constituent=x.constituent,
        kind=x.kind,
        sign_convention="",
        derivation=f"{x.label} - {y.label}",
    )


def step_length(data: xr.DataArray | xr.Dataset, units: StepLengthUnits = "d") -> xr.DataArray:
    """The length of each of *data*'s timesteps, as a float array in *units*.

    Reads the ``time_step_length`` coordinate every pySIPNET output carries
    and returns it with the coordinate's own dimensions and coordinates, in
    ``"d"``, ``"h"`` or ``"s"``, with a ``units`` attribute and no ``kind``, so
    that ``divide_with_units(total, step_length(total))`` is a rate.  A missing
    length (``NaT``, the padding in a stack of records of unequal length) is
    ``NaN``.

    The coordinate itself cannot be used as an operand: it is a ``timedelta64``
    with no ``units``, and it declares the kind ``timestep_total``, so dividing
    by it would be refused.

    Raises ``ValueError`` if *data* has no ``time_step_length`` coordinate or
    *units* is not one of the three, and ``TypeError`` for anything but a
    ``DataArray`` or ``Dataset``.
    """
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"step_length() takes an xarray DataArray or Dataset, not {type(data).__name__}."
        )
    if units not in _NANOSECONDS_PER:
        raise ValueError(f"step_length() units must be one of 'd', 'h' or 's', not {units!r}.")
    if _STEP_LENGTH_COORDINATE not in data.coords:
        what = f"DataArray {data.name!r}" if isinstance(data, xr.DataArray) else "this Dataset"
        raise ValueError(
            f"{what} has no {_STEP_LENGTH_COORDINATE!r} coordinate. Only pySIPNET output "
            "and ClimateDrivers.xarray declare their step lengths; take the array from one "
            "of them, before any reduction that drops the coordinate."
        )
    coordinate = data.coords[_STEP_LENGTH_COORDINATE]
    lengths = np.asarray(coordinate.values).astype("timedelta64[ns]")
    nanoseconds = np.where(np.isnat(lengths), np.nan, lengths.astype("int64").astype("float64"))
    return xr.DataArray(
        nanoseconds / _NANOSECONDS_PER[units],
        dims=coordinate.dims,
        coords=coordinate.coords,
        name=_STEP_LENGTH_COORDINATE,
        attrs={"units": units, "long_name": "Timestep length"},
    )


def _operands(a: Any, b: Any, what: str) -> tuple[_Labeled, _Labeled]:
    if not isinstance(a, xr.DataArray) and not isinstance(b, xr.DataArray):
        raise TypeError(
            f"{what}() needs at least one xarray DataArray operand; for two numbers, use "
            "plain arithmetic."
        )
    return _labeled(a, what), _labeled(b, what)


def _labeled(operand: Any, what: str) -> _Labeled:
    """An operand's value and attributes; a number is dimensionless with no constituent or kind."""
    if isinstance(operand, xr.DataArray):
        label = _operand_label(operand)
        units = operand.attrs.get("units")
        if not isinstance(units, str):
            hint = (
                " Use step_length(), which gives the step length as a float with units."
                if operand.name == _STEP_LENGTH_COORDINATE
                else ""
            )
            raise ValueError(
                f"{what}(): {label} has no 'units' attribute, so the result's units would be "
                "unknown. Take the operand from pySIPNET output or build it from a "
                f"parameter's spec, or set attrs['units'] first.{hint}"
            )
        try:
            validate_units(units)
        except ValueError as exc:
            raise ValueError(f"{what}(): {label}: {exc}") from None
        constituent = operand.attrs.get("constituent", "")
        if not isinstance(constituent, str):
            raise ValueError(
                f"{what}(): {label} has a 'constituent' attribute of {constituent!r}; it must "
                "be a str, '' or absent for none."
            )
        raw_kind = operand.attrs.get("kind")
        try:
            kind = VariableKind(raw_kind) if raw_kind is not None else None
        except ValueError:
            valid = ", ".join(repr(k.value) for k in VariableKind)
            raise ValueError(
                f"{what}(): {label} has a 'kind' attribute of {raw_kind!r}, which is not one "
                f"of pySIPNET's kinds ({valid})."
            ) from None
        if kind is VariableKind.TIMESTEP_START_COORDINATE:
            raise ValueError(
                f"{what}(): {label} is a time coordinate, not a quantity; it does not enter "
                "arithmetic."
            )
        sign = operand.attrs.get("sign_convention", "")
        return _Labeled(
            operand, units, constituent, kind, label, sign if isinstance(sign, str) else ""
        )
    if isinstance(operand, (bool, np.bool_)) or not isinstance(
        operand, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"{what}() takes an xarray DataArray with a 'units' attribute or a real number, "
            f"not {type(operand).__name__}."
        )
    number = operand.item() if isinstance(operand, np.generic) else operand
    return _Labeled(number, "1", "", None, repr(number), "")


def _operand_label(array: xr.DataArray) -> str:
    """How an array is named in a derivation: its name, its own derivation, or ``array``."""
    if array.name is not None:
        return str(array.name)
    derivation = array.attrs.get("derivation")
    if isinstance(derivation, str) and derivation:
        return f"({derivation})"
    return "array"


def _refuse_offset(x: _Labeled, y: _Labeled, verb: str) -> None:
    for operand in (x, y):
        if _is_offset(operand.units):
            raise ValueError(
                f"cannot {verb} {operand.label}, which is in {operand.units!r}: a temperature "
                "on an offset scale is not a multiple of anything, so its products, "
                "quotients and sums have no units. Only the difference of two is defined "
                "(subtract_with_units, which gives K); express the temperature in K instead."
            )


def _kelvin_per_degree(units: str) -> float:
    to_kelvin = [unit_registry.Quantity(t, units).to("K").magnitude for t in (0.0, 1.0)]
    return float(to_kelvin[1] - to_kelvin[0])


def _exponents(units: str) -> dict[str, int]:
    """A UDUNITS string as ``{symbol: exponent}``, in the order the symbols appear."""
    exponents: dict[str, int] = {}
    for token in units.split():
        if token == "1":
            continue
        match = _UNIT_TOKEN.match(token)
        if match is None:
            raise ValueError(f"cannot combine the unit token {token!r} of {units!r}.")
        symbol = match.group(1)
        exponents[symbol] = exponents.get(symbol, 0) + int(match.group(2) or 1)
    return exponents


def _render(exponents: dict[str, int]) -> str:
    parts = [
        symbol if exponent == 1 else f"{symbol}{exponent}"
        for symbol, exponent in exponents.items()
        if exponent != 0
    ]
    return " ".join(parts) if parts else "1"


def _combine_units(
    lead: _Labeled, other: _Labeled, sign: int, what: str, *, keep_lead: bool | None = None
) -> str:
    """*lead*'s units times (``sign=+1``) or over (``sign=-1``) *other*'s, *lead*'s symbols first.

    When the result has a constituent (*keep_lead*, which defaults to whether
    *lead* names one), *lead*'s first unit is the one it qualifies, and it must
    come through first and unchanged.
    """
    exponents = _exponents(lead.units)
    for symbol, exponent in _exponents(other.units).items():
        exponents[symbol] = exponents.get(symbol, 0) + sign * exponent
    units = _render(exponents)
    if lead.constituent if keep_lead is None else keep_lead:
        first = lead.units.split()[0]
        if units.split()[0] != first:
            op = "*" if sign > 0 else "/"
            raise ValueError(
                f"{what}(): {lead.label} is {lead.units!r} of {lead.constituent!r}, and the "
                f"constituent qualifies its first unit, {first!r}; {lead.label} {op} "
                f"{other.label} is {units!r}, which changes that unit, so the result would "
                f"not say what quantity of {lead.constituent!r} it measures. Convert "
                f"{other.label} so that {first!r} does not cancel or combine."
            )
    validate_units(units)
    return units


def _quotient_constituent(x: _Labeled, y: _Labeled) -> str:
    if not y.constituent:
        return x.constituent
    if x.constituent == y.constituent:
        return ""
    if not x.constituent:
        raise ValueError(
            f"divide_with_units(): {y.label} is a quantity of {y.constituent!r} but {x.label} "
            "names no constituent, so the result would be per unit of a substance the "
            "numerator does not measure. Divide by a quantity with no constituent, or give "
            "the numerator its constituent."
        )
    raise ValueError(
        f"divide_with_units(): {x.label} is a quantity of {x.constituent!r} and {y.label} a "
        f"quantity of {y.constituent!r}. Convert one with "
        "pysipnet.units.convert_dataarray_units so both name the same constituent, and "
        "the quotient is a ratio."
    )


def _time_power(units: str) -> float:
    power: Any = unit_registry.Quantity(1.0, units).dimensionality.get("[time]", 0)
    return float(power)


def _product_or_quotient_kind(x: _Labeled, y: _Labeled, op: str) -> VariableKind | None:
    verb = "multiply" if op == "*" else "divide"
    if x.kind is not None and y.kind is not None:
        raise ValueError(
            f"cannot {verb} {x.label} (kind {x.kind.value!r}) and {y.label} (kind "
            f"{y.kind.value!r}): at most one operand of a product or quotient may have a "
            "kind, since combining two is not something SIPNET reports. Combine the "
            "variable with a parameter, a number or step_length()."
        )
    if y.kind is not None and op == "/":
        raise ValueError(
            f"cannot divide by {y.label}, which has the kind {y.kind.value!r}: no pySIPNET "
            "kind names a value per pool or per total. Put the variable in the numerator, "
            "or divide a total by step_length() first."
        )
    if x.kind is None and y.kind is None:
        return None
    kinded, other = (x, y) if x.kind is not None else (y, x)
    assert kinded.kind is not None
    power = _time_power(other.units)
    change = power if op == "*" else -power
    if change == 0:
        return kinded.kind
    changed = KIND_AFTER_TIME_POWER.get((kinded.kind, int(change))) if change.is_integer() else None
    if changed is not None:
        return changed
    defined = "; ".join(
        f"a {kind.value!r} {'times' if p > 0 else 'per'} a time is a {result.value!r}"
        for (kind, p), result in KIND_AFTER_TIME_POWER.items()
    )
    raise ValueError(
        f"cannot {verb} {kinded.label} (kind {kinded.kind.value!r}) {op} {other.label} "
        f"({other.units!r}): that multiplies it by time to the power {change:g}, which "
        "changes what the value is over a step, and no pySIPNET kind names the result. "
        f"Defined: {defined}. Anything else, set the attributes yourself."
    )


def _scaled_sign_convention(x: _Labeled, y: _Labeled) -> str:
    """The kinded operand's ``sign_convention``, if the other operand cannot flip a sign."""
    source, other = (x, y) if x.sign_convention else (y, x)
    if not source.sign_convention:
        return ""
    if isinstance(other.value, xr.DataArray):
        flips = bool((other.value <= 0).any())
    else:
        flips = not other.value > 0
    return "" if flips else source.sign_convention


def _require_agreement(x: _Labeled, y: _Labeled, what: str) -> None:
    def described(operand: _Labeled) -> str:
        kind = operand.kind.value if operand.kind is not None else None
        return f"{operand.label} ({operand.units!r}, {operand.constituent!r}, kind {kind!r})"

    if (x.units, x.constituent, x.kind) != (y.units, y.constituent, y.kind):
        raise ValueError(
            f"{what}() needs operands that agree in units, constituent and kind; got "
            f"{described(x)} and {described(y)}. Convert one with "
            "pysipnet.units.convert_dataarray_units first, or resample both to the same kind."
        )


def _apply(x: _Labeled, y: _Labeled, op: str) -> xr.DataArray:
    operations = {
        "*": lambda p, q: p * q,
        "/": lambda p, q: p / q,
        "+": lambda p, q: p + q,
        "-": lambda p, q: p - q,
    }
    try:
        with xr.set_options(arithmetic_join="exact", keep_attrs=False):  # type: ignore[no-untyped-call]
            result: xr.DataArray = operations[op](x.value, y.value)
    except ValueError as exc:
        # xr.AlignmentError, a ValueError, only exists from xarray 2024.10.
        if "exact" not in str(exc):
            raise
        raise ValueError(
            f"{x.label} {op} {y.label}: the operands' index coordinates differ, and "
            "silently keeping only the labels they share would drop rows. Select the same "
            f"labels on both first (for example with xr.align(..., join='inner')). {exc}"
        ) from None
    return result


def _labeled_result(
    values: xr.DataArray,
    *,
    units: str,
    constituent: str,
    kind: VariableKind | None,
    sign_convention: str,
    derivation: str,
) -> xr.DataArray:
    attrs: dict[str, Any] = {"units": units}
    if constituent:
        attrs["constituent"] = constituent
    if kind is not None:
        attrs["kind"] = kind.value
        attrs["time_reference"] = TIME_REFERENCE_FOR_KIND[kind]
        cell_methods = CELL_METHODS_FOR_KIND[kind]
        if cell_methods is not None:
            attrs["cell_methods"] = cell_methods
    if sign_convention:
        attrs["sign_convention"] = sign_convention
    attrs["long_name"] = derivation
    attrs["derivation"] = derivation
    values.attrs = attrs
    values.name = None
    return values


__all__ = [
    "Operand",
    "StepLengthUnits",
    "add_with_units",
    "divide_with_units",
    "multiply_with_units",
    "step_length",
    "subtract_with_units",
]
