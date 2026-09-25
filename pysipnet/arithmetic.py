"""Arithmetic on labeled ``DataArray`` objects, with a result that describes itself.

xarray drops attributes in arithmetic, so ``nee / days`` comes back with no
``units`` and :func:`~pysipnet.units.convert_dataarray_units` has nothing to
read.  The functions here do the arithmetic and write the ``units``,
``constituent`` and ``kind`` that are true of the result:

- :func:`multiply_with_units` and :func:`divide_with_units`;
- :func:`add_with_units` and :func:`subtract_with_units`, which require the
  operands to agree;
- :func:`step_length`, a pySIPNET array's ``time_step_length`` as a float
  array with units, so that a per-step total divides into a rate.

Leaf area index, for example, is ``leaf_carbon`` over the
``leaf_carbon_per_area`` parameter, both ``g m-2`` of C, which is a
dimensionless pool; and ``net_ecosystem_exchange`` over the step length in days
is a rate in ``g m-2 d-1`` of C, which converts to ``umol m-2 s-1`` of CO2::

    rate = divide_with_units(nee, step_length(nee))
    convert_dataarray_units(rate, to_units="umol m-2 s-1", to_constituent="CO2")

Operands
--------
An operand is a ``DataArray`` with a ``units`` attribute (and optionally
``constituent`` and ``kind``), or a plain real number, which is dimensionless
with no constituent and no kind.  At least one operand must be a
``DataArray``.  A parameter from
:func:`~pysipnet.parameters.model.parameter_dataarray` (or
:meth:`SIPNETParameters.dataarray
<pysipnet.parameters.model.SIPNETParameters.dataarray>`) carries ``units`` and
``constituent`` and no ``kind``; a model variable carries all three.

Values
------
Plain xarray arithmetic, broadcasting as usual, with nothing rescaled.  Index
coordinates must match exactly where both operands have them: two time axes
that differ are refused rather than silently cut to the labels they share.
Non-index coordinates of the operands carry through, so the result keeps a
model variable's ``time_step_start`` and ``time_step_length`` and can still be
passed to :func:`pysipnet.resample.resample`; a coordinate both operands carry
with different values, which xarray would silently drop, is refused.

Units and constituent
---------------------
As :func:`~pysipnet.units.product_units`,
:func:`~pysipnet.units.quotient_units`, :func:`~pysipnet.units.sum_units`
and :func:`~pysipnet.units.difference_units` give them, by the rules under
"Combination" in :mod:`pysipnet.units`; a refusal there is raised here,
prefixed by the operation.  ``per_day * nee`` is ``"g m-2 d-1"`` of C, the
constituent's operand leading.

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
when the other operand has no values at or below zero, in a sum when both
operands state the same one, and never in a difference.  ``long_name`` and
``derivation`` name the operation (``"net_ecosystem_exchange /
time_step_length"``), with an unnamed operand given by its own derivation in
parentheses.  Nothing describing a source rather than the result is carried:
not ``description``, ``sipnet_name``, ``output_decimals`` or SIPNET's
internal-conversion attributes.  The result's name is ``None``, and the
operands are not modified.

Refusals raise ``ValueError`` naming the operation and what would work, or
``TypeError`` for an operand that is neither a ``DataArray`` nor a real
number.  The algebra is deliberately not closed: a derivation the rules do not
cover sets the attributes itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr

from pysipnet.units import (
    difference_units,
    product_units,
    quotient_units,
    read_dataarray_units,
    sum_units,
    unit_registry,
)
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
_ALIGNMENT_ERROR: type[Exception] = getattr(xr, "AlignmentError", ValueError)


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
    derivation = f"{x.label} * {y.label}"
    kind = _product_or_quotient_kind(x, y, "*")
    units, constituent = _units_of(
        derivation,
        product_units,
        units=x.units,
        constituent=x.constituent,
        other_units=y.units,
        other_constituent=y.constituent,
    )
    return _labeled_result(
        _apply(x, y, "*"),
        units=units,
        constituent=constituent,
        kind=kind,
        sign_convention=_scaled_sign_convention(x, y),
        derivation=derivation,
    )


def divide_with_units(a: Operand, b: Operand) -> xr.DataArray:
    """``a / b``, labeled with the ``units``, ``constituent`` and ``kind`` true of it.

    See the module docstring for the rules.  ``nee / step_length(nee)`` is a
    ``daily_rate`` in ``"g m-2 d-1"`` of C.
    """
    x, y = _operands(a, b, "divide_with_units")
    derivation = f"{x.label} / {y.label}"
    kind = _product_or_quotient_kind(x, y, "/")
    units, constituent = _units_of(
        derivation,
        quotient_units,
        units=x.units,
        constituent=x.constituent,
        divisor_units=y.units,
        divisor_constituent=y.constituent,
    )
    return _labeled_result(
        _apply(x, y, "/"),
        units=units,
        constituent=constituent,
        kind=kind,
        sign_convention=_scaled_sign_convention(x, y),
        derivation=derivation,
    )


def add_with_units(a: Operand, b: Operand) -> xr.DataArray:
    """``a + b``; the operands must agree in ``units``, ``constituent`` and ``kind``."""
    x, y = _operands(a, b, "add_with_units")
    derivation = f"{x.label} + {y.label}"
    _require_same_kind(x, y, derivation)
    units, constituent = _units_of(
        derivation,
        sum_units,
        units=x.units,
        constituent=x.constituent,
        other_units=y.units,
        other_constituent=y.constituent,
    )
    return _labeled_result(
        _apply(x, y, "+"),
        units=units,
        constituent=constituent,
        kind=x.kind,
        sign_convention=x.sign_convention if x.sign_convention == y.sign_convention else "",
        derivation=derivation,
    )


def subtract_with_units(a: Operand, b: Operand) -> xr.DataArray:
    """``a - b``; the operands must agree in ``units``, ``constituent`` and ``kind``.

    The difference of two temperatures in ``degC`` is a temperature difference
    and is labeled ``K``.  No ``sign_convention`` is kept: a positive
    difference of two upward fluxes is not an upward flux.
    """
    x, y = _operands(a, b, "subtract_with_units")
    derivation = f"{x.label} - {y.label}"
    _require_same_kind(x, y, derivation)
    units, constituent = _units_of(
        derivation,
        difference_units,
        units=x.units,
        constituent=x.constituent,
        other_units=y.units,
        other_constituent=y.constituent,
    )
    return _labeled_result(
        _apply(x, y, "-"),
        units=units,
        constituent=constituent,
        kind=x.kind,
        sign_convention="",
        derivation=derivation,
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
        coords=coordinate.drop_vars(_STEP_LENGTH_COORDINATE).coords,
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
        try:
            units, constituent = read_dataarray_units(operand)
        except ValueError as exc:
            hint = (
                " Use step_length(), which gives the step length as a float with units."
                if operand.name == _STEP_LENGTH_COORDINATE
                else ""
            )
            raise ValueError(f"{what}(): {exc}{hint}") from None
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


def _units_of(
    derivation: str, rule: Callable[..., tuple[str, str]], **units: str
) -> tuple[str, str]:
    """``rule(**units)``, with a refusal prefixed by the operation it refused."""
    try:
        return rule(**units)
    except ValueError as exc:
        raise ValueError(f"{derivation}: {exc}") from None


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
    """An operand's ``sign_convention``, if the other operand cannot flip its sign."""
    source, other = (x, y) if x.sign_convention else (y, x)
    if not source.sign_convention:
        return ""
    if isinstance(other.value, xr.DataArray):
        flips = bool((other.value <= 0).any())
    else:
        flips = not other.value > 0
    return "" if flips else source.sign_convention


def _require_same_kind(x: _Labeled, y: _Labeled, derivation: str) -> None:
    if x.kind != y.kind:
        kinds = [k.value if k is not None else None for k in (x.kind, y.kind)]
        raise ValueError(
            f"{derivation}: a sum or difference needs operands of the same kind; got "
            f"{kinds[0]!r} and {kinds[1]!r}. Resample one of them first, or combine a total "
            "with a rate through step_length()."
        )


def _apply(x: _Labeled, y: _Labeled, op: str) -> xr.DataArray:
    if isinstance(x.value, xr.DataArray) and isinstance(y.value, xr.DataArray):
        try:
            xr.align(x.value, y.value, join="exact", copy=False)
        except ValueError as exc:
            # xr.AlignmentError only exists from xarray 2024.10; before, a plain ValueError.
            if not isinstance(exc, _ALIGNMENT_ERROR) or "join='exact'" not in str(exc):
                raise
            raise ValueError(
                f"{x.label} {op} {y.label}: the operands' index coordinates differ, and "
                "silently keeping only the labels they share would drop rows. Select the "
                "same labels on both first (for example with xr.align(..., join='inner')). "
                f"{exc}"
            ) from None
        _refuse_conflicting_coordinates(x, y, op)
    operations = {
        "*": lambda p, q: p * q,
        "/": lambda p, q: p / q,
        "+": lambda p, q: p + q,
        "-": lambda p, q: p - q,
    }
    with xr.set_options(keep_attrs=False):  # type: ignore[no-untyped-call]
        result: xr.DataArray = operations[op](x.value, y.value)
    return result


def _refuse_conflicting_coordinates(x: _Labeled, y: _Labeled, op: str) -> None:
    """Refuse non-index coordinates both operands carry with different values.

    xarray drops such a coordinate from the result without a word, so the
    result would lose ``time_step_start`` or ``time_step_length`` and could no
    longer be resampled.
    """
    a, b = x.value, y.value
    shared = (set(a.coords) & set(b.coords)) - set(a.indexes) - set(b.indexes)
    differing = sorted(
        str(name) for name in shared if not a.coords[name].variable.equals(b.coords[name].variable)
    )
    if differing:
        raise ValueError(
            f"{x.label} {op} {y.label}: the operands carry different values of the "
            f"coordinates {differing}, which the result would silently lose. Take both "
            "from the same record (step_length() of the array itself, for example), or "
            "drop the coordinate from one of them first."
        )


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
