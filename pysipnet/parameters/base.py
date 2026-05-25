"""Base types shared across all SIPNET version parameter models.

Unit convention
---------------
Unit strings follow **Pint** format and are validated at class-definition time
via :func:`validate_unit_string`.  Examples:

+----------------------------------+-------------------------------------+
| Quantity                         | ``unit`` string                     |
+==================================+=====================================+
| nmol CO₂ g⁻¹ leaf s⁻¹           | ``"nmol / (g * s)"``                |
+----------------------------------+-------------------------------------+
| g C m⁻²                         | ``"g / m**2"``                      |
+----------------------------------+-------------------------------------+
| dimensionless fraction           | ``"1"``                             |
+----------------------------------+-------------------------------------+
| °C                               | ``"degC"``                          |
+----------------------------------+-------------------------------------+
| year⁻¹                          | ``"1 / year"``                      |
+----------------------------------+-------------------------------------+
| K·day (growing degree-days)      | ``"K * day"`` (= °C·day; K used to avoid Pint offset-unit error) |
+----------------------------------+-------------------------------------+
| cm K⁻¹ day⁻¹                    | ``"cm / (K * day)"``                |
+----------------------------------+-------------------------------------+
| mol photons m⁻² (total per step)| ``"mol / m**2"`` (1 Einstein = 1 mol photons) |
+----------------------------------+-------------------------------------+

When a parameter is expressed in physical units that do not fully capture the
domain-specific substance (e.g., "g C" vs. generic "g"), the ``constituent``
field on :class:`ParameterSpec` provides the qualifier (e.g., ``"C"``,
``"N"``, ``"CO2 g-1 leaf"``).

Parameter domains
-----------------
:class:`ParameterDomain` encodes the mathematical support of each parameter.
This is the primary piece of information needed to select a bijector for
unconstrained optimisation or MCMC:

+---------------------+--------------------+------------------------------+
| Domain              | Support            | Typical bijector             |
+=====================+====================+==============================+
| ``REAL``            | (−∞, ∞)            | identity                     |
+---------------------+--------------------+------------------------------+
| ``POSITIVE``        | (0, ∞)             | log / softplus               |
+---------------------+--------------------+------------------------------+
| ``NON_NEGATIVE``    | [0, ∞)             | softplus                     |
+---------------------+--------------------+------------------------------+
| ``UNIT_INTERVAL``   | [0, 1]             | logistic / sigmoid           |
+---------------------+--------------------+------------------------------+
| ``OPEN_UNIT_INTERVAL`` | (0, 1)          | logit                        |
+---------------------+--------------------+------------------------------+

Querying parameter specs
------------------------
Every Pydantic model that uses :func:`param_field` stores a
:class:`ParameterSpec` in ``field.json_schema_extra``.  Use
:func:`get_parameter_specs` to retrieve a flat ``{dotted.path: ParameterSpec}``
dict from a model class, suitable for building bijector maps::

    from pysipnet.parameters.v1 import SIPNETParametersV1
    from pysipnet.parameters.base import get_parameter_specs, ParameterDomain

    specs = get_parameter_specs(SIPNETParametersV1)
    positive_params = {k for k, s in specs.items() if s.domain == ParameterDomain.POSITIVE}
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any

_MISSING: Any = dataclasses.MISSING

import pint
from pydantic import Field
from pydantic import BaseModel

_ureg = pint.UnitRegistry()


class ParameterDomain(str, Enum):
    """Mathematical support of a scalar parameter.

    Use this when constructing bijectors for unconstrained optimisation or
    sampling (e.g., TensorFlow Probability, NumPyro, PyMC).
    """

    REAL = "real"
    """(−∞, ∞) — identity bijector."""

    POSITIVE = "positive"
    """(0, ∞) — log / softplus bijector."""

    NON_NEGATIVE = "non_negative"
    """[0, ∞) — softplus bijector."""

    UNIT_INTERVAL = "unit_interval"
    """[0, 1] — logistic / sigmoid bijector."""

    OPEN_UNIT_INTERVAL = "open_unit_interval"
    """(0, 1) — logit bijector."""


@dataclass(frozen=True)
class ParameterSpec:
    """Complete specification for a single scalar parameter.

    Stored in ``field.json_schema_extra`` for every field produced by
    :func:`param_field`.  Access via :func:`get_parameter_specs`.
    """

    unit: str
    """Pint-parseable unit string for the physical dimension (see module docs)."""

    domain: ParameterDomain
    """Mathematical support; drives bijector selection in calibration."""

    description: str
    """Human-readable description of the parameter."""

    constituent: str = ""
    """Substance qualifier not captured by the physical unit, e.g. ``"C"``,
    ``"N"``, ``"CO2 g-1 leaf"``.  Empty string when not applicable."""

    per_year: bool = False
    """``True`` when SIPNET reads this value as a per-year rate and converts
    it to per-day internally (divide by 365).  The Python interface always
    works in per-year units for these parameters, matching the SIPNET param
    file convention."""


def validate_unit_string(unit: str) -> None:
    """Raise :class:`pint.UndefinedUnitError` if *unit* is not Pint-parseable."""
    try:
        _ureg.parse_expression(unit)
    except pint.errors.UndefinedUnitError as exc:
        raise ValueError(f"Unit string {unit!r} is not recognised by Pint: {exc}") from exc


def param_field(
    *,
    unit: str,
    domain: ParameterDomain,
    description: str,
    constituent: str = "",
    per_year: bool = False,
    default: Any = _MISSING,
) -> Any:
    """Factory for a Pydantic ``Field`` with embedded :class:`ParameterSpec`.

    Pydantic validators (``gt``, ``ge``, ``le``, ``lt``) are derived
    automatically from *domain* so there is no risk of the two diverging.

    Parameters
    ----------
    unit:
        Pint-parseable unit string (see module-level docs for the convention).
    domain:
        Mathematical support — controls both Pydantic validators and the
        ``ParameterDomain`` stored in ``json_schema_extra``.
    description:
        Human-readable description included in the JSON schema.
    constituent:
        Substance qualifier not captured by the physical unit.
    per_year:
        Set to ``True`` for parameters SIPNET reads as annual rates.
    default:
        Field default value.  Omit (or pass ``_MISSING``) to make the field
        required.  Pass ``None`` to make it optional with a ``None`` default.
    """
    validate_unit_string(unit)

    pydantic_kwargs: dict[str, Any] = {}
    if domain == ParameterDomain.POSITIVE:
        pydantic_kwargs["gt"] = 0
    elif domain == ParameterDomain.NON_NEGATIVE:
        pydantic_kwargs["ge"] = 0
    elif domain == ParameterDomain.UNIT_INTERVAL:
        pydantic_kwargs["ge"] = 0
        pydantic_kwargs["le"] = 1
    elif domain == ParameterDomain.OPEN_UNIT_INTERVAL:
        pydantic_kwargs["gt"] = 0
        pydantic_kwargs["lt"] = 1

    spec = ParameterSpec(
        unit=unit,
        domain=domain,
        description=description,
        constituent=constituent,
        per_year=per_year,
    )

    return Field(
        ... if default is _MISSING else default,
        description=description,
        json_schema_extra={
            "unit": unit,
            "constituent": constituent,
            "domain": domain.value,
            "per_year": per_year,
            "_spec": spec,
        },
        **pydantic_kwargs,
    )


def get_parameter_specs(model_cls: type[BaseModel], prefix: str = "") -> dict[str, ParameterSpec]:
    """Return a flat ``{dotted.field.path: ParameterSpec}`` dict for *model_cls*.

    Recurses into nested :class:`pydantic.BaseModel` fields.  Only fields
    produced by :func:`param_field` (i.e., those with ``json_schema_extra``
    containing a ``"_spec"`` key) are included.

    Example::

        specs = get_parameter_specs(SIPNETParametersV1)
        # {"photosynthesis.a_max": ParameterSpec(...), ...}
    """
    result: dict[str, ParameterSpec] = {}
    for name, field_info in model_cls.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        extra = field_info.json_schema_extra
        if isinstance(extra, dict) and "_spec" in extra:
            result[path] = extra["_spec"]
        else:
            annotation = field_info.annotation
            if annotation is not None and isinstance(annotation, type) and issubclass(annotation, BaseModel):
                result.update(get_parameter_specs(annotation, prefix=path))
    return result
