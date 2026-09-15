"""Base types shared across all SIPNET version parameter models.

Every parameter field carries a :class:`ParameterSpec`: the SIPNET name it is
written under, its units, its mathematical domain, a description, plot labels,
aliases, and (for initial conditions) which output state it initializes.
:func:`get_parameter_specs` returns them all as a flat ``{dotted.path: spec}``
dict, which is what calibration tooling, the parameter-file writer and the
generated documentation read.

Naming convention
-----------------
The same as for output variables (:mod:`pysipnet.variables`): lower-case
words joined by underscores, no acronyms, no truncations.
``max_photosynthesis_rate`` rather than ``a_max``; ``soil_water_holding_capacity``
rather than ``soil_whc``. SIPNET's own names are recorded in ``sipnet_name``
and accepted as aliases by :func:`resolve_parameter_name`, as are the names
pySIPNET used before this convention.

Unit convention
---------------
Unit strings are UDUNITS-style, validated at class-definition time by
:func:`pysipnet.units.validate_units`, with the substance in ``constituent``
rather than in the string (see :mod:`pysipnet.units` for why). Examples:

+-----------------------------------+------------------+---------------+
| Quantity                          | ``units``        | ``constituent`` |
+===================================+==================+===============+
| nmol CO₂ g⁻¹ leaf s⁻¹            | ``"nmol g-1 s-1"`` | ``"CO2"``   |
+-----------------------------------+------------------+---------------+
| g C m⁻²                          | ``"g m-2"``      | ``"C"``       |
+-----------------------------------+------------------+---------------+
| dimensionless fraction            | ``"1"``          |               |
+-----------------------------------+------------------+---------------+
| °C                                | ``"degC"``       |               |
+-----------------------------------+------------------+---------------+
| year⁻¹                           | ``"yr-1"``       |               |
+-----------------------------------+------------------+---------------+
| growing degree-days               | ``"K d"``        |               |
+-----------------------------------+------------------+---------------+
| cm water K⁻¹ day⁻¹               | ``"cm K-1 d-1"`` | ``"H2O"``     |
+-----------------------------------+------------------+---------------+
| leaf area index                   | ``"m2 m-2"``     |               |
+-----------------------------------+------------------+---------------+

Parameter domains
-----------------
:class:`ParameterDomain` encodes the mathematical support of each parameter.
This is the primary piece of information needed to select a bijector for
unconstrained optimization or MCMC:

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
::

    from pysipnet.parameters.model import SIPNETParameters
    from pysipnet.parameters.base import get_parameter_specs, ParameterDomain

    specs = get_parameter_specs(SIPNETParameters)
    positive_params = {k for k, s in specs.items() if s.domain == ParameterDomain.POSITIVE}
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from pysipnet.units import UnitStyle, format_units, validate_units
from pysipnet.variables import NAME_PATTERN

_MISSING: Any = dataclasses.MISSING


class ParameterDomain(StrEnum):
    """Mathematical support of a scalar parameter.

    Use this when constructing bijectors for unconstrained optimization or
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


class ParameterGroup(BaseModel):
    """Base class for the parameter sub-models.

    Unknown field names are an error rather than silently ignored, so a
    parameter set saved under a name this version no longer uses fails loudly
    with the offending key in the message.
    """

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ParameterSpec:
    """Complete specification for a single scalar parameter.

    Stored in ``field.json_schema_extra`` for every field produced by
    :func:`param_field`. Access via :func:`get_parameter_specs`.
    """

    sipnet_name: str
    """The name SIPNET reads this parameter under in the ``.param`` file."""

    units: str
    """UDUNITS-style unit string, physical units only (see :mod:`pysipnet.units`)."""

    domain: ParameterDomain
    """Mathematical support; drives bijector selection in calibration."""

    description: str
    """What the parameter is and how SIPNET uses it."""

    long_label: str
    """Plot-ready name without units, e.g. ``"Maximum photosynthesis rate"``."""

    constituent: str = ""
    """Substance the unit refers to: ``"C"``, ``"N"``, ``"CO2"``, ``"H2O"`` or empty."""

    short_label: str = ""
    """Compact label for legends; empty means use *long_label*."""

    aliases: tuple[str, ...] = ()
    """Other names :func:`resolve_parameter_name` accepts, e.g. the pre-convention name."""

    per_year: bool = False
    """``True`` when SIPNET reads this value as a per-year rate and converts
    it to per-day internally (divide by 365). The Python interface always
    works in per-year units for these parameters, matching the param file."""

    initializes: tuple[str, ...] = ()
    """Output variable(s) from :mod:`pysipnet.variables` that this initial condition sets."""

    initializes_via: str = ""
    """How, when the relation is not the identity, e.g.
    ``"leaf_carbon = leaf_area_index × leaf_carbon_per_area"``."""

    @property
    def label(self) -> str:
        """*short_label* if set, else *long_label*."""
        return self.short_label or self.long_label

    def formatted_units(self, style: UnitStyle = "unicode") -> str:
        """Units rendered for display, constituent included, e.g. ``"g C m⁻²"``."""
        return format_units(self.units, constituent=self.constituent, style=style)

    def axis_label(self, style: UnitStyle = "unicode") -> str:
        """``"Maximum photosynthesis rate (nmol CO2 g⁻¹ s⁻¹)"``, for a plot axis."""
        return f"{self.long_label} ({self.formatted_units(style)})"

    def to_record(self) -> dict[str, Any]:
        """A plain, JSON-serializable dict of every field."""
        record = asdict(self)
        record["domain"] = self.domain.value
        record["aliases"] = list(self.aliases)
        record["initializes"] = list(self.initializes)
        record["formatted_units"] = self.formatted_units()
        return record


def param_field(
    *,
    sipnet_name: str,
    units: str,
    domain: ParameterDomain,
    description: str,
    long_label: str,
    constituent: str = "",
    short_label: str = "",
    aliases: tuple[str, ...] = (),
    per_year: bool = False,
    initializes: tuple[str, ...] = (),
    initializes_via: str = "",
    default: Any = _MISSING,
) -> Any:
    """Factory for a Pydantic ``Field`` with embedded :class:`ParameterSpec`.

    Pydantic validators (``gt``, ``ge``, ``le``, ``lt``) are derived
    automatically from *domain* so there is no risk of the two diverging.

    Parameters
    ----------
    sipnet_name:
        Name in SIPNET's ``.param`` file.
    units:
        UDUNITS-style unit string (see module-level docs for the convention).
    domain:
        Mathematical support — controls both Pydantic validators and the
        ``ParameterDomain`` stored in ``json_schema_extra``.
    description:
        Human-readable description included in the JSON schema.
    long_label, short_label:
        Plot labels without units.
    constituent:
        Substance qualifier not captured by the physical unit.
    aliases:
        Other names accepted for this parameter in lookups.
    per_year:
        Set to ``True`` for parameters SIPNET reads as annual rates.
    initializes, initializes_via:
        For initial conditions: the output state variable(s) set, and how.
    default:
        Field default value. Omit (or pass ``_MISSING``) to make the field
        required. Pass ``None`` to make it optional with a ``None`` default.
    """
    validate_units(units)

    # Pydantic allows NaN and inf through by default, and the gt/ge bounds do
    # not catch them: comparisons against NaN are always false and inf passes
    # any lower bound. Either reaches SIPNET, which parses it with strtod and
    # runs to completion — a NaN temperature parameter produces a whole run of
    # zero productivity, exit code 0, and no warning anywhere.
    pydantic_kwargs: dict[str, Any] = {"allow_inf_nan": False}
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
        sipnet_name=sipnet_name,
        units=units,
        domain=domain,
        description=description,
        long_label=long_label,
        constituent=constituent,
        short_label=short_label,
        aliases=aliases,
        per_year=per_year,
        initializes=initializes,
        initializes_via=initializes_via,
    )

    return Field(
        ... if default is _MISSING else default,
        description=description,
        json_schema_extra={
            "sipnet_name": sipnet_name,
            "units": units,
            "constituent": constituent,
            "domain": domain.value,
            "per_year": per_year,
            # ParameterSpec is a dataclass, not JSON — it is carried here so
            # get_parameter_specs can recover the full spec from a model class.
            # Pydantic does not inspect this key.
            "_spec": spec,  # type: ignore[dict-item]
        },
        **pydantic_kwargs,
    )


def get_parameter_specs(model_cls: type[BaseModel], prefix: str = "") -> dict[str, ParameterSpec]:
    """Return a flat ``{dotted.field.path: ParameterSpec}`` dict for *model_cls*.

    Recurses into nested :class:`pydantic.BaseModel` fields. Only fields
    produced by :func:`param_field` (i.e., those with ``json_schema_extra``
    containing a ``"_spec"`` key) are included.

    Example::

        specs = get_parameter_specs(SIPNETParameters)
        # {"photosynthesis.max_photosynthesis_rate": ParameterSpec(...), ...}
    """
    result: dict[str, ParameterSpec] = {}
    for name, field_info in model_cls.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        extra = field_info.json_schema_extra
        if isinstance(extra, dict) and "_spec" in extra:
            if not NAME_PATTERN.match(name):
                raise ValueError(f"Parameter field {path!r} violates the naming convention.")
            result[path] = cast("ParameterSpec", extra["_spec"])
        else:
            annotation = field_info.annotation
            if (
                annotation is not None
                and isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
            ):
                result.update(get_parameter_specs(annotation, prefix=path))
    return result
