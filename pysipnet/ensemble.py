"""PyEns field-spec helpers for pySIPNET.

This module provides :func:`sipnet_site_fields` and
:func:`sipnet_member_fields`, which build PyEns
:class:`~pyens.FieldSpec` objects pre-validated against the SIPNET v1
parameter schema.

The primary model interface — :class:`~pysipnet.model.SIPNETModel` — lives
in :mod:`pysipnet.model` and has no PyEns dependency.  Import it directly
for single-run use or to pass to any framework that expects a callable::

    from pysipnet import SIPNETModel

This module is only needed when constructing PyEns
:class:`~pyens.EnsembleSpec` objects using the pySIPNET-aware helpers below.

Optional dependency
-------------------
This module requires ``pyens``, which is not yet on PyPI.  Install it from
source::

    pip install git+https://github.com/arob5/PyEns.git

or, if you have a local clone::

    pip install -e /path/to/pyens
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from pyens import Axis, FieldSpec, Grid  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "pysipnet.ensemble requires pyens, which is not yet on PyPI. "
        "Install it from source: pip install git+https://github.com/arob5/PyEns.git"
    ) from _exc

from pysipnet.model import _PARAM_TO_GROUP, _RESERVED_FIELDS

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers


def sipnet_site_fields(
    sites: Axis,
    *,
    climates: list[ClimateDrivers],
    **site_params: list[Any],
) -> dict[str, FieldSpec]:
    """Build field specs for site-level variation, all aligned on the same axis.

    All returned fields share *sites* as their axis, so they co-vary: run *i*
    uses ``climates[i]`` together with ``site_params[field][i]`` for every
    field.  Pass the returned dict into ``EnsembleSpec(inputs={...})``
    alongside other field specs.

    Parameters
    ----------
    sites:
        The :class:`~pyens.Axis` defining the set of sites.
    climates:
        Climate drivers for each site, one per site in axis order.
    **site_params:
        Parameter values for each site, one per site in axis order.  Each
        keyword argument key must be a valid SIPNET v1 parameter name.

    Returns
    -------
    dict[str, FieldSpec]
        Mapping ``"climate"`` and each parameter name to a
        :class:`~pyens.Grid` aligned on *sites*.

    Raises
    ------
    ValueError
        If any parameter name is unrecognised or is a reserved field name.

    Examples
    --------
    .. code-block:: python

        sites = Axis("site", labels=["harvard_forest", "niwot_ridge"])
        fields = sipnet_site_fields(
            sites,
            climates=[clim_hf, clim_nr],
            soil=[8000.0, 12000.0],
            plant_wood=[30000.0, 25000.0],
        )
        spec = EnsembleSpec(inputs={**fields, ...})
    """
    _validate_param_names(site_params.keys(), context="sipnet_site_fields")
    result: dict[str, FieldSpec] = {"climate": Grid(climates, along=sites)}
    for name, values in site_params.items():
        result[name] = Grid(values, along=sites)
    return result


def sipnet_member_fields(
    members: Axis,
    **param_samples: list[Any],
) -> dict[str, FieldSpec]:
    """Build field specs for ensemble member variation, all aligned on the same axis.

    All returned fields share *members* as their axis, so they co-vary: run
    *i* uses ``param_samples[field][i]`` for every field.  Crossing this dict
    with site-level fields (a different axis) produces a full Cartesian product.

    Parameters
    ----------
    members:
        The :class:`~pyens.Axis` defining the set of ensemble members.
    **param_samples:
        Parameter values for each member, one per member in axis order.  Each
        keyword argument key must be a valid SIPNET v1 parameter name.

    Returns
    -------
    dict[str, FieldSpec]
        Mapping each parameter name to a :class:`~pyens.Grid` aligned on
        *members*.

    Raises
    ------
    ValueError
        If any parameter name is unrecognised or is a reserved field name.

    Examples
    --------
    .. code-block:: python

        import numpy as np

        members = Axis("member", size=100)
        rng     = np.random.default_rng(42)
        fields  = sipnet_member_fields(
            members,
            a_max=rng.uniform(80, 140, 100).tolist(),
            base_veg_resp=rng.uniform(0.01, 0.05, 100).tolist(),
        )
    """
    _validate_param_names(param_samples.keys(), context="sipnet_member_fields")
    return {name: Grid(values, along=members) for name, values in param_samples.items()}


def _validate_param_names(names: Any, *, context: str) -> None:
    """Raise ValueError if any name is reserved or unrecognised."""
    reserved = _RESERVED_FIELDS & set(names)
    if reserved:
        raise ValueError(
            f"{context}: {sorted(reserved)} are reserved field names. "
            "Pass climates= as a keyword argument to sipnet_site_fields, "
            "not as a site_params entry."
        )
    unknown = {n for n in names if n not in _PARAM_TO_GROUP}
    if unknown:
        raise ValueError(
            f"{context}: unrecognised parameter name(s): {sorted(unknown)}. "
            "Use a name from pysipnet.parameters.SIPNET_PARAMS_BY_GROUP."
        )
