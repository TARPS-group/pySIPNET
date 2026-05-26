"""pySIPNET ensemble integration via PyEns.

This module provides :class:`SIPNETModel`, a wrapper that adapts a
:class:`~pysipnet.runner.SIPNETRunner` to the callable interface expected by
a PyEns :class:`~pyens.EnsembleRunner`, plus convenience functions for
constructing :class:`~pyens.FieldSpec` objects for common ensemble patterns.

Optional dependency
-------------------
This module requires the ``pyens`` package::

    pip install pysipnet[ensemble]

Importing this module without pyens installed raises :class:`ImportError` with
a clear installation hint.

Usage pattern
-------------
Construct a :class:`SIPNETModel` from a runner and a baseline parameter set.
Then pass it to a PyEns ``EnsembleRunner`` as the model callable::

    from pysipnet import SIPNETRunner, ModelPreset, SIPNETParametersV1, ClimateDrivers
    from pysipnet.ensemble import SIPNETModel, sipnet_site_fields, sipnet_member_fields
    from pyens import Axis, EnsembleSpec, Fixed, Grid, EnsembleRunner
    from pyens.backends import SequentialBackend

    runner = SIPNETRunner(preset=ModelPreset.STANDARD)
    model  = SIPNETModel(runner, base_params=params, base_climate=climate)

    sites   = Axis("site",   labels=["harvard_forest", "niwot_ridge"])
    members = Axis("member", size=50)

    spec = EnsembleSpec(inputs={
        **sipnet_site_fields(sites, climates=[clim_hf, clim_nr], soil=[8000.0, 12000.0]),
        **sipnet_member_fields(members, a_max=sampled_a_max, base_veg_resp=sampled_resp),
    })

    ensemble_runner = EnsembleRunner(model, SequentialBackend())
    result = ensemble_runner.run(spec)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from pyens import Axis, FieldSpec, Fixed, Grid  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "The ensemble module requires pyens. Install with: pip install pysipnet[ensemble]"
    ) from _exc

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.events import EventSequence
    from pysipnet.parameters.v1 import SIPNETParametersV1
    from pysipnet.result import SIPNETResult
    from pysipnet.runner import SIPNETRunner


# ── Parameter group map ────────────────────────────────────────────────────────


def _build_param_group_map() -> dict[str, str]:
    """Return a mapping from flat parameter name to group attribute name.

    Inspects the Pydantic model fields of :class:`~pysipnet.parameters.v1.SIPNETParametersV1`
    at import time.  Raises :class:`AssertionError` if any parameter name
    appears in two groups — this would make group routing ambiguous and
    indicates a structural bug in the parameter model.

    Returns
    -------
    dict[str, str]
        Keys are flat parameter field names (e.g. ``"a_max"``); values are
        the corresponding group attribute names on ``SIPNETParametersV1``
        (e.g. ``"photosynthesis"``).
    """
    from pydantic import BaseModel

    from pysipnet.parameters.v1 import SIPNETParametersV1

    mapping: dict[str, str] = {}
    for group_name, field_info in SIPNETParametersV1.model_fields.items():
        annotation = field_info.annotation
        if (
            annotation is None
            or not isinstance(annotation, type)
            or not issubclass(annotation, BaseModel)
        ):
            continue
        for param_name in annotation.model_fields:
            assert param_name not in mapping, (
                f"Parameter name '{param_name}' appears in both "
                f"'{mapping[param_name]}' and '{group_name}'. "
                "Parameter names must be unique across all groups."
            )
            mapping[param_name] = group_name
    return mapping


_PARAM_GROUP: dict[str, str] = _build_param_group_map()
"""Mapping from flat parameter name to the group attribute name on SIPNETParametersV1.

Built once at module import time.  Used internally by :class:`SIPNETModel`
and the helper functions to route keyword arguments to the right sub-model.
"""

_RESERVED_FIELDS: frozenset[str] = frozenset({"climate", "events"})
"""Field names reserved for non-parameter inputs in an ensemble spec.

These names cannot be used as parameter override keys because they refer to
:class:`~pysipnet.climate.ClimateDrivers` and
:class:`~pysipnet.events.EventSequence`, not scalar parameters.
"""


# ── Internal helpers ───────────────────────────────────────────────────────────


def _apply_overrides(
    base: SIPNETParametersV1,
    overrides: dict[str, Any],
) -> SIPNETParametersV1:
    """Return a new parameter set with the specified values overridden.

    Runs the full Pydantic validation chain so cross-group constraints (such
    as the allocation triangle constraint) are always checked.

    Parameters
    ----------
    base:
        Baseline parameter set to copy and modify.
    overrides:
        Flat dict mapping parameter name to new value.  All keys must be
        present in :data:`_PARAM_GROUP`.

    Returns
    -------
    SIPNETParametersV1
        A new instance with the specified parameters updated.  All other
        parameters are unchanged.
    """
    from pysipnet.parameters.v1 import SIPNETParametersV1

    # Serialise to plain Python dict, apply overrides at the right group level,
    # then reconstruct — this ensures every validator (including cross-group
    # constraints) is re-run on the modified values.
    current = base.model_dump()
    for param_name, value in overrides.items():
        group = _PARAM_GROUP[param_name]
        current[group][param_name] = value
    return SIPNETParametersV1.model_validate(current)


# ── SIPNETModel ────────────────────────────────────────────────────────────────


class SIPNETModel:
    """PyEns-compatible callable that wraps a :class:`~pysipnet.runner.SIPNETRunner`.

    :class:`SIPNETModel` bridges the PyEns ensemble interface and the pySIPNET
    single-run interface.  It accepts any subset of SIPNET v1 parameter names
    as keyword arguments, applies them as overrides on top of a baseline
    parameter set, and delegates to the underlying runner.

    The reserved keyword arguments ``climate`` and ``events`` pass
    :class:`~pysipnet.climate.ClimateDrivers` and
    :class:`~pysipnet.events.EventSequence` objects directly through to the
    runner.  All other keyword arguments are interpreted as parameter names and
    routed to the appropriate sub-model.

    Parameters
    ----------
    runner:
        The :class:`~pysipnet.runner.SIPNETRunner` to use for execution.
    base_params:
        Baseline parameter set.  Any parameter not overridden in a given call
        takes its value from here.
    base_climate:
        Optional default climate drivers.  Used when the ensemble spec does
        not include a ``"climate"`` field.  At least one of *base_climate* or
        a ``climate`` field in the spec must be provided; calling
        :meth:`__call__` without either raises :class:`ValueError`.

    Examples
    --------
    Single-parameter sweep — vary ``a_max`` across 20 values:

    .. code-block:: python

        from pysipnet.ensemble import SIPNETModel
        from pyens import Axis, EnsembleSpec, Fixed, Grid, EnsembleRunner
        from pyens.backends import SequentialBackend

        model = SIPNETModel(runner, base_params=params, base_climate=climate)
        ax    = Axis("a_max_sweep", size=20)
        spec  = EnsembleSpec(inputs={
            "a_max": Grid([80.0 + i * 4 for i in range(20)], along=ax),
        })
        result = EnsembleRunner(model, SequentialBackend()).run(spec)
    """

    def __init__(
        self,
        runner: SIPNETRunner,
        base_params: SIPNETParametersV1,
        base_climate: ClimateDrivers | None = None,
    ) -> None:
        self._runner = runner
        self._base_params = base_params
        self._base_climate = base_climate

    def __call__(
        self,
        climate: ClimateDrivers | None = None,
        events: EventSequence | None = None,
        **param_overrides: Any,
    ) -> SIPNETResult:
        """Execute one SIPNET run with the given overrides applied.

        Called by the PyEns backend for each run in the ensemble.  Keyword
        arguments that match SIPNET v1 parameter names are applied on top of
        ``base_params``; the ``climate`` and ``events`` arguments are passed
        directly to the runner.

        Parameters
        ----------
        climate:
            Climate drivers for this run.  If ``None``, the ``base_climate``
            provided at construction is used.
        events:
            Optional management event sequence for this run.
        **param_overrides:
            Parameter values to override.  Each key must be a valid SIPNET v1
            parameter name (i.e., a key in :data:`_PARAM_GROUP`).  Keys that
            are not recognised raise :class:`ValueError` immediately rather
            than silently being ignored.

        Returns
        -------
        SIPNETResult
            The parsed output from the SIPNET binary.

        Raises
        ------
        ValueError
            If any key in *param_overrides* is not a recognised parameter name,
            or if no climate is available (neither passed here nor set as
            ``base_climate``).
        pydantic.ValidationError
            If applying the overrides produces an invalid parameter set (e.g.,
            violating the allocation constraint).
        """
        unknown = {k for k in param_overrides if k not in _PARAM_GROUP}
        if unknown:
            raise ValueError(
                f"SIPNETModel: unrecognised parameter name(s): {sorted(unknown)}. "
                "Expected keys from SIPNETParametersV1 (e.g. 'a_max', 'soil'). "
                "Use 'climate' for ClimateDrivers and 'events' for EventSequence."
            )

        effective_climate = climate if climate is not None else self._base_climate
        if effective_climate is None:
            raise ValueError(
                "SIPNETModel: no climate available for this run. "
                "Provide climate= to SIPNETModel() or include a 'climate' field in the spec."
            )

        effective_params = (
            _apply_overrides(self._base_params, param_overrides)
            if param_overrides
            else self._base_params
        )

        return self._runner.run(effective_params, effective_climate, events=events)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        preset = self._runner.preset
        has_climate = self._base_climate is not None
        return f"SIPNETModel(preset={preset!r}, base_climate={'set' if has_climate else 'None'})"


# ── Convenience helpers ────────────────────────────────────────────────────────


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
        If any parameter name is not recognised or is a reserved field name.

    Examples
    --------
    .. code-block:: python

        sites = Axis("site", labels=["harvard_forest", "niwot_ridge"])
        fields = sipnet_site_fields(
            sites,
            climates=[clim_hf, clim_nr],
            soil=[8000.0, 12000.0],           # initial soil C per site
            plant_wood=[30000.0, 25000.0],    # initial wood C per site
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
    with site-level fields (different axis) produces a full Cartesian product.

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
        If any parameter name is not recognised or is a reserved field name.

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
        spec = EnsembleSpec(inputs={**fields, ...})
    """
    _validate_param_names(param_samples.keys(), context="sipnet_member_fields")
    return {name: Grid(values, along=members) for name, values in param_samples.items()}


# ── Validation ─────────────────────────────────────────────────────────────────


def _validate_param_names(names: Any, *, context: str) -> None:
    """Raise ValueError if any name is reserved or unrecognised."""
    reserved = _RESERVED_FIELDS & set(names)
    if reserved:
        raise ValueError(
            f"{context}: {sorted(reserved)} are reserved field names and cannot "
            "be used as parameter names. Use sipnet_site_fields(..., climates=...) "
            "for climate, not a keyword argument."
        )
    unknown = {n for n in names if n not in _PARAM_GROUP}
    if unknown:
        raise ValueError(
            f"{context}: unrecognised parameter name(s): {sorted(unknown)}. "
            "Expected keys from SIPNETParametersV1 (e.g. 'a_max', 'soil')."
        )
