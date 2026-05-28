"""SIPNETModel — high-level model interface with parameter and climate overrides.

:class:`SIPNETModel` is the recommended entry point for running SIPNET.  It
wraps a :class:`~pysipnet.runner.SIPNETRunner` and a baseline parameter set,
and exposes a single ``__call__`` interface::

    runner = SIPNETRunner(preset=ModelPreset.STANDARD)
    model  = SIPNETModel(runner, base_params=params, base_climate=climate)

    result           = model()                          # baseline run
    result_tuned     = model(a_max=120.0)               # single override
    result_site_b    = model(climate=other_climate)     # different drivers
    result_both      = model(a_max=120.0, climate=other_climate)

Any SIPNET v1 parameter name (see
:data:`~pysipnet.parameters.SIPNET_PARAMS_BY_GROUP`) can be passed as a keyword
argument to override the baseline value for that run.  The reserved names
``climate`` and ``events`` pass a
:class:`~pysipnet.climate.ClimateDrivers` or
:class:`~pysipnet.events.EventSequence` directly through to the runner.

Ensemble workflows
------------------
Because :class:`SIPNETModel` is a plain Python callable that accepts
``**kwargs`` and returns a :class:`~pysipnet.result.SIPNETResult`, it is
directly compatible with any ensemble or optimisation framework that expects
a ``(**inputs) -> output`` function.  For PyEns, no adapter layer is needed::

    from pyens import EnsembleRunner
    from pyens.backends import SequentialBackend

    ensemble_runner = EnsembleRunner(model, SequentialBackend())
    result = ensemble_runner.run(spec)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pysipnet.parameters.v1 import SIPNET_PARAMS_BY_GROUP

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.events import EventSequence
    from pysipnet.parameters.v1 import SIPNETParametersV1
    from pysipnet.result import SIPNETResult
    from pysipnet.runner import SIPNETRunner


# Flat param-name → group-name reverse lookup, derived from the public constant
# so the two are always in sync.
_PARAM_TO_GROUP: dict[str, str] = {
    param: group for group, params in SIPNET_PARAMS_BY_GROUP.items() for param in params
}

_RESERVED_FIELDS: frozenset[str] = frozenset({"climate", "events"})


# ── Internal helpers ───────────────────────────────────────────────────────────


def _apply_overrides(
    base: SIPNETParametersV1,
    overrides: dict[str, Any],
) -> SIPNETParametersV1:
    """Return a new parameter set with the given values overridden.

    Serialises *base* to a plain dict, applies the overrides at the
    appropriate group level, then reconstructs via ``model_validate`` so that
    every Pydantic validator — including cross-group constraints such as the
    allocation triangle — is re-run on the modified values.

    Parameters
    ----------
    base:
        Baseline parameter set to copy and modify.
    overrides:
        Flat ``{param_name: value}`` dict.  All keys must be present in
        :data:`_PARAM_TO_GROUP`.

    Returns
    -------
    SIPNETParametersV1
        New instance with the specified parameters updated; all others
        unchanged.
    """
    from pysipnet.parameters.v1 import SIPNETParametersV1

    current = base.model_dump()
    for param_name, value in overrides.items():
        group = _PARAM_TO_GROUP[param_name]
        current[group][param_name] = value
    return SIPNETParametersV1.model_validate(current)


# ── SIPNETModel ────────────────────────────────────────────────────────────────


class SIPNETModel:
    """High-level SIPNET model interface with an override-friendly callable API.

    :class:`SIPNETModel` wraps a :class:`~pysipnet.runner.SIPNETRunner` and a
    baseline :class:`~pysipnet.parameters.v1.SIPNETParametersV1`.  Each call
    accepts any combination of parameter overrides, a climate replacement, and
    an optional event sequence::

        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        model  = SIPNETModel(runner, base_params=params, base_climate=climate)

        result        = model()                         # baseline run
        result_tuned  = model(a_max=140.0)              # single parameter override
        result_site_b = model(climate=other_climate)    # swap climate drivers
        result_both   = model(a_max=140.0, climate=other_climate)

    Any SIPNET v1 parameter name is accepted as a keyword argument.
    Unrecognised names raise :class:`ValueError` immediately.  Invalid
    parameter values (e.g., a negative ``a_max``) raise
    :class:`pydantic.ValidationError` before the binary is invoked.

    Because :class:`SIPNETModel` is a plain callable, it is directly compatible
    with PyEns, Parsl, Dask, Ray, and any other framework that treats the model
    as a ``(**inputs) -> output`` function.  For PyEns:

    .. code-block:: python

        from pyens import EnsembleRunner
        from pyens.backends import LocalBackend

        ensemble_runner = EnsembleRunner(model, LocalBackend(n_workers=8))
        result = ensemble_runner.run(spec)

    Parameters
    ----------
    runner:
        The :class:`~pysipnet.runner.SIPNETRunner` to use for execution.
        Determines the binary preset and any execution options
        (``keep_workdir``, ``timeout``, etc.).
    base_params:
        Baseline parameter set.  Any parameter not overridden in a given call
        takes its value from here.
    base_climate:
        Default climate drivers.  Used when ``climate=`` is not passed to
        ``__call__``.  At least one of *base_climate* or a ``climate`` keyword
        argument must be present at call time.

    Examples
    --------
    Explore the sensitivity of annual NEE to ``a_max``:

    .. code-block:: python

        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        model  = SIPNETModel(runner, base_params=params, base_climate=climate)

        nee_by_a_max = {
            v: model(a_max=v).nee().sum()
            for v in [80.0, 100.0, 120.0, 140.0]
        }
    """

    def __init__(
        self,
        runner: SIPNETRunner,
        *,
        base_params: SIPNETParametersV1,
        base_climate: ClimateDrivers | None = None,
    ) -> None:
        self._runner = runner
        self._base_params = base_params
        self._base_climate = base_climate

    def __call__(
        self,
        *,
        climate: ClimateDrivers | None = None,
        events: EventSequence | None = None,
        **param_overrides: Any,
    ) -> SIPNETResult:
        """Execute one SIPNET run with the given overrides applied.

        All arguments are keyword-only.

        Parameters
        ----------
        climate:
            Climate drivers for this run.  Falls back to ``base_climate`` if
            not provided.  Raises :class:`ValueError` if neither is set.
        events:
            Optional management event sequence.
        **param_overrides:
            Parameter values to override for this run.  Each key must be a
            valid SIPNET v1 parameter name (see
            :data:`~pysipnet.parameters.SIPNET_PARAMS_BY_GROUP`).
            Unrecognised keys raise :class:`ValueError` immediately.

        Returns
        -------
        SIPNETResult
            Parsed output from the SIPNET binary.

        Raises
        ------
        ValueError
            If *param_overrides* contains unrecognised parameter names, or if
            no climate is available for this run.
        pydantic.ValidationError
            If applying the overrides produces an invalid parameter set (e.g.,
            violating the allocation triangle constraint).
        """
        unknown = {k for k in param_overrides if k not in _PARAM_TO_GROUP}
        if unknown:
            raise ValueError(
                f"SIPNETModel: unrecognised parameter name(s): {sorted(unknown)}. "
                "Use a name from pysipnet.parameters.SIPNET_PARAMS_BY_GROUP, "
                "or climate= / events= for non-parameter inputs."
            )

        effective_climate = climate if climate is not None else self._base_climate
        if effective_climate is None:
            raise ValueError(
                "SIPNETModel: no climate available for this run. "
                "Provide base_climate= to SIPNETModel() or climate= at call time."
            )

        effective_params = (
            _apply_overrides(self._base_params, param_overrides)
            if param_overrides
            else self._base_params
        )

        return self._runner.run(effective_params, effective_climate, events=events)

    @property
    def runner(self) -> SIPNETRunner:
        """The underlying :class:`~pysipnet.runner.SIPNETRunner`."""
        return self._runner

    @property
    def base_params(self) -> SIPNETParametersV1:
        """The baseline :class:`~pysipnet.parameters.v1.SIPNETParametersV1`."""
        return self._base_params

    @property
    def base_climate(self) -> ClimateDrivers | None:
        """The default climate drivers, or ``None`` if not set."""
        return self._base_climate

    def __repr__(self) -> str:
        """Return a concise string representation."""
        has_climate = self._base_climate is not None
        return (
            f"SIPNETModel(preset={self._runner.preset!r}, "
            f"base_climate={'set' if has_climate else 'None'})"
        )
