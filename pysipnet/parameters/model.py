"""SIPNET parameter models.

All inputs to a SIPNET run are captured here as Pydantic models.  The
models mirror the structure of SIPNET's ``.param`` file but use snake_case
field names and explicit units.  The IO layer (:mod:`pysipnet.io.param_io`)
handles translation to SIPNET's camelCase names.

Model structure
---------------
``SIPNETParameters`` is a flat composition of domain-grouped sub-models::

    params = SIPNETParameters(
        initial_conditions=InitialConditions(total_wood_carbon=30000, ...),
        photosynthesis=PhotosynthesisParams(max_photosynthesis_rate=112.0, ...),
        ...
    )

Use :func:`pysipnet.parameters.base.get_parameter_specs` to retrieve the full
``{path: ParameterSpec}`` dict for calibration tooling::

    from pysipnet.parameters.base import get_parameter_specs
    specs = get_parameter_specs(SIPNETParameters)

Naming convention
-----------------
Field names are lower-case words joined by underscores, with no acronyms or
truncations: ``max_photosynthesis_rate``, not ``a_max``. SIPNET's own names are
in each field's ``ParameterSpec.sipnet_name``; the names pySIPNET used before
this convention are in ``aliases``, and :func:`resolve_parameter_name` maps
either to the current name.

Per-year rate parameters
------------------------
The following parameters are specified in the ``.param`` file as **per-year**
rates and are converted to per-day internally by SIPNET (÷ 365). The Python
interface works in per-year units throughout, matching SIPNET's convention.
Each of these fields has ``per_year=True`` in its
:class:`~pysipnet.parameters.base.ParameterSpec`.

- ``respiration.base_wood_respiration_rate``
- ``respiration.base_fine_root_respiration_rate``
- ``respiration.base_coarse_root_respiration_rate``
- ``respiration.base_soil_respiration_rate``
- ``respiration.litter_breakdown_rate``
- ``allocation.fine_root_turnover_rate``
- ``allocation.coarse_root_turnover_rate``
- ``allocation.wood_turnover_rate``
- ``phenology.leaf_turnover_rate``

Allocation constraint
---------------------
SIPNET derives coarse-root allocation as::

    coarse_root_allocation = 1 − leaf_allocation − fine_root_allocation − wood_allocation

A model validator on :class:`AllocationParams` enforces that the three
explicit fractions sum to strictly less than 1.

Flag-dependent parameters
-------------------------
Some parameters are only meaningful when the corresponding model flag is on.
These fields are ``Optional[float]`` with a default of ``None``.
:class:`SIPNETParameters` validates that required-by-flag parameters are
provided given the active :class:`ModelFlags`.

+-------------------------------------------------+----------------------------+
| Parameter                                       | Required when flag is on   |
+=================================================+============================+
| ``phenology.leaf_on_growing_degree_days``       | ``ModelFlags.gdd``         |
+-------------------------------------------------+----------------------------+
| ``phenology.leaf_on_soil_temperature``          | ``ModelFlags.soil_phenol`` |
+-------------------------------------------------+----------------------------+
| ``initial_conditions.snow_water_equivalent``    | ``ModelFlags.snow``        |
| (optional, default 0)                           |                            |
+-------------------------------------------------+----------------------------+
| ``water.snow_melt_rate``                        | ``ModelFlags.snow``        |
+-------------------------------------------------+----------------------------+
| ``water.leaf_water_pool_depth``                 | ``ModelFlags.leaf_water``  |
+-------------------------------------------------+----------------------------+
| ``respiration.litter_breakdown_rate``           | ``ModelFlags.litter_pool`` |
+-------------------------------------------------+----------------------------+
| ``respiration.litter_respired_fraction``        | ``ModelFlags.litter_pool`` |
+-------------------------------------------------+----------------------------+
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from pysipnet.parameters.base import (
    ParameterDomain,
    ParameterGroup,
    ParameterSpec,
    get_parameter_specs,
    param_field,
)

_D = ParameterDomain  # local alias for brevity


# Processes SIPNET supports at the pinned version that pySIPNET cannot yet drive.
#
# Each of these flags reaches SIPNET correctly, but the parameters SIPNET then
# demands are not in SIPNETParameters, so the run would fail inside SIPNET with
# "Did not find required parameter". Rather than let that happen, ModelFlags
# refuses the flag up front and says why.
#
# To add support for one: model its parameters, mark them required under the
# flag in SIPNETParameters.validate_for_flags, then delete its entry here.
# Tracked in issue #26.
UNSUPPORTED_FLAGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "nitrogen_cycle": (
        "nitrogen pools and fluxes",
        (
            "mineralNInit",
            "soilOrgNInit",
            "litterOrgNInit",
            "plantStorageNInit",
            "nVolatilizationFrac",
            "nLeachingFrac",
            "leafCN",
            "woodCN",
            "fineRootCN",
            "kCN",
            "nFixationFracMax",
            "halfNFixationMax",
            "leafNResorptionFrac",
        ),
    ),
    "anaerobic": (
        "methane production and waterlogged-soil decomposition",
        (
            "fAnoxia",
            "anaerobicDecompRate",
            "anaerobicTransExp",
            "soilMethaneRate",
            "litterMethaneRate",
        ),
    ),
    "flooding": (
        "soil moisture above water holding capacity",
        ("waterDrainFrac",),
    ),
    "carbon_saturation": (
        "soil and litter carbon pools that saturate rather than grow without limit",
        ("soilCSaturation",),
    ),
}


# ── Model feature flags ───────────────────────────────────────────────────────


class ModelFlags(BaseModel):
    """Which optional SIPNET processes are switched on for a run.

    Immutable. The checks below run when an instance is built, so a mutable
    model would let ``flags.nitrogen_cycle = True`` walk straight past them and
    reach SIPNET, which is the failure this class exists to prevent. Build a
    new instance to change a flag.

    SIPNET calls these its "model feature flags". They turn whole processes on
    or off — snow, a separate litter pool, the nitrogen cycle — and they do two
    things in pySIPNET:

    1. They are written into the ``sipnet.in`` config file, which is how SIPNET
       learns about them. See :meth:`to_config_keys`.
    2. They decide which parameters SIPNET will insist on finding in the
       ``.param`` file. See :meth:`SIPNETParameters.validate_for_flags`.

    Because of (2), changing a flag can change which parameters you must
    supply, so the flags belong in the saved record of a run rather than being
    passed in ad hoc.

    Named starting points
    ---------------------
    :meth:`standard` and :meth:`forest` return the two configurations pySIPNET
    has always shipped. Any other combination is equally valid — build one
    directly with keyword arguments.

    Not every flag can be used
    --------------------------
    ``nitrogen_cycle``, ``anaerobic``, ``flooding`` and ``carbon_saturation``
    are rejected. SIPNET
    supports all four, but each needs parameters that
    :class:`SIPNETParameters` does not define yet, so a run would fail inside
    SIPNET rather than here. :data:`UNSUPPORTED_FLAGS` lists what each one
    needs.

    Restrictions
    ------------
    SIPNET refuses to start on four flag combinations, and this model rejects
    them first so the problem is reported in Python rather than as an exit
    code from the binary:

    - ``gdd`` and ``soil_phenol`` cannot both be on. They are two different
      ways of deciding when leaves appear, so exactly one must be chosen.
    - ``anaerobic`` requires ``water_hresp``.
    - ``nitrogen_cycle`` requires both ``litter_pool`` and ``anaerobic``.
    """

    model_config = ConfigDict(frozen=True)

    # ── Processes on by default ──
    snow: bool = True
    """Require the snow melt rate parameter.

    SIPNET's documentation says this flag switches the snowpack on or off, but
    at the pinned version the snowpack is always simulated: precipitation
    below 0 °C falls as snow regardless of the flag. The only effect of turning
    it off is that ``water.snow_melt_rate`` is no longer required. If the
    parameter is supplied anyway (pySIPNET writes every parameter that is not
    ``None``) it is used exactly as with the flag on; if it is omitted SIPNET
    leaves it at zero and snow that falls never melts. Leave this on unless you
    want exactly that. ``tests/test_integration.py`` pins this behavior so an
    upstream fix is noticed.
    """

    gdd: bool = True
    """Decide when leaves appear from accumulated growing degree-days.

    Degree-days accumulate from January 1 and are compared against the
    ``phenology.leaf_on_growing_degree_days`` parameter. Mutually exclusive with
    ``soil_phenol``.
    """

    water_hresp: bool = True
    """Let soil moisture affect how fast soil carbon decomposes."""

    # ── Processes off by default ──
    growth_resp: bool = False
    """Track growth respiration separately from maintenance respiration."""

    leaf_water: bool = False
    """Give leaves their own water pool and evaporate from it.

    Mainly matters for timesteps shorter than a day, where treating canopy
    interception as instantaneous is too coarse.
    """

    litter_pool: bool = False
    """Add a litter carbon pool that feeds the soil carbon pool.

    Without this, plant litter goes straight into soil carbon.
    """

    soil_phenol: bool = False
    """Decide when leaves appear from a soil temperature threshold.

    An alternative to ``gdd``; the two cannot both be on.
    """

    nitrogen_cycle: bool = False
    """Track nitrogen pools and fluxes alongside carbon.

    Requires ``litter_pool`` and ``anaerobic``.

    **Not usable yet**: setting this raises, because the nitrogen parameters
    SIPNET would require are not modeled. See :data:`UNSUPPORTED_FLAGS`.
    """

    anaerobic: bool = False
    """Model methane production and waterlogged-soil effects on decomposition.

    Requires ``water_hresp``.

    **Not usable yet**: setting this raises, because the methane and anaerobic
    parameters SIPNET would require are not modeled. See
    :data:`UNSUPPORTED_FLAGS`.
    """

    carbon_saturation: bool = False
    """Let soil and litter carbon saturate instead of accumulating without limit.

    Requires ``litter_pool``.

    **Not usable yet**: setting this raises, because ``soilCSaturation`` is not
    modeled. See :data:`UNSUPPORTED_FLAGS`.
    """

    flooding: bool = False
    """Allow soil moisture to rise above the soil's water holding capacity.

    **Not usable yet**: setting this raises, because ``waterDrainFrac`` is not
    modeled. See :data:`UNSUPPORTED_FLAGS`.
    """

    # ── Provenance ──
    name: str | None = None
    """Optional label for this configuration, for run records and plots.

    Purely descriptive: it is never written to ``sipnet.in`` and never affects
    the model. :meth:`standard` and :meth:`forest` set it for you.

    Note that it does take part in equality, so a labeled configuration is
    not equal to an identical unlabeled one. Compare
    :meth:`to_config_keys` output when you want to compare only the flags.
    """

    @model_validator(mode="after")
    def _check_flags_are_supported(self) -> ModelFlags:
        """Reject flags whose parameters pySIPNET does not model yet.

        Runs before :meth:`_check_flag_restrictions` so that turning on an
        unsupported flag reports that, rather than sending the caller to
        satisfy a dependency that would still not work.

        Separate from :meth:`_check_flag_restrictions` because the reason is
        different. That method mirrors combinations SIPNET itself rejects;
        this one is about a gap on the Python side. SIPNET would happily run
        these processes if we could supply their parameters.

        Without this check the flag would reach SIPNET, which would then stop
        with "Did not find required parameter" — a failure a long way from its
        cause, and one the parameter model is supposed to prevent.
        """
        unsupported = [name for name in UNSUPPORTED_FLAGS if getattr(self, name)]
        if not unsupported:
            return self

        lines = [
            "These model flags are not supported by pySIPNET yet, "
            "even though SIPNET itself supports them:",
            "",
        ]
        for name in unsupported:
            description, params = UNSUPPORTED_FLAGS[name]
            lines.append(f"  {name}={getattr(self, name)!r} would enable {description}.")
            lines.append(
                f"    SIPNET would then require {len(params)} parameter(s) that "
                f"SIPNETParameters does not define: {', '.join(params)}."
            )
        lines += [
            "",
            "Leave these flags off. Turning one on would produce a run that fails "
            "inside SIPNET rather than here.",
            "To add support, model the listed parameters and remove the flag from "
            "UNSUPPORTED_FLAGS in this module. Tracked in issue #26.",
        ]
        raise ValueError("\n".join(lines))

    @model_validator(mode="after")
    def _check_flag_restrictions(self) -> ModelFlags:
        """Reject the flag combinations SIPNET itself refuses to run.

        Mirrors ``validateContext()`` in SIPNET's ``src/common/context.c``.
        All problems are collected so a caller fixing several at once sees
        them together.
        """
        problems: list[str] = []

        if self.gdd and self.soil_phenol:
            problems.append(
                "gdd and soil_phenol are two different ways of triggering leaf-out "
                "and cannot both be on; set exactly one to True"
            )
        if self.anaerobic and not self.water_hresp:
            problems.append("anaerobic requires water_hresp to be True")
        if self.nitrogen_cycle and not (self.litter_pool and self.anaerobic):
            problems.append("nitrogen_cycle requires both litter_pool and anaerobic to be True")
        if self.carbon_saturation and not self.litter_pool:
            problems.append("carbon_saturation requires litter_pool to be True")

        if problems:
            raise ValueError(
                "Invalid combination of model flags:\n" + "\n".join(f"  - {p}" for p in problems)
            )
        return self

    def to_config_keys(self) -> dict[str, int]:
        """Return these flags as the ``sipnet.in`` keys SIPNET expects.

        Keys are SIPNET's own uppercase names and values are 1 or 0. Every
        flag is written, including the ones left at their default, so the
        config file is a complete and unambiguous record of the run rather
        than something whose meaning depends on the binary's built-in
        defaults.

        The ``name`` label is descriptive only and is not included.

        Example::

            >>> ModelFlags.forest().to_config_keys()["LITTER_POOL"]
            1
        """
        return {
            "SNOW": int(self.snow),
            "GDD": int(self.gdd),
            "WATER_HRESP": int(self.water_hresp),
            "GROWTH_RESP": int(self.growth_resp),
            "LEAF_WATER": int(self.leaf_water),
            "LITTER_POOL": int(self.litter_pool),
            "SOIL_PHENOL": int(self.soil_phenol),
            "CARBON_SATURATION": int(self.carbon_saturation),
            "NITROGEN_CYCLE": int(self.nitrogen_cycle),
            "ANAEROBIC": int(self.anaerobic),
            "FLOODING": int(self.flooding),
        }

    @classmethod
    def standard(cls) -> ModelFlags:
        """Snow, degree-day leaf-out, and moisture-sensitive soil respiration.

        The sensible default for most sites, and what you get from
        ``ModelFlags()``.
        """
        return cls(name="standard")

    @classmethod
    def forest(cls) -> ModelFlags:
        """:meth:`standard` plus a separate litter carbon pool.

        Suited to sites where litter accumulates and decomposes on a
        noticeably different timescale from soil carbon, such as boreal or
        deciduous forest.
        """
        return cls(litter_pool=True, name="forest")


# ── Sub-models ─────────────────────────────────────────────────────────────────


class InitialConditions(ParameterGroup):
    """Carbon pool and hydrological state at the start of the simulation.

    These are written to the ``.param`` file alongside model parameters
    (SIPNET makes no file-format distinction between the two). Each field's
    ``initializes`` records the output state variable it sets, and
    ``initializes_via`` how, when the relation is not the identity.

    Every field here is read exactly once, in SIPNET's ``setupModel()``
    before the first timestep. The root fractions, for example, only split
    ``total_wood_carbon`` at the start; root growth afterwards is governed by
    the allocation and turnover parameters. None of these values influences a
    later timestep.

    Restart caveat: when SIPNET resumes from a checkpoint (``RESTART_IN`` in
    ``sipnet.in``), ``restartLoadCheckpoint()`` runs after ``setupModel()``
    and overwrites every pool, so these fields have no effect at all on a
    restarted run. pySIPNET does not use restarts yet; this is recorded for
    when it does.
    """

    total_wood_carbon: float = param_field(
        sipnet_name="plantWoodInit",
        units="g m-2",
        constituent="C",
        domain=_D.NON_NEGATIVE,
        description="Total woody carbon at the start of the run: above-ground wood plus "
        "coarse and fine roots. SIPNET splits it three ways using fine_root_fraction and "
        "coarse_root_fraction; the remainder is wood_carbon.",
        long_label="Initial total wood carbon",
        aliases=("plant_wood",),
        initializes=("wood_carbon", "coarse_root_carbon", "fine_root_carbon"),
        initializes_via="wood_carbon = (1 − fine_root_fraction − coarse_root_fraction) × "
        "total_wood_carbon; coarse_root_carbon = coarse_root_fraction × total_wood_carbon; "
        "fine_root_carbon = fine_root_fraction × total_wood_carbon",
    )
    leaf_area_index: float = param_field(
        sipnet_name="laiInit",
        units="m2 m-2",
        domain=_D.NON_NEGATIVE,
        description="Leaf area index at the start of the run. SIPNET uses it only to set the "
        "initial leaf carbon pool; leaf area is not tracked afterwards.",
        long_label="Initial leaf area index",
        short_label="Initial LAI",
        aliases=("lai",),
        initializes=("leaf_carbon",),
        initializes_via="leaf_carbon = leaf_area_index × leaf_carbon_per_area",
    )
    litter_carbon: float = param_field(
        sipnet_name="litterInit",
        units="g m-2",
        constituent="C",
        domain=_D.NON_NEGATIVE,
        default=0.0,
        description="Litter carbon pool at the start of the run. Only affects dynamics when "
        "ModelFlags.litter_pool is on.",
        long_label="Initial litter carbon",
        aliases=("litter",),
        initializes=("litter_carbon",),
    )
    soil_carbon: float = param_field(
        sipnet_name="soilInit",
        units="g m-2",
        constituent="C",
        domain=_D.NON_NEGATIVE,
        description="Soil organic carbon pool at the start of the run.",
        long_label="Initial soil carbon",
        aliases=("soil",),
        initializes=("soil_carbon",),
    )
    soil_wetness_fraction: float = param_field(
        sipnet_name="soilWFracInit",
        units="1",
        domain=_D.NON_NEGATIVE,
        description="Soil water at the start of the run as a fraction of water holding "
        "capacity. May exceed 1 in flooding scenarios.",
        long_label="Initial soil wetness fraction",
        aliases=("soil_water_frac",),
        initializes=("soil_water",),
        initializes_via="soil_water = soil_wetness_fraction × soil_water_holding_capacity",
    )
    snow_water_equivalent: float = param_field(
        sipnet_name="snowInit",
        units="cm",
        constituent="H2O",
        domain=_D.NON_NEGATIVE,
        default=0.0,
        description="Snowpack at the start of the run as a depth of liquid water. The "
        "snowpack is simulated whether or not ModelFlags.snow is on.",
        long_label="Initial snow water equivalent",
        short_label="Initial SWE",
        aliases=("snow",),
        initializes=("snow_water_equivalent",),
    )
    fine_root_fraction: float = param_field(
        sipnet_name="fineRootFrac",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of total_wood_carbon that is fine roots at the start of the run.",
        long_label="Initial fine root fraction",
        aliases=("fine_root_frac",),
        initializes=("fine_root_carbon",),
        initializes_via="fine_root_carbon = fine_root_fraction × total_wood_carbon",
    )
    coarse_root_fraction: float = param_field(
        sipnet_name="coarseRootFrac",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of total_wood_carbon that is coarse roots at the start of the run.",
        long_label="Initial coarse root fraction",
        aliases=("coarse_root_frac",),
        initializes=("coarse_root_carbon",),
        initializes_via="coarse_root_carbon = coarse_root_fraction × total_wood_carbon",
    )


class PhotosynthesisParams(ParameterGroup):
    """Parameters governing gross primary production.

    Vapor pressure deficit (VPD) effect: ``1 − slope × vpd^exponent``
    multiplies photosynthesis, so a larger slope means stronger suppression.

    The maximum photosynthesis temperature is derived internally as
    ``2 × optimum − minimum`` and is therefore not a free parameter.
    """

    max_photosynthesis_rate: float = param_field(
        sipnet_name="aMax",
        units="nmol g-1 s-1",
        constituent="CO2",
        domain=_D.POSITIVE,
        description="Maximum net photosynthesis rate per gram of leaf at saturating light "
        "and no environmental stress.",
        long_label="Maximum photosynthesis rate",
        aliases=("a_max",),
    )
    daily_mean_photosynthesis_fraction: float = param_field(
        sipnet_name="aMaxFrac",
        units="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Daily mean photosynthesis as a fraction of the instantaneous maximum, "
        "accounting for within-day variation in light and temperature.",
        long_label="Daily mean photosynthesis fraction",
        aliases=("a_max_frac",),
    )
    foliar_respiration_fraction: float = param_field(
        sipnet_name="baseFolRespFrac",
        units="1",
        domain=_D.POSITIVE,
        description="Basal foliar maintenance respiration as a fraction of the maximum "
        "photosynthesis rate.",
        long_label="Foliar respiration fraction",
        aliases=("base_fol_resp_frac",),
    )
    min_photosynthesis_temperature: float = param_field(
        sipnet_name="psnTMin",
        units="degC",
        domain=_D.REAL,
        description="Air temperature at or below which net photosynthesis is zero.",
        long_label="Minimum photosynthesis temperature",
        aliases=("psn_t_min",),
    )
    optimum_photosynthesis_temperature: float = param_field(
        sipnet_name="psnTOpt",
        units="degC",
        domain=_D.REAL,
        description="Air temperature at which photosynthesis is maximal. The maximum "
        "temperature is derived as 2 × optimum − minimum.",
        long_label="Optimum photosynthesis temperature",
        aliases=("psn_t_opt",),
    )
    vapor_pressure_deficit_slope: float = param_field(
        sipnet_name="dVpdSlope",
        units="kPa-1",
        domain=_D.POSITIVE,
        description="Slope of the vapor pressure deficit reduction of photosynthesis: "
        "the multiplier is 1 − slope × vpd^exponent.",
        long_label="Vapor pressure deficit slope",
        aliases=("d_vpd_slope", "vapour_pressure_deficit_slope"),
    )
    vapor_pressure_deficit_exponent: float = param_field(
        sipnet_name="dVpdExp",
        units="1",
        domain=_D.POSITIVE,
        description="Exponent of the vapor pressure deficit reduction of photosynthesis.",
        long_label="Vapor pressure deficit exponent",
        aliases=("d_vpd_exp", "vapour_pressure_deficit_exponent"),
    )
    half_saturation_light: float = param_field(
        sipnet_name="halfSatPar",
        units="mol m-2 d-1",
        constituent="photons",
        domain=_D.POSITIVE,
        description="Photosynthetically active radiation at which photosynthesis is half its "
        "maximum, as moles of photons per square meter of ground per day "
        "(1 Einstein = 1 mol photons).",
        long_label="Half-saturation light",
        aliases=("half_sat_par",),
    )
    light_extinction_coefficient: float = param_field(
        sipnet_name="attenuation",
        units="1",
        domain=_D.POSITIVE,
        description="Canopy light extinction coefficient in Beer's law.",
        long_label="Light extinction coefficient",
        aliases=("attenuation",),
    )


class PhenologyParams(ParameterGroup):
    """Parameters controlling leaf phenology (growing season timing and leaf dynamics).

    Exactly one of ``leaf_on_day``, ``leaf_on_growing_degree_days`` or
    ``leaf_on_soil_temperature`` is used, depending on the ``gdd`` and
    ``soil_phenol`` model flags. The other two may still be set here; SIPNET
    ignores them.
    """

    leaf_on_day: float | None = param_field(
        sipnet_name="leafOnDay",
        units="d",
        domain=_D.POSITIVE,
        description="Day of year on which leaves appear. Used when both ModelFlags.gdd and "
        "ModelFlags.soil_phenol are off.",
        long_label="Leaf-on day",
        default=None,
    )
    leaf_off_day: float = param_field(
        sipnet_name="leafOffDay",
        units="d",
        domain=_D.POSITIVE,
        description="Day of year on which leaves fall.",
        long_label="Leaf-off day",
    )
    leaf_on_growing_degree_days: float | None = param_field(
        sipnet_name="gddLeafOn",
        units="K d",
        domain=_D.NON_NEGATIVE,
        description="Accumulated growing degree-days (air temperature × timestep length, "
        "summed from 1 January) at which leaves appear. Used when ModelFlags.gdd is on.",
        long_label="Leaf-on growing degree-days",
        aliases=("gdd_leaf_on",),
        default=None,
    )
    leaf_on_soil_temperature: float | None = param_field(
        sipnet_name="soilTempLeafOn",
        units="degC",
        domain=_D.REAL,
        description="Soil temperature at which leaves appear. Used when ModelFlags.soil_phenol "
        "is on.",
        long_label="Leaf-on soil temperature",
        aliases=("soil_temp_leaf_on",),
        default=None,
    )
    leaf_on_growth: float = param_field(
        sipnet_name="leafGrowth",
        units="g m-2",
        constituent="C",
        domain=_D.NON_NEGATIVE,
        description="Leaf carbon grown at leaf-on, drawn from wood and coarse roots subject to "
        "leaf_on_reallocation_fraction.",
        long_label="Leaf-on growth",
        aliases=("leaf_growth",),
    )
    leaf_off_fall_fraction: float = param_field(
        sipnet_name="fracLeafFall",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of the standing leaf carbon that falls at leaf-off.",
        long_label="Leaf-off fall fraction",
        aliases=("frac_leaf_fall",),
    )
    leaf_allocation: float = param_field(
        sipnet_name="leafAllocation",
        units="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Fraction of net primary production allocated to leaves. Enters the "
        "allocation constraint; see AllocationParams.",
        long_label="Leaf allocation",
    )
    leaf_turnover_rate: float = param_field(
        sipnet_name="leafTurnoverRate",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Fraction of leaf carbon lost to litter per year; SIPNET divides by 365 "
        "for daily use.",
        long_label="Leaf turnover rate",
    )
    leaf_on_reallocation_fraction: float = param_field(
        sipnet_name="leafOnReallocFrac",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Largest fraction of wood plus coarse-root carbon that leaf-on may draw "
        "on. SIPNET compares the leaf-on demand against (wood_carbon + coarse_root_carbon) × "
        "this fraction and scales the transfer down if it would exceed that.",
        long_label="Leaf-on reallocation fraction",
        aliases=("leaf_on_realloc_frac",),
    )


class RespirationParams(ParameterGroup):
    """Autotrophic and heterotrophic respiration parameters.

    Per-year rate parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``base_wood_respiration_rate``, ``base_fine_root_respiration_rate``,
    ``base_coarse_root_respiration_rate``, ``base_soil_respiration_rate`` and
    ``litter_breakdown_rate`` are specified as per-year rates, matching the
    SIPNET param file convention. SIPNET divides by 365 internally. Their
    :class:`~pysipnet.parameters.base.ParameterSpec` has ``per_year=True``.

    Litter parameters
    ~~~~~~~~~~~~~~~~~
    ``litter_breakdown_rate`` and ``litter_respired_fraction`` are only
    meaningful when ``ModelFlags.litter_pool`` is on. They may be ``None``
    otherwise; the validator on :class:`SIPNETParameters` enforces this.
    """

    base_wood_respiration_rate: float = param_field(
        sipnet_name="baseVegResp",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Wood maintenance respiration at 0 °C as a fraction of wood carbon per "
        "year. SIPNET's name for it is baseVegResp, but it is applied to wood only.",
        long_label="Base wood respiration rate",
        aliases=("base_veg_resp",),
    )
    wood_respiration_q10: float = param_field(
        sipnet_name="vegRespQ10",
        units="1",
        domain=_D.POSITIVE,
        description="Q10 temperature sensitivity of wood maintenance respiration.",
        long_label="Wood respiration Q10",
        aliases=("veg_resp_q10",),
    )
    growth_respiration_fraction: float = param_field(
        sipnet_name="growthRespFrac",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Growth respiration as a fraction of running-mean net primary production. "
        "Only used when ModelFlags.growth_resp is on.",
        long_label="Growth respiration fraction",
        aliases=("growth_resp_frac",),
        default=0.0,
    )
    frozen_soil_foliar_respiration_factor: float = param_field(
        sipnet_name="frozenSoilFolREff",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Multiplier on foliar respiration when the soil is frozen: 0 shuts it "
        "down, 1 leaves it unchanged.",
        long_label="Frozen-soil foliar respiration factor",
        aliases=("frozen_soil_fol_r_eff",),
    )
    frozen_soil_threshold: float = param_field(
        sipnet_name="frozenSoilThreshold",
        units="degC",
        domain=_D.REAL,
        description="Soil temperature below which the soil counts as frozen.",
        long_label="Frozen-soil threshold",
    )
    base_fine_root_respiration_rate: float = param_field(
        sipnet_name="baseFineRootResp",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Fine-root respiration at 0 °C as a fraction of fine-root carbon per "
        "year; SIPNET divides by 365.",
        long_label="Base fine root respiration rate",
        aliases=("base_fine_root_resp",),
    )
    base_coarse_root_respiration_rate: float = param_field(
        sipnet_name="baseCoarseRootResp",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Coarse-root respiration at 0 °C as a fraction of coarse-root carbon per "
        "year; SIPNET divides by 365.",
        long_label="Base coarse root respiration rate",
        aliases=("base_coarse_root_resp",),
    )
    fine_root_respiration_q10: float = param_field(
        sipnet_name="fineRootQ10",
        units="1",
        domain=_D.POSITIVE,
        description="Q10 temperature sensitivity of fine-root respiration.",
        long_label="Fine root respiration Q10",
        aliases=("fine_root_q10",),
    )
    coarse_root_respiration_q10: float = param_field(
        sipnet_name="coarseRootQ10",
        units="1",
        domain=_D.POSITIVE,
        description="Q10 temperature sensitivity of coarse-root respiration.",
        long_label="Coarse root respiration Q10",
        aliases=("coarse_root_q10",),
    )
    base_soil_respiration_rate: float = param_field(
        sipnet_name="baseSoilResp",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Heterotrophic soil respiration at 0 °C and saturated moisture as a "
        "fraction of soil carbon per year; SIPNET divides by 365.",
        long_label="Base soil respiration rate",
        aliases=("base_soil_resp",),
    )
    soil_respiration_q10: float = param_field(
        sipnet_name="soilRespQ10",
        units="1",
        domain=_D.POSITIVE,
        description="Q10 temperature sensitivity of heterotrophic soil respiration.",
        long_label="Soil respiration Q10",
        aliases=("soil_resp_q10",),
    )
    soil_respiration_moisture_exponent: float = param_field(
        sipnet_name="soilRespMoistEffect",
        units="1",
        domain=_D.NON_NEGATIVE,
        description="Exponent of the soil-moisture dependence of heterotrophic respiration. "
        "Only used when ModelFlags.water_hresp is on.",
        long_label="Soil respiration moisture exponent",
        aliases=("soil_resp_moist_effect",),
    )
    litter_breakdown_rate: float | None = param_field(
        sipnet_name="litterBreakdownRate",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Litter carbon broken down per year at 0 °C, as a fraction of the litter "
        "pool; SIPNET divides by 365. Required when ModelFlags.litter_pool is on.",
        long_label="Litter breakdown rate",
        default=None,
    )
    litter_respired_fraction: float | None = param_field(
        sipnet_name="fracLitterRespired",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of broken-down litter that is respired rather than transferred "
        "to the soil carbon pool. Required when ModelFlags.litter_pool is on.",
        long_label="Litter respired fraction",
        aliases=("frac_litter_respired",),
        default=None,
    )


class AllocationParams(ParameterGroup):
    """Carbon allocation fractions and pool turnover rates.

    Constraint
    ----------
    ``leaf_allocation + fine_root_allocation + wood_allocation < 1``.
    Coarse-root allocation is the residual and is derived by SIPNET, not read
    from the param file.

    All turnover rates are per-year and have ``per_year=True`` in their
    :class:`~pysipnet.parameters.base.ParameterSpec`.
    """

    fine_root_allocation: float = param_field(
        sipnet_name="fineRootAllocation",
        units="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Fraction of net primary production allocated to fine roots.",
        long_label="Fine root allocation",
    )
    wood_allocation: float = param_field(
        sipnet_name="woodAllocation",
        units="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Fraction of net primary production allocated to wood.",
        long_label="Wood allocation",
    )
    fine_root_turnover_rate: float = param_field(
        sipnet_name="fineRootTurnoverRate",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Fraction of fine-root carbon lost per year; SIPNET divides by 365.",
        long_label="Fine root turnover rate",
    )
    coarse_root_turnover_rate: float = param_field(
        sipnet_name="coarseRootTurnoverRate",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Fraction of coarse-root carbon lost per year; SIPNET divides by 365.",
        long_label="Coarse root turnover rate",
    )
    wood_turnover_rate: float = param_field(
        sipnet_name="woodTurnoverRate",
        units="yr-1",
        domain=_D.POSITIVE,
        per_year=True,
        description="Fraction of wood carbon lost to litter per year; SIPNET divides by 365.",
        long_label="Wood turnover rate",
    )

    @model_validator(mode="after")
    def _check_allocation_sum(self) -> AllocationParams:
        # leaf_allocation lives in PhenologyParams; the cross-model constraint
        # (leaf + fine_root + wood < 1) is checked on SIPNETParameters.
        total = self.fine_root_allocation + self.wood_allocation
        if total >= 1.0:
            raise ValueError(
                f"fine_root_allocation + wood_allocation = {total:.4f} ≥ 1.0. "
                "Coarse-root allocation (the residual) would be non-positive."
            )
        return self


class WaterParams(ParameterGroup):
    """Soil water, evapotranspiration, and snow parameters.

    Flag-dependent fields
    ~~~~~~~~~~~~~~~~~~~~~
    ``snow_melt_rate`` is only used when ``ModelFlags.snow`` is on.
    ``leaf_water_pool_depth`` is only used when ``ModelFlags.leaf_water`` is on.
    Both are ``Optional[float]`` and validated by :class:`SIPNETParameters`.
    """

    water_removal_fraction: float = param_field(
        sipnet_name="waterRemoveFrac",
        units="d-1",
        domain=_D.POSITIVE,
        description="Fraction of plant-available soil water that can be removed per day "
        "without inducing water stress.",
        long_label="Water removal fraction",
        aliases=("water_remove_frac",),
    )
    frozen_soil_water_fraction: float = param_field(
        sipnet_name="frozenSoilEff",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of soil water available to plants when the soil is frozen; "
        "0 means none.",
        long_label="Frozen-soil water fraction",
        aliases=("frozen_soil_eff",),
    )
    water_use_efficiency: float = param_field(
        sipnet_name="wueConst",
        units="mg g-1 kPa",
        constituent="CO2",
        domain=_D.POSITIVE,
        description="Water use efficiency constant linking transpiration to gross primary "
        "production: water use efficiency in mg CO2 per g water is this value divided by "
        "the vapor pressure deficit in kPa.",
        long_label="Water use efficiency",
        short_label="WUE",
        aliases=("wue_const",),
    )
    soil_water_holding_capacity: float = param_field(
        sipnet_name="soilWHC",
        units="cm",
        constituent="H2O",
        domain=_D.POSITIVE,
        description="Plant-available soil water at saturation, as a depth of water.",
        long_label="Soil water holding capacity",
        aliases=("soil_whc",),
    )
    interception_evaporation_fraction: float = param_field(
        sipnet_name="immedEvapFrac",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of precipitation intercepted by the canopy and evaporated "
        "immediately.",
        long_label="Interception evaporation fraction",
        aliases=("immed_evap_frac",),
    )
    fast_flow_fraction: float = param_field(
        sipnet_name="fastFlowFrac",
        units="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of water reaching the soil that drains immediately without "
        "entering the soil water pool.",
        long_label="Fast flow fraction",
        aliases=("fast_flow_frac",),
    )
    snow_melt_rate: float | None = param_field(
        sipnet_name="snowMelt",
        units="cm K-1 d-1",
        constituent="H2O",
        domain=_D.POSITIVE,
        description="Snow melted per degree of air temperature above freezing per day. "
        "Required when ModelFlags.snow is on. With the flag off it is optional but still "
        "used if supplied; if omitted SIPNET uses zero and snow never melts.",
        long_label="Snow melt rate",
        aliases=("snow_melt",),
        default=None,
    )
    aerodynamic_resistance_constant: float = param_field(
        sipnet_name="rdConst",
        units="1",
        domain=_D.POSITIVE,
        description="Scalar in the aerodynamic resistance used for soil evaporation.",
        long_label="Aerodynamic resistance constant",
        aliases=("rd_const",),
    )
    soil_resistance_intercept: float = param_field(
        sipnet_name="rSoilConst1",
        units="1",
        domain=_D.REAL,
        description="Intercept of the soil evaporation resistance model "
        "exp(intercept − slope × soil_wetness_fraction).",
        long_label="Soil resistance intercept",
        aliases=("r_soil_const1",),
    )
    soil_resistance_slope: float = param_field(
        sipnet_name="rSoilConst2",
        units="1",
        domain=_D.POSITIVE,
        description="Slope of the soil evaporation resistance model "
        "exp(intercept − slope × soil_wetness_fraction); larger values mean stronger "
        "resistance when the soil is dry.",
        long_label="Soil resistance slope",
        aliases=("r_soil_const2",),
    )
    leaf_water_pool_depth: float | None = param_field(
        sipnet_name="leafPoolDepth",
        units="cm d-1",
        constituent="H2O",
        domain=_D.NON_NEGATIVE,
        description="Cap on interception evaporation per unit leaf area index: the canopy "
        "can evaporate at most this rate times the leaf area index. Required when "
        "ModelFlags.leaf_water is on.",
        long_label="Leaf water pool depth",
        aliases=("leaf_pool_depth",),
        default=None,
    )


class LeafPhysiologyParams(ParameterGroup):
    """Leaf structural and carbon-fraction parameters."""

    leaf_carbon_per_area: float = param_field(
        sipnet_name="leafCSpWt",
        units="g m-2",
        constituent="C",
        domain=_D.POSITIVE,
        description="Leaf carbon per unit leaf area (specific leaf weight expressed as "
        "carbon); the reciprocal of specific leaf area times leaf_carbon_fraction. Converts "
        "leaf_area_index to leaf_carbon at the start of the run.",
        long_label="Leaf carbon per area",
        aliases=("leaf_c_sp_wt",),
    )
    leaf_carbon_fraction: float = param_field(
        sipnet_name="cFracLeaf",
        units="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Carbon as a fraction of leaf dry mass.",
        long_label="Leaf carbon fraction",
        aliases=("c_frac_leaf",),
    )


# ── Top-level model ────────────────────────────────────────────────────────────


class SIPNETParameters(BaseModel):
    """Complete parameter set for a SIPNET run.

    Composed of domain-grouped sub-models.  All fields are required unless
    otherwise noted.  The companion :class:`ModelFlags` decides which optional
    processes are on, and therefore which parameters SIPNET requires.

    Serialization / deserialization::

        params_dict = params.model_dump()
        params      = SIPNETParameters.model_validate(params_dict)

    Calibration tooling::

        from pysipnet.parameters.base import get_parameter_specs
        specs = get_parameter_specs(SIPNETParameters)
        domains = {k: v.domain for k, v in specs.items()}
    """

    initial_conditions: InitialConditions
    photosynthesis: PhotosynthesisParams
    phenology: PhenologyParams
    respiration: RespirationParams
    allocation: AllocationParams
    water: WaterParams
    leaf: LeafPhysiologyParams

    @model_validator(mode="after")
    def _check_allocation_triangle(self) -> SIPNETParameters:
        total = (
            self.phenology.leaf_allocation
            + self.allocation.fine_root_allocation
            + self.allocation.wood_allocation
        )
        if total >= 1.0:
            raise ValueError(
                f"leaf_allocation + fine_root_allocation + wood_allocation = {total:.4f} ≥ 1.0. "
                "Coarse-root allocation (the residual 1 − total) would be non-positive."
            )
        return self

    def validate_for_flags(self, flags: ModelFlags) -> None:
        """Raise :class:`ValueError` if any flag-required parameter is ``None``.

        Call this before writing the param file to surface configuration
        mismatches early.
        """
        errors: list[str] = []
        if flags.snow and self.water.snow_melt_rate is None:
            errors.append("water.snow_melt_rate is required when ModelFlags.snow is True")
        if flags.leaf_water and self.water.leaf_water_pool_depth is None:
            errors.append(
                "water.leaf_water_pool_depth is required when ModelFlags.leaf_water is True"
            )
        if flags.litter_pool and self.respiration.litter_breakdown_rate is None:
            errors.append(
                "respiration.litter_breakdown_rate is required when ModelFlags.litter_pool is True"
            )
        if flags.litter_pool and self.respiration.litter_respired_fraction is None:
            errors.append(
                "respiration.litter_respired_fraction is required when "
                "ModelFlags.litter_pool is True"
            )
        if flags.gdd and self.phenology.leaf_on_growing_degree_days is None:
            errors.append(
                "phenology.leaf_on_growing_degree_days is required when ModelFlags.gdd is True"
            )
        if flags.soil_phenol and self.phenology.leaf_on_soil_temperature is None:
            errors.append(
                "phenology.leaf_on_soil_temperature is required when ModelFlags.soil_phenol is True"
            )
        if not flags.gdd and not flags.soil_phenol and self.phenology.leaf_on_day is None:
            errors.append(
                "phenology.leaf_on_day is required when both gdd and soil_phenol are False"
            )
        if errors:
            raise ValueError("Parameter–flag mismatch:\n" + "\n".join(f"  • {e}" for e in errors))


# ── Parameter group map ────────────────────────────────────────────────────────


def _build_param_groups() -> dict[str, list[str]]:
    """Return a mapping from group name to the list of parameter field names.

    Asserts that all parameter names are unique across groups — a structural
    invariant of :class:`SIPNETParameters`.
    """
    groups: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for group_name, field_info in SIPNETParameters.model_fields.items():
        annotation = field_info.annotation
        if (
            annotation is None
            or not isinstance(annotation, type)
            or not issubclass(annotation, BaseModel)
        ):
            continue
        param_names = list(annotation.model_fields.keys())
        for name in param_names:
            assert name not in seen, (
                f"Parameter '{name}' appears in both '{seen[name]}' and '{group_name}'. "
                "Parameter names must be unique across all groups."
            )
            seen[name] = group_name
        groups[group_name] = param_names
    return groups


SIPNET_PARAMS_BY_GROUP: dict[str, list[str]] = _build_param_groups()
"""Mapping from parameter group name to the list of field names in that group.

Built once at module import time by inspecting :class:`SIPNETParameters`.
All parameter names are guaranteed unique across groups.

The field names in each group are listed on the generated "Parameters"
documentation page, or via :data:`PARAMETER_SPECS`.

Examples
--------
List all photosynthesis parameter names::

    from pysipnet.parameters import SIPNET_PARAMS_BY_GROUP
    SIPNET_PARAMS_BY_GROUP["photosynthesis"]
    # ['max_photosynthesis_rate', 'daily_mean_photosynthesis_fraction', ...]

Check which group a parameter belongs to::

    group = next(
        g for g, ps in SIPNET_PARAMS_BY_GROUP.items() if "max_photosynthesis_rate" in ps
    )
    # 'photosynthesis'
"""


PARAMETER_SPECS: dict[str, ParameterSpec] = get_parameter_specs(SIPNETParameters)
"""``{"group.field": ParameterSpec}`` for every parameter, in declaration order."""


def _build_parameter_alias_index() -> dict[str, str]:
    """Map every accepted spelling of a parameter to its flat field name.

    Accepted spellings are the field name itself, SIPNET's name, and the
    aliases recorded on the spec (the names pySIPNET used before the naming
    convention). Collisions are a programming error and fail at import.
    """
    index: dict[str, str] = {}
    for path, spec in PARAMETER_SPECS.items():
        field = path.split(".", 1)[1]
        for key in (field, spec.sipnet_name, *spec.aliases):
            existing = index.get(key)
            if existing is not None and existing != field:
                raise ValueError(
                    f"Parameter alias {key!r} is claimed by {existing!r} and {field!r}."
                )
            index[key] = field
    return index


_PARAMETER_ALIASES: dict[str, str] = _build_parameter_alias_index()


def resolve_parameter_name(name: str) -> str:
    """Return the current flat field name for a parameter name, alias or SIPNET name.

    ``resolve_parameter_name("aMax")``, ``resolve_parameter_name("a_max")``
    and ``resolve_parameter_name("max_photosynthesis_rate")`` all give
    ``"max_photosynthesis_rate"``. Raises ``KeyError`` for anything else.
    """
    try:
        return _PARAMETER_ALIASES[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a SIPNET parameter name or alias.") from None
