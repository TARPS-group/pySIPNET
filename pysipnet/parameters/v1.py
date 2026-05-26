"""SIPNET v1 parameter models.

All inputs to a SIPNET v1 run are captured here as Pydantic models.  The
models mirror the structure of SIPNET's ``.param`` file but use snake_case
field names and explicit units.  The IO layer (:mod:`pysipnet.io.param_io`)
handles translation to SIPNET's camelCase names.

Model structure
---------------
``SIPNETParametersV1`` is a flat composition of domain-grouped sub-models::

    params = SIPNETParametersV1(
        initial_conditions=InitialConditions(plant_wood=30000, ...),
        photosynthesis=PhotosynthesisParams(a_max=112.0, ...),
        ...
    )

Use :func:`pysipnet.parameters.base.get_parameter_specs` to retrieve the full
``{path: ParameterSpec}`` dict for calibration tooling::

    from pysipnet.parameters.base import get_parameter_specs
    specs = get_parameter_specs(SIPNETParametersV1)

Per-year rate parameters
------------------------
The following parameters are specified in the ``.param`` file as **per-year**
rates and are converted to per-day internally by SIPNET (÷ 365).  The Python
interface works in per-year units throughout, matching SIPNET's convention.
Each of these fields has ``per_year=True`` in its :class:`ParameterSpec`.

- ``respiration.base_veg_resp``
- ``respiration.base_fine_root_resp``
- ``respiration.base_coarse_root_resp``
- ``respiration.base_soil_resp``
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
Some parameters are only meaningful when the corresponding compile-time flag is
active.  These fields are ``Optional[float]`` with a default of ``None``.
:class:`SIPNETParametersV1` validates that required-by-flag parameters are
provided given the active :class:`ModelFlagsV1`.

+---------------------------+----------------------------------+
| Parameter                 | Required when flag is active     |
+===========================+==================================+
| ``phenology.gdd_leaf_on`` | ``ModelFlagsV1.gdd = True``      |
+---------------------------+----------------------------------+
| ``phenology.soil_temp_leaf_on`` | ``ModelFlagsV1.soil_phenol`` |
+---------------------------+----------------------------------+
| ``initial_conditions.snow`` (optional, default 0) | ``ModelFlagsV1.snow`` |
+---------------------------+----------------------------------+
| ``water.snow_melt``       | ``ModelFlagsV1.snow = True``     |
+---------------------------+----------------------------------+
| ``water.leaf_pool_depth`` | ``ModelFlagsV1.leaf_water``      |
+---------------------------+----------------------------------+
| ``respiration.litter_breakdown_rate`` | ``ModelFlagsV1.litter_pool`` |
+---------------------------+----------------------------------+
| ``respiration.frac_litter_respired``  | ``ModelFlagsV1.litter_pool`` |
+---------------------------+----------------------------------+
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from pysipnet.parameters.base import (
    ParameterDomain,
    param_field,
)

_D = ParameterDomain  # local alias for brevity


# ── Compile-time flag model ────────────────────────────────────────────────────


class ModelFlagsV1(BaseModel):
    """Compile-time feature flags for the SIPNET v1 binary.

    Each flag controls a ``#define`` in the SIPNET source.  The combination of
    flags determines which binary preset to use (see :class:`pysipnet.runner.ModelPreset`)
    and which parameters are required.

    ``gdd`` and ``soil_phenol`` are mutually exclusive.

    Presets
    -------
    ``ModelFlagsV1.standard()``  — SNOW=1 GDD=1 WATER_HRESP=1 (all others off)
    ``ModelFlagsV1.forest()``    — standard + LITTER_POOL=1
    """

    snow: bool = True
    """Track snowpack rather than treating all precipitation as liquid."""

    gdd: bool = True
    """Use growing degree-days (accumulated from Jan 1) for leaf-out phenology."""

    water_hresp: bool = True
    """Allow soil moisture to affect heterotrophic respiration."""

    growth_resp: bool = False
    """Model growth respiration explicitly, separate from maintenance respiration."""

    leaf_water: bool = False
    """Calculate a leaf water pool for sub-daily evaporation realism."""

    litter_pool: bool = False
    """Enable a separate litter carbon pool in addition to the soil C pool."""

    soil_phenol: bool = False
    """Use soil temperature threshold for leaf-out (mutually exclusive with ``gdd``)."""

    @model_validator(mode="after")
    def _check_phenology_exclusivity(self) -> ModelFlagsV1:
        if self.gdd and self.soil_phenol:
            raise ValueError("gdd and soil_phenol are mutually exclusive: set exactly one to True.")
        return self

    @classmethod
    def standard(cls) -> ModelFlagsV1:
        """Default v1 configuration: SNOW, GDD, WATER_HRESP on; everything else off."""
        return cls()

    @classmethod
    def forest(cls) -> ModelFlagsV1:
        """Standard configuration with an additional explicit litter C pool."""
        return cls(litter_pool=True)


# ── Sub-models ─────────────────────────────────────────────────────────────────


class InitialConditions(BaseModel):
    """Carbon pool and hydrological state at the start of the simulation.

    All C pool values are in g C m⁻².  These are written to the ``.param``
    file alongside model parameters (SIPNET makes no file-format distinction
    between initial conditions and parameters).
    """

    plant_wood: float = param_field(
        unit="g / m**2",
        constituent="C",
        domain=_D.NON_NEGATIVE,
        description="Initial aboveground wood + root C (SIPNET param: plantWoodInit).",
    )
    lai: float = param_field(
        unit="m**2 / m**2",
        domain=_D.NON_NEGATIVE,
        description="Initial leaf area index; used internally to derive initial leaf C "
        "(SIPNET param: laiInit).",
    )
    litter: float = param_field(
        unit="g / m**2",
        constituent="C",
        domain=_D.NON_NEGATIVE,
        default=0.0,
        description="Initial litter C pool (SIPNET param: litterInit). "
        "Only affects dynamics when ModelFlagsV1.litter_pool is True.",
    )
    soil: float = param_field(
        unit="g / m**2",
        constituent="C",
        domain=_D.NON_NEGATIVE,
        description="Initial soil C pool (SIPNET param: soilInit).",
    )
    soil_water_frac: float = param_field(
        unit="1",
        domain=_D.NON_NEGATIVE,
        description="Initial soil water as a fraction of water holding capacity "
        "(SIPNET param: soilWFracInit). May exceed 1 in flooding scenarios.",
    )
    litter_water_frac: float = param_field(
        unit="1",
        domain=_D.NON_NEGATIVE,
        default=0.0,
        description="Initial litter water as a fraction of litter water holding capacity "
        "(SIPNET param: litterWFracInit). Used to initialise litterWater = "
        "litterWFracInit × litterWHC.",
    )
    snow: float = param_field(
        unit="cm",
        constituent="water equiv.",
        domain=_D.NON_NEGATIVE,
        default=0.0,
        description="Initial snowpack in cm water equivalent (SIPNET param: snowInit). "
        "Only used when ModelFlagsV1.snow is True.",
    )
    fine_root_frac: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of plantWoodInit allocated to fine roots at initialisation "
        "(SIPNET param: fineRootFrac).",
    )
    coarse_root_frac: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of plantWoodInit allocated to coarse roots at initialisation "
        "(SIPNET param: coarseRootFrac).",
    )


class PhotosynthesisParams(BaseModel):
    """Parameters governing gross primary production.

    VPD effect: ``dVpd = 1 − dVpd_slope × vpd^dVpd_exp``.  GPP is scaled by
    this factor, so a larger ``dVpd_slope`` produces stronger VPD suppression.

    ``psn_t_max`` is derived internally as ``2 × psn_t_opt − psn_t_min`` and
    is therefore not a free parameter.
    """

    a_max: float = param_field(
        unit="nmol / (g * s)",
        constituent="CO2 g-1 leaf",
        domain=_D.POSITIVE,
        description="Maximum photosynthesis rate at saturating PAR and no environmental "
        "stress (SIPNET param: aMax).",
    )
    a_max_frac: float = param_field(
        unit="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Average daily aMax as a fraction of the instantaneous peak, "
        "accounting for within-day variation in light and temperature "
        "(SIPNET param: aMaxFrac).",
    )
    base_fol_resp_frac: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Basal foliar (maintenance) respiration as a fraction of aMax "
        "(SIPNET param: baseFolRespFrac).",
    )
    psn_t_min: float = param_field(
        unit="degC",
        domain=_D.REAL,
        description="Minimum air temperature for net photosynthesis (SIPNET param: psnTMin). "
        "Net PSN is zero at or below this temperature.",
    )
    psn_t_opt: float = param_field(
        unit="degC",
        domain=_D.REAL,
        description="Optimum air temperature for photosynthesis (SIPNET param: psnTOpt). "
        "psnTMax is derived symmetrically: psnTMax = 2 × psnTOpt − psnTMin.",
    )
    d_vpd_slope: float = param_field(
        unit="1 / kPa",
        domain=_D.POSITIVE,
        description="Slope of the VPD–photosynthesis reduction: "
        "dVpd = 1 − dVpdSlope × vpd^dVpdExp (SIPNET param: dVpdSlope).",
    )
    d_vpd_exp: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Exponent for VPD effect on photosynthesis (SIPNET param: dVpdExp).",
    )
    half_sat_par: float = param_field(
        unit="mol / (m**2 * day)",
        constituent="photons",
        domain=_D.POSITIVE,
        description="PAR at which photosynthesis equals half its theoretical maximum "
        "(SIPNET param: halfSatPar). Units: mol photons m⁻² ground day⁻¹ "
        "(1 Einstein = 1 mol photons).",
    )
    attenuation: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Canopy PAR extinction coefficient (Beer's law k) (SIPNET param: attenuation).",
    )


class PhenologyParams(BaseModel):
    """Parameters controlling leaf phenology (growing season timing and leaf dynamics).

    Exactly one of ``leaf_on_day``, ``gdd_leaf_on``, or ``soil_temp_leaf_on``
    will be active depending on the compile-time flags (``GDD`` or
    ``SOIL_PHENOL``).  The inactive alternatives are still stored here but are
    ignored by SIPNET.
    """

    leaf_on_day: float | None = param_field(
        unit="day",
        domain=_D.POSITIVE,
        description="Day of year on which leaves appear (SIPNET param: leafOnDay). "
        "Active when both ModelFlagsV1.gdd and ModelFlagsV1.soil_phenol are False.",
        default=None,
    )
    leaf_off_day: float = param_field(
        unit="day",
        domain=_D.POSITIVE,
        description="Day of year on which leaves fall (SIPNET param: leafOffDay).",
    )
    gdd_leaf_on: float | None = param_field(
        unit="K * day",
        domain=_D.NON_NEGATIVE,
        description="Growing degree-day (GDD) threshold for leaf appearance "
        "(SIPNET param: gddLeafOn). Unit is K·day = °C·day (temperature differences; "
        "Kelvin used instead of degC to avoid Pint offset-unit ambiguity). "
        "Active when ModelFlagsV1.gdd is True.",
        default=None,
    )
    soil_temp_leaf_on: float | None = param_field(
        unit="degC",
        domain=_D.REAL,
        description="Soil temperature threshold for leaf appearance "
        "(SIPNET param: soilTempLeafOn). Active when ModelFlagsV1.soil_phenol is True.",
        default=None,
    )
    leaf_growth: float = param_field(
        unit="g / m**2",
        constituent="C",
        domain=_D.NON_NEGATIVE,
        description="Additional leaf C grown at the start of the growing season "
        "(SIPNET param: leafGrowth).",
    )
    frac_leaf_fall: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Additional fraction of the standing leaf C that falls at "
        "season end (SIPNET param: fracLeafFall).",
    )
    leaf_allocation: float = param_field(
        unit="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Fraction of NPP allocated to leaf growth (SIPNET param: leafAllocation). "
        "Also used in the allocation constraint; see AllocationParams.",
    )
    leaf_turnover_rate: float = param_field(
        unit="1 / year",
        domain=_D.POSITIVE,
        per_year=True,
        description="Average leaf turnover rate (SIPNET param: leafTurnoverRate). "
        "Specified as year⁻¹; SIPNET divides by 365 for daily use.",
    )


class RespirationParams(BaseModel):
    """Autotrophic and heterotrophic respiration parameters.

    Per-year rate parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``base_veg_resp``, ``base_fine_root_resp``, ``base_coarse_root_resp``,
    ``base_soil_resp``, and ``litter_breakdown_rate`` are all specified as
    per-year rates (matching the SIPNET param file convention).  SIPNET divides
    by 365 internally.  Their :class:`~pysipnet.parameters.base.ParameterSpec`
    has ``per_year=True``.

    Litter parameters
    ~~~~~~~~~~~~~~~~~
    ``litter_breakdown_rate`` and ``frac_litter_respired`` are only meaningful
    when ``ModelFlagsV1.litter_pool`` is ``True``.  They may be ``None``
    otherwise; the validator on :class:`SIPNETParametersV1` enforces this.
    """

    base_veg_resp: float = param_field(
        unit="1 / year",
        constituent="C g-1 plant C",
        domain=_D.POSITIVE,
        per_year=True,
        description="Wood maintenance respiration rate at 0 °C (SIPNET param: baseVegResp). "
        "Units: g C respired g⁻¹ plant C year⁻¹.",
    )
    veg_resp_q10: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Q10 temperature sensitivity of vegetation (wood) respiration "
        "(SIPNET param: vegRespQ10).",
    )
    growth_resp_frac: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Growth respiration as a fraction of running-mean NPP "
        "(SIPNET param: growthRespFrac). "
        "Only used when ModelFlagsV1.growth_resp is True.",
        default=0.0,
    )
    frozen_soil_fol_r_eff: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Foliar respiration reduction factor when soil is frozen "
        "(SIPNET param: frozenSoilFolREff). "
        "0 = full shutdown; 1 = no reduction.",
    )
    frozen_soil_threshold: float = param_field(
        unit="degC",
        domain=_D.REAL,
        description="Soil temperature below which frozen-soil effects activate "
        "(SIPNET param: frozenSoilThreshold).",
    )
    base_fine_root_resp: float = param_field(
        unit="1 / year",
        domain=_D.POSITIVE,
        per_year=True,
        description="Base fine-root respiration rate at 0 °C (SIPNET param: baseFineRootResp). "
        "Year⁻¹; divided by 365 internally.",
    )
    base_coarse_root_resp: float = param_field(
        unit="1 / year",
        domain=_D.POSITIVE,
        per_year=True,
        description="Base coarse-root respiration rate at 0 °C "
        "(SIPNET param: baseCoarseRootResp). Year⁻¹; divided by 365 internally.",
    )
    fine_root_q10: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Q10 for fine-root respiration (SIPNET param: fineRootQ10).",
    )
    coarse_root_q10: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Q10 for coarse-root respiration (SIPNET param: coarseRootQ10).",
    )
    base_soil_resp: float = param_field(
        unit="1 / year",
        constituent="C g-1 soil C",
        domain=_D.POSITIVE,
        per_year=True,
        description="Soil respiration rate at 0 °C and saturated moisture "
        "(SIPNET param: baseSoilResp). "
        "Units: g C respired g⁻¹ soil C year⁻¹.",
    )
    soil_resp_q10: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Q10 temperature sensitivity of soil (heterotrophic) respiration "
        "(SIPNET param: soilRespQ10).",
    )
    soil_resp_moist_effect: float = param_field(
        unit="1",
        domain=_D.NON_NEGATIVE,
        description="Exponent controlling the effect of soil moisture on heterotrophic "
        "respiration (SIPNET param: soilRespMoistEffect). "
        "Only used when ModelFlagsV1.water_hresp is True.",
    )
    litter_breakdown_rate: float | None = param_field(
        unit="1 / year",
        constituent="C g-1 litter C",
        domain=_D.POSITIVE,
        per_year=True,
        description="Litter-to-soil carbon transfer rate at 0 °C "
        "(SIPNET param: litterBreakdownRate). "
        "Required when ModelFlagsV1.litter_pool is True.",
        default=None,
    )
    frac_litter_respired: float | None = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of broken-down litter that is respired rather than "
        "transferred to the soil C pool (SIPNET param: fracLitterRespired). "
        "Required when ModelFlagsV1.litter_pool is True.",
        default=None,
    )


class AllocationParams(BaseModel):
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
        unit="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Fraction of NPP allocated to fine roots (SIPNET param: fineRootAllocation).",
    )
    fine_root_exudation: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        default=0.0,
        description="Fraction of GPP exuded from fine roots to soil "
        "(SIPNET param: fineRootExudation).",
    )
    coarse_root_exudation: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        default=0.0,
        description="Fraction of NPP exuded from coarse roots to soil "
        "(SIPNET param: coarseRootExudation).",
    )
    wood_allocation: float = param_field(
        unit="1",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Fraction of NPP allocated to wood (SIPNET param: woodAllocation).",
    )
    fine_root_turnover_rate: float = param_field(
        unit="1 / year",
        domain=_D.POSITIVE,
        per_year=True,
        description="Fine-root turnover rate (SIPNET param: fineRootTurnoverRate). "
        "Year⁻¹; divided by 365 internally.",
    )
    coarse_root_turnover_rate: float = param_field(
        unit="1 / year",
        domain=_D.POSITIVE,
        per_year=True,
        description="Coarse-root turnover rate (SIPNET param: coarseRootTurnoverRate). "
        "Year⁻¹; divided by 365 internally.",
    )
    wood_turnover_rate: float = param_field(
        unit="1 / year",
        domain=_D.POSITIVE,
        per_year=True,
        description="Wood turnover rate (SIPNET param: woodTurnoverRate). "
        "Year⁻¹; divided by 365 internally.",
    )

    @model_validator(mode="after")
    def _check_allocation_sum(self) -> AllocationParams:
        # leaf_allocation lives in PhenologyParams; the cross-model constraint
        # (leaf + fine_root + wood < 1) is checked on SIPNETParametersV1.
        total = self.fine_root_allocation + self.wood_allocation
        if total >= 1.0:
            raise ValueError(
                f"fine_root_allocation + wood_allocation = {total:.4f} ≥ 1.0. "
                "Coarse-root allocation (the residual) would be non-positive."
            )
        return self


class WaterParams(BaseModel):
    """Soil water, evapotranspiration, and snow parameters.

    Flag-dependent fields
    ~~~~~~~~~~~~~~~~~~~~~
    ``snow_melt`` is only used when ``ModelFlagsV1.snow`` is ``True``.
    ``leaf_pool_depth`` is only used when ``ModelFlagsV1.leaf_water`` is ``True``.
    Both are ``Optional[float]`` and validated by :class:`SIPNETParametersV1`.
    """

    water_remove_frac: float = param_field(
        unit="1 / day",
        domain=_D.POSITIVE,
        description="Fraction of plant-available soil water that can be removed per day "
        "without inducing water stress (SIPNET param: waterRemoveFrac).",
    )
    frozen_soil_eff: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of soil water available to plants when the soil is frozen "
        "(SIPNET param: frozenSoilEff). 0 = fully unavailable.",
    )
    wue_const: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Water use efficiency constant linking transpiration to GPP "
        "(SIPNET param: wueConst).",
    )
    soil_whc: float = param_field(
        unit="cm",
        domain=_D.POSITIVE,
        description="Soil water holding capacity (SIPNET param: soilWHC).",
    )
    immed_evap_frac: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of precipitation immediately intercepted and evaporated "
        "from the canopy (SIPNET param: immedEvapFrac).",
    )
    fast_flow_frac: float = param_field(
        unit="1",
        domain=_D.UNIT_INTERVAL,
        description="Fraction of incoming water going directly to drainage without "
        "entering the soil reservoir (SIPNET param: fastFlowFrac).",
    )
    snow_melt: float | None = param_field(
        unit="cm / (K * day)",
        domain=_D.POSITIVE,
        description="Snowmelt rate per Kelvin above freezing per day "
        "(SIPNET param: snowMelt). Unit is cm K⁻¹ day⁻¹ = cm °C⁻¹ day⁻¹ "
        "(Kelvin used instead of degC to avoid Pint offset-unit ambiguity). "
        "Required when ModelFlagsV1.snow is True.",
        default=None,
    )
    rd_const: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Aerodynamic resistance scalar (SIPNET param: rdConst).",
    )
    r_soil_const1: float = param_field(
        unit="1",
        domain=_D.REAL,
        description="Soil resistance constant 1 in the exponential resistance model "
        "rSoil = exp(rSoilConst1 − rSoilConst2 × W/WHC) (SIPNET param: rSoilConst1).",
    )
    r_soil_const2: float = param_field(
        unit="1",
        domain=_D.POSITIVE,
        description="Soil resistance constant 2 (SIPNET param: rSoilConst2). "
        "Larger values produce stronger soil resistance at low soil moisture.",
    )
    litter_whc: float = param_field(
        unit="cm",
        domain=_D.POSITIVE,
        description="Litter layer water holding capacity (SIPNET param: litterWHC). "
        "Used together with litter_water_frac to initialise litterWater.",
    )
    leaf_pool_depth: float | None = param_field(
        unit="cm",
        domain=_D.NON_NEGATIVE,
        description="Leaf water pool capacity (cm per unit LAI per day cap on "
        "interception evaporation) (SIPNET param: leafPoolDepth). "
        "Required when ModelFlagsV1.leaf_water is True.",
        default=None,
    )


class LeafPhysiologyParams(BaseModel):
    """Leaf structural and carbon-fraction parameters."""

    leaf_c_sp_wt: float = param_field(
        unit="g / m**2",
        constituent="C m-2 leaf",
        domain=_D.POSITIVE,
        description="Specific leaf weight: carbon per unit leaf area "
        "(SIPNET param: leafCSpWt). "
        "Related to SLA: leafCSpWt = cFracLeaf / SLA.",
    )
    c_frac_leaf: float = param_field(
        unit="1",
        constituent="C g-1 leaf",
        domain=_D.OPEN_UNIT_INTERVAL,
        description="Carbon fraction of leaf dry mass (SIPNET param: cFracLeaf).",
    )


# ── Top-level model ────────────────────────────────────────────────────────────


class SIPNETParametersV1(BaseModel):
    """Complete parameter set for a SIPNET v1 run.

    Composed of domain-grouped sub-models.  All fields are required unless
    otherwise noted.  The companion :class:`ModelFlagsV1` controls which
    compile-time features are active and therefore which parameters are
    actually used by SIPNET.

    Serialisation / deserialisation::

        params_dict = params.model_dump()
        params      = SIPNETParametersV1.model_validate(params_dict)

    Calibration tooling::

        from pysipnet.parameters.base import get_parameter_specs
        specs = get_parameter_specs(SIPNETParametersV1)
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
    def _check_allocation_triangle(self) -> SIPNETParametersV1:
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

    def validate_for_flags(self, flags: ModelFlagsV1) -> None:
        """Raise :class:`ValueError` if any flag-required parameter is ``None``.

        Call this before writing the param file to surface configuration
        mismatches early.
        """
        errors: list[str] = []
        if flags.snow and self.water.snow_melt is None:
            errors.append("water.snow_melt is required when ModelFlagsV1.snow is True")
        if flags.leaf_water and self.water.leaf_pool_depth is None:
            errors.append("water.leaf_pool_depth is required when ModelFlagsV1.leaf_water is True")
        if flags.litter_pool and self.respiration.litter_breakdown_rate is None:
            errors.append(
                "respiration.litter_breakdown_rate is required"
                " when ModelFlagsV1.litter_pool is True"
            )
        if flags.litter_pool and self.respiration.frac_litter_respired is None:
            errors.append(
                "respiration.frac_litter_respired is required when ModelFlagsV1.litter_pool is True"
            )
        if flags.gdd and self.phenology.gdd_leaf_on is None:
            errors.append("phenology.gdd_leaf_on is required when ModelFlagsV1.gdd is True")
        if flags.soil_phenol and self.phenology.soil_temp_leaf_on is None:
            errors.append(
                "phenology.soil_temp_leaf_on is required when ModelFlagsV1.soil_phenol is True"
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
    invariant of :class:`SIPNETParametersV1`.
    """
    groups: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for group_name, field_info in SIPNETParametersV1.model_fields.items():
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

Built once at module import time by inspecting :class:`SIPNETParametersV1`.
All parameter names are guaranteed unique across groups.

Groups and their parameters:

- ``initial_conditions`` — ``plant_wood``, ``lai``, ``litter``, ``soil``,
  ``soil_water_frac``, ``litter_water_frac``, ``snow``, ``fine_root_frac``,
  ``coarse_root_frac``
- ``photosynthesis`` — ``a_max``, ``a_max_frac``, ``base_fol_resp_frac``,
  ``psn_t_min``, ``psn_t_opt``, ``d_vpd_slope``, ``d_vpd_exp``,
  ``half_sat_par``, ``attenuation``
- ``phenology`` — ``leaf_on_day``, ``leaf_off_day``, ``gdd_leaf_on``,
  ``soil_temp_leaf_on``, ``leaf_growth``, ``frac_leaf_fall``,
  ``leaf_allocation``, ``leaf_turnover_rate``
- ``respiration`` — ``base_veg_resp``, ``veg_resp_q10``,
  ``growth_resp_frac``, ``frozen_soil_fol_r_eff``,
  ``frozen_soil_threshold``, ``base_fine_root_resp``,
  ``base_coarse_root_resp``, ``fine_root_q10``, ``coarse_root_q10``,
  ``base_soil_resp``, ``soil_resp_q10``, ``soil_resp_moist_effect``,
  ``litter_breakdown_rate``, ``frac_litter_respired``
- ``allocation`` — ``fine_root_allocation``, ``fine_root_exudation``,
  ``coarse_root_exudation``, ``wood_allocation``,
  ``fine_root_turnover_rate``, ``coarse_root_turnover_rate``,
  ``wood_turnover_rate``
- ``water`` — ``water_remove_frac``, ``frozen_soil_eff``, ``wue_const``,
  ``soil_whc``, ``immed_evap_frac``, ``fast_flow_frac``, ``snow_melt``,
  ``rd_const``, ``r_soil_const1``, ``r_soil_const2``, ``litter_whc``,
  ``leaf_pool_depth``
- ``leaf`` — ``leaf_c_sp_wt``, ``c_frac_leaf``

Examples
--------
List all photosynthesis parameter names::

    from pysipnet.parameters import SIPNET_PARAMS_BY_GROUP
    SIPNET_PARAMS_BY_GROUP["photosynthesis"]
    # ['a_max', 'a_max_frac', 'base_fol_resp_frac', ...]

Check which group a parameter belongs to::

    group = next(g for g, ps in SIPNET_PARAMS_BY_GROUP.items() if "a_max" in ps)
    # 'photosynthesis'
"""
