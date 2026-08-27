"""Read and write SIPNET ``.param`` files.

SIPNET parameter file format
----------------------------------
Space/tab-delimited, two used columns::

    parameterName  value   [changeable  min  max  sigma ...]

- Comment character: ``!``  (everything after ``!`` on a line is ignored)
- Column order is irrelevant (SIPNET reads by name, not position)
- Columns beyond the second are silently accepted (legacy 5+ column format)

Python-to-SIPNET name mapping
------------------------------
Python field names use ``snake_case``; SIPNET uses ``camelCase``.  The mapping
is defined in :data:`PYTHON_TO_SIPNET` and is the single source of truth for
the translation between the two naming conventions.

Unit contract
~~~~~~~~~~~~~
Values are written as-is.  For parameters with ``per_year=True`` in their
:class:`~pysipnet.parameters.base.ParameterSpec`, the value written is the
per-year rate — matching what SIPNET expects in the param file (SIPNET divides
by 365 internally).
"""

from __future__ import annotations

from pathlib import Path

from pysipnet.parameters.model import ModelFlags, SIPNETParameters

# Maps dot-separated Python path → SIPNET param file name.
PYTHON_TO_SIPNET: dict[str, str] = {
    # Initial conditions
    "initial_conditions.plant_wood": "plantWoodInit",
    "initial_conditions.lai": "laiInit",
    "initial_conditions.litter": "litterInit",
    "initial_conditions.soil": "soilInit",
    "initial_conditions.soil_water_frac": "soilWFracInit",
    "initial_conditions.snow": "snowInit",
    "initial_conditions.fine_root_frac": "fineRootFrac",
    "initial_conditions.coarse_root_frac": "coarseRootFrac",
    # Photosynthesis
    "photosynthesis.a_max": "aMax",
    "photosynthesis.a_max_frac": "aMaxFrac",
    "photosynthesis.base_fol_resp_frac": "baseFolRespFrac",
    "photosynthesis.psn_t_min": "psnTMin",
    "photosynthesis.psn_t_opt": "psnTOpt",
    "photosynthesis.d_vpd_slope": "dVpdSlope",
    "photosynthesis.d_vpd_exp": "dVpdExp",
    "photosynthesis.half_sat_par": "halfSatPar",
    "photosynthesis.attenuation": "attenuation",
    # Phenology
    "phenology.leaf_on_day": "leafOnDay",
    "phenology.leaf_off_day": "leafOffDay",
    "phenology.gdd_leaf_on": "gddLeafOn",
    "phenology.soil_temp_leaf_on": "soilTempLeafOn",
    "phenology.leaf_growth": "leafGrowth",
    "phenology.frac_leaf_fall": "fracLeafFall",
    "phenology.leaf_allocation": "leafAllocation",
    "phenology.leaf_turnover_rate": "leafTurnoverRate",
    "phenology.leaf_on_realloc_frac": "leafOnReallocFrac",
    # Respiration
    "respiration.base_veg_resp": "baseVegResp",
    "respiration.veg_resp_q10": "vegRespQ10",
    "respiration.growth_resp_frac": "growthRespFrac",
    "respiration.frozen_soil_fol_r_eff": "frozenSoilFolREff",
    "respiration.frozen_soil_threshold": "frozenSoilThreshold",
    "respiration.base_fine_root_resp": "baseFineRootResp",
    "respiration.base_coarse_root_resp": "baseCoarseRootResp",
    "respiration.fine_root_q10": "fineRootQ10",
    "respiration.coarse_root_q10": "coarseRootQ10",
    "respiration.base_soil_resp": "baseSoilResp",
    "respiration.soil_resp_q10": "soilRespQ10",
    "respiration.soil_resp_moist_effect": "soilRespMoistEffect",
    "respiration.litter_breakdown_rate": "litterBreakdownRate",
    "respiration.frac_litter_respired": "fracLitterRespired",
    # Allocation
    "allocation.fine_root_allocation": "fineRootAllocation",
    "allocation.wood_allocation": "woodAllocation",
    "allocation.fine_root_turnover_rate": "fineRootTurnoverRate",
    "allocation.coarse_root_turnover_rate": "coarseRootTurnoverRate",
    "allocation.wood_turnover_rate": "woodTurnoverRate",
    # Water
    "water.water_remove_frac": "waterRemoveFrac",
    "water.frozen_soil_eff": "frozenSoilEff",
    "water.wue_const": "wueConst",
    "water.soil_whc": "soilWHC",
    "water.immed_evap_frac": "immedEvapFrac",
    "water.fast_flow_frac": "fastFlowFrac",
    "water.snow_melt": "snowMelt",
    "water.rd_const": "rdConst",
    "water.r_soil_const1": "rSoilConst1",
    "water.r_soil_const2": "rSoilConst2",
    "water.leaf_pool_depth": "leafPoolDepth",
    # Leaf physiology
    "leaf.leaf_c_sp_wt": "leafCSpWt",
    "leaf.c_frac_leaf": "cFracLeaf",
}

SIPNET_TO_PYTHON: dict[str, str] = {v: k for k, v in PYTHON_TO_SIPNET.items()}


def _flatten(params: SIPNETParameters) -> dict[str, float]:
    """Return a flat ``{sipnet_name: value}`` dict, omitting ``None`` values."""
    result: dict[str, float] = {}
    dump = params.model_dump()
    for python_path, sipnet_name in PYTHON_TO_SIPNET.items():
        group, field = python_path.split(".", 1)
        value = dump[group][field]
        if value is not None:
            result[sipnet_name] = value
    return result


def write_param_file(
    parameters: SIPNETParameters,
    flags: ModelFlags,
    path: Path,
) -> None:
    """Write the ``.param`` file SIPNET reads its parameters from.

    Parameters left as ``None`` are omitted. Those are the ones only some
    model configurations need, so writing them unconditionally would mean
    inventing values SIPNET would then ignore.

    The flags are checked against the parameters first, so a configuration
    asking for a process whose parameters are missing fails here rather than
    inside SIPNET.

    Parameters
    ----------
    parameters:
        The full parameter set.
    flags:
        Model options for this run, used to decide which parameters are
        required.
    path:
        Where to write the file, normally ``<workdir>/sipnet.param``.
    """
    parameters.validate_for_flags(flags)
    flat = _flatten(parameters)
    lines = ["! SIPNET parameter file - generated by pySIPNET\n"]
    for sipnet_name, value in flat.items():
        lines.append(f"{sipnet_name}\t{value!r}\n")
    path.write_text("".join(lines))


def read_param_file(path: Path) -> dict[str, float]:
    """Read a SIPNET ``.param`` file and return a ``{sipnet_name: value}`` dict.

    Comments (``!``) and extra columns beyond the second are ignored.
    """
    result: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.split("!")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            result[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return result
