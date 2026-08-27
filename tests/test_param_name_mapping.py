"""The parameter-name mapping, written out independently of the code.

`PYTHON_TO_SIPNET` in `pysipnet/io/param_io.py` is the only thing that decides
which SIPNET parameter each Python field becomes. Nothing else in the test
suite can catch an error in it, because the test helper that reads the
reference `.param` file reads *through the same table* the writer writes
through — so a swapped or misspelled pair cancels itself out and every test
still passes while every run gets the wrong value.

This file breaks that circle. The table below is written out by hand rather
than derived, so it is an independent statement of what the mapping should be.
Changing `PYTHON_TO_SIPNET` without changing this file fails, which is the
point: a mapping change should be deliberate and reviewed, because there is no
other signal that it is wrong.

When a mapping genuinely changes, update this table in the same commit and say
why in the message.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pysipnet.io.param_io import PYTHON_TO_SIPNET

EXPECTED_PARAM_NAMES: dict[str, str] = {
    # initial_conditions
    "initial_conditions.plant_wood": "plantWoodInit",
    "initial_conditions.lai": "laiInit",
    "initial_conditions.litter": "litterInit",
    "initial_conditions.soil": "soilInit",
    "initial_conditions.soil_water_frac": "soilWFracInit",
    "initial_conditions.snow": "snowInit",
    "initial_conditions.fine_root_frac": "fineRootFrac",
    "initial_conditions.coarse_root_frac": "coarseRootFrac",
    # photosynthesis
    "photosynthesis.a_max": "aMax",
    "photosynthesis.a_max_frac": "aMaxFrac",
    "photosynthesis.base_fol_resp_frac": "baseFolRespFrac",
    "photosynthesis.psn_t_min": "psnTMin",
    "photosynthesis.psn_t_opt": "psnTOpt",
    "photosynthesis.d_vpd_slope": "dVpdSlope",
    "photosynthesis.d_vpd_exp": "dVpdExp",
    "photosynthesis.half_sat_par": "halfSatPar",
    "photosynthesis.attenuation": "attenuation",
    # phenology
    "phenology.leaf_on_day": "leafOnDay",
    "phenology.leaf_off_day": "leafOffDay",
    "phenology.gdd_leaf_on": "gddLeafOn",
    "phenology.soil_temp_leaf_on": "soilTempLeafOn",
    "phenology.leaf_growth": "leafGrowth",
    "phenology.frac_leaf_fall": "fracLeafFall",
    "phenology.leaf_allocation": "leafAllocation",
    "phenology.leaf_turnover_rate": "leafTurnoverRate",
    "phenology.leaf_on_realloc_frac": "leafOnReallocFrac",
    # respiration
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
    # allocation
    "allocation.fine_root_allocation": "fineRootAllocation",
    "allocation.wood_allocation": "woodAllocation",
    "allocation.fine_root_turnover_rate": "fineRootTurnoverRate",
    "allocation.coarse_root_turnover_rate": "coarseRootTurnoverRate",
    "allocation.wood_turnover_rate": "woodTurnoverRate",
    # water
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
    # leaf
    "leaf.leaf_c_sp_wt": "leafCSpWt",
    "leaf.c_frac_leaf": "cFracLeaf",
}


def _names_registered_by_sipnet() -> set[str]:
    """Every parameter name the pinned SIPNET source registers."""
    source = Path(__file__).parent.parent / "sipnet" / "src"
    if not source.exists():
        pytest.skip("SIPNET submodule not populated")
    pattern = re.compile(r'initializeOneModelParam\(\s*\w+\s*,\s*"([A-Za-z_0-9]+)"')
    names: set[str] = set()
    for path in source.rglob("*.c"):
        names.update(pattern.findall(path.read_text()))
    return names


class TestParameterNameMapping:
    """Pin every mapping independently of the code that defines it."""

    def test_mapping_matches_the_expected_table_exactly(self):
        """A swapped pair here would otherwise be invisible.

        Swapping psnTMin and psnTOpt, for instance, gives every run the wrong
        photosynthesis temperature response and passes every other test.
        """
        assert PYTHON_TO_SIPNET == EXPECTED_PARAM_NAMES, (
            "the parameter-name mapping changed. If that was deliberate, update "
            "EXPECTED_PARAM_NAMES in this file in the same commit and explain why."
        )

    @pytest.mark.parametrize(("python_path", "sipnet_name"), sorted(EXPECTED_PARAM_NAMES.items()))
    def test_each_name_is_one_sipnet_registers(self, python_path, sipnet_name, request):
        """Guards the table itself against naming something SIPNET lacks."""
        assert sipnet_name in _names_registered_by_sipnet(), (
            f"{python_path} maps to {sipnet_name!r}, which the pinned SIPNET "
            "does not register. An unknown name is only a warning in SIPNET, so "
            "the parameter would silently have no effect."
        )

    def test_no_two_fields_map_to_the_same_sipnet_name(self):
        names = list(EXPECTED_PARAM_NAMES.values())
        duplicates = {n for n in names if names.count(n) > 1}
        assert not duplicates, f"two Python fields map to the same SIPNET name: {duplicates}"

    def test_every_python_path_is_a_real_field(self):
        """A path naming a field that does not exist would be written as nothing."""
        from pysipnet.parameters.model import SIPNETParameters

        for python_path in EXPECTED_PARAM_NAMES:
            group, field = python_path.split(".", 1)
            assert group in SIPNETParameters.model_fields, f"no such group: {group}"
            group_model = SIPNETParameters.model_fields[group].annotation
            assert field in group_model.model_fields, f"no such field: {python_path}"
