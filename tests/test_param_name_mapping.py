"""The parameter-name mapping, written out independently of the code.

`PYTHON_TO_SIPNET` in `pysipnet/io/param_io.py`, derived from each field's
`ParameterSpec.sipnet_name`, is the only thing that decides which SIPNET
parameter each Python field becomes. Nothing else in the test
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
    "initial_conditions.total_wood_carbon": "plantWoodInit",
    "initial_conditions.leaf_area_index": "laiInit",
    "initial_conditions.litter_carbon": "litterInit",
    "initial_conditions.soil_carbon": "soilInit",
    "initial_conditions.soil_wetness_fraction": "soilWFracInit",
    "initial_conditions.snow_water_equivalent": "snowInit",
    "initial_conditions.fine_root_fraction": "fineRootFrac",
    "initial_conditions.coarse_root_fraction": "coarseRootFrac",
    # photosynthesis
    "photosynthesis.max_photosynthesis_rate": "aMax",
    "photosynthesis.daily_mean_photosynthesis_fraction": "aMaxFrac",
    "photosynthesis.foliar_respiration_fraction": "baseFolRespFrac",
    "photosynthesis.min_photosynthesis_temperature": "psnTMin",
    "photosynthesis.optimum_photosynthesis_temperature": "psnTOpt",
    "photosynthesis.vapour_pressure_deficit_slope": "dVpdSlope",
    "photosynthesis.vapour_pressure_deficit_exponent": "dVpdExp",
    "photosynthesis.half_saturation_light": "halfSatPar",
    "photosynthesis.light_extinction_coefficient": "attenuation",
    # phenology
    "phenology.leaf_on_day": "leafOnDay",
    "phenology.leaf_off_day": "leafOffDay",
    "phenology.leaf_on_growing_degree_days": "gddLeafOn",
    "phenology.leaf_on_soil_temperature": "soilTempLeafOn",
    "phenology.leaf_on_growth": "leafGrowth",
    "phenology.leaf_off_fall_fraction": "fracLeafFall",
    "phenology.leaf_allocation": "leafAllocation",
    "phenology.leaf_turnover_rate": "leafTurnoverRate",
    "phenology.leaf_on_reallocation_fraction": "leafOnReallocFrac",
    # respiration
    "respiration.base_wood_respiration_rate": "baseVegResp",
    "respiration.wood_respiration_q10": "vegRespQ10",
    "respiration.growth_respiration_fraction": "growthRespFrac",
    "respiration.frozen_soil_foliar_respiration_factor": "frozenSoilFolREff",
    "respiration.frozen_soil_threshold": "frozenSoilThreshold",
    "respiration.base_fine_root_respiration_rate": "baseFineRootResp",
    "respiration.base_coarse_root_respiration_rate": "baseCoarseRootResp",
    "respiration.fine_root_respiration_q10": "fineRootQ10",
    "respiration.coarse_root_respiration_q10": "coarseRootQ10",
    "respiration.base_soil_respiration_rate": "baseSoilResp",
    "respiration.soil_respiration_q10": "soilRespQ10",
    "respiration.soil_respiration_moisture_exponent": "soilRespMoistEffect",
    "respiration.litter_breakdown_rate": "litterBreakdownRate",
    "respiration.litter_respired_fraction": "fracLitterRespired",
    # allocation
    "allocation.fine_root_allocation": "fineRootAllocation",
    "allocation.wood_allocation": "woodAllocation",
    "allocation.fine_root_turnover_rate": "fineRootTurnoverRate",
    "allocation.coarse_root_turnover_rate": "coarseRootTurnoverRate",
    "allocation.wood_turnover_rate": "woodTurnoverRate",
    # water
    "water.water_removal_fraction": "waterRemoveFrac",
    "water.frozen_soil_water_fraction": "frozenSoilEff",
    "water.water_use_efficiency": "wueConst",
    "water.soil_water_holding_capacity": "soilWHC",
    "water.interception_evaporation_fraction": "immedEvapFrac",
    "water.fast_flow_fraction": "fastFlowFrac",
    "water.snow_melt_rate": "snowMelt",
    "water.aerodynamic_resistance_constant": "rdConst",
    "water.soil_resistance_intercept": "rSoilConst1",
    "water.soil_resistance_slope": "rSoilConst2",
    "water.leaf_water_pool_depth": "leafPoolDepth",
    # leaf
    "leaf.leaf_carbon_per_area": "leafCSpWt",
    "leaf.leaf_carbon_fraction": "cFracLeaf",
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
