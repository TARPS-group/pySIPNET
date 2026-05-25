"""Parameter model tests.

These tests do not require a compiled SIPNET binary.
"""

import pytest
from pydantic import ValidationError

from pysipnet.parameters.base import ParameterDomain, get_parameter_specs
from pysipnet.parameters.v1 import (
    AllocationParams,
    ModelFlagsV1,
    PhotosynthesisParams,
    SIPNETParametersV1,
)


class TestParameterSpec:
    def test_all_fields_have_spec(self):
        specs = get_parameter_specs(SIPNETParametersV1)
        assert len(specs) > 0
        for path, spec in specs.items():
            assert spec.unit, f"{path}: unit string is empty"
            assert spec.domain in ParameterDomain, f"{path}: invalid domain"
            assert spec.description, f"{path}: description is empty"

    def test_per_year_params_are_flagged(self):
        specs = get_parameter_specs(SIPNETParametersV1)
        per_year = {k for k, s in specs.items() if s.per_year}
        expected = {
            "respiration.base_veg_resp",
            "respiration.base_fine_root_resp",
            "respiration.base_coarse_root_resp",
            "respiration.base_soil_resp",
            "respiration.litter_breakdown_rate",
            "allocation.fine_root_turnover_rate",
            "allocation.coarse_root_turnover_rate",
            "allocation.wood_turnover_rate",
            "phenology.leaf_turnover_rate",
        }
        assert expected.issubset(per_year), f"Missing per_year flags: {expected - per_year}"

    def test_positive_params_reject_zero(self):
        with pytest.raises(ValidationError):
            PhotosynthesisParams(
                a_max=0.0,  # must be > 0
                a_max_frac=0.76,
                base_fol_resp_frac=0.1,
                psn_t_min=2.0,
                psn_t_opt=24.0,
                d_vpd_slope=0.05,
                d_vpd_exp=1.0,
                half_sat_par=300.0,
                attenuation=0.5,
            )

    def test_unit_interval_rejects_out_of_range(self):
        with pytest.raises(ValidationError):
            AllocationParams(
                fine_root_allocation=0.9,  # fine + wood > 1
                wood_allocation=0.9,
                fine_root_turnover_rate=1.0,
                coarse_root_turnover_rate=0.1,
                wood_turnover_rate=0.02,
            )


class TestModelFlagsV1:
    def test_standard_preset(self):
        flags = ModelFlagsV1.standard()
        assert flags.snow
        assert flags.gdd
        assert flags.water_hresp
        assert not flags.litter_pool
        assert not flags.growth_resp

    def test_forest_preset(self):
        flags = ModelFlagsV1.forest()
        assert flags.litter_pool

    def test_gdd_soil_phenol_exclusive(self):
        with pytest.raises(ValidationError):
            ModelFlagsV1(gdd=True, soil_phenol=True)

    def test_gdd_off_soil_phenol_on(self):
        flags = ModelFlagsV1(gdd=False, soil_phenol=True)
        assert flags.soil_phenol

    def test_serialisation_roundtrip(self):
        flags = ModelFlagsV1.forest()
        assert ModelFlagsV1.model_validate(flags.model_dump()) == flags


class TestSIPNETParametersV1:
    def test_construction(self, minimal_params):
        assert minimal_params.photosynthesis.a_max == 112.0

    def test_allocation_triangle_constraint(self, minimal_params):
        data = minimal_params.model_dump()
        data["allocation"]["fine_root_allocation"] = 0.8
        data["allocation"]["wood_allocation"] = 0.3
        with pytest.raises(ValidationError):
            SIPNETParametersV1.model_validate(data)

    def test_serialisation_roundtrip(self, minimal_params):
        dumped = minimal_params.model_dump()
        restored = SIPNETParametersV1.model_validate(dumped)
        assert restored.photosynthesis.a_max == minimal_params.photosynthesis.a_max
        assert restored.water.snow_melt == minimal_params.water.snow_melt

    def test_validate_for_flags_snow_missing(self, minimal_params):
        data = minimal_params.model_dump()
        data["water"]["snow_melt"] = None
        params = SIPNETParametersV1.model_validate(data)
        with pytest.raises(ValueError, match="snow_melt"):
            params.validate_for_flags(ModelFlagsV1.standard())

    def test_validate_for_flags_litter_missing(self, minimal_params):
        data = minimal_params.model_dump()
        data["respiration"]["litter_breakdown_rate"] = None
        params = SIPNETParametersV1.model_validate(data)
        with pytest.raises(ValueError, match="litter_breakdown_rate"):
            params.validate_for_flags(ModelFlagsV1.forest())

    def test_validate_for_flags_standard_ok(self, minimal_params):
        minimal_params.validate_for_flags(ModelFlagsV1.standard())
