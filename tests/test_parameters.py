"""Parameter model tests.

These tests do not require a compiled SIPNET binary.
"""

import pytest
from pydantic import ValidationError

from pysipnet.parameters.base import ParameterDomain, get_parameter_specs
from pysipnet.parameters.model import (
    AllocationParams,
    ModelFlags,
    PhotosynthesisParams,
    SIPNETParameters,
)


class TestParameterSpec:
    def test_all_fields_have_spec(self):
        specs = get_parameter_specs(SIPNETParameters)
        assert len(specs) > 0
        for path, spec in specs.items():
            assert spec.unit, f"{path}: unit string is empty"
            assert spec.domain in ParameterDomain, f"{path}: invalid domain"
            assert spec.description, f"{path}: description is empty"

    def test_per_year_params_are_flagged(self):
        specs = get_parameter_specs(SIPNETParameters)
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


class TestModelFlagsDefaults:
    def test_standard_named_constructor(self):
        flags = ModelFlags.standard()
        assert flags.snow
        assert flags.gdd
        assert flags.water_hresp
        assert not flags.litter_pool
        assert not flags.growth_resp

    def test_forest_named_constructor(self):
        assert ModelFlags.forest().litter_pool

    def test_bare_constructor_matches_standard_flags(self):
        """ModelFlags() and ModelFlags.standard() differ only by the label."""
        assert ModelFlags().to_config_keys() == ModelFlags.standard().to_config_keys()

    def test_new_processes_are_off_by_default(self):
        """Nitrogen, methane and flooding must stay opt-in.

        These arrived with the SIPNET v2.1.0 pin. If any of them defaulted on,
        existing parameter sets would suddenly be incomplete.
        """
        flags = ModelFlags()
        assert not flags.nitrogen_cycle
        assert not flags.anaerobic
        assert not flags.flooding


class TestModelFlagsRestrictions:
    """SIPNET refuses three flag combinations; we must refuse them first.

    Each of these mirrors a check in ``validateContext()`` in SIPNET's
    ``src/common/context.c``. Catching them in Python turns an opaque exit
    code into a message naming the flags involved.
    """

    def test_gdd_and_soil_phenol_are_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="cannot both be on"):
            ModelFlags(gdd=True, soil_phenol=True)

    def test_soil_phenol_alone_is_allowed(self):
        assert ModelFlags(gdd=False, soil_phenol=True).soil_phenol

    def test_anaerobic_requires_water_hresp(self):
        with pytest.raises(ValidationError, match="anaerobic requires water_hresp"):
            ModelFlags(anaerobic=True, water_hresp=False)

    def test_anaerobic_with_water_hresp_is_allowed(self):
        assert ModelFlags(anaerobic=True, water_hresp=True).anaerobic

    def test_nitrogen_cycle_requires_litter_pool_and_anaerobic(self):
        with pytest.raises(ValidationError, match="nitrogen_cycle requires"):
            ModelFlags(nitrogen_cycle=True)

    def test_nitrogen_cycle_requires_anaerobic_too(self):
        with pytest.raises(ValidationError, match="nitrogen_cycle requires"):
            ModelFlags(nitrogen_cycle=True, litter_pool=True)

    def test_full_nitrogen_configuration_is_allowed(self):
        flags = ModelFlags(nitrogen_cycle=True, litter_pool=True, anaerobic=True)
        assert flags.nitrogen_cycle

    def test_all_problems_are_reported_together(self):
        """A caller fixing several mistakes should see them in one message."""
        with pytest.raises(ValidationError) as exc:
            ModelFlags(gdd=True, soil_phenol=True, anaerobic=True, water_hresp=False)
        message = str(exc.value)
        assert "cannot both be on" in message
        assert "anaerobic requires water_hresp" in message


class TestModelFlagsConfigKeys:
    """to_config_keys produces what SIPNET actually reads from sipnet.in."""

    def test_every_flag_is_written(self):
        """Writing all keys, not just the non-default ones, keeps runs reproducible."""
        keys = ModelFlags().to_config_keys()
        flag_fields = {
            name for name in ModelFlags.model_fields if name != "name"
        }
        assert len(keys) == len(flag_fields)

    def test_keys_are_sipnet_uppercase_names(self):
        keys = ModelFlags().to_config_keys()
        assert "LITTER_POOL" in keys
        assert "WATER_HRESP" in keys
        assert "NITROGEN_CYCLE" in keys

    def test_values_are_integers_not_booleans(self):
        """SIPNET parses these with strtol, so they must render as 1/0."""
        for value in ModelFlags().to_config_keys().values():
            assert value in (0, 1)
            assert not isinstance(value, bool) or isinstance(value, int)

    def test_reflects_the_flag_values(self):
        assert ModelFlags.standard().to_config_keys()["LITTER_POOL"] == 0
        assert ModelFlags.forest().to_config_keys()["LITTER_POOL"] == 1

    def test_label_is_not_a_config_key(self):
        """The name is for humans; SIPNET would reject it as an unknown key."""
        keys = ModelFlags(name="my-site").to_config_keys()
        assert "NAME" not in keys
        assert "my-site" not in keys.values()


class TestModelFlagsName:
    def test_defaults_to_unset(self):
        assert ModelFlags().name is None

    def test_named_constructors_set_it(self):
        assert ModelFlags.standard().name == "standard"
        assert ModelFlags.forest().name == "forest"

    def test_can_be_set_directly(self):
        assert ModelFlags(litter_pool=True, name="niwot-forest").name == "niwot-forest"

    def test_does_not_change_the_model_configuration(self):
        labelled = ModelFlags(litter_pool=True, name="anything")
        unlabelled = ModelFlags(litter_pool=True)
        assert labelled.to_config_keys() == unlabelled.to_config_keys()

    def test_participates_in_equality(self):
        """Documented consequence of keeping the label on the model itself."""
        assert ModelFlags(name="a") != ModelFlags(name="b")
        assert ModelFlags.standard() != ModelFlags()


class TestModelFlagsSerialisation:
    def test_roundtrips_through_a_dict(self):
        flags = ModelFlags.forest()
        assert ModelFlags.model_validate(flags.model_dump()) == flags

    def test_roundtrips_through_json(self):
        flags = ModelFlags(nitrogen_cycle=True, litter_pool=True, anaerobic=True, name="n-cycle")
        assert ModelFlags.model_validate_json(flags.model_dump_json()) == flags

    def test_restrictions_are_enforced_on_load(self):
        """A hand-edited or corrupted saved config must not load silently."""
        with pytest.raises(ValidationError):
            ModelFlags.model_validate({"gdd": True, "soil_phenol": True})


class TestSIPNETParameters:
    def test_construction(self, minimal_params):
        assert minimal_params.photosynthesis.a_max == 112.0

    def test_allocation_triangle_constraint(self, minimal_params):
        data = minimal_params.model_dump()
        data["allocation"]["fine_root_allocation"] = 0.8
        data["allocation"]["wood_allocation"] = 0.3
        with pytest.raises(ValidationError):
            SIPNETParameters.model_validate(data)

    def test_serialisation_roundtrip(self, minimal_params):
        dumped = minimal_params.model_dump()
        restored = SIPNETParameters.model_validate(dumped)
        assert restored.photosynthesis.a_max == minimal_params.photosynthesis.a_max
        assert restored.water.snow_melt == minimal_params.water.snow_melt

    def test_validate_for_flags_snow_missing(self, minimal_params):
        data = minimal_params.model_dump()
        data["water"]["snow_melt"] = None
        params = SIPNETParameters.model_validate(data)
        with pytest.raises(ValueError, match="snow_melt"):
            params.validate_for_flags(ModelFlags.standard())

    def test_validate_for_flags_litter_missing(self, minimal_params):
        data = minimal_params.model_dump()
        data["respiration"]["litter_breakdown_rate"] = None
        params = SIPNETParameters.model_validate(data)
        with pytest.raises(ValueError, match="litter_breakdown_rate"):
            params.validate_for_flags(ModelFlags.forest())

    def test_validate_for_flags_standard_ok(self, minimal_params):
        minimal_params.validate_for_flags(ModelFlags.standard())
