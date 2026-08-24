"""Parameter model tests.

These tests do not require a compiled SIPNET binary.
"""

import pytest
from pydantic import ValidationError

from pysipnet.parameters.base import ParameterDomain, get_parameter_specs
from pysipnet.parameters.model import (
    UNSUPPORTED_FLAGS,
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

    def test_anaerobic_and_nitrogen_rules_are_still_correct(self):
        """Check the SIPNET-mirroring rules that the unsupported-flag gate hides.

        ``anaerobic`` and ``nitrogen_cycle`` cannot be switched on through the
        constructor at present, so these two rules are unreachable that way.
        They still have to be right for when the flags become usable, so call
        the validator directly on an instance built without validation.
        """
        # anaerobic without water_hresp
        flags = ModelFlags.model_construct(anaerobic=True, water_hresp=False)
        with pytest.raises(ValueError, match="anaerobic requires water_hresp"):
            flags._check_flag_restrictions()

        # nitrogen_cycle without either dependency
        flags = ModelFlags.model_construct(
            nitrogen_cycle=True, litter_pool=False, anaerobic=False, gdd=True, soil_phenol=False
        )
        with pytest.raises(ValueError, match="nitrogen_cycle requires"):
            flags._check_flag_restrictions()

        # nitrogen_cycle with only one dependency
        flags = ModelFlags.model_construct(
            nitrogen_cycle=True, litter_pool=True, anaerobic=False, gdd=True, soil_phenol=False
        )
        with pytest.raises(ValueError, match="nitrogen_cycle requires"):
            flags._check_flag_restrictions()

        # both dependencies present: the rule is satisfied
        flags = ModelFlags.model_construct(
            nitrogen_cycle=True,
            litter_pool=True,
            anaerobic=True,
            water_hresp=True,
            gdd=True,
            soil_phenol=False,
        )
        assert flags._check_flag_restrictions() is flags

    def test_all_problems_are_reported_together(self):
        """A caller fixing several mistakes should see them in one message."""
        flags = ModelFlags.model_construct(
            gdd=True, soil_phenol=True, anaerobic=True, water_hresp=False
        )
        with pytest.raises(ValueError) as exc:
            flags._check_flag_restrictions()
        message = str(exc.value)
        assert "cannot both be on" in message
        assert "anaerobic requires water_hresp" in message


class TestUnsupportedFlags:
    """Flags SIPNET supports but pySIPNET cannot yet supply parameters for.

    Without this gate the flag reaches SIPNET, which stops with "Did not find
    required parameter" — a failure far from its cause, and one the parameter
    model exists to prevent. These tests pin both the refusal and the quality
    of the message, since the message is the whole point.
    """

    @pytest.mark.parametrize("flag", sorted(UNSUPPORTED_FLAGS))
    def test_each_unsupported_flag_is_refused(self, flag):
        with pytest.raises(ValidationError, match="not supported by pySIPNET yet"):
            ModelFlags(**{flag: True})

    @pytest.mark.parametrize("flag", sorted(UNSUPPORTED_FLAGS))
    def test_message_names_the_flag_and_its_parameters(self, flag):
        """A caller must learn what is missing, not merely that something is."""
        with pytest.raises(ValidationError) as exc:
            ModelFlags(**{flag: True})
        message = str(exc.value)
        assert flag in message
        for param in UNSUPPORTED_FLAGS[flag][1]:
            assert param in message, f"{param} missing from the error message"

    def test_message_says_sipnet_itself_supports_it(self):
        """The limitation is ours, and saying so saves a hunt through SIPNET."""
        with pytest.raises(ValidationError, match="even though SIPNET itself supports them"):
            ModelFlags(flooding=True)

    def test_message_says_how_to_add_support(self):
        with pytest.raises(ValidationError, match="UNSUPPORTED_FLAGS"):
            ModelFlags(flooding=True)

    def test_several_unsupported_flags_reported_together(self):
        with pytest.raises(ValidationError) as exc:
            ModelFlags(flooding=True, anaerobic=True)
        message = str(exc.value)
        assert "flooding" in message
        assert "anaerobic" in message

    def test_refusal_takes_precedence_over_dependency_advice(self):
        """nitrogen_cycle alone must not be answered with 'set litter_pool'.

        That advice is a dead end: satisfying it still leaves the flag
        unsupported. The more fundamental problem has to be reported first.
        """
        with pytest.raises(ValidationError) as exc:
            ModelFlags(nitrogen_cycle=True)
        message = str(exc.value)
        assert "not supported by pySIPNET yet" in message
        assert "nitrogen_cycle requires both litter_pool" not in message

    def test_supported_flags_are_unaffected(self):
        """The gate must not catch anything it should not."""
        assert ModelFlags(litter_pool=True, growth_resp=True, leaf_water=True).litter_pool

    def test_flags_default_to_off_so_the_gate_is_invisible(self):
        for flag in UNSUPPORTED_FLAGS:
            assert getattr(ModelFlags(), flag) is False

    def test_a_saved_config_with_an_unsupported_flag_is_refused_on_load(self):
        """Deserialisation must not be a way around the gate."""
        with pytest.raises(ValidationError, match="not supported"):
            ModelFlags.model_validate({"flooding": True})

    def test_every_sipnet_parameter_is_either_modelled_or_listed(self, sipnet_source_params):
        """Nothing SIPNET registers may go unaccounted for.

        If a pin bump adds a parameter, it is either modelled or it belongs to
        a flag we refuse. Anything else is a silent gap: SIPNET would require
        it under some configuration and we would have no way to supply it.
        """
        from pysipnet.io.param_io import PYTHON_TO_SIPNET

        modelled = set(PYTHON_TO_SIPNET.values())
        listed = {p for _, params in UNSUPPORTED_FLAGS.values() for p in params}
        unaccounted = sipnet_source_params - modelled - listed
        assert not unaccounted, (
            f"SIPNET registers {sorted(unaccounted)}, which pySIPNET neither models "
            "nor lists in UNSUPPORTED_FLAGS."
        )

    def test_we_do_not_write_parameters_sipnet_does_not_have(self, sipnet_source_params):
        """The other direction: an unknown name is only a warning in SIPNET."""
        from pysipnet.io.param_io import PYTHON_TO_SIPNET

        unknown = set(PYTHON_TO_SIPNET.values()) - sipnet_source_params
        assert not unknown, f"pySIPNET writes parameters SIPNET does not know: {sorted(unknown)}"

    def test_the_table_matches_what_sipnet_actually_requires(self, sipnet_source_params):
        """Every parameter named in the message must be real, and still absent.

        Guards two ways of going stale: naming a parameter SIPNET does not
        have, and keeping a flag listed after its parameters were modelled.
        """
        from pysipnet.io.param_io import PYTHON_TO_SIPNET

        modelled = set(PYTHON_TO_SIPNET.values())
        for flag, (_, params) in UNSUPPORTED_FLAGS.items():
            for param in params:
                assert param in sipnet_source_params, (
                    f"{flag}: {param} is not a parameter SIPNET v2.1.0 registers"
                )
                assert param not in modelled, (
                    f"{flag}: {param} is modelled now — remove {flag} from UNSUPPORTED_FLAGS"
                )


class TestModelFlagsConfigKeys:
    """to_config_keys produces what SIPNET actually reads from sipnet.in."""

    def test_every_flag_is_written(self):
        """Writing all keys, not just the non-default ones, keeps runs reproducible."""
        keys = ModelFlags().to_config_keys()
        flag_fields = {name for name in ModelFlags.model_fields if name != "name"}
        assert len(keys) == len(flag_fields)

    def test_keys_are_sipnet_uppercase_names(self):
        keys = ModelFlags().to_config_keys()
        assert "LITTER_POOL" in keys
        assert "WATER_HRESP" in keys
        assert "NITROGEN_CYCLE" in keys

    def test_values_are_integers_not_booleans(self):
        """SIPNET parses these with strtol, so they must render as 1/0.

        `type(value) is int` rather than isinstance: bool subclasses int, so an
        isinstance check passes for True/False and tests nothing.
        """
        for key, value in ModelFlags().to_config_keys().items():
            assert type(value) is int, f"{key} is {type(value).__name__}, not int"
            assert value in (0, 1)

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


class TestModelFlagsAreImmutable:
    """Validation runs at construction, so the model must not be mutable.

    A mutable model lets an unsupported flag be switched on after the checks
    have run, defeating them entirely and producing the deep SIPNET failure
    they exist to prevent.
    """

    def test_assignment_is_refused(self):
        flags = ModelFlags.standard()
        with pytest.raises(ValidationError):
            flags.flooding = True

    def test_a_supported_flag_cannot_be_reassigned_either(self):
        """Not about which flag; the object is a run specification."""
        flags = ModelFlags.standard()
        with pytest.raises(ValidationError):
            flags.litter_pool = True

    def test_config_keys_cannot_be_changed_after_construction(self):
        flags = ModelFlags.standard()
        before = flags.to_config_keys()
        with pytest.raises(ValidationError):
            flags.nitrogen_cycle = True
        assert flags.to_config_keys() == before

    def test_flags_are_hashable(self):
        """A useful consequence: they can key a cache of ensemble runs."""
        assert len({ModelFlags.standard(), ModelFlags.standard()}) == 1
        assert len({ModelFlags.standard(), ModelFlags.forest()}) == 2


class TestModelFlagsSerialisation:
    def test_roundtrips_through_a_dict(self):
        flags = ModelFlags.forest()
        assert ModelFlags.model_validate(flags.model_dump()) == flags

    def test_roundtrips_through_json(self):
        flags = ModelFlags(litter_pool=True, growth_resp=True, name="forest-growth-resp")
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


class TestValidateForFlags:
    """Every flag-conditional parameter must be checked before SIPNET runs.

    This is design principle 2 in practice: a parameter SIPNET will demand
    should be reported as missing here, naming the field, rather than as an
    exit code from a subprocess. Each branch is exercised because an unchecked
    one produces exactly the deep failure the method exists to avoid.
    """

    def _without(self, params, group, field):
        """Return a copy of *params* with one field cleared."""
        data = params.model_dump()
        data[group][field] = None
        return type(params).model_validate(data)

    @pytest.mark.parametrize(
        ("flags", "group", "field"),
        [
            (ModelFlags(snow=True), "water", "snow_melt"),
            (ModelFlags(leaf_water=True), "water", "leaf_pool_depth"),
            (ModelFlags(litter_pool=True), "respiration", "litter_breakdown_rate"),
            (ModelFlags(litter_pool=True), "respiration", "frac_litter_respired"),
            (ModelFlags(gdd=True), "phenology", "gdd_leaf_on"),
            (ModelFlags(gdd=False, soil_phenol=True), "phenology", "soil_temp_leaf_on"),
        ],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_missing_required_parameter_is_reported(self, minimal_params, flags, group, field):
        stripped = self._without(minimal_params, group, field)
        with pytest.raises(ValueError, match=field):
            stripped.validate_for_flags(flags)

    def test_leaf_on_day_is_required_when_neither_trigger_is_on(self, minimal_params):
        """With gdd and soil_phenol both off, leaf-out falls back to a fixed day."""
        flags = ModelFlags(gdd=False, soil_phenol=False)
        stripped = self._without(minimal_params, "phenology", "leaf_on_day")
        with pytest.raises(ValueError, match="leaf_on_day"):
            stripped.validate_for_flags(flags)

    def test_a_parameter_is_not_demanded_when_its_flag_is_off(self, minimal_params):
        """The complement: an off flag must not make its parameter required."""
        stripped = self._without(minimal_params, "water", "leaf_pool_depth")
        stripped.validate_for_flags(ModelFlags(leaf_water=False))

    def test_all_missing_parameters_are_reported_together(self, minimal_params):
        """One round trip should surface every problem, not the first."""
        data = minimal_params.model_dump()
        data["respiration"]["litter_breakdown_rate"] = None
        data["respiration"]["frac_litter_respired"] = None
        stripped = type(minimal_params).model_validate(data)
        with pytest.raises(ValueError) as exc:
            stripped.validate_for_flags(ModelFlags(litter_pool=True))
        message = str(exc.value)
        assert "litter_breakdown_rate" in message
        assert "frac_litter_respired" in message

    def test_a_complete_parameter_set_passes(self, minimal_params):
        minimal_params.validate_for_flags(ModelFlags.standard())
