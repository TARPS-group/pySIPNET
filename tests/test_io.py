"""Unit tests for the param IO layer (pysipnet.io.param_io).

The climate IO roundtrip is already covered by tests/test_climate.py.
The output reader is exercised by the integration tests.  This file
focuses on the param writer/reader, which has no other direct coverage.
"""

from __future__ import annotations

import pytest

from pysipnet.io.param_io import (
    _OBSOLETE_DEFAULTS,
    PYTHON_TO_SIPNET,
    SIPNET_TO_PYTHON,
    _flatten,
    read_param_file,
    write_param_file,
)
from pysipnet.parameters.v1 import ModelFlagsV1


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flags():
    return ModelFlagsV1.standard()


# ---------------------------------------------------------------------------
# Name mapping tables
# ---------------------------------------------------------------------------

class TestNameMappings:
    def test_python_to_sipnet_non_empty(self):
        assert len(PYTHON_TO_SIPNET) > 0

    def test_sipnet_to_python_is_exact_inverse(self):
        assert SIPNET_TO_PYTHON == {v: k for k, v in PYTHON_TO_SIPNET.items()}

    def test_no_duplicate_sipnet_names(self):
        values = list(PYTHON_TO_SIPNET.values())
        assert len(values) == len(set(values))

    def test_all_paths_have_exactly_one_dot(self):
        for path in PYTHON_TO_SIPNET:
            assert path.count(".") == 1, f"Expected single dot in {path!r}"

    def test_obsolete_defaults_names_not_in_main_mapping(self):
        """Obsolete params must not accidentally shadow a real param."""
        main_names = set(PYTHON_TO_SIPNET.values())
        for name in _OBSOLETE_DEFAULTS:
            assert name not in main_names, f"{name!r} appears in both mappings"


# ---------------------------------------------------------------------------
# _flatten
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_returns_dict_of_floats(self, minimal_params):
        flat = _flatten(minimal_params)
        assert all(isinstance(v, (int, float)) for v in flat.values())

    def test_keys_are_sipnet_camel_case(self, minimal_params):
        flat = _flatten(minimal_params)
        for key in flat:
            assert key in PYTHON_TO_SIPNET.values(), f"{key!r} not a known SIPNET name"

    def test_none_fields_omitted(self, minimal_params):
        """snow_melt has a value but snow=False fields like leafPoolDepth are None."""
        flat = _flatten(minimal_params)
        # leaf_pool_depth defaults to None and should not appear
        assert "leafPoolDepth" not in flat

    def test_non_none_optional_included(self, minimal_params):
        """snow_melt=0.15 is set in minimal_params — must appear in flat output."""
        flat = _flatten(minimal_params)
        assert "snowMelt" in flat
        assert flat["snowMelt"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# write_param_file
# ---------------------------------------------------------------------------

class TestWriteParamFile:
    def test_creates_file(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        assert path.exists()

    def test_file_is_nonempty(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        assert path.stat().st_size > 0

    def test_comment_line_present(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        assert any(line.startswith("!") for line in path.read_text().splitlines())

    def test_all_obsolete_defaults_written(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        written = read_param_file(path)
        for name in _OBSOLETE_DEFAULTS:
            assert name in written, f"Obsolete param {name!r} missing from file"

    def test_obsolete_values_match_defaults(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        written = read_param_file(path)
        for name, expected in _OBSOLETE_DEFAULTS.items():
            assert written[name] == pytest.approx(expected)

    def test_none_fields_not_written(self, tmp_path, minimal_params, flags):
        """Fields with None values must not appear in the file at all."""
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        written = read_param_file(path)
        assert "leafPoolDepth" not in written

    def test_known_param_value_correct(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        written = read_param_file(path)
        assert written["aMax"] == pytest.approx(112.0)
        assert written["soilWHC"] == pytest.approx(12.0)

    def test_validate_for_flags_called(self, tmp_path, flags):
        """write_param_file must fail if snow_melt is missing with SNOW=1."""
        from pysipnet.parameters.v1 import (
            AllocationParams, InitialConditions, LeafPhysiologyParams,
            PhenologyParams, PhotosynthesisParams, RespirationParams,
            SIPNETParametersV1, WaterParams,
        )
        params = SIPNETParametersV1(
            initial_conditions=InitialConditions(
                plant_wood=1.0, lai=0.0, soil=1.0, soil_water_frac=0.5,
                fine_root_frac=0.05, coarse_root_frac=0.15,
            ),
            photosynthesis=PhotosynthesisParams(
                a_max=100.0, a_max_frac=0.76, base_fol_resp_frac=0.1,
                psn_t_min=2.0, psn_t_opt=24.0, d_vpd_slope=0.05,
                d_vpd_exp=1.0, half_sat_par=300.0, attenuation=0.5,
            ),
            phenology=PhenologyParams(
                leaf_off_day=270.0, gdd_leaf_on=100.0,
                leaf_growth=50.0, frac_leaf_fall=0.95,
                leaf_allocation=0.25, leaf_turnover_rate=1.0,
            ),
            respiration=RespirationParams(
                base_veg_resp=0.02, veg_resp_q10=2.0, growth_resp_frac=0.0,
                frozen_soil_fol_r_eff=0.5, frozen_soil_threshold=-1.0,
                base_fine_root_resp=0.5, base_coarse_root_resp=0.1,
                fine_root_q10=2.0, coarse_root_q10=2.0,
                base_soil_resp=0.06, soil_resp_q10=2.0,
                soil_resp_moist_effect=1.5,
            ),
            allocation=AllocationParams(
                fine_root_allocation=0.35, wood_allocation=0.30,
                fine_root_turnover_rate=1.0, coarse_root_turnover_rate=0.1,
                wood_turnover_rate=0.02,
            ),
            water=WaterParams(
                water_remove_frac=0.1, frozen_soil_eff=0.1, wue_const=10.0,
                soil_whc=12.0, litter_whc=5.0, immed_evap_frac=0.1,
                fast_flow_frac=0.1, rd_const=100.0,
                r_soil_const1=3.0, r_soil_const2=2.0,
                # snow_melt is None — should fail with SNOW=True flags
            ),
            leaf=LeafPhysiologyParams(leaf_c_sp_wt=32.0, c_frac_leaf=0.45),
        )
        with pytest.raises(ValueError, match="snow_melt"):
            write_param_file(params, flags, tmp_path / "sipnet.param")


# ---------------------------------------------------------------------------
# read_param_file
# ---------------------------------------------------------------------------

class TestReadParamFile:
    def test_reads_simple_file(self, tmp_path):
        path = tmp_path / "test.param"
        path.write_text("aMax\t112.0\naMaxFrac\t0.76\n")
        result = read_param_file(path)
        assert result["aMax"] == pytest.approx(112.0)
        assert result["aMaxFrac"] == pytest.approx(0.76)

    def test_ignores_comment_lines(self, tmp_path):
        path = tmp_path / "test.param"
        path.write_text("! this is a comment\naMax\t112.0\n")
        result = read_param_file(path)
        assert "aMax" in result
        assert len(result) == 1

    def test_ignores_inline_comments(self, tmp_path):
        path = tmp_path / "test.param"
        path.write_text("aMax\t112.0\t! some note\n")
        result = read_param_file(path)
        assert result["aMax"] == pytest.approx(112.0)

    def test_ignores_extra_columns(self, tmp_path):
        """Legacy 5-column format: name value changeable min max."""
        path = tmp_path / "test.param"
        path.write_text("aMax\t112.0\t1\t50.0\t200.0\n")
        result = read_param_file(path)
        assert result["aMax"] == pytest.approx(112.0)

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "test.param"
        path.write_text("\naMax\t112.0\n\naMaxFrac\t0.76\n\n")
        result = read_param_file(path)
        assert len(result) == 2

    def test_skips_non_numeric_value(self, tmp_path):
        path = tmp_path / "test.param"
        path.write_text("aMax\tNOT_A_NUMBER\ngddLeafOn\t100.0\n")
        result = read_param_file(path)
        assert "aMax" not in result
        assert result["gddLeafOn"] == pytest.approx(100.0)

    def test_tab_and_space_delimited(self, tmp_path):
        path = tmp_path / "test.param"
        path.write_text("aMax 112.0\naMaxFrac\t0.76\n")
        result = read_param_file(path)
        assert result["aMax"] == pytest.approx(112.0)
        assert result["aMaxFrac"] == pytest.approx(0.76)


# ---------------------------------------------------------------------------
# write → read roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_all_non_none_params_survive_roundtrip(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        written = read_param_file(path)

        flat = _flatten(minimal_params)
        for sipnet_name, expected in flat.items():
            assert sipnet_name in written, f"{sipnet_name!r} missing after roundtrip"
            assert written[sipnet_name] == pytest.approx(expected, rel=1e-6), (
                f"{sipnet_name}: expected {expected}, got {written[sipnet_name]}"
            )

    def test_written_file_is_human_readable(self, tmp_path, minimal_params, flags):
        """Every non-comment, non-blank line should parse as 'name value'."""
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        for raw_line in path.read_text().splitlines():
            line = raw_line.split("!")[0].strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) >= 2, f"Unparseable line: {raw_line!r}"
            float(parts[1])  # should not raise
