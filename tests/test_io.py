"""Unit tests for the param IO layer (pysipnet.io.param_io).

The climate IO roundtrip is already covered by tests/test_climate.py.
The output reader is exercised by the integration tests.  This file
focuses on the param writer/reader, which has no other direct coverage.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from pysipnet.io.param_io import (
    PYTHON_TO_SIPNET,
    SIPNET_REQUIRES_DESPITE_DEFAULT,
    SIPNET_TO_PYTHON,
    UnknownParameterWarning,
    _flatten,
    read_param_file,
    read_parameters,
    write_param_file,
)
from pysipnet.parameters.model import ModelFlags, SIPNETParameters

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flags():
    return ModelFlags.standard()


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
        """snow_melt_rate has a value but snow=False fields like leafPoolDepth are None."""
        flat = _flatten(minimal_params)
        # leaf_water_pool_depth defaults to None and should not appear
        assert "leafPoolDepth" not in flat

    def test_non_none_optional_included(self, minimal_params):
        """snow_melt_rate=0.15 is set in minimal_params — must appear in flat output."""
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
        """write_param_file must fail if snow_melt_rate is missing with SNOW=1."""
        from pysipnet.parameters.model import (
            AllocationParams,
            InitialConditions,
            LeafPhysiologyParams,
            PhenologyParams,
            PhotosynthesisParams,
            RespirationParams,
            SIPNETParameters,
            WaterParams,
        )

        params = SIPNETParameters(
            initial_conditions=InitialConditions(
                total_wood_carbon=1.0,
                leaf_area_index=0.0,
                soil_carbon=1.0,
                soil_wetness_fraction=0.5,
                fine_root_fraction=0.05,
                coarse_root_fraction=0.15,
            ),
            photosynthesis=PhotosynthesisParams(
                max_photosynthesis_rate=100.0,
                daily_mean_photosynthesis_fraction=0.76,
                foliar_respiration_fraction=0.1,
                min_photosynthesis_temperature=2.0,
                optimum_photosynthesis_temperature=24.0,
                vapor_pressure_deficit_slope=0.05,
                vapor_pressure_deficit_exponent=1.0,
                half_saturation_light=300.0,
                light_extinction_coefficient=0.5,
            ),
            phenology=PhenologyParams(
                leaf_off_day=270.0,
                leaf_on_growing_degree_days=100.0,
                leaf_on_growth=50.0,
                leaf_off_fall_fraction=0.95,
                leaf_allocation=0.25,
                leaf_turnover_rate=1.0,
                leaf_on_reallocation_fraction=0.2,
            ),
            respiration=RespirationParams(
                base_wood_respiration_rate=0.02,
                wood_respiration_q10=2.0,
                growth_respiration_fraction=0.0,
                frozen_soil_foliar_respiration_factor=0.5,
                frozen_soil_threshold=-1.0,
                base_fine_root_respiration_rate=0.5,
                base_coarse_root_respiration_rate=0.1,
                fine_root_respiration_q10=2.0,
                coarse_root_respiration_q10=2.0,
                base_soil_respiration_rate=0.06,
                soil_respiration_q10=2.0,
                soil_respiration_moisture_exponent=1.5,
            ),
            allocation=AllocationParams(
                fine_root_allocation=0.35,
                wood_allocation=0.30,
                fine_root_turnover_rate=1.0,
                coarse_root_turnover_rate=0.1,
                wood_turnover_rate=0.02,
            ),
            water=WaterParams(
                water_removal_fraction=0.1,
                frozen_soil_water_fraction=0.1,
                water_use_efficiency=10.0,
                soil_water_holding_capacity=12.0,
                interception_evaporation_fraction=0.1,
                fast_flow_fraction=0.1,
                aerodynamic_resistance_constant=100.0,
                soil_resistance_intercept=3.0,
                soil_resistance_slope=2.0,
                # snow_melt_rate is None — should fail with SNOW=True flags
            ),
            leaf=LeafPhysiologyParams(leaf_carbon_per_area=32.0, leaf_carbon_fraction=0.45),
        )
        with pytest.raises(ValueError, match="snow_melt_rate"):
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

    def test_keeps_the_file_s_spelling_of_names(self, tmp_path):
        path = tmp_path / "test.param"
        path.write_text("AMAX\t112.0\n")
        assert read_param_file(path) == {"AMAX": 112.0}

    def test_reads_crlf_line_endings(self, tmp_path):
        path = tmp_path / "test.param"
        path.write_bytes(b"aMax\t112.0\r\naMaxFrac 0.76\r\n")
        assert read_param_file(path) == {"aMax": 112.0, "aMaxFrac": 0.76}

    def test_an_unregistered_name_may_repeat(self, tmp_path):
        """SIPNET ignores a name it does not register, however often it appears."""
        path = tmp_path / "test.param"
        path.write_text("microbeInit 0.5\nmicrobeInit 0.6\n")
        assert read_param_file(path) == {"microbeInit": 0.6}


class TestReadParamFileRefusals:
    """What SIPNET refuses, crashes on or misreads silently, refused by line."""

    @pytest.mark.parametrize(
        ("contents", "message"),
        [
            pytest.param("aMax *\n", "spatially-varying marker", id="star"),
            pytest.param("aMax 1\naMax 2\n", "already set on line 1", id="duplicate"),
            pytest.param("aMax 1\nAMAX 2\n", "case-insensitive", id="duplicate-other-case"),
            pytest.param("waterDrainFrac 1\nwaterDrainFrac 2\n", "already set", id="dup-unmodeled"),
            pytest.param("aMax\n", "has no value", id="no-value"),
            pytest.param("aMax ! 112\n", "has no value", id="value-commented-out"),
            pytest.param("aMax abc\n", "not a decimal number", id="word"),
            pytest.param("aMax 8.3x\n", "not a decimal number", id="trailing-garbage"),
            pytest.param("aMax 1_000\n", "not a decimal number", id="python-underscore"),
            pytest.param("aMax 0x10\n", "not a decimal number", id="hex"),
            pytest.param("aMax nan\n", "must be finite", id="nan"),
            pytest.param("aMax -inf\n", "must be finite", id="inf"),
            pytest.param("aMax 1" + "0" * 40 + "\n", "overflows", id="long-value"),
            pytest.param("x" * 64 + " 1\n", "overflows", id="long-name"),
            pytest.param("aMax 1 !" + "!" * 260 + "\n", "255-character", id="long-line"),
        ],
    )
    def test_refused(self, tmp_path, contents, message):
        path = tmp_path / "test.param"
        path.write_text(contents)
        with pytest.raises(ValueError, match=message) as info:
            read_param_file(path)
        assert f"{path}:" in str(info.value)

    def test_the_line_limit_is_sipnet_s_buffer(self, tmp_path):
        """fgets(line, 256) holds 255 characters; a line that long is read whole."""
        path = tmp_path / "test.param"
        line = "aMax 1 !"
        path.write_text(line + "!" * (255 - len(line)) + "\n")
        assert read_param_file(path) == {"aMax": 1.0}

    @pytest.mark.parametrize("token", ["1", "-1", "+1.5", ".5", "5.", "1e3", "1.5E-2"])
    def test_decimal_forms_are_read(self, tmp_path, token):
        path = tmp_path / "test.param"
        path.write_text(f"aMax {token}\n")
        assert read_param_file(path) == {"aMax": float(token)}

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


# ---------------------------------------------------------------------------
# read_parameters: the file back into the model
# ---------------------------------------------------------------------------

_EVERY_OPTIONAL_VALUE = {
    ("phenology", "leaf_on_day"): 140.0,
    ("phenology", "leaf_on_soil_temperature"): 8.5,
    ("respiration", "litter_breakdown_rate"): 0.4,
    ("respiration", "litter_respired_fraction"): 0.6,
    ("water", "leaf_water_pool_depth"): 0.05,
}


@pytest.fixture
def every_optional_params(minimal_params):
    """minimal_params with every optional field set, so nothing is left out of the file."""
    data = minimal_params.model_dump()
    for (group, field), value in _EVERY_OPTIONAL_VALUE.items():
        data[group][field] = value
    params = SIPNETParameters.model_validate(data)
    unset = [
        f"{group}.{field}"
        for group, values in params.model_dump().items()
        for field, value in values.items()
        if value is None
    ]
    assert not unset, f"extend _EVERY_OPTIONAL_VALUE with {unset}"
    return params


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def _written_lines(params, flags, tmp_path) -> list[str]:
    path = tmp_path / "written.param"
    write_param_file(params, flags, path)
    return [line for line in path.read_text().splitlines() if not line.startswith("!")]


class TestReadParameters:
    def test_round_trip_minimal(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        assert read_parameters(path).model_dump() == minimal_params.model_dump()

    def test_round_trip_every_optional(self, tmp_path, every_optional_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(every_optional_params, flags, path)
        assert read_parameters(path).model_dump() == every_optional_params.model_dump()

    def test_round_trip_is_exact_for_awkward_floats(self, tmp_path, minimal_params, flags):
        data = minimal_params.model_dump()
        data["photosynthesis"]["max_photosynthesis_rate"] = 0.1 + 0.2
        data["water"]["soil_water_holding_capacity"] = 1e-300
        params = SIPNETParameters.model_validate(data)
        path = tmp_path / "sipnet.param"
        write_param_file(params, flags, path)
        assert read_parameters(path).model_dump() == params.model_dump()

    def test_classmethod_takes_a_string(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        assert SIPNETParameters.from_param_file(str(path)) == minimal_params

    def test_names_match_case_insensitively(self, tmp_path, minimal_params, flags):
        lines = [line.upper() for line in _written_lines(minimal_params, flags, tmp_path)]
        path = _write_lines(tmp_path / "upper.param", lines)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert read_parameters(path) == minimal_params

    def test_a_clean_file_warns_nothing(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            read_parameters(path)

    def test_absent_optional_fields_are_none(self, tmp_path, minimal_params, flags):
        path = tmp_path / "sipnet.param"
        write_param_file(minimal_params, flags, path)
        params = read_parameters(path)
        assert params.phenology.leaf_on_day is None
        assert params.water.leaf_water_pool_depth is None

    def test_every_missing_required_parameter_is_listed_by_both_names(
        self, tmp_path, minimal_params, flags
    ):
        dropped = ("aMax", "soilWHC")
        lines = [
            line
            for line in _written_lines(minimal_params, flags, tmp_path)
            if line.split()[0] not in dropped
        ]
        path = _write_lines(tmp_path / "short.param", lines)
        with pytest.raises(ValueError, match="lacks 2 required") as info:
            read_parameters(path)
        message = str(info.value)
        assert "aMax (photosynthesis.max_photosynthesis_rate)" in message
        assert "soilWHC (water.soil_water_holding_capacity)" in message

    @pytest.mark.parametrize("python_path", SIPNET_REQUIRES_DESPITE_DEFAULT)
    def test_a_default_is_not_filled_in_where_sipnet_requires_the_name(
        self, tmp_path, minimal_params, flags, python_path
    ):
        sipnet_name = PYTHON_TO_SIPNET[python_path]
        lines = [
            line
            for line in _written_lines(minimal_params, flags, tmp_path)
            if line.split()[0] != sipnet_name
        ]
        path = _write_lines(tmp_path / "short.param", lines)
        with pytest.raises(ValueError, match=re.escape(f"{sipnet_name} ({python_path})")):
            read_parameters(path)

    def test_a_default_sipnet_shares_is_filled_in(self, tmp_path, minimal_params, flags):
        """growthRespFrac is required only under growth_resp; SIPNET runs on 0 otherwise."""
        lines = [
            line
            for line in _written_lines(minimal_params, flags, tmp_path)
            if line.split()[0] != "growthRespFrac"
        ]
        path = _write_lines(tmp_path / "short.param", lines)
        assert read_parameters(path).respiration.growth_respiration_fraction == 0.0

    def test_a_flag_dependent_parameter_may_be_absent(self, tmp_path, minimal_params):
        """soilRespMoistEffect is needed only under water_hresp; without it the field is None."""
        flags = ModelFlags(water_hresp=False)
        data = minimal_params.model_dump()
        data["respiration"]["soil_respiration_moisture_exponent"] = None
        params = SIPNETParameters.model_validate(data)
        path = tmp_path / "sipnet.param"
        write_param_file(params, flags, path)
        assert "soilRespMoistEffect" not in path.read_text()
        assert read_parameters(path) == params

    def test_a_value_outside_its_domain_is_named_by_both_names(
        self, tmp_path, minimal_params, flags
    ):
        lines = [
            "aMax -5" if line.split()[0] == "aMax" else line
            for line in _written_lines(minimal_params, flags, tmp_path)
        ]
        path = _write_lines(tmp_path / "bad.param", lines)
        with pytest.raises(ValueError, match="invalid parameters") as info:
            read_parameters(path)
        assert "aMax (photosynthesis.max_photosynthesis_rate) = -5.0" in str(info.value)
        assert isinstance(info.value.__cause__, ValidationError)

    def test_the_allocation_triangle_is_checked(self, tmp_path, minimal_params, flags):
        lines = [
            "woodAllocation 0.6" if line.split()[0] == "woodAllocation" else line
            for line in _written_lines(minimal_params, flags, tmp_path)
        ]
        path = _write_lines(tmp_path / "bad.param", lines)
        with pytest.raises(ValueError, match="Coarse-root allocation"):
            read_parameters(path)

    def test_a_malformed_file_is_refused_as_read_param_file_refuses_it(
        self, tmp_path, minimal_params, flags
    ):
        lines = [*_written_lines(minimal_params, flags, tmp_path), "AMAX 1"]
        path = _write_lines(tmp_path / "dup.param", lines)
        with pytest.raises(ValueError, match="SIPNET refuses a parameter given twice"):
            read_parameters(path)


class TestUnknownParameterWarning:
    def _read_with_extra(self, tmp_path, params, flags, extra):
        lines = [*_written_lines(params, flags, tmp_path), *extra]
        path = _write_lines(tmp_path / "extra.param", lines)
        with pytest.warns(UnknownParameterWarning) as record:
            result = read_parameters(path)
        assert len(record) == 1
        return result, str(record[0].message)

    def test_unregistered_names_are_dropped_and_listed(self, tmp_path, minimal_params, flags):
        result, message = self._read_with_extra(
            tmp_path, minimal_params, flags, ["microbeInit 0.5", "psnTMax 40"]
        )
        assert result == minimal_params
        assert "SIPNET does not register these either" in message
        assert "microbeInit, psnTMax" in message
        assert "flags pySIPNET refuses" not in message

    def test_unmodeled_names_say_which_flag_reads_them(self, tmp_path, minimal_params, flags):
        result, message = self._read_with_extra(
            tmp_path, minimal_params, flags, ["waterDrainFrac 0.1", "LEAFCN 30"]
        )
        assert result == minimal_params
        assert "waterDrainFrac (flooding)" in message
        assert "LEAFCN (nitrogen_cycle)" in message
        assert "SIPNET does not register these either" not in message

    def test_the_warning_comes_before_a_missing_parameter_error(
        self, tmp_path, minimal_params, flags
    ):
        """A misspelled required name is both unknown and missing; the warning explains it."""
        lines = [
            "aMaxx 112" if line.split()[0] == "aMax" else line
            for line in _written_lines(minimal_params, flags, tmp_path)
        ]
        path = _write_lines(tmp_path / "typo.param", lines)
        with pytest.warns(UnknownParameterWarning, match="aMaxx"):
            with pytest.raises(ValueError, match="aMax"):
                read_parameters(path)


class TestRequiredAgreesWithSipnet:
    """The reader requires what SIPNET requires unconditionally, and no default hides it."""

    @pytest.fixture(scope="class")
    def unconditionally_required(self) -> set[str]:
        source = Path(__file__).parent.parent / "sipnet" / "src" / "sipnet" / "sipnet.c"
        if not source.exists():
            pytest.skip("SIPNET submodule not populated")
        pattern = re.compile(r'initializeOneModelParam\(\s*\w+\s*,\s*"(\w+)",[^;]*,\s*1\s*\);')
        names = set(pattern.findall(source.read_text()))
        assert names, "no unconditional registrations found; has the C API changed?"
        return names

    def test_every_field_with_a_default_is_listed_exactly_when_sipnet_requires_it(
        self, unconditionally_required
    ):
        from pysipnet.parameters.model import PARAMETER_SPECS

        for python_path, spec in PARAMETER_SPECS.items():
            group, field = python_path.split(".")
            info = SIPNETParameters.model_fields[group].annotation.model_fields[field]
            if info.is_required() or info.default is None:
                continue
            listed = python_path in SIPNET_REQUIRES_DESPITE_DEFAULT
            required = spec.sipnet_name in unconditionally_required
            assert listed == required, (
                f"{python_path} ({spec.sipnet_name}) defaults to {info.default!r}; SIPNET "
                f"{'requires' if required else 'does not require'} it unconditionally"
            )

    def test_the_reader_requires_exactly_what_sipnet_requires_unconditionally(
        self, unconditionally_required
    ):
        from pysipnet.io.param_io import _REQUIRED_FIELDS

        required = {PYTHON_TO_SIPNET[f"{group}.{field}"] for group, field in _REQUIRED_FIELDS}
        assert required == unconditionally_required

    def test_nothing_sipnet_requires_can_be_left_out(
        self, tmp_path, minimal_params, flags, unconditionally_required
    ):
        written = {line.split()[0] for line in _written_lines(minimal_params, flags, tmp_path)}
        for name in sorted(unconditionally_required):
            assert name in written
            lines = [
                line
                for line in _written_lines(minimal_params, flags, tmp_path)
                if line.split()[0] != name
            ]
            path = _write_lines(tmp_path / f"without_{name}.param", lines)
            with pytest.raises(ValueError, match=f"{name} \\("):
                read_parameters(path)


# ---------------------------------------------------------------------------
# read_output_file — header detection
# ---------------------------------------------------------------------------


_EXPECTED_COLUMNS = [
    "year",
    "day_of_year",
    "hour_of_day",
    "wood_carbon",
    "net_ecosystem_exchange",
]


class TestOutputHeaderDetection:
    """The reader must find the header in all three layouts SIPNET produces.

    Getting this wrong is quiet rather than loud. When the reader mistakes the
    header for something else it falls back to numbered columns, so every
    lookup by name fails later and far from the cause — or worse, the header
    row is parsed as data and every column is off by one row.
    """

    HEADER = "year day time plantWoodC nee"
    ROW_1 = "1998 305 0.00 5759.61 0.742"
    ROW_2 = "1998 306 0.00 5760.10 0.751"

    def _write(self, tmp_path, *lines):
        path = tmp_path / "sipnet.out"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_header_then_data(self, tmp_path):
        """The layout the pinned SIPNET version writes."""
        from pysipnet.io.output_reader import read_output_file

        df = read_output_file(self._write(tmp_path, self.HEADER, self.ROW_1, self.ROW_2))
        assert list(df.columns) == _EXPECTED_COLUMNS
        assert len(df) == 2

    def test_notes_line_then_header_then_data(self, tmp_path):
        """The layout SIPNET wrote up to and including v2.0.0."""
        from pysipnet.io.output_reader import read_output_file

        df = read_output_file(
            self._write(tmp_path, "Notes: (PlantWoodC in g C/m^2;", self.HEADER, self.ROW_1)
        )
        assert list(df.columns) == _EXPECTED_COLUMNS
        assert len(df) == 1

    def test_data_only(self, tmp_path):
        """A binary run with --no-print-header; columns can only be positional."""
        from pysipnet.io.output_reader import read_output_file

        df = read_output_file(self._write(tmp_path, self.ROW_1, self.ROW_2))
        assert list(df.columns) == [0, 1, 2, 3, 4]
        assert len(df) == 2

    def test_header_row_is_not_counted_as_data(self, tmp_path):
        """The classic off-by-one: a header parsed as a timestep."""
        from pysipnet.io.output_reader import read_output_file

        df = read_output_file(self._write(tmp_path, self.HEADER, self.ROW_1, self.ROW_2))
        assert df["year"].tolist() == [1998, 1998]

    def test_unmapped_column_keeps_its_sipnet_name_and_warns(self, tmp_path):
        """A column added by a future SIPNET version must still be readable.

        It must also be noticed: a silently passed-through column would never
        get a description or units, so the reader warns.
        """
        from pysipnet.io.output_reader import UnknownOutputColumnWarning, read_output_file

        with pytest.warns(UnknownOutputColumnWarning, match="somethingNew"):
            df = read_output_file(
                self._write(tmp_path, "year day time somethingNew", "1998 305 0.00 1.5")
            )
        assert "somethingNew" in df.columns

    def test_legacy_column_reads_under_a_convention_name(self, tmp_path):
        """Output saved from SIPNET v2.1.0 still reads, without a warning."""
        import warnings

        from pysipnet.io.output_reader import read_output_file

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = read_output_file(
                self._write(tmp_path, "year day time bcdeltaC", "1998 305 0.00 0.0")
            )
        assert "carbon_balance_error" in df.columns

    def test_empty_file_gives_an_empty_frame(self, tmp_path):
        from pysipnet.io.output_reader import read_output_file

        path = tmp_path / "sipnet.out"
        path.write_text("")
        assert read_output_file(path).empty

    def test_variable_selection_keeps_the_time_coordinates(self, tmp_path):
        """year/day_of_year/hour_of_day identify each row, so they survive any selection."""
        from pysipnet.io.output_reader import read_output_file

        df = read_output_file(self._write(tmp_path, self.HEADER, self.ROW_1), variables=["nee"])
        assert set(df.columns) == {"year", "day_of_year", "hour_of_day", "net_ecosystem_exchange"}

    def test_variable_selection_accepts_names_aliases_and_sipnet_tokens(self, tmp_path):
        from pysipnet.io.output_reader import read_output_file

        path = self._write(tmp_path, self.HEADER, self.ROW_1)
        for name in ("nee", "NEE", "net_ecosystem_exchange"):
            df = read_output_file(path, variables=[name])
            assert "net_ecosystem_exchange" in df.columns

    def test_unknown_variable_selection_is_an_error(self, tmp_path):
        from pysipnet.io.output_reader import read_output_file

        with pytest.raises(KeyError, match="not a SIPNET output variable"):
            read_output_file(self._write(tmp_path, self.HEADER, self.ROW_1), variables=["wood"])


class TestNonFiniteValuesAreRefused:
    """NaN and inf must never reach SIPNET.

    SIPNET parses both with strtod and runs to completion. A NaN temperature
    parameter gives a whole run of zero productivity, exit code 0, and no
    warning — output that looks entirely plausible. In a calibration loop a NaN
    proposal becomes a finite, wrong likelihood instead of an error.
    """

    def test_nan_is_refused_at_construction(self):
        from pysipnet.parameters.model import PhotosynthesisParams

        with pytest.raises(ValidationError):
            PhotosynthesisParams(
                max_photosynthesis_rate=112.0,
                daily_mean_photosynthesis_fraction=0.76,
                foliar_respiration_fraction=0.1,
                min_photosynthesis_temperature=float("nan"),
                optimum_photosynthesis_temperature=24.0,
                vapor_pressure_deficit_slope=0.05,
                vapor_pressure_deficit_exponent=1.0,
                half_saturation_light=17.0,
                light_extinction_coefficient=0.5,
            )

    def test_inf_is_refused_at_construction(self):
        from pysipnet.parameters.model import PhotosynthesisParams

        with pytest.raises(ValidationError):
            PhotosynthesisParams(
                max_photosynthesis_rate=float("inf"),
                daily_mean_photosynthesis_fraction=0.76,
                foliar_respiration_fraction=0.1,
                min_photosynthesis_temperature=2.0,
                optimum_photosynthesis_temperature=24.0,
                vapor_pressure_deficit_slope=0.05,
                vapor_pressure_deficit_exponent=1.0,
                half_saturation_light=17.0,
                light_extinction_coefficient=0.5,
            )

    def test_writer_refuses_a_non_finite_value_that_skipped_validation(
        self, tmp_path, minimal_params
    ):
        """model_construct bypasses validators, so the writer checks too."""
        from pysipnet.parameters.model import ModelFlags

        data = minimal_params.model_dump()
        data["photosynthesis"]["max_photosynthesis_rate"] = float("nan")
        sneaked = type(minimal_params).model_construct(
            **{
                k: type(getattr(minimal_params, k)).model_construct(**v)
                if isinstance(v, dict)
                else v
                for k, v in data.items()
            }
        )
        with pytest.raises(ValueError, match="Refusing to write"):
            write_param_file(sneaked, ModelFlags.standard(), tmp_path / "sipnet.param")

    def test_a_numpy_scalar_is_written_as_a_plain_number(self, tmp_path, minimal_params):
        """repr() of a numpy scalar is 'np.float64(8.3)', which SIPNET reads as 0."""
        import numpy as np

        from pysipnet.parameters.model import ModelFlags

        data = minimal_params.model_dump()
        data["photosynthesis"]["max_photosynthesis_rate"] = np.float64(112.5)
        params = type(minimal_params).model_validate(data)
        path = tmp_path / "sipnet.param"
        write_param_file(params, ModelFlags.standard(), path)
        line = next(ln for ln in path.read_text().splitlines() if ln.startswith("aMax"))
        assert line.split()[1] == "112.5", f"unparseable value written: {line!r}"
