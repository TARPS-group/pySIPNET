"""The output variable registry: its contract with SIPNET and with itself.

Three things can go wrong quietly here, and each gets a test:

- SIPNET adds, drops or renames a column. The run succeeds, the reader keeps
  the unknown name, and nothing downstream has a description or units for it.
  ``test_binary_header_matches_registry`` runs the pinned binary and compares
  its header token-for-token with the registry.
- A mapping in the registry is wrong. Every other test reads *through* the
  registry, so a swapped pair cancels itself out. ``EXPECTED_OUTPUT_NAMES``
  below is written out by hand as an independent statement of the mapping.
- A name, alias or unit string drifts from the convention. Those are checked
  on every spec.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysipnet.variables import (
    LEGACY_OUTPUT_COLUMNS,
    NAME_PATTERN,
    OUTPUT_VARIABLES,
    OUTPUT_VARIABLES_BY_NAME,
    TIME_COORDINATE_NAMES,
    Aggregation,
    VariableKind,
    VariableSpec,
    output_variable_records,
    resolve_output_variable,
    resolve_output_variable_names,
)

GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "niwot_standard.out.csv"

# Written by hand from outputHeader() in sipnet/src/sipnet/sipnet.c. Do not
# derive this from the registry; the point is that it is independent.
EXPECTED_OUTPUT_NAMES: dict[str, str] = {
    "year": "year",
    "day": "day_of_year",
    "time": "hour_of_day",
    "plantWoodC": "wood_carbon",
    "plantLeafC": "leaf_carbon",
    "woodCreation": "wood_growth",
    "soil": "soil_carbon",
    "coarseRootC": "coarse_root_carbon",
    "fineRootC": "fine_root_carbon",
    "litter": "litter_carbon",
    "soilWater": "soil_water",
    "soilWetnessFrac": "soil_wetness_fraction",
    "snow": "snow_water_equivalent",
    "npp": "net_primary_production",
    "nee": "net_ecosystem_exchange",
    "cumNEE": "cumulative_net_ecosystem_exchange",
    "gpp": "gross_primary_production",
    "rAboveground": "above_ground_respiration",
    "rSoil": "soil_respiration",
    "rRoot": "root_respiration",
    "ra": "autotrophic_respiration",
    "rh": "heterotrophic_respiration",
    "rtot": "ecosystem_respiration",
    "evapotranspiration": "evapotranspiration",
    "fluxestranspiration": "transpiration_rate",
    "minN": "mineral_nitrogen",
    "soilOrgN": "soil_organic_nitrogen",
    "litterN": "litter_nitrogen",
    "plantStorageN": "plant_nitrogen_storage",
    "n2o": "nitrogen_volatilization",
    "nLeaching": "nitrogen_leaching",
    "nFixation": "nitrogen_fixation",
    "nUptake": "nitrogen_uptake",
    "ch4": "methane_production",
    "nppStorage": "wood_storage_carbon",
}

# Word fragments the naming convention forbids. Each is a truncation or an
# acronym that a reader would have to decode.
FORBIDDEN_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "frac", "resp", "temp", "psn", "whc", "vpd", "par", "lai", "nee", "gpp",
        "npp", "et", "wue", "cum", "swe", "evap", "init", "const", "eff", "wt",
        "sp", "veg", "fol", "rd", "tot", "ra", "rh", "n", "c", "ch4", "n2o",
    }
)  # fmt: skip


# ---------------------------------------------------------------------------
# The mapping, stated independently
# ---------------------------------------------------------------------------


def test_registry_matches_the_independent_mapping():
    actual = {v.sipnet_name: v.name for v in OUTPUT_VARIABLES}
    assert actual == EXPECTED_OUTPUT_NAMES


def test_registry_order_is_sipnet_column_order():
    """SIPNET writes columns in a fixed order; the registry keeps it so docs read like the file."""
    assert [v.sipnet_name for v in OUTPUT_VARIABLES] == list(EXPECTED_OUTPUT_NAMES)


def test_golden_fixture_header_is_the_registry():
    golden_header = pd.read_csv(GOLDEN, nrows=0).columns.tolist()
    assert golden_header == [v.name for v in OUTPUT_VARIABLES]


# ---------------------------------------------------------------------------
# Conventions every spec must follow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", OUTPUT_VARIABLES, ids=lambda s: s.name)
def test_name_follows_convention(spec: VariableSpec):
    assert NAME_PATTERN.match(spec.name)
    tokens = set(spec.name.split("_"))
    assert not (tokens & FORBIDDEN_NAME_TOKENS), f"{spec.name} contains a forbidden token"


@pytest.mark.parametrize("name", LEGACY_OUTPUT_COLUMNS.values())
def test_legacy_names_follow_convention(name: str):
    assert NAME_PATTERN.match(name)
    assert not (set(name.split("_")) & FORBIDDEN_NAME_TOKENS)


@pytest.mark.parametrize("spec", OUTPUT_VARIABLES, ids=lambda s: s.name)
def test_spec_is_complete(spec: VariableSpec):
    assert spec.description.strip()
    assert spec.long_label.strip()
    assert spec.group
    if spec.kind is VariableKind.TIMESTEP_START_COORDINATE:
        assert spec.aggregation is Aggregation.NONE
    else:
        assert spec.output_decimals is not None, "every numeric column has a printf precision"
    if spec.constituent:
        assert spec.units != "1", "a dimensionless quantity has no constituent"


@pytest.mark.parametrize("spec", OUTPUT_VARIABLES, ids=lambda s: s.name)
def test_labels_render(spec: VariableSpec):
    label = spec.axis_label()
    assert label.startswith(spec.long_label)
    assert "(" in label and label.endswith(")")
    assert spec.axis_label("latex")
    assert spec.axis_label("html")


@pytest.mark.parametrize("spec", OUTPUT_VARIABLES, ids=lambda s: s.name)
def test_requires_flag_names_a_model_flag(spec: VariableSpec):
    """A typo in requires_flag would otherwise pass every test."""
    from pysipnet.parameters.model import ModelFlags

    assert spec.requires_flag is None or spec.requires_flag in ModelFlags.model_fields


def test_legacy_columns_are_selectable_by_either_name():
    """Output saved by SIPNET v2.1.0 can still be read column-by-column."""
    assert resolve_output_variable_names(["bcdeltaC", "carbon_balance_error", "nee"]) == [
        "carbon_balance_error",
        "net_ecosystem_exchange",
    ]


def test_units_must_be_udunits_syntax():
    from pysipnet.units import validate_units

    for bad in ("g/m2", "m**-2", "g m^-2", "kg C m-2"):
        with pytest.raises(ValueError):
            validate_units(bad)
    for good in ("g m-2 d-1", "1", "degC", "m2 m-2", "mg g-1 kPa"):
        validate_units(good)


def test_aliases_are_unique_and_resolve_to_their_owner():
    seen: dict[str, str] = {}
    for spec in OUTPUT_VARIABLES:
        for key in (spec.name, spec.sipnet_name, *spec.aliases):
            assert seen.setdefault(key, spec.name) == spec.name, f"{key} claimed twice"
            assert resolve_output_variable(key) is spec


def test_previous_pysipnet_names_still_resolve():
    """The names in use before the registry existed, so old scripts fail loudly at most once."""
    old_to_new = {
        "plant_wood_c": "wood_carbon",
        "soil_c": "soil_carbon",
        "litter_c": "litter_carbon",
        "cum_nee": "cumulative_net_ecosystem_exchange",
        "r_soil": "soil_respiration",
        "transpiration": "transpiration_rate",
        "npp_storage": "wood_storage_carbon",
        "mineral_n": "mineral_nitrogen",
    }
    for old, new in old_to_new.items():
        assert resolve_output_variable(old).name == new


def test_resolve_unknown_name_suggests_near_matches():
    with pytest.raises(KeyError, match="wood_carbon"):
        resolve_output_variable("wood")


def test_resolve_names_dedupes_and_keeps_order():
    assert resolve_output_variable_names(["gpp", "NEE", "net_ecosystem_exchange", "GPP"]) == [
        "gross_primary_production",
        "net_ecosystem_exchange",
    ]


def test_time_coordinates():
    assert TIME_COORDINATE_NAMES == ("year", "day_of_year", "hour_of_day")


def test_records_are_json_serialisable():
    records = output_variable_records()
    json.dumps(records)
    assert len(records) == len(OUTPUT_VARIABLES)
    nee = next(r for r in records if r["name"] == "net_ecosystem_exchange")
    assert nee["formatted_units"] == "g C m⁻²"
    assert nee["aggregation"] == "sum"


def test_spec_refuses_bad_names_and_units():
    good = OUTPUT_VARIABLES_BY_NAME["soil_water"]
    with pytest.raises(ValueError, match="naming convention"):
        VariableSpec(**{**good.__dict__, "name": "soilWater"})
    with pytest.raises(ValueError, match="substance token"):
        VariableSpec(**{**good.__dict__, "name": "soil_thing", "units": "g C m-2"})
    with pytest.raises(ValueError, match="not a valid unit"):
        VariableSpec(**{**good.__dict__, "name": "soil_thing", "units": "blorf m-2"})


# ---------------------------------------------------------------------------
# Precision: SIPNET's fixed printf formats
# ---------------------------------------------------------------------------


def test_golden_values_respect_declared_precision():
    """No column in real output carries more decimals than the registry says it can."""
    text = GOLDEN.read_text().splitlines()
    header = text[0].split(",")
    max_decimals = dict.fromkeys(header, 0)
    for line in text[1:]:
        for col, value in zip(header, line.split(","), strict=True):
            if "." in value:
                max_decimals[col] = max(max_decimals[col], len(value.split(".")[1]))
    for spec in OUTPUT_VARIABLES:
        if spec.output_decimals is None:
            assert max_decimals[spec.name] == 0
        else:
            assert max_decimals[spec.name] <= spec.output_decimals, spec.name


# ---------------------------------------------------------------------------
# xarray attributes and Dataset construction (no binary needed)
# ---------------------------------------------------------------------------


def test_xarray_attributes_carry_the_time_reference():
    attrs = OUTPUT_VARIABLES_BY_NAME["wood_carbon"].xarray_attributes()
    assert attrs["time_reference"] == "value at the end of the timestep"
    assert attrs["cell_methods"] == "time: point"
    attrs = OUTPUT_VARIABLES_BY_NAME["net_ecosystem_exchange"].xarray_attributes()
    assert attrs["cell_methods"] == "time: sum"
    assert "sign_convention" in attrs
    assert attrs["units"] == "g m-2" and attrs["constituent"] == "C"


def _frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2020] * n,
            "day_of_year": [1, 1, 2, 2],
            "hour_of_day": [0.0, 12.0, 0.0, 12.0],
            "net_ecosystem_exchange": np.arange(n, dtype=float),
            "wood_carbon": 100.0,
            "somethingNew": 1.0,
        }
    )


def test_dataset_is_one_dimensional_in_time():
    from pysipnet.output import output_dataframe_to_dataset

    ds = output_dataframe_to_dataset(_frame())
    assert dict(ds.sizes) == {"time": 4, "bounds": 2}
    assert ds["time"].values[1] == np.datetime64("2020-01-01T12:00")
    assert ds["net_ecosystem_exchange"].attrs["units"] == "g m-2"
    assert ds["somethingNew"].attrs == {}
    assert ds["net_ecosystem_exchange"].dims == ("time",)


def test_dataset_infers_the_step_length_when_it_is_not_given():
    """Without the drivers the interval is reconstructed, and the Dataset says so."""
    from pysipnet.output import output_dataframe_to_dataset

    ds = output_dataframe_to_dataset(_frame())
    assert ds.attrs["time_step_length_source"].startswith("inferred")
    # Rows are 12 h apart, and the last step repeats the one before it.
    assert list(ds["time_step_length"].values) == [np.timedelta64(12, "h")] * 4
    assert ds["time_step_end"].values[-1] == np.datetime64("2020-01-03T00:00")


def test_dataset_step_bounds_from_lengths():
    from pysipnet.output import output_dataframe_to_dataset

    ds = output_dataframe_to_dataset(_frame(), time_step_length=np.full(4, 0.5))
    assert ds.attrs["time_step_length_source"] == "climate drivers"
    assert ds["time_step_end"].values[0] == np.datetime64("2020-01-01T12:00")
    assert ds["time_step_length"].values[0] == np.timedelta64(12, "h")
    with pytest.raises(ValueError, match="values but the data has"):
        output_dataframe_to_dataset(_frame(), time_step_length=np.ones(3))


def test_dataset_carries_cf_time_bounds():
    """The interval each row covers, in the form CF-aware tooling looks for."""
    from pysipnet.output import output_dataframe_to_dataset

    ds = output_dataframe_to_dataset(_frame(), time_step_length=np.full(4, 0.5))
    assert ds["time"].attrs["bounds"] == "time_bounds"
    assert ds["time_bounds"].dims == ("time", "bounds")
    np.testing.assert_array_equal(ds["time_bounds"].values[:, 0], ds["time"].values)
    np.testing.assert_array_equal(ds["time_bounds"].values[:, 1], ds["time_step_end"].values)


def test_dataset_says_when_the_step_length_is_unknowable():
    """One row gives nothing to measure an interval against."""
    from pysipnet.output import output_dataframe_to_dataset

    ds = output_dataframe_to_dataset(_frame().head(1))
    assert ds.attrs["time_step_length_source"] == "unknown"
    assert "time_step_end" not in ds.coords
    assert "time_bounds" not in ds.coords


def test_dataset_needs_the_time_coordinates():
    from pysipnet.output import output_dataframe_to_dataset

    with pytest.raises(ValueError, match="missing"):
        output_dataframe_to_dataset(_frame().drop(columns=["hour_of_day"]))


def test_output_getitem_and_variables():
    from pysipnet.output import SIPNETOutput

    out = SIPNETOutput.from_dataframe(_frame())
    assert out["nee"].attrs["long_name"] == "Net ecosystem exchange"
    assert out["NEE"].values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert [v.name for v in out.variables] == [
        "year",
        "day_of_year",
        "hour_of_day",
        "net_ecosystem_exchange",
        "wood_carbon",
    ]
    subset = out.select(["nee"], format="pandas")
    assert list(subset.columns) == ["year", "day_of_year", "hour_of_day", "net_ecosystem_exchange"]
    assert set(out[["nee", "wood_carbon"]].data_vars) == {"net_ecosystem_exchange", "wood_carbon"}


def test_output_refuses_a_variable_the_flags_leave_at_zero():
    """Selecting a flag-zeroed column by name is a mistake, not a request for zeros."""
    from pysipnet.output import SIPNETOutput
    from pysipnet.parameters.model import ModelFlags

    frame = _frame().assign(litter_carbon=0.0)
    out = SIPNETOutput.from_dataframe(frame, flags=ModelFlags.standard())
    with pytest.raises(ValueError, match="litter_pool"):
        out["litter_carbon"]

    # The faithful view of the file still has it, and so does a run that asked for it.
    assert "litter_carbon" in out.xarray.data_vars
    with_pool = SIPNETOutput.from_dataframe(frame, flags=ModelFlags(litter_pool=True))
    assert with_pool["litter_carbon"].sizes["time"] == 4


# ---------------------------------------------------------------------------
# The binary itself
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_binary_header_matches_registry(minimal_params, tmp_path):
    """The pinned SIPNET writes exactly the registry's columns, in order.

    An unknown column would only produce a warning on read, so this is the
    test that makes an upstream output change loud.
    """
    from pysipnet.parameters.model import ModelFlags
    from pysipnet.runner import SIPNETRunner
    from tests.test_integration import _make_climate

    runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path)
    if not runner.binary_path.exists():
        pytest.skip(f"SIPNET binary not found at {runner.binary_path}; run 'make sipnet'")
    result = runner.run(minimal_params, _make_climate(n_days=3))
    assert result.outputs.source_path is not None
    header = result.outputs.source_path.read_text().splitlines()[0].split()
    assert header == [v.sipnet_name for v in OUTPUT_VARIABLES]
    assert re.match(r"^\d{4} ", result.outputs.source_path.read_text().splitlines()[1])


# ---------------------------------------------------------------------------
# Climate drivers
# ---------------------------------------------------------------------------

# Written by hand from readClimData() in sipnet/src/sipnet/sipnet.c (12-column layout).
EXPECTED_CLIMATE_NAMES: dict[str, str] = {
    "year": "year",
    "day": "day_of_year",
    "time": "hour_of_day",
    "length": "time_step_length",
    "tair": "air_temperature",
    "tsoil": "soil_temperature",
    "par": "photosynthetically_active_radiation",
    "precip": "precipitation",
    "vpd": "vapor_pressure_deficit",
    "vpdSoil": "soil_vapor_pressure_deficit",
    "vPress": "vapor_pressure",
    "wspd": "wind_speed",
}


def test_climate_registry_matches_the_independent_mapping():
    from pysipnet.variables import CLIMATE_VARIABLES

    assert {v.sipnet_name: v.name for v in CLIMATE_VARIABLES} == EXPECTED_CLIMATE_NAMES
    assert [v.sipnet_name for v in CLIMATE_VARIABLES] == list(EXPECTED_CLIMATE_NAMES)


@pytest.mark.parametrize(
    "spec",
    __import__("pysipnet.variables", fromlist=["CLIMATE_VARIABLES"]).CLIMATE_VARIABLES,
    ids=lambda s: s.name,
)
def test_climate_spec_follows_conventions(spec):
    assert NAME_PATTERN.match(spec.name)
    assert not (set(spec.name.split("_")) & FORBIDDEN_NAME_TOKENS), spec.name
    assert spec.description.strip() and spec.long_label.strip()
    if spec.internal_units:
        from pysipnet.units import validate_units

        validate_units(spec.internal_units)
        assert spec.internal_conversion, f"{spec.name}: say how SIPNET converts it"
    attrs = spec.xarray_attributes()
    assert attrs["units"] == spec.units
    assert "time_reference" in attrs


def test_climate_conversion_note_survives_without_internal_units():
    """wind_speed is clamped but not converted; the note must still reach the Dataset."""
    from pysipnet.variables import CLIMATE_VARIABLES_BY_NAME

    attrs = CLIMATE_VARIABLES_BY_NAME["wind_speed"].xarray_attributes()
    assert "sipnet_internal_conversion" in attrs and "sipnet_internal_units" not in attrs
    length = CLIMATE_VARIABLES_BY_NAME["time_step_length"]
    assert length.aggregation.value == "sum", "a duration sums when resampling"


def test_climate_time_coordinates_match_output_coordinates():
    """Outputs and drivers share the same time columns so they can be aligned directly."""
    from pysipnet.variables import CLIMATE_COLUMN_NAMES

    assert CLIMATE_COLUMN_NAMES[:3] == TIME_COORDINATE_NAMES


def test_previous_climate_names_still_resolve():
    from pysipnet.variables import resolve_climate_variable

    for old, new in {
        "tair": "air_temperature",
        "vpd_soil": "soil_vapor_pressure_deficit",
        "vPress": "vapor_pressure",
        "length": "time_step_length",
        "day": "day_of_year",
    }.items():
        assert resolve_climate_variable(old).name == new
    with pytest.raises(KeyError, match="not a SIPNET climate driver"):
        resolve_climate_variable("rain")


def test_british_spellings_resolve_as_aliases():
    """Names use American spelling; the British forms shipped briefly and still resolve."""
    from pysipnet.parameters.model import resolve_parameter_name
    from pysipnet.variables import resolve_climate_variable

    assert resolve_climate_variable("vapour_pressure_deficit").name == "vapor_pressure_deficit"
    assert resolve_climate_variable("soil_vapour_pressure_deficit").name == (
        "soil_vapor_pressure_deficit"
    )
    assert resolve_climate_variable("vapour_pressure").name == "vapor_pressure"
    assert resolve_parameter_name("vapour_pressure_deficit_slope") == "vapor_pressure_deficit_slope"


def test_timestep_start_matches_the_calendar():
    """Integer datetime64 arithmetic replaces a string round-trip; it must agree with one."""
    import pandas as pd

    from pysipnet.dataset import timestep_start

    frame = pd.DataFrame(
        {
            "year": [2000, 2000, 1999, 2021],
            "day_of_year": [60, 366, 365, 1],
            "hour_of_day": [0.0, 23.5, 12.25, 0.0],
        }
    )
    assert list(timestep_start(frame)) == [
        np.datetime64("2000-02-29T00:00"),  # leap day
        np.datetime64("2000-12-31T23:30"),  # day 366 of a leap year
        np.datetime64("1999-12-31T12:15"),  # quarter-hour offsets
        np.datetime64("2021-01-01T00:00"),
    ]


def test_there_is_no_dataset_method():
    """`dataset` was a property until d7bd6a8; a method of that name is a trap.

    Old code writing `outputs.dataset` would get a bound method — truthy, no
    error — instead of the Dataset it expected. `out[[...]]` is the spelling,
    and it is exactly what a `dataset([...])` method would have returned.
    """
    from pysipnet.output import SIPNETOutput

    out = SIPNETOutput.from_dataframe(_frame())
    assert not hasattr(out, "dataset")
    assert not hasattr(out, "dataframe")
    assert set(out[["nee"]].data_vars) == {"net_ecosystem_exchange"}


def test_output_rejects_a_bare_string_where_a_sequence_is_expected():
    """dataset("nee") would otherwise select the variables 'n', 'e', 'e'."""
    from pysipnet.output import SIPNETOutput

    out = SIPNETOutput.from_dataframe(_frame())
    with pytest.raises(TypeError, match="sequence of variable names"):
        out.select("nee")


def test_output_reports_an_unknown_variable_name():
    from pysipnet.output import SIPNETOutput

    out = SIPNETOutput.from_dataframe(_frame())
    with pytest.raises(KeyError, match="not a SIPNET output variable"):
        out["nonexistent"]


def test_memory_backed_output_says_when_a_column_is_simply_absent():
    """No file to fall back on, so the message must not suggest a read failed."""
    from pysipnet.output import SIPNETOutput

    out = SIPNETOutput.from_dataframe(_frame().drop(columns=["wood_carbon"]))
    with pytest.raises(KeyError, match="no file to read"):
        out["wood_carbon"]


def test_timestep_start_refuses_years_it_cannot_represent():
    """Nanosecond datetimes wrap silently outside 1678-2261; that must not pass."""
    import pandas as pd

    from pysipnet.dataset import timestep_start

    frame = pd.DataFrame({"year": [2300], "day_of_year": [1], "hour_of_day": [0.0]})
    with pytest.raises(ValueError, match="wrap silently"):
        timestep_start(frame)


def test_timestep_start_refuses_missing_time_values():
    """np.rint(nan).astype(int64) is undefined and differs between platforms."""
    import pandas as pd

    from pysipnet.dataset import timestep_start

    frame = pd.DataFrame(
        {"year": [2020, 2020], "day_of_year": [1, 2], "hour_of_day": [0.0, np.nan]}
    )
    with pytest.raises(ValueError, match="non-finite"):
        timestep_start(frame)


def test_step_lengths_must_increase_the_clock():
    """Duplicate or backwards timestamps would give bounds that run backwards."""
    import pandas as pd

    from pysipnet.output import output_dataframe_to_dataset

    repeated = pd.DataFrame(
        {
            "year": [2020] * 3,
            "day_of_year": [1, 1, 2],
            "hour_of_day": [0.0, 0.0, 0.0],
            "net_ecosystem_exchange": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="do not increase"):
        output_dataframe_to_dataset(repeated)


def test_supplied_step_lengths_must_be_positive():
    from pysipnet.output import output_dataframe_to_dataset

    with pytest.raises(ValueError, match="positive duration"):
        output_dataframe_to_dataset(_frame(), time_step_length=np.array([0.5, 0.5, 0.0, 0.5]))


def test_a_frames_own_step_lengths_beat_the_inferred_ones():
    """The drivers state their lengths; measuring the gaps would discard the last one."""
    from pysipnet.dataset import build_xarray_dataset

    frame = _frame().assign(time_step_length=[0.5, 0.5, 0.5, 0.25])
    ds = build_xarray_dataset(frame, attributes_for=lambda _: {}, source="test")
    assert ds.attrs["time_step_length_source"] == "climate drivers"
    assert ds["time_step_length"].values[-1] == np.timedelta64(6, "h")


def test_select_returns_the_format_it_was_asked_for():
    import pandas as pd
    import xarray as xr

    from pysipnet.output import SIPNETOutput

    out = SIPNETOutput.from_dataframe(_frame())
    assert isinstance(out.select(["nee"]), xr.Dataset)
    assert isinstance(out.select(["nee"], format="pandas"), pd.DataFrame)
    assert out.select(["nee"]).identical(out[["nee"]])
    with pytest.raises(ValueError, match="'xarray' or 'pandas'"):
        out.select(["nee"], format="dataframe")
