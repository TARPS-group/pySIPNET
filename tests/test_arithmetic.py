"""Arithmetic on labeled DataArrays: values, and the units, constituent and kind of the result."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pysipnet import resample
from pysipnet.arithmetic import (
    add_with_units,
    divide_with_units,
    multiply_with_units,
    step_length,
    subtract_with_units,
)
from pysipnet.io.reference import niwot_reference_output
from pysipnet.parameters.model import PARAMETER_SPECS
from pysipnet.units import conversion_factor, convert_dataarray_units
from pysipnet.variables import (
    KIND_AFTER_TIME_POWER,
    RESAMPLING_METHODS_FOR_KIND,
    TIME_REFERENCE_FOR_KIND,
    VariableKind,
)

EXACT = {"rtol": 1e-12, "atol": 0.0}


@pytest.fixture(scope="module")
def niwot() -> xr.Dataset:
    return niwot_reference_output().xarray


@pytest.fixture
def nee(niwot) -> xr.DataArray:
    return niwot["net_ecosystem_exchange"]


def _parameter(path: str, value: float | list[float], **dims: list[str]) -> xr.DataArray:
    """A parameter as a DataArray labeled from its spec, as a caller would build one."""
    spec = PARAMETER_SPECS[path]
    attrs = {"units": spec.units}
    if spec.constituent:
        attrs["constituent"] = spec.constituent
    return xr.DataArray(value, dims=list(dims), coords=dims, name=path.split(".")[1], attrs=attrs)


def _per_day() -> xr.DataArray:
    return xr.DataArray(0.5, name="per_day", attrs={"units": "d-1"})


def _days(niwot: xr.Dataset) -> np.ndarray:
    return niwot["time_step_length"].values / np.timedelta64(1, "D")


# ── step_length ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("units, per_day", [("d", 1.0), ("h", 24.0), ("s", 86_400.0)])
def test_step_length_is_the_coordinate_in_the_units_asked_for(niwot, nee, units, per_day):
    lengths = step_length(nee, units)
    np.testing.assert_allclose(lengths.values, _days(niwot) * per_day, **EXACT)
    assert lengths.attrs == {"units": units, "long_name": "Timestep length"}
    assert lengths.dims == ("time",)
    assert (lengths["time"] == nee["time"]).all()
    assert "kind" not in lengths.attrs


def test_step_length_goes_into_a_dataset(nee):
    lengths = step_length(nee)
    assert "time_step_length" not in lengths.coords
    assert "time_step_length" in lengths.to_dataset().data_vars


def test_step_length_reads_a_dataset(niwot):
    np.testing.assert_allclose(step_length(niwot).values, _days(niwot), **EXACT)


def test_step_length_of_a_missing_step_is_nan(nee):
    lengths = nee["time_step_length"].values.copy()
    lengths[3] = np.timedelta64("NaT")
    padded = nee.assign_coords(time_step_length=("time", lengths))
    days = step_length(padded)
    assert np.isnan(days.values[3])
    assert np.isfinite(np.delete(days.values, 3)).all()


def test_step_length_keeps_a_two_dimensional_coordinate(nee):
    stacked = xr.concat([nee, nee], dim="site")
    lengths = np.stack([nee["time_step_length"].values] * 2)
    lengths[1, -1] = np.timedelta64("NaT")
    stacked = stacked.assign_coords(time_step_length=(("site", "time"), lengths))
    days = step_length(stacked)
    assert days.dims == ("site", "time")
    assert np.isnan(days.values[1, -1]) and not np.isnan(days.values[0, -1])


def test_step_length_refuses_an_array_without_the_coordinate(nee):
    with pytest.raises(ValueError, match="no 'time_step_length' coordinate"):
        step_length(nee.drop_vars("time_step_length"))


def test_step_length_refuses_other_units_and_types(nee):
    with pytest.raises(ValueError, match="'d', 'h' or 's'"):
        step_length(nee, "fortnight")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DataArray or Dataset"):
        step_length(nee.values)  # type: ignore[arg-type]


def test_the_raw_coordinate_is_not_an_operand(nee):
    with pytest.raises(ValueError, match="Use step_length()"):
        divide_with_units(nee, nee["time_step_length"])


# ── the two uses ─────────────────────────────────────────────────────────────


def test_nee_over_step_length_is_a_daily_rate_that_converts_to_co2(niwot, nee):
    rate = divide_with_units(nee, step_length(nee))
    np.testing.assert_allclose(rate.values, nee.values / _days(niwot), **EXACT)
    assert rate.attrs["units"] == "g m-2 d-1"
    assert rate.attrs["constituent"] == "C"
    assert rate.attrs["kind"] == VariableKind.DAILY_RATE
    assert rate.attrs["time_reference"] == TIME_REFERENCE_FOR_KIND[VariableKind.DAILY_RATE]
    assert rate.attrs["cell_methods"] == "time: mean"
    assert rate.attrs["sign_convention"] == nee.attrs["sign_convention"]
    assert rate.attrs["derivation"] == "net_ecosystem_exchange / time_step_length"
    assert rate.attrs["long_name"] == rate.attrs["derivation"]

    factor = conversion_factor(
        units="g m-2 d-1", constituent="C", to_units="umol m-2 s-1", to_constituent="CO2"
    )
    assert factor == pytest.approx(0.96362, abs=1e-5)
    flux = convert_dataarray_units(rate, to_units="umol m-2 s-1", to_constituent="CO2")
    np.testing.assert_allclose(flux.values, rate.values * factor, **EXACT)


@pytest.mark.parametrize("order", ["per_day first", "nee first"])
def test_a_product_puts_the_constituent_first_whichever_operand_it_is(niwot, nee, order):
    operands = (_per_day(), nee) if order == "per_day first" else (nee, _per_day())
    product = multiply_with_units(*operands)
    assert product.attrs["units"] == "g m-2 d-1"
    assert product.attrs["constituent"] == "C"
    assert product.attrs["kind"] == VariableKind.DAILY_RATE
    np.testing.assert_allclose(product.values, nee.values * 0.5, **EXACT)
    convert_dataarray_units(product, to_units="umol m-2 s-1", to_constituent="CO2")


def test_leaf_carbon_over_leaf_carbon_per_area_is_leaf_area_index(niwot):
    leaf_carbon = niwot["leaf_carbon"]
    per_area = _parameter("leaf.leaf_carbon_per_area", 270.0)
    lai = divide_with_units(leaf_carbon, per_area)
    np.testing.assert_allclose(lai.values, leaf_carbon.values / 270.0, **EXACT)
    assert lai.attrs["units"] == "1"
    assert "constituent" not in lai.attrs
    assert lai.attrs["kind"] == VariableKind.TIMESTEP_END_STATE
    assert lai.attrs["cell_methods"] == "time: point"
    in_m2 = convert_dataarray_units(lai, to_units="m2 m-2")
    np.testing.assert_allclose(in_m2.values, lai.values, **EXACT)


# ── units and constituent (the rules are pysipnet.units', tested there) ─────


def test_prefixes_are_not_merged():
    depth = xr.DataArray([2.0], dims="x", name="depth", attrs={"units": "cm"})
    height = xr.DataArray([4.0], dims="x", name="height", attrs={"units": "m"})
    ratio = divide_with_units(depth, height)
    assert ratio.attrs["units"] == "cm m-1"
    np.testing.assert_allclose(
        convert_dataarray_units(ratio, to_units="1").values, [0.005], **EXACT
    )


def test_a_number_is_dimensionless(nee):
    doubled = multiply_with_units(2, nee)
    assert doubled.attrs["units"] == "g m-2"
    assert doubled.attrs["kind"] == VariableKind.TIMESTEP_TOTAL
    assert doubled.attrs["derivation"] == "2 * net_ecosystem_exchange"
    np.testing.assert_allclose(doubled.values, 2 * nee.values, **EXACT)


@pytest.mark.parametrize(
    "other_units, op, symbol",
    [("g-1", multiply_with_units, "*"), ("g", divide_with_units, "/")],
)
def test_a_unit_refusal_names_the_operation(nee, other_units, op, symbol):
    other = xr.DataArray(2.0, name="leaf_mass", attrs={"units": other_units})
    with pytest.raises(
        ValueError,
        match=f"^net_ecosystem_exchange \\{symbol} leaf_mass: .*qualifies the first unit",
    ):
        op(nee, other)


def test_degree_days_multiply_but_a_temperature_does_not():
    gdd = xr.DataArray([100.0], dims="x", name="gdd", attrs={"units": "degC d"})
    assert multiply_with_units(gdd, 2).attrs["units"] == "degC d"
    celsius = xr.DataArray([1.0], dims="x", name="air_temperature", attrs={"units": "degC"})
    with pytest.raises(ValueError, match="offset temperature scale"):
        multiply_with_units(celsius, 2)


def _carbon(name: str = "c") -> xr.DataArray:
    return xr.DataArray([2.0], dims="x", name=name, attrs={"units": "g m-2", "constituent": "C"})


def test_the_same_constituent_cancels_in_a_quotient():
    ratio = divide_with_units(_carbon("a"), _carbon("b"))
    assert ratio.attrs["units"] == "1"
    assert "constituent" not in ratio.attrs


def test_a_constituent_refusal_names_the_operation():
    nitrogen = xr.DataArray([4.0], dims="x", name="n", attrs={"units": "g m-2", "constituent": "N"})
    with pytest.raises(ValueError, match="^c \\* n: .*at most one factor"):
        multiply_with_units(_carbon(), nitrogen)
    with pytest.raises(ValueError, match="^c / n: .*same constituent"):
        divide_with_units(_carbon(), nitrogen)


# ── kind ─────────────────────────────────────────────────────────────────────


def test_a_rate_times_step_length_is_the_original_total(niwot, nee):
    rate = divide_with_units(nee, step_length(nee))
    total = multiply_with_units(rate, step_length(nee))
    assert total.attrs["kind"] == VariableKind.TIMESTEP_TOTAL
    assert total.attrs["units"] == "g m-2"
    np.testing.assert_allclose(total.values, nee.values, **EXACT)


def test_the_models_own_rate_times_step_length_is_a_total(niwot):
    transpiration = niwot["transpiration_rate"]
    total = multiply_with_units(transpiration, step_length(transpiration))
    assert total.attrs["units"] == "cm"
    assert total.attrs["kind"] == VariableKind.TIMESTEP_TOTAL
    assert total.attrs["constituent"] == "H2O"


def test_a_total_times_a_per_time_is_a_rate_and_a_rate_per_per_time_is_a_total(nee):
    rate = multiply_with_units(nee, _per_day())
    assert rate.attrs["kind"] == VariableKind.DAILY_RATE
    assert divide_with_units(rate, _per_day()).attrs["kind"] == VariableKind.TIMESTEP_TOTAL


def test_a_pool_times_a_turnover_rate_is_refused(niwot):
    turnover = _parameter("phenology.leaf_turnover_rate", 0.3)
    with pytest.raises(ValueError, match="no pySIPNET kind names the result"):
        multiply_with_units(niwot["leaf_carbon"], turnover)


def test_a_cumulative_value_per_day_is_refused(niwot):
    with pytest.raises(ValueError, match="time to the power -1"):
        divide_with_units(niwot["cumulative_net_ecosystem_exchange"], step_length(niwot))


def test_a_squared_time_is_refused(nee):
    with pytest.raises(ValueError, match="time to the power -2"):
        divide_with_units(nee, xr.DataArray(1.0, name="t2", attrs={"units": "d2"}))


def test_two_kinded_operands_are_refused(niwot, nee):
    with pytest.raises(ValueError, match="at most one operand of a product or quotient"):
        multiply_with_units(niwot["leaf_carbon"], nee)


def test_dividing_by_a_kinded_operand_is_refused(niwot):
    with pytest.raises(ValueError, match="cannot divide by leaf_carbon"):
        divide_with_units(1.0, niwot["leaf_carbon"])


def test_a_time_coordinate_is_not_an_operand(nee):
    year = nee["year"].copy()
    year.attrs = {"units": "1", "kind": VariableKind.TIMESTEP_START_COORDINATE.value}
    with pytest.raises(ValueError, match="time coordinate"):
        multiply_with_units(year, 2)


def test_every_kind_transition_is_between_kinds_that_resample():
    for (kind, power), result in KIND_AFTER_TIME_POWER.items():
        assert power in (-1, 1)
        assert RESAMPLING_METHODS_FOR_KIND[kind] and RESAMPLING_METHODS_FOR_KIND[result]


# ── result attributes and coordinates ────────────────────────────────────────


def test_source_attributes_are_not_carried_and_the_result_is_unnamed(nee):
    result = divide_with_units(nee, step_length(nee))
    for attr in ("description", "sipnet_name", "output_decimals"):
        assert attr not in result.attrs
    assert result.name is None


def test_the_operands_are_not_modified(nee):
    before = (nee.attrs.copy(), nee.name, nee.values.copy())
    divide_with_units(nee, step_length(nee))
    assert (nee.attrs, nee.name) == before[:2]
    np.testing.assert_array_equal(nee.values, before[2])


def test_a_chained_derivation_names_every_step(nee):
    rate = divide_with_units(nee, step_length(nee))
    assert divide_with_units(rate, 2.0).attrs["derivation"] == (
        "(net_ecosystem_exchange / time_step_length) / 2.0"
    )


@pytest.mark.parametrize("other", [-1.0, xr.DataArray(-1.0, attrs={"units": "1"})])
def test_a_negative_factor_drops_the_sign_convention(nee, other):
    assert "sign_convention" not in multiply_with_units(nee, other).attrs


def test_the_result_keeps_the_kinded_operands_time_coordinates(niwot, nee):
    rate = divide_with_units(nee, step_length(nee))
    for name in ("time_step_start", "time_step_length", "year", "day_of_year", "hour_of_day"):
        xr.testing.assert_identical(rate[name], nee[name])
    daily = resample(rate, "1D", how="mean")
    assert daily.attrs["kind"] == VariableKind.DAILY_RATE
    assert daily.name is None


def test_a_site_parameter_broadcasts_over_a_stack_of_records(niwot):
    leaf_carbon = xr.concat([niwot["leaf_carbon"]] * 2, dim="site").assign_coords(site=["a", "b"])
    per_area = _parameter("leaf.leaf_carbon_per_area", [200.0, 300.0], site=["a", "b"])
    lai = divide_with_units(leaf_carbon, per_area)
    assert lai.dims == ("site", "time")
    np.testing.assert_allclose(
        lai.sel(site="a").values, niwot["leaf_carbon"].values / 200.0, **EXACT
    )
    np.testing.assert_allclose(
        lai.sel(site="b").values, niwot["leaf_carbon"].values / 300.0, **EXACT
    )


def test_conflicting_time_coordinates_are_refused_not_dropped(nee):
    shifted = nee.assign_coords(time_step_start=nee["time_step_start"] + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="time_step_start"):
        divide_with_units(nee, step_length(shifted))


def test_misaligned_index_coordinates_are_refused_not_intersected(nee):
    per_step = step_length(nee).isel(time=slice(0, 10))
    with pytest.raises(ValueError, match="index coordinates differ"):
        divide_with_units(nee, per_step)


# ── operands ─────────────────────────────────────────────────────────────────


def test_an_array_without_units_is_refused(nee):
    with pytest.raises(ValueError, match="no 'units' attribute"):
        multiply_with_units(nee, xr.DataArray(2.0, name="bare"))


@pytest.mark.parametrize("operand", [True, np.bool_(False), "2", None, [1.0]])
def test_an_operand_that_is_not_a_number_or_dataarray_is_refused(nee, operand):
    with pytest.raises(TypeError, match="real number"):
        multiply_with_units(nee, operand)  # type: ignore[arg-type]


def test_two_numbers_are_refused():
    with pytest.raises(TypeError, match="at least one xarray DataArray"):
        divide_with_units(2.0, 4.0)


def test_an_unknown_kind_is_refused(nee):
    odd = nee.copy()
    odd.attrs["kind"] = "flux"
    with pytest.raises(ValueError, match="not one of pySIPNET's kinds"):
        multiply_with_units(odd, 2)


# ── sums and differences ─────────────────────────────────────────────────────


def test_a_sum_of_agreeing_operands_keeps_their_labels(niwot, nee):
    total = add_with_units(nee, niwot["gross_primary_production"])
    assert total.attrs["units"] == "g m-2"
    assert total.attrs["constituent"] == "C"
    assert total.attrs["kind"] == VariableKind.TIMESTEP_TOTAL
    np.testing.assert_allclose(
        total.values, nee.values + niwot["gross_primary_production"].values, **EXACT
    )


def test_a_sum_keeps_a_sign_convention_only_when_both_state_it(nee):
    assert add_with_units(nee, nee).attrs["sign_convention"] == nee.attrs["sign_convention"]
    assert "sign_convention" not in subtract_with_units(nee, nee).attrs


def test_a_sum_needs_the_same_kind(niwot, nee):
    with pytest.raises(
        ValueError, match="same kind; got 'timestep_total' and 'timestep_end_state'"
    ):
        add_with_units(nee, niwot["leaf_carbon"])


def test_a_difference_needs_the_same_units(nee):
    rate = multiply_with_units(nee, _per_day())
    with pytest.raises(ValueError, match="same kind"):
        subtract_with_units(nee, rate)
    with pytest.raises(ValueError, match="same units and constituent"):
        subtract_with_units(nee, convert_dataarray_units(nee, to_units="kg m-2"))


def test_the_difference_of_two_celsius_temperatures_is_in_kelvin():
    warm = xr.DataArray([20.0], dims="x", name="warm", attrs={"units": "degC"})
    cool = xr.DataArray([5.0], dims="x", name="cool", attrs={"units": "degC"})
    difference = subtract_with_units(warm, cool)
    assert difference.attrs["units"] == "K"
    np.testing.assert_allclose(difference.values, [15.0], **EXACT)
    with pytest.raises(ValueError, match="offset temperature scale"):
        add_with_units(warm, cool)
