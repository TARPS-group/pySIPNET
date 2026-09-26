"""resample(): explicit, kind-checked coarsening of the time axis.

Checked against SIPNET's own Niwot record, whose day and night steps differ in
length (0.29 to 0.63 days), so a plain mean and a length-weighted mean disagree
and the tests can tell them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pysipnet import check_resampling_method, resample
from pysipnet.climate import ClimateDrivers
from pysipnet.dataset import assemble_time_coords
from pysipnet.io.reference import niwot_reference_climate, niwot_reference_output
from pysipnet.resample import (
    check_frequency,
    check_not_upsampling,
    drop_padding,
    resampled_attributes,
)
from pysipnet.variables import RESAMPLED_KIND, RESAMPLING_METHODS_FOR_KIND, VariableKind


@pytest.fixture(scope="module")
def niwot() -> tuple[pd.DataFrame, np.ndarray, xr.Dataset]:
    output = niwot_reference_output()
    length = output.timestep_length
    assert length is not None
    return output.pandas, length, output.xarray


def _daily_key(ds: xr.Dataset) -> np.ndarray:
    """The calendar day each step ends in, with a step ending at midnight in the day that ended."""
    return (pd.DatetimeIndex(ds["time"].values) - pd.Timedelta(1, "ns")).floor("1D").to_numpy()


def test_totals_sum_over_the_day_the_step_ends_in(niwot):
    frame, _, ds = niwot
    daily = resample(ds[["net_ecosystem_exchange", "gross_primary_production"]], "1D", how="sum")
    expected = frame["net_ecosystem_exchange"].groupby(_daily_key(ds)).sum()
    np.testing.assert_allclose(daily["net_ecosystem_exchange"].values, expected.values)
    assert daily["net_ecosystem_exchange"].attrs["kind"] == "timestep_total"
    assert daily["net_ecosystem_exchange"].attrs["cell_methods"] == "time: sum"
    assert "output_decimals" not in daily["net_ecosystem_exchange"].attrs
    assert daily.attrs["resampling_frequency"] == "1D"


def test_pools_keep_their_last_value(niwot):
    frame, _, ds = niwot
    daily = resample(ds[["wood_carbon"]], "1D", how="last")
    expected = frame["wood_carbon"].groupby(_daily_key(ds)).last()
    np.testing.assert_allclose(daily["wood_carbon"].values, expected.values)
    assert daily["wood_carbon"].attrs["kind"] == "timestep_end_state"
    assert daily["wood_carbon"].attrs["cell_methods"] == "time: point"


def test_means_are_weighted_by_step_length(niwot):
    frame, length, ds = niwot
    daily = resample(ds["soil_wetness_fraction"], "1D", how="mean")
    key = _daily_key(ds)
    weighted = (frame["soil_wetness_fraction"] * length).groupby(key).sum() / pd.Series(
        length
    ).groupby(key).sum()
    plain = frame["soil_wetness_fraction"].groupby(key).mean()
    np.testing.assert_allclose(daily.values, weighted.values)
    assert not np.allclose(daily.values, plain.values), "unequal steps make the two differ"
    assert daily.attrs["resampling"].endswith("weighted by timestep_length")
    # The two-point qualification no longer applies once steps are combined.
    assert daily.attrs["cell_methods"] == "time: mean"


def test_a_pool_averaged_becomes_a_mean(niwot):
    _, _, ds = niwot
    daily = resample(ds[["soil_water"]], "1D", how="mean")
    assert daily["soil_water"].attrs["kind"] == "timestep_mean"
    assert daily["soil_water"].attrs["cell_methods"] == "time: mean"
    assert daily["soil_water"].attrs["time_reference"] == "mean over the timestep"


def test_result_has_the_same_time_layout(niwot):
    _, _, ds = niwot
    daily = resample(ds[["net_ecosystem_exchange"]], "1D", how="sum")
    assert set(daily.coords) == set(ds.coords)
    assert daily["time"].attrs["standard_name"] == "time"
    assert daily["time"].attrs["bounds"] == "time_bounds"
    np.testing.assert_array_equal(daily["time_bounds"].values[:, 1], daily["time"].values)
    np.testing.assert_array_equal(daily["time_bounds"].values[:, 0], daily["timestep_start"].values)
    # Cells are contiguous and tile the record exactly.
    assert daily["timestep_start"].values[0] == ds["timestep_start"].values[0]
    assert daily["time"].values[-1] == ds["time"].values[-1]
    np.testing.assert_array_equal(daily["timestep_start"].values[1:], daily["time"].values[:-1])
    # timestep_length is the coverage: the declared lengths that went in.
    total = ds["timestep_length"].values.astype("int64").sum()
    assert daily["timestep_length"].values.astype("int64").sum() == total
    assert daily.attrs["timestep_length_source"].startswith("sum of the declared")
    for name in daily.coords:
        assert daily[name].encoding["_FillValue"] is None


def test_row_labels_are_the_start_of_the_coarser_step(niwot):
    _, _, ds = niwot
    daily = resample(ds[["net_ecosystem_exchange"]], "1D", how="sum")
    starts = pd.DatetimeIndex(daily["timestep_start"].values)
    np.testing.assert_array_equal(daily["year"].values, starts.year)
    np.testing.assert_array_equal(daily["day_of_year"].values, starts.dayofyear)
    np.testing.assert_allclose(daily["hour_of_day"].values, starts.hour + starts.minute / 60)


def test_mapping_picks_a_method_per_variable_and_keeps_only_those(niwot):
    _, _, ds = niwot
    annual = resample(
        ds,
        "YS",
        how={
            "wood_carbon": "last",
            "net_ecosystem_exchange": "sum",
            "cumulative_net_ecosystem_exchange": "last",
        },
    )
    assert list(annual.data_vars) == [
        "wood_carbon",
        "net_ecosystem_exchange",
        "cumulative_net_ecosystem_exchange",
    ]
    assert annual.sizes["time"] == 1
    assert annual["wood_carbon"].values[0] == ds["wood_carbon"].values[-1]
    assert annual["net_ecosystem_exchange"].values[0] == pytest.approx(
        float(ds["net_ecosystem_exchange"].sum())
    )
    cumulative = annual["cumulative_net_ecosystem_exchange"]
    assert cumulative.values[0] == ds["cumulative_net_ecosystem_exchange"].values[-1]
    assert "cell_methods" not in cumulative.attrs


def test_how_is_required():
    with pytest.raises(TypeError, match="how"):
        resample(xr.Dataset(), "1D")  # type: ignore[call-overload]


@pytest.mark.parametrize(
    ("name", "how", "why"),
    [
        ("wood_carbon", "sum", "not additive"),
        ("net_ecosystem_exchange", "mean", "sum them"),
        ("net_ecosystem_exchange", "last", "not the total"),
        ("cumulative_net_ecosystem_exchange", "sum", "double counts"),
        ("cumulative_net_ecosystem_exchange", "mean", "take 'last'"),
        ("transpiration_rate", "sum", "multiply by"),
        ("soil_wetness_fraction", "last", "does not represent"),
    ],
)
def test_an_invalid_method_is_refused_with_the_reason_and_the_menu(niwot, name, how, why):
    _, _, ds = niwot
    with pytest.raises(ValueError) as excinfo:
        resample(ds[[name]], "1D", how=how)
    message = str(excinfo.value)
    assert f"Cannot resample {name!r} with {how!r}" in message
    assert why in message
    kind = VariableKind(ds[name].attrs["kind"])
    for valid in RESAMPLING_METHODS_FOR_KIND[kind]:
        assert f"'{valid}' gives" in message


def test_every_valid_method_has_a_result_kind():
    for kind, methods in RESAMPLING_METHODS_FOR_KIND.items():
        for method in methods:
            assert (kind, method) in RESAMPLED_KIND
    assert set(RESAMPLED_KIND) == {
        (kind, method)
        for kind, methods in RESAMPLING_METHODS_FOR_KIND.items()
        for method in methods
    }


def test_unknown_method_and_unknown_variable(niwot):
    _, _, ds = niwot
    with pytest.raises(ValueError, match="Unknown resampling method"):
        resample(ds[["net_ecosystem_exchange"]], "1D", how="median")
    with pytest.raises(KeyError, match="registry names"):
        resample(ds, "1D", how={"nee": "sum"})
    with pytest.raises(TypeError, match="mapping"):
        resample(ds, "1D", how=["sum"])  # type: ignore[arg-type]


def test_a_variable_of_unknown_kind_must_declare_one(niwot):
    _, _, ds = niwot
    mystery = ds[["net_ecosystem_exchange"]].rename(net_ecosystem_exchange="mystery")
    mystery["mystery"].attrs.clear()
    with pytest.raises(ValueError, match="attrs\\['kind'\\]"):
        resample(mystery, "1D", how="sum")
    mystery["mystery"].attrs["kind"] = "timestep_total"
    assert resample(mystery, "1D", how="sum")["mystery"].attrs["kind"] == "timestep_total"


def test_a_dataarray_resamples_to_a_dataarray(niwot):
    _, _, ds = niwot
    nee = resample(ds["net_ecosystem_exchange"], "7D", how="sum")
    assert isinstance(nee, xr.DataArray)
    assert nee.name == "net_ecosystem_exchange"
    with pytest.raises(TypeError, match="single method"):
        resample(ds["net_ecosystem_exchange"], "7D", how={"net_ecosystem_exchange": "sum"})


def test_a_dataset_with_only_part_of_the_time_layout_is_refused(niwot):
    half = niwot[2][["net_ecosystem_exchange"]].drop_vars("timestep_length")
    with pytest.raises(ValueError, match="lacks the pySIPNET time coordinates.*timestep_length"):
        resample(half, "1D", how="sum")


def test_a_daily_record_resamples_daily_to_itself():
    frame = pd.DataFrame(
        {
            "year": 2020,
            "day_of_year": range(100, 105),
            "hour_of_day": 0.0,
            "timestep_length": 1.0,
            "air_temperature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "soil_temperature": 0.0,
            "photosynthetically_active_radiation": 1.0,
            "precipitation": 2.0,
            "vapor_pressure_deficit": 100.0,
            "soil_vapor_pressure_deficit": 100.0,
            "vapor_pressure": 100.0,
            "wind_speed": 1.0,
        }
    )
    ds = ClimateDrivers.from_dataframe(frame).xarray
    daily = resample(ds, "1D", how={"air_temperature": "mean", "precipitation": "sum"})
    np.testing.assert_array_equal(daily["time"].values, ds["time"].values)
    np.testing.assert_array_equal(daily["air_temperature"].values, ds["air_temperature"].values)
    np.testing.assert_array_equal(daily["precipitation"].values, ds["precipitation"].values)
    assert daily["precipitation"].attrs["cell_methods"] == "time: sum"


# ── Ensembles: dimensions besides time ───────────────────────────────────────

MEMBERS = [0, 1]
SITES = ["niwot", "elsewhere"]


@pytest.fixture(scope="module")
def climate() -> xr.Dataset:
    # 800 rows over fourteen months, so monthly and annual cells are more than one.
    return niwot_reference_climate().xarray


def _scaled(ds: xr.Dataset, factor: float) -> xr.Dataset:
    with xr.set_options(keep_attrs=True):
        return ds.map(lambda variable: variable * factor)


def _factor(member: int, site: str) -> float:
    return (member + 1) * (1.0 + 10.0 * SITES.index(site))


def _stack(ds: xr.Dataset) -> xr.Dataset:
    """Members × sites of copies of *ds*, each scaled so the slices differ."""
    by_site = [
        xr.concat([_scaled(ds, _factor(m, site)) for m in MEMBERS], dim="member") for site in SITES
    ]
    stack = xr.concat(by_site, dim="site").assign_coords(
        member=MEMBERS,
        site=SITES,
        lon=("site", [-105.5, -72.2], {"units": "degrees_east"}),
        lat=("site", [40.0, 42.5], {"units": "degrees_north"}),
    )
    first = next(iter(stack.data_vars))
    # Dimension order is the variable's own business; it must survive.
    return stack.assign({first: stack[first].transpose("time", "site", "member")})


_ENSEMBLE_CASES = {
    "output": {
        "net_ecosystem_exchange": "sum",
        "wood_carbon": "last",
        "soil_wetness_fraction": "mean",
    },
    "climate": {"precipitation": "sum", "air_temperature": "mean", "vapor_pressure": "mean"},
}


def _source(name: str, niwot, climate) -> xr.Dataset:
    ds = niwot[2] if name == "output" else climate
    return ds[list(_ENSEMBLE_CASES[name])]


@pytest.mark.parametrize("source", ["output", "climate"])
@pytest.mark.parametrize("freq", ["1D", "MS", "YS"])
def test_every_slice_of_a_stack_resamples_as_its_own_run(niwot, climate, source, freq):
    ds = _source(source, niwot, climate)
    how = _ENSEMBLE_CASES[source]
    stack = _stack(ds)
    resampled = resample(stack, freq, how=how)

    for name in how:
        assert resampled[name].dims == stack[name].dims
    np.testing.assert_array_equal(resampled["lon"].values, stack["lon"].values)
    assert resampled["lat"].attrs == {"units": "degrees_north"}
    for member in MEMBERS:
        for site in SITES:
            one = resampled.sel(member=member, site=site).drop_vars(
                ["member", "site", "lon", "lat"]
            )
            alone = resample(_scaled(ds, _factor(member, site)), freq, how=how)
            xr.testing.assert_identical(one, alone)


@pytest.mark.parametrize("freq", ["1D", "MS"])
@pytest.mark.parametrize(("name", "how"), [("precipitation", "sum"), ("air_temperature", "mean")])
def test_a_stacked_dataarray_resamples_slice_by_slice(climate, freq, name, how):
    stack = _stack(climate[[name]])[name]
    resampled = resample(stack, freq, how=how)
    assert isinstance(resampled, xr.DataArray)
    assert resampled.dims == stack.dims
    np.testing.assert_array_equal(resampled["lat"].values, stack["lat"].values)
    for member in MEMBERS:
        for site in SITES:
            alone = resample(_scaled(climate[[name]], _factor(member, site))[name], freq, how=how)
            one = resampled.sel(member=member, site=site).drop_vars(
                ["member", "site", "lon", "lat"]
            )
            xr.testing.assert_identical(one, alone)


def test_the_last_value_of_a_stack_is_each_slices_last(niwot):
    stack = _stack(niwot[2][["wood_carbon"]])
    daily = resample(stack["wood_carbon"], "1D", how="last")
    for member in MEMBERS:
        for site in SITES:
            expected = niwot[0]["wood_carbon"].groupby(_daily_key(niwot[2])).last()
            np.testing.assert_allclose(
                daily.sel(member=member, site=site).values,
                expected.values * _factor(member, site),
            )


def test_scalar_coordinates_of_one_run_are_carried(niwot):
    one = niwot[2][["net_ecosystem_exchange"]].assign_coords(site="niwot", member=3)
    daily = resample(one, "1D", how="sum")
    assert daily["site"].item() == "niwot"
    assert daily["member"].item() == 3
    assert daily["net_ecosystem_exchange"].dims == ("time",)


def test_a_nan_stays_in_its_own_slice(climate):
    stack = _stack(climate[["precipitation"]])
    values = stack["precipitation"].transpose("member", "site", "time").values.copy()
    values[1, 0, 5] = np.nan
    stack["precipitation"] = (("member", "site", "time"), values, stack["precipitation"].attrs)
    daily = resample(stack, "1D", how="sum")["precipitation"]
    missing = np.isnan(daily.values)
    assert missing.sum() == 1
    assert np.isnan(daily.sel(member=1, site="niwot").values).sum() == 1


def _two_runs_on_different_axes(ds: xr.Dataset) -> xr.Dataset:
    short = ds.isel(time=slice(0, 40))
    return xr.concat(
        [ds, short], dim="site", coords="different", compat="equals", join="outer"
    ).assign_coords(site=SITES)


def test_interval_coordinates_that_vary_by_site_are_refused(niwot):
    stack = _two_runs_on_different_axes(niwot[2][["net_ecosystem_exchange"]])
    assert stack["timestep_start"].dims == ("site", "time")
    with pytest.raises(ValueError, match="timestep_start.*each site, separately"):
        resample(stack, "1D", how="sum")


def test_a_run_selected_from_such_a_stack_drops_its_padding(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]]
    stack = _two_runs_on_different_axes(ds)
    padded = stack.sel(site=SITES[1]).drop_vars("site")
    assert np.isnat(padded["timestep_length"].values).sum() == 20
    alone = resample(ds.isel(time=slice(0, 40)), "1D", how="sum")
    xr.testing.assert_identical(resample(padded, "1D", how="sum"), alone)


def test_padding_in_mid_record_is_dropped():
    # A truncated record ends at its declared end, not snapped to the next start
    # as the full record's step is, so the union axis gains a timestamp inside
    # the full record, where the full record's row is padding.
    drivers = niwot_reference_climate()
    full = drivers.xarray[["precipitation"]]
    short = drivers.head(40).xarray[["precipitation"]]
    assert short["time"].values[-1] not in full["time"].values
    stack = xr.concat(
        [full, short], dim="site", coords="different", compat="equals", join="outer"
    ).assign_coords(site=SITES)
    padded_full = stack.sel(site=SITES[0]).drop_vars("site")
    assert np.isnat(padded_full["timestep_length"].values).sum() == 1
    for run, padded in ((full, padded_full), (short, stack.sel(site=SITES[1]).drop_vars("site"))):
        xr.testing.assert_identical(
            resample(padded, "1D", how="sum"), resample(run, "1D", how="sum")
        )


def test_a_value_on_a_row_with_no_interval_is_refused_not_dropped(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]]
    lengths = ds["timestep_length"].values.copy()
    lengths[10] = np.timedelta64("NaT")
    ds = ds.assign_coords(timestep_length=("time", lengths, ds["timestep_length"].attrs))
    with pytest.raises(ValueError, match="net_ecosystem_exchange.*1 rows.*not padding"):
        resample(ds, "MS", how="sum")


def test_a_transposed_time_bounds_is_accepted(niwot):
    ds = niwot[2]
    assert ds.transpose()["time_bounds"].dims == ("bounds", "time")
    how = {"net_ecosystem_exchange": "sum"}
    xr.testing.assert_identical(
        resample(ds.transpose(), "1D", how=how), resample(ds, "1D", how=how)
    )


def test_time_coordinates_of_the_wrong_dtype_are_refused(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]]
    days = ds["timestep_length"].values.astype("int64") / 86_400e9
    in_days = ds.assign_coords(timestep_length=("time", days))
    with pytest.raises(ValueError, match="'timestep_length': 'float64'.*days_to_timedelta"):
        resample(in_days, "1D", how="sum")


def test_an_invalid_kind_attribute_is_named(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]].copy()
    ds["net_ecosystem_exchange"].attrs["kind"] = "flux"
    with pytest.raises(ValueError, match="'net_ecosystem_exchange' has kind 'flux', which is not"):
        resample(ds, "1D", how="sum")


def test_a_variable_without_time_is_refused(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]].assign(
        static=("site", [1.0], {"kind": "timestep_total"})
    )
    with pytest.raises(ValueError, match="'static' has dims.*nothing to resample"):
        resample(ds, "1D", how="sum")


# ── Exact step lengths ───────────────────────────────────────────────────────


def _record(length: np.timedelta64, n: int, start: str) -> xr.Dataset:
    starts = np.datetime64(start, "ns") + np.arange(n) * length
    lengths = np.full(n, length)
    coords = assemble_time_coords(
        start=starts,
        end=starts + lengths,
        length=lengths,
        attributes_for=lambda _name: {},
        length_source="declared",
        time_zone="UTC",
    )
    return xr.Dataset({"flux": ("time", np.ones(n), {"kind": "timestep_total"})}, coords=coords)


def test_combined_lengths_are_exact_to_the_nanosecond():
    step = np.timedelta64(3_600 * 10**9 + 1, "ns")
    # 2000 is a leap year, so the 8760 steps end well inside it.
    annual = resample(_record(step, 8760, "2000-01-01"), "YS", how="sum")
    assert annual.sizes["time"] == 1
    assert int(annual["timestep_length"].values.astype("int64")[0]) == 8760 * (3_600 * 10**9 + 1)


@pytest.mark.parametrize("freq", ["1D", "7D", "MS", "YS"])
def test_combined_lengths_are_the_sum_over_each_cell(climate, freq):
    resampled = resample(climate[["precipitation"]], freq, how="sum")
    ends = pd.DatetimeIndex(climate["time"].values)
    cells = pd.Series(climate["timestep_length"].values.astype("int64"), index=ends)
    expected = cells.resample(freq, closed="right", label="right").sum()
    expected = expected[cells.resample(freq, closed="right", label="right").count() > 0]
    np.testing.assert_array_equal(
        resampled["timestep_length"].values.astype("int64"), expected.to_numpy()
    )


# ── The public validity check ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kind", "method"),
    [
        (kind, method)
        for kind, valid in RESAMPLING_METHODS_FOR_KIND.items()
        for method in ("sum", "mean", "last")
        if method not in valid
    ],
)
def test_the_public_check_raises_what_resample_raises(kind, method):
    ds = _record(np.timedelta64(1, "h").astype("timedelta64[ns]"), 4, "2000-01-01")
    ds["flux"].attrs["kind"] = kind.value
    with pytest.raises(ValueError) as from_resample:
        resample(ds, "1D", how=method)
    with pytest.raises(ValueError) as from_check:
        check_resampling_method(kind, method, name="flux")
    assert str(from_check.value) == str(from_resample.value)
    expected = "Nothing is valid" if not RESAMPLING_METHODS_FOR_KIND[kind] else "Valid for this"
    assert expected in str(from_check.value)


def test_the_public_check_passes_valid_pairs_and_refuses_unknown_names():
    for kind, valid in RESAMPLING_METHODS_FOR_KIND.items():
        for method in valid:
            check_resampling_method(kind, method, name="x")
            check_resampling_method(kind.value, method, name="x")
    with pytest.raises(ValueError, match="Unknown resampling method 'max' for 'x'"):
        check_resampling_method(VariableKind.TIMESTEP_END_STATE, "max", name="x")
    with pytest.raises(ValueError, match="'x' has kind 'flux', which is not one of"):
        check_resampling_method("flux", "sum", name="x")


# ── Frequencies ──────────────────────────────────────────────────────────────


def test_upsampling_is_refused(niwot):
    with pytest.raises(
        ValueError,
        match="freq='1h' makes cells of at most 0 days 01:00:00.*"
        "shortest step in the data \\(0 days 07:00",
    ):
        resample(niwot[2][["net_ecosystem_exchange"]], "1h", how="sum")


def test_steps_a_hair_longer_than_the_frequency_are_not_upsampled():
    # 0.0416667 days is 1 h 2.88 ms, within the time axis's own tolerance.
    step = np.timedelta64(round(0.0416667 * 86_400e9), "ns")
    hourly = _record(step, 48, "2000-01-01")
    resampled = resample(hourly, "1h", how="sum")
    assert resampled.sizes["time"] == 48
    np.testing.assert_array_equal(resampled["flux"].values, hourly["flux"].values)


def test_a_frequency_between_unequal_steps_is_allowed(niwot):
    # Niwot's steps are about 7 and 17 hours; 12-hour cells hold at most one, but some do.
    resample(niwot[2][["net_ecosystem_exchange"]], "12h", how="sum")


@pytest.mark.parametrize("freq", ["0D", "-1D"])
def test_a_period_that_is_not_positive_is_refused(niwot, freq):
    with pytest.raises(ValueError, match="pass a positive frequency"):
        resample(niwot[2][["net_ecosystem_exchange"]], freq, how="sum")


def test_a_frequency_that_is_not_one_is_refused_with_pandas_reason(niwot):
    with pytest.raises(ValueError, match="pandas offset alias.*'bogus': Invalid frequency"):
        resample(niwot[2][["net_ecosystem_exchange"]], "bogus", how="sum")


def test_the_frequency_check_returns_the_offset():
    assert check_frequency("MS") == pd.offsets.MonthBegin(1)
    with pytest.raises(ValueError, match="pandas offset alias.*'bogus': Invalid frequency"):
        check_frequency("bogus")
    with pytest.raises(ValueError, match="pass a positive frequency"):
        check_frequency("0D")


def test_the_upsampling_check_is_resamples_own(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]]
    check_not_upsampling(ds, "12h")
    check_not_upsampling(ds["net_ecosystem_exchange"], "1D")
    with pytest.raises(ValueError, match="shortest step in the data \\(0 days 07:00"):
        check_not_upsampling(ds["net_ecosystem_exchange"], "1h")
    with pytest.raises(ValueError, match="pass a positive frequency"):
        check_not_upsampling(ds, "-1D")


def test_the_upsampling_check_ignores_padding(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]]
    padded = _two_runs_on_different_axes(ds).sel(site=SITES[1]).drop_vars("site")
    with pytest.raises(ValueError, match="shortest step in the data \\(0 days 07:00"):
        check_not_upsampling(padded, "1h")


def test_the_upsampling_check_measures_label_spacing_without_lengths():
    daily = _observed([1.0, 2.0, 3.0], "2020-01-02", "1D")
    check_not_upsampling(daily, "1D")
    with pytest.raises(ValueError, match="shortest step in the data \\(1 days"):
        check_not_upsampling(daily, "12h")
    check_not_upsampling(daily.isel(time=[0]), "1h")


# ── Public pieces of resample ────────────────────────────────────────────────


def test_a_dataarray_result_names_no_bounds_it_cannot_carry():
    output = niwot_reference_output()
    daily = resample(output["nee"], "1D", how="sum")
    assert "time_bounds" not in daily.coords
    assert "bounds" not in daily["time"].attrs
    assert daily["time"].attrs["time_zone"] == output["nee"]["time"].attrs["time_zone"]
    dataset = resample(output[["nee"]], "1D", how="sum")
    assert dataset["time"].attrs["bounds"] == "time_bounds"
    assert "time_bounds" in dataset.coords


def test_padding_is_dropped_from_a_dataset_and_a_dataarray(niwot):
    ds = niwot[2][["net_ecosystem_exchange", "wood_carbon"]]
    padded = _two_runs_on_different_axes(ds).sel(site=SITES[1]).drop_vars("site")
    assert padded.sizes["time"] == 60
    alone = ds.isel(time=slice(0, 40))
    kept = drop_padding(padded)
    xr.testing.assert_identical(kept["net_ecosystem_exchange"], alone["net_ecosystem_exchange"])
    xr.testing.assert_identical(
        drop_padding(padded["wood_carbon"]), padded["wood_carbon"].isel(time=slice(0, 40))
    )
    assert drop_padding(ds) is ds


def test_padding_with_a_value_is_refused_by_name(niwot):
    ds = niwot[2][["net_ecosystem_exchange", "wood_carbon"]]
    lengths = ds["timestep_length"].values.copy()
    lengths[10] = np.timedelta64("NaT")
    ds = ds.assign_coords(timestep_length=("time", lengths, ds["timestep_length"].attrs))
    with pytest.raises(ValueError, match="\\['net_ecosystem_exchange', 'wood_carbon'\\].*1 rows"):
        drop_padding(ds)
    with pytest.raises(ValueError, match="\\['wood_carbon'\\] have values.*not padding"):
        drop_padding(ds["wood_carbon"])
    unnamed = ds["wood_carbon"].rename(None)
    with pytest.raises(ValueError, match="\\['array'\\] have values"):
        drop_padding(unnamed)


def test_padding_needs_one_run_and_some_steps(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]]
    with pytest.raises(ValueError, match="vary along more than time"):
        drop_padding(_two_runs_on_different_axes(ds))
    empty = ds.assign_coords(
        timestep_start=("time", np.full(ds.sizes["time"], np.datetime64("NaT", "ns")))
    ).assign(net_ecosystem_exchange=ds["net_ecosystem_exchange"] * np.nan)
    with pytest.raises(ValueError, match="Every row's timestep_start"):
        drop_padding(empty)


def test_a_record_without_interval_coordinates_has_no_padding():
    observed = _observed([1.0, np.nan, 3.0], "2020-01-02", "1D")
    assert drop_padding(observed) is observed


def test_resampled_attributes_describe_the_combined_value(niwot):
    attrs = dict(niwot[2]["wood_carbon"].attrs)
    assert "output_decimals" in attrs
    averaged = resampled_attributes(attrs, "timestep_end_state", "mean", name="wood_carbon")
    assert averaged["kind"] == "timestep_mean"
    assert averaged["time_reference"] == "mean over the timestep"
    assert averaged["cell_methods"] == "time: mean"
    assert averaged["units"] == attrs["units"]
    assert "output_decimals" not in averaged
    assert "resampling" not in averaged
    assert attrs["kind"] == "timestep_end_state"

    cumulative = niwot[2]["cumulative_net_ecosystem_exchange"].attrs
    last = resampled_attributes(cumulative, VariableKind.CUMULATIVE, "last", name="x")
    assert "cell_methods" not in last
    with pytest.raises(ValueError, match="Cannot resample 'wood_carbon' with 'sum'"):
        resampled_attributes(attrs, "timestep_end_state", "sum", name="wood_carbon")


def test_resample_sets_the_public_attributes_and_says_how(niwot):
    ds = niwot[2]
    daily = resample(ds[["wood_carbon"]], "1D", how="mean")["wood_carbon"].attrs
    expected = resampled_attributes(
        ds["wood_carbon"].attrs, "timestep_end_state", "mean", name="wood_carbon"
    )
    assert {k: v for k, v in daily.items() if k != "resampling"} == expected
    assert daily["resampling"] == (
        "mean of timestep_end_state values over 1D, weighted by timestep_length"
    )


def test_a_variable_named_by_alias_resamples_by_its_registry_kind(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]].rename(net_ecosystem_exchange="nee")
    ds["nee"].attrs.clear()
    daily = resample(ds, "1D", how="sum")
    reference = resample(niwot[2]["net_ecosystem_exchange"], "1D", how="sum")
    np.testing.assert_array_equal(daily["nee"].values, reference.values)
    assert daily["nee"].attrs["kind"] == "timestep_total"
    with pytest.raises(ValueError, match="Cannot resample 'nee' with 'last'"):
        resample(ds, "1D", how="last")


# ── Records without interval coordinates ─────────────────────────────────────


def _observed(
    values: list[float], start: str, freq: str, kind: str = "timestep_total"
) -> xr.Dataset:
    time = pd.date_range(start, periods=len(values), freq=freq)
    return xr.Dataset(
        {"flux": ("time", np.asarray(values), {"kind": kind, "units": "g m-2"})},
        coords={"time": ("time", time, {"time_zone": "UTC"})},
        attrs={"source": "a flux tower"},
    )


def _without_intervals(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars([name for name in ds.coords if name != "time" and "time" in ds[name].dims])


@pytest.mark.parametrize("freq", ["1D", "7D", "MS"])
def test_labels_alone_sum_to_what_the_intervals_sum_to(niwot, freq):
    ds = niwot[2][["net_ecosystem_exchange", "wood_carbon"]]
    how = {"net_ecosystem_exchange": "sum", "wood_carbon": "last"}
    with_intervals = resample(ds, freq, how=how)
    labels_only = resample(_without_intervals(ds), freq, how=how)
    for name in how:
        np.testing.assert_array_equal(labels_only[name].values, with_intervals[name].values)


def test_labels_alone_are_labeled_at_the_cell_edge_and_carry_no_intervals():
    observed = _observed([1.0, 2.0, 3.0, 4.0], "2020-01-01 06:00", "12h")
    daily = resample(observed, "1D", how="sum")
    np.testing.assert_array_equal(
        daily["time"].values, pd.to_datetime(["2020-01-02", "2020-01-03"]).values
    )
    np.testing.assert_array_equal(daily["flux"].values, [3.0, 7.0])
    assert not {"timestep_start", "timestep_length", "time_bounds"} & set(daily.coords)
    assert "bounds" not in daily["time"].attrs
    assert daily["time"].attrs["time_zone"] == "UTC"
    assert daily["time"].attrs["long_name"] == "End of calendar cell"
    assert daily["flux"].attrs["resampling"] == (
        "sum of timestep_total values over calendar cells of 1D, labeled at each cell's right edge"
    )
    assert daily["flux"].attrs["cell_methods"] == "time: sum"
    assert daily.attrs["resampling_frequency"] == "1D"
    assert daily.attrs["source"] == "a flux tower"
    assert "timestep_length_source" not in daily.attrs


def test_labels_alone_drop_empty_cells_and_keep_nan_cells():
    time = pd.to_datetime(["2020-01-01 12:00", "2020-01-01 18:00", "2020-01-04 12:00"])
    observed = xr.Dataset(
        {"flux": ("time", [1.0, np.nan, 2.0], {"kind": "timestep_total"})}, coords={"time": time}
    )
    daily = resample(observed, "1D", how="sum")
    np.testing.assert_array_equal(
        daily["time"].values, pd.to_datetime(["2020-01-02", "2020-01-05"]).values
    )
    np.testing.assert_array_equal(daily["flux"].values, [np.nan, 2.0])


def test_labels_alone_take_the_last_value_as_the_intervals_do(niwot):
    # 'last' is the cell's last value on both paths: NaN only when that value is.
    ds = niwot[2][["wood_carbon"]].copy(deep=True)
    day = _daily_key(ds)
    last_of_its_day = np.flatnonzero(day[:-1] != day[1:])[3]
    ds["wood_carbon"][[last_of_its_day - 1, last_of_its_day]] = np.nan
    assert day[last_of_its_day - 1] == day[last_of_its_day]
    ds["wood_carbon"][last_of_its_day + 1] = np.nan
    with_intervals = resample(ds, "1D", how="last")["wood_carbon"].values
    labels_only = resample(_without_intervals(ds), "1D", how="last")["wood_carbon"].values
    np.testing.assert_array_equal(labels_only, with_intervals)
    # Three NaN steps, one of them the last of its day.
    assert np.isnan(labels_only).sum() == 1


def test_labels_alone_average_equally_spaced_steps_with_equal_weights():
    observed = _observed([1.0, 2.0, 3.0, 6.0], "2020-01-01 06:00", "12h", kind="timestep_mean")
    daily = resample(observed, "1D", how="mean")
    np.testing.assert_array_equal(daily["flux"].values, [1.5, 4.5])
    assert daily["flux"].attrs["resampling"].endswith("right edge, weighted equally")


def test_labels_alone_refuse_a_mean_over_unequal_steps(niwot):
    unequal = _without_intervals(niwot[2][["soil_wetness_fraction"]])
    with pytest.raises(ValueError, match="Cannot average \\['soil_wetness_fraction'\\].*not eq"):
        resample(unequal, "1D", how="mean")


def test_labels_alone_keep_the_kind_check_and_the_upsampling_refusal():
    observed = _observed([1.0, 2.0, 3.0], "2020-01-02", "1D")
    with pytest.raises(ValueError, match="Cannot resample 'flux' with 'mean'"):
        resample(observed, "7D", how="mean")
    observed["flux"].attrs.pop("kind")
    with pytest.raises(ValueError, match="attrs\\['kind'\\]"):
        resample(observed, "7D", how="sum")
    observed["flux"].attrs["kind"] = "timestep_total"
    with pytest.raises(ValueError, match="makes cells of at most 0 days 06:00"):
        resample(observed, "6h", how="sum")


def test_labels_alone_must_be_increasing_datetimes():
    observed = _observed([1.0, 2.0, 3.0], "2020-01-02", "1D")
    with pytest.raises(ValueError, match="do not strictly increase: row 2"):
        resample(observed.isel(time=[0, 1, 1]), "7D", how="sum")
    with pytest.raises(ValueError, match="datetime64"):
        resample(observed.assign_coords(time=[1, 2, 3]), "7D", how="sum")
    with pytest.raises(ValueError, match="no timesteps"):
        resample(observed.isel(time=slice(0, 0)), "7D", how="sum")


def test_labels_alone_carry_other_dimensions_and_resample_a_dataarray(climate):
    stacked = _without_intervals(_stack(climate[["precipitation"]]))
    monthly = resample(stacked["precipitation"], "MS", how="sum")
    assert isinstance(monthly, xr.DataArray)
    assert monthly.dims == stacked["precipitation"].dims
    for member in MEMBERS:
        for site in SITES:
            alone = resample(
                stacked["precipitation"].sel(member=member, site=site), "MS", how="sum"
            )
            np.testing.assert_allclose(monthly.sel(member=member, site=site).values, alone.values)


def test_padding_is_dropped_from_an_array_named_like_a_coordinate(niwot):
    ds = niwot[2][["net_ecosystem_exchange"]]
    padded = _two_runs_on_different_axes(ds).sel(site=SITES[1]).drop_vars("site")
    clash = padded["net_ecosystem_exchange"].rename("year")
    assert drop_padding(clash).sizes["time"] == 40


def test_labels_alone_average_labels_equal_within_the_step_tolerance():
    observed = _observed([1.0, 2.0, 3.0, 6.0], "2020-01-01 06:00", "12h", kind="timestep_mean")
    jittered = observed["time"].values + np.array([0, 3, -2, 1], dtype="timedelta64[ns]")
    daily = resample(observed.assign_coords(time=jittered), "1D", how="mean")
    np.testing.assert_array_equal(daily["flux"].values, [1.5, 4.5])


def test_the_upsampling_check_refuses_unsorted_labels_in_its_own_words():
    observed = _observed([1.0, 2.0, 3.0], "2020-01-02", "1D")
    with pytest.raises(ValueError, match="do not strictly increase: row 1"):
        check_not_upsampling(observed.isel(time=[1, 0, 2]), "7D")
