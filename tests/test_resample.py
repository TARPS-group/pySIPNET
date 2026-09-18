"""resample(): explicit, kind-checked coarsening of the time axis.

Checked against SIPNET's own Niwot record, whose day and night steps differ in
length (0.29 to 0.63 days), so a plain mean and a length-weighted mean disagree
and the tests can tell them apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pysipnet import resample
from pysipnet.climate import ClimateDrivers
from pysipnet.io.clim_io import read_clim_file
from pysipnet.output import build_output_dataset
from pysipnet.variables import RESAMPLED_KIND, RESAMPLING_METHODS_FOR_KIND, VariableKind

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def niwot() -> tuple[pd.DataFrame, np.ndarray, xr.Dataset]:
    frame = pd.read_csv(FIXTURES / "golden" / "niwot_standard.out.csv")
    climate = read_clim_file(FIXTURES / "niwot_reference" / "sipnet.clim", n_columns=14)
    length = climate.pandas["time_step_length"].to_numpy()[: len(frame)]
    return frame, length, build_output_dataset(frame, time_step_length=length)


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
    assert daily.attrs["resampling"].endswith("weighted by time_step_length")
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
    np.testing.assert_array_equal(
        daily["time_bounds"].values[:, 0], daily["time_step_start"].values
    )
    # Cells are contiguous and tile the record exactly.
    assert daily["time_step_start"].values[0] == ds["time_step_start"].values[0]
    assert daily["time"].values[-1] == ds["time"].values[-1]
    np.testing.assert_array_equal(daily["time_step_start"].values[1:], daily["time"].values[:-1])
    # time_step_length is the coverage: the declared lengths that went in.
    total = ds["time_step_length"].values.astype("int64").sum()
    assert daily["time_step_length"].values.astype("int64").sum() == total
    assert daily.attrs["time_step_length_source"].startswith("sum of the declared")
    for name in daily.coords:
        assert daily[name].encoding["_FillValue"] is None


def test_row_labels_are_the_start_of_the_coarser_step(niwot):
    _, _, ds = niwot
    daily = resample(ds[["net_ecosystem_exchange"]], "1D", how="sum")
    starts = pd.DatetimeIndex(daily["time_step_start"].values)
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


def test_a_dataset_without_the_time_layout_is_refused():
    plain = xr.Dataset(
        {"x": ("time", [1.0, 2.0])}, coords={"time": pd.date_range("2020", periods=2)}
    )
    with pytest.raises(ValueError, match="time_step_start"):
        resample(plain, "1D", how="sum")


def test_a_daily_record_resamples_daily_to_itself():
    frame = pd.DataFrame(
        {
            "year": 2020,
            "day_of_year": range(100, 105),
            "hour_of_day": 0.0,
            "time_step_length": 1.0,
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
