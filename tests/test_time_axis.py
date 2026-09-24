"""The time axis: built from the climate drivers, checked against their step lengths.

SIPNET has no clock. It echoes each climate row's labels into its output row,
rounded to 0.01 h, and integrates every row over its declared length without
ever checking the labels against it. These tests pin the three things pySIPNET
does about that:

- an output that knows its drivers takes their axis, row for row, and refuses
  drivers it cannot have been produced from;
- labels that overlap or drift from the declared lengths are refused, and a
  gap is warned about, with tolerances that Niwot's rounded lengths pass;
- the drivers can declare which clock they are on, and the declaration travels
  with the run.

The run tests use the real binary, because the claim is about what SIPNET
prints; they are skipped when it is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pysipnet.climate import ClimateDrivers, normalize_time_zone
from pysipnet.dataset import (
    DRIFT_TOLERANCE,
    STEP_TOLERANCE,
    TIME_AXIS_FROM_DRIVERS,
    TIME_AXIS_FROM_PRINTED_LABELS,
    check_step_continuity,
    days_to_timedelta,
    timestep_start,
)
from pysipnet.io.reference import (
    niwot_reference_climate,
    niwot_reference_files,
    niwot_reference_output,
)
from pysipnet.output import SIPNETOutput
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import SIPNETRunner

_SIPNET_BINARY = SIPNETRunner(flags=ModelFlags.standard()).binary_path
needs_binary = pytest.mark.skipif(
    not _SIPNET_BINARY.exists(),
    reason=f"SIPNET binary not found at {_SIPNET_BINARY}; run 'make sipnet'",
)


def _climate_frame(start: str, n_steps: int, step: pd.Timedelta) -> pd.DataFrame:
    """Drivers on a regular grid, labeled exactly, with plausible weather."""
    stamps = pd.date_range(start, periods=n_steps, freq=step)
    phase = 2 * np.pi * (stamps.hour + stamps.minute / 60) / 24
    return pd.DataFrame(
        {
            "year": stamps.year,
            "day_of_year": stamps.dayofyear,
            "hour_of_day": stamps.hour + stamps.minute / 60 + stamps.second / 3600,
            "time_step_length": step / pd.Timedelta(days=1),
            "air_temperature": 12.0 + 6.0 * np.sin(phase),
            "soil_temperature": 9.0,
            "photosynthetically_active_radiation": np.clip(np.sin(phase), 0, None)
            * 2.0
            * (step / pd.Timedelta(days=1)),
            "precipitation": 0.05,
            "vapor_pressure_deficit": 700.0,
            "soil_vapor_pressure_deficit": 350.0,
            "vapor_pressure": 900.0,
            "wind_speed": 2.0,
        }
    )


def _drifted_frame(years: list[int]) -> pd.DataFrame:
    """Three-hourly drivers whose hour labels are ``linspace(0, 24n - 1, 8n) % 24``.

    The labels advance 3.0007 h per step while the declared length stays
    0.125 days, so they run about 2.5 s per step late, reach 2 h late by
    31 December and reset each 1 January. The same construction as the ERA5
    files that prompted the check.
    """
    frames = []
    for year in years:
        n_days = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
        hours = np.linspace(0, 24 * n_days - 1, 8 * n_days)
        frame = _climate_frame(f"{year}-01-01", 8 * n_days, pd.Timedelta(hours=3))
        frame["day_of_year"] = (hours // 24).astype(int) + 1
        frame["hour_of_day"] = hours % 24
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _axis_parts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return timestep_start(frame), days_to_timedelta(frame["time_step_length"].to_numpy())


@pytest.fixture
def niwot_params():
    from tests.helpers import params_from_sipnet_file

    return params_from_sipnet_file(niwot_reference_files().param)


# ---------------------------------------------------------------------------
# The continuity check
# ---------------------------------------------------------------------------


class TestStepContinuity:
    def test_niwot_passes_with_margin(self):
        """SIPNET's own fixture rounds its lengths to three decimals; that is not an error.

        Also records how close it comes, so a change to the tolerances is
        measured against the case they were set for.
        """
        frame = niwot_reference_climate().pandas
        start, length = _axis_parts(frame)
        assert len(check_step_continuity(start, length)) == 0

        mismatch = (start[1:] - (start[:-1] + length[:-1])).astype("int64") / 1e9
        running = np.cumsum(mismatch)
        assert np.abs(mismatch).max() == pytest.approx(43.2)
        assert np.abs(running).max() == pytest.approx(100.8)
        assert np.abs(mismatch).max() < STEP_TOLERANCE / np.timedelta64(1, "s")
        assert 2 * np.abs(running).max() < DRIFT_TOLERANCE / np.timedelta64(1, "s")

    def test_a_regular_grid_passes(self):
        start, length = _axis_parts(_climate_frame("2012-01-01", 5000, pd.Timedelta(hours=3)))
        assert len(check_step_continuity(start, length)) == 0

    def test_a_slow_drift_is_caught(self):
        """2.5 s per step is far inside any per-step tolerance; only the running sum sees it."""
        with pytest.raises(ValueError, match="labels drift") as caught:
            ClimateDrivers.from_dataframe(_drifted_frame([2012, 2013]))
        message = str(caught.value)
        assert "row 122 " in message
        assert "2.46 s more" in message
        assert "also start before the previous one ends" in message

    def test_an_overlap_is_caught(self):
        frame = _climate_frame("2012-01-01", 200, pd.Timedelta(hours=3))
        frame.loc[100, "time_step_length"] = 0.25
        with pytest.raises(ValueError, match="Row 101 starts .* 3 h before row 100 ends"):
            ClimateDrivers.from_dataframe(frame)

    def test_an_overlap_too_short_to_drift_is_caught(self):
        """Two minutes is under the drift tolerance, so only the per-step check can see it."""
        frame = _climate_frame("2012-01-01", 200, pd.Timedelta(hours=3))
        frame.loc[100, "time_step_length"] += 120 / 86_400
        frame.loc[101, "time_step_length"] -= 120 / 86_400
        with pytest.raises(ValueError, match="overlap"):
            ClimateDrivers.from_dataframe(frame)

    def test_a_gap_is_a_warning_and_restarts_the_reconstruction(self):
        frame = _climate_frame("2012-01-01", 400, pd.Timedelta(hours=3))
        gapped = frame.drop(index=range(200, 208)).reset_index(drop=True)
        with pytest.warns(UserWarning, match="1 gap.* follows row 199.* 24 h later"):
            climate = ClimateDrivers.from_dataframe(gapped)
        start, length = _axis_parts(climate.pandas)
        assert list(check_step_continuity(start, length)) == [199]

        ds = climate.xarray
        assert ds["time"].values[199] == np.datetime64("2012-01-26T00:00")
        assert ds["time_step_start"].values[200] == np.datetime64("2012-01-27T00:00")

    def test_hourly_lengths_rounded_to_three_decimals_are_refused(self):
        """0.042 days is 24.19 h of forcing a day: the lengths, not the labels, are wrong."""
        frame = _climate_frame("2012-01-01", 48, pd.Timedelta(hours=1))
        frame["time_step_length"] = 0.042
        with pytest.raises(ValueError, match="labels drift.* 28.8 s less"):
            ClimateDrivers.from_dataframe(frame)

    def test_the_last_row_of_a_record_can_end_anywhere(self):
        frame = _climate_frame("2012-01-01", 10, pd.Timedelta(hours=3))
        frame.loc[9, "time_step_length"] = 5.0
        ClimateDrivers.from_dataframe(frame)


# ---------------------------------------------------------------------------
# The axis comes from the drivers
# ---------------------------------------------------------------------------


def _printed(climate: pd.DataFrame, **columns: np.ndarray) -> pd.DataFrame:
    """An output frame as SIPNET would print it from these drivers: hours to 0.01."""
    frame = climate[["year", "day_of_year", "hour_of_day"]].copy()
    frame["hour_of_day"] = frame["hour_of_day"].round(2)
    for name, values in columns.items():
        frame[name] = values
    return frame


class TestAxisFromDrivers:
    def test_the_axis_is_the_climate_axis_even_where_labels_do_not_print_exactly(self):
        """Twenty-minute steps print as 0.33, 0.67, ...; the axis must not."""
        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 72, pd.Timedelta(minutes=20))
        )
        output = SIPNETOutput.from_dataframe(
            _printed(climate.pandas, net_ecosystem_exchange=np.arange(72.0)), climate=climate
        )
        ds, cx = output.xarray, climate.xarray
        for name in ("time", "time_step_start", "time_step_length", "time_bounds", "hour_of_day"):
            np.testing.assert_array_equal(ds[name].values, cx[name].values)
        assert ds["time"].values[0] == np.datetime64("2012-06-01T00:20")
        assert ds.attrs["time_axis_source"] == TIME_AXIS_FROM_DRIVERS
        assert output.time_step_length is not None
        np.testing.assert_array_equal(output.time_step_length, climate.pandas["time_step_length"])

    def test_without_drivers_the_axis_is_the_printed_labels(self):
        """An output re-opened on its own can only use SIPNET's rounded labels, and says so."""
        climate = _climate_frame("2012-06-01", 72, pd.Timedelta(minutes=20))
        alone = SIPNETOutput.from_dataframe(
            _printed(climate, net_ecosystem_exchange=np.arange(72.0))
        ).xarray
        assert alone.attrs["time_axis_source"] == TIME_AXIS_FROM_PRINTED_LABELS
        assert alone.attrs["time_step_length_source"].startswith("inferred")
        assert alone["time"].attrs["time_zone"] == "undeclared"
        off = np.abs(alone["time_step_start"].values - timestep_start(climate))
        assert off.max() == np.timedelta64(12, "s")

    def test_a_row_count_mismatch_is_refused(self):
        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 10, pd.Timedelta(hours=3))
        )
        output = SIPNETOutput.from_dataframe(
            _printed(climate.pandas.head(9), net_ecosystem_exchange=np.arange(9.0)),
            climate=climate,
        )
        with pytest.raises(ValueError, match="9 rows but its climate drivers have 10"):
            _ = output.xarray
        with pytest.raises(ValueError, match="9 rows but its climate drivers have 10"):
            _ = output["nee"]

    def test_labels_from_other_drivers_are_refused(self):
        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 10, pd.Timedelta(hours=3))
        )
        printed = _printed(climate.pandas, net_ecosystem_exchange=np.arange(10.0))
        printed.loc[4, "hour_of_day"] += 0.01
        output = SIPNETOutput.from_dataframe(printed, climate=climate)
        with pytest.raises(ValueError, match="Row 4 is labelled .* not produced from these"):
            _ = output.xarray

    def test_the_niwot_reference_is_on_the_climate_axis(self):
        output = niwot_reference_output().xarray
        climate = niwot_reference_climate().xarray.isel(time=slice(0, output.sizes["time"]))
        np.testing.assert_array_equal(output["time"].values, climate["time"].values)
        assert output.attrs["time_axis_source"] == TIME_AXIS_FROM_DRIVERS

    def test_a_bare_frame_carrying_its_own_lengths_is_checked_when_placed(self):
        """Nothing else has checked a plain DataFrame, so building its axis does."""
        from pysipnet.output import build_output_dataset

        frame = _climate_frame("2012-06-01", 10, pd.Timedelta(hours=3))
        frame.loc[4, "time_step_length"] = 0.25
        with pytest.raises(ValueError, match="overlap"):
            build_output_dataset(frame[["year", "day_of_year", "hour_of_day", "time_step_length"]])


# ---------------------------------------------------------------------------
# Validation runs once, when the data is loaded
# ---------------------------------------------------------------------------


def _write_unchecked_clim(frame: pd.DataFrame, path) -> None:
    """Write a 12-column .clim without going through ClimateDrivers, which would refuse it."""
    from pysipnet.climate import CLIMATE_COLUMNS

    np.savetxt(path, frame[CLIMATE_COLUMNS].to_numpy(dtype=float), fmt="%.10g")


@pytest.fixture
def count_checks(monkeypatch):
    """How many times the continuity check has run, anywhere."""
    import pysipnet.dataset

    calls = {"n": 0}
    real = pysipnet.dataset.check_step_continuity

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pysipnet.dataset, "check_step_continuity", counting)
    return calls


class TestValidatesOnce:
    def test_the_constructor_validates(self):
        with pytest.raises(ValueError, match="labels drift"):
            ClimateDrivers(data=_drifted_frame([2012]))

    def test_the_constructor_normalizes_columns_like_from_dataframe(self):
        frame = _climate_frame("2012-06-01", 4, pd.Timedelta(hours=3)).rename(
            columns={"air_temperature": "tair"}
        )
        frame["extra"] = 1.0
        climate = ClimateDrivers(data=frame)
        assert "air_temperature" in climate.pandas.columns
        assert "extra" not in climate.pandas.columns
        with pytest.raises(ValueError, match="missing required columns"):
            ClimateDrivers(data=frame.drop(columns=["wind_speed"]))

    def test_in_memory_drivers_are_checked_once_however_they_are_used(self, count_checks):
        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 10, pd.Timedelta(hours=3))
        )
        assert count_checks["n"] == 1
        _ = climate.xarray
        output = SIPNETOutput.from_dataframe(
            _printed(climate.pandas, net_ecosystem_exchange=np.arange(10.0)), climate=climate
        )
        _ = output.xarray
        _ = climate.head(5).xarray
        assert count_checks["n"] == 1

    def test_a_file_is_checked_once_on_read(self, count_checks, tmp_path):
        path = tmp_path / "site.clim"
        _write_unchecked_clim(_climate_frame("2012-06-01", 10, pd.Timedelta(hours=3)), path)
        ClimateDrivers.from_file(path)
        assert count_checks["n"] == 1

    def test_a_file_backed_climate_is_checked_when_first_loaded(self, count_checks, tmp_path):
        path = tmp_path / "site.clim"
        _write_unchecked_clim(_climate_frame("2012-06-01", 10, pd.Timedelta(hours=3)), path)
        climate = ClimateDrivers.from_path(path)
        assert count_checks["n"] == 0
        _ = climate.pandas
        _ = climate.xarray
        climate.validate()  # already loaded: runs the checks again, on request
        assert count_checks["n"] == 2

    def test_a_file_backed_climate_defers_the_failure_until_loaded(self, tmp_path):
        path = tmp_path / "drifted.clim"
        _write_unchecked_clim(_drifted_frame([2012]), path)
        climate = ClimateDrivers.from_path(path)
        assert climate.n_timesteps == 8 * 366
        with pytest.raises(ValueError, match="labels drift"):
            climate.validate()
        with pytest.raises(ValueError, match="labels drift"):
            _ = climate.pandas


# ---------------------------------------------------------------------------
# The drivers declare the clock
# ---------------------------------------------------------------------------


class TestTimeZone:
    @pytest.mark.parametrize(
        ("given", "stored"),
        [
            (None, None),
            ("UTC", "UTC"),
            ("UTC-07:00", "UTC-07:00"),
            ("UTC+05:30", "UTC+05:30"),
            ("UTC+00:00", "UTC"),
            ("UTC-00:00", "UTC"),
        ],
    )
    def test_accepted_declarations(self, given, stored):
        assert normalize_time_zone(given) == stored

    @pytest.mark.parametrize("given", ["America/Denver", "MST", "utc", "UTC-7", "UTC+15:00", ""])
    def test_refused_declarations(self, given):
        with pytest.raises(ValueError, match="time_zone"):
            normalize_time_zone(given)

    def test_undeclared_by_default(self):
        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 4, pd.Timedelta(hours=3))
        )
        assert climate.time_zone is None
        assert climate.xarray["time"].attrs["time_zone"] == "undeclared"
        assert "no time zone" in climate.xarray.attrs["time_convention"]

    def test_declared_on_the_climate_dataset(self):
        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 4, pd.Timedelta(hours=3)), time_zone="UTC-07:00"
        )
        assert climate.xarray["time"].attrs["time_zone"] == "UTC-07:00"
        assert climate.xarray.attrs["time_zone"] == "UTC-07:00"
        assert "UTC-07:00" in repr(climate)

    def test_every_constructor_takes_it(self, tmp_path):
        path = tmp_path / "site.clim"
        ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 4, pd.Timedelta(hours=3))
        ).to_file(path)
        assert ClimateDrivers.from_file(path, time_zone="UTC").time_zone == "UTC"
        lazy = ClimateDrivers.from_path(path, time_zone="UTC+01:00")
        assert lazy.time_zone == "UTC+01:00"
        assert lazy.xarray["time"].attrs["time_zone"] == "UTC+01:00"

    def test_it_survives_a_saved_config(self, tmp_path, minimal_params):
        from pysipnet.config import RunConfig

        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 4, pd.Timedelta(hours=3)), time_zone="UTC-05:00"
        )
        config = RunConfig(flags=ModelFlags.standard(), params=minimal_params, climate=climate)
        config.save(tmp_path / "copy")
        assert (
            json.loads((tmp_path / "copy" / "config.json").read_text())["climate"]["time_zone"]
            == "UTC-05:00"
        )
        assert RunConfig.load(tmp_path / "copy").climate.time_zone == "UTC-05:00"

        referenced = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=ClimateDrivers.from_path(tmp_path / "copy" / "sipnet.clim", time_zone="UTC"),
        )
        referenced.save(tmp_path / "reference", reference_only=True)
        assert RunConfig.load(tmp_path / "reference").climate.time_zone == "UTC"

    def test_it_survives_resampling(self):
        from pysipnet.resample import resample

        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 16, pd.Timedelta(hours=3)), time_zone="UTC+02:00"
        )
        output = SIPNETOutput.from_dataframe(
            _printed(climate.pandas, net_ecosystem_exchange=np.arange(16.0)), climate=climate
        )
        daily = resample(output[["nee"]], "1D", how="sum")
        assert daily["time"].attrs["time_zone"] == "UTC+02:00"
        assert daily.attrs["time_zone"] == "UTC+02:00"


# ---------------------------------------------------------------------------
# Against the binary
# ---------------------------------------------------------------------------


@needs_binary
class TestRuns:
    def test_a_run_is_on_its_climate_axis_exactly(self, niwot_params):
        """Twenty-minute labels print as 0.33, 0.67; the run's axis must be the climate's."""
        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 3 * 72, pd.Timedelta(minutes=20)), time_zone="UTC-07:00"
        )
        result = SIPNETRunner(flags=ModelFlags.standard()).run(niwot_params, climate)
        assert result.provenance.success

        printed = result.outputs.pandas["hour_of_day"].to_numpy()
        assert not np.array_equal(printed, climate.pandas["hour_of_day"].to_numpy())

        ds, cx = result.outputs.xarray, climate.xarray
        for name in ("time", "time_step_start", "time_bounds", "hour_of_day", "day_of_year"):
            np.testing.assert_array_equal(ds[name].values, cx[name].values)
        assert ds.attrs["time_axis_source"] == TIME_AXIS_FROM_DRIVERS
        assert ds["time"].attrs["time_zone"] == ds.attrs["time_zone"] == "UTC-07:00"

    def test_a_file_backed_run_is_on_its_climate_axis(self, niwot_params, tmp_path):
        path = tmp_path / "site.clim"
        ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 72, pd.Timedelta(minutes=20))
        ).to_file(path)
        climate = ClimateDrivers.from_path(path, time_zone="UTC")
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "out")
        result = runner.run(niwot_params, climate, run_id="lazy")

        assert result.outputs.source_path is not None
        ds = result.outputs[["nee"]]
        np.testing.assert_array_equal(ds["time"].values, climate.xarray["time"].values)
        assert ds["time"].attrs["time_zone"] == "UTC"

    def test_niwot_runs_on_its_climate_axis(self, niwot_params):
        climate = niwot_reference_climate()
        result = SIPNETRunner(flags=ModelFlags.standard()).run(niwot_params, climate)
        np.testing.assert_array_equal(
            result.outputs.xarray["time"].values, climate.xarray["time"].values
        )

    def test_a_drifted_file_runs_but_has_no_time_axis(self, niwot_params, tmp_path):
        """SIPNET runs it happily; it is the time axis that would be wrong, so that refuses."""
        path = tmp_path / "drifted.clim"
        drifted = _drifted_frame([2012])
        _write_unchecked_clim(drifted, path)
        climate = ClimateDrivers.from_path(path)
        result = SIPNETRunner(flags=ModelFlags.standard()).run(niwot_params, climate)

        assert result.provenance.success
        assert len(result.outputs.pandas) == len(drifted)
        with pytest.raises(ValueError, match="labels drift"):
            _ = result.outputs["nee"]

    def test_a_real_output_with_the_wrong_climate_is_refused(self, niwot_params, tmp_path):
        climate = ClimateDrivers.from_dataframe(
            _climate_frame("2012-06-01", 24, pd.Timedelta(hours=3))
        )
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "out")
        result = runner.run(niwot_params, climate, run_id="full")
        assert result.outputs.source_path is not None

        shorter = climate.head(23)
        with pytest.raises(ValueError, match="24 rows but its climate drivers have 23"):
            _ = SIPNETOutput.from_path(result.outputs.source_path, climate=shorter).xarray
