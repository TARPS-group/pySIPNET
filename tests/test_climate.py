"""Tests for ClimateDrivers: construction, validation, properties, and file IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysipnet.climate import CLIMATE_COLUMNS, ClimateDrivers
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import ClimateStaging, SIPNETRunner


def _make_df(
    n_days: int = 5,
    start_doy: int = 100,
    year: int = 2020,
) -> pd.DataFrame:
    """Return a minimal valid daily climate DataFrame."""
    return pd.DataFrame(
        {
            "year": year,
            "day_of_year": range(start_doy, start_doy + n_days),
            "hour_of_day": 0.0,
            "timestep_length": 1.0,
            # Every column a distinct, non-round value. With repeated or round
            # numbers a round trip proves only the shape: swapping two columns
            # in the writer, or truncating precision, would still compare equal.
            "air_temperature": 15.3125,
            "soil_temperature": 10.0625,
            "photosynthetically_active_radiation": 21.8437,
            "precipitation": 2.1875,
            "vapor_pressure_deficit": 803.40625,
            "soil_vapor_pressure_deficit": 401.703125,
            "vapor_pressure": 1203.28125,
            "wind_speed": 2.546875,
        }
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFromDataframe:
    def test_happy_path(self):
        cd = ClimateDrivers.from_dataframe(_make_df())
        assert cd.n_timesteps == 5
        assert list(cd.pandas.columns) == CLIMATE_COLUMNS

    def test_extra_columns_ignored(self):
        df = _make_df()
        df["extra"] = 99.0
        cd = ClimateDrivers.from_dataframe(df)
        assert "extra" not in cd.pandas.columns

    def test_column_order_normalised(self):
        df = _make_df()[list(reversed(CLIMATE_COLUMNS))]
        cd = ClimateDrivers.from_dataframe(df)
        assert list(cd.pandas.columns) == CLIMATE_COLUMNS

    def test_missing_column_raises(self):
        df = _make_df().drop(columns=["photosynthetically_active_radiation"])
        with pytest.raises(ValueError, match="missing required columns"):
            ClimateDrivers.from_dataframe(df)

    def test_layout_stored(self):
        cd = ClimateDrivers.from_dataframe(_make_df(), n_columns=14, loc=7)
        assert cd.n_columns == 14
        assert cd.loc == 7

    def test_data_is_a_copy(self):
        df = _make_df()
        cd = ClimateDrivers.from_dataframe(df)
        df["air_temperature"] = 999.0
        assert (cd.pandas["air_temperature"] != 999.0).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_null_values_raise(self):
        df = _make_df()
        df.loc[2, "air_temperature"] = float("nan")
        with pytest.raises(ValueError, match="Missing values"):
            ClimateDrivers.from_dataframe(df)

    def test_zero_length_raises(self):
        df = _make_df()
        df.loc[0, "timestep_length"] = 0.0
        with pytest.raises(ValueError, match="timestep_length"):
            ClimateDrivers.from_dataframe(df)

    def test_negative_length_raises(self):
        df = _make_df()
        df.loc[0, "timestep_length"] = -1.0
        with pytest.raises(ValueError, match="timestep_length"):
            ClimateDrivers.from_dataframe(df)

    def test_non_monotonic_doy_raises(self):
        df = _make_df(n_days=5)
        df.loc[1, "day_of_year"] = 99  # goes backward
        with pytest.raises(ValueError, match="chronological"):
            ClimateDrivers.from_dataframe(df)

    def test_non_monotonic_year_raises(self):
        df = _make_df(n_days=4)
        df.loc[2, "year"] = 2019  # year goes backward
        with pytest.raises(ValueError, match="chronological"):
            ClimateDrivers.from_dataframe(df)

    def test_zero_vpd_warns(self):
        df = _make_df()
        df.loc[0, "vapor_pressure_deficit"] = 0.0
        with pytest.warns(UserWarning, match="vapor_pressure_deficit"):
            ClimateDrivers.from_dataframe(df)

    def test_negative_vpd_warns(self):
        df = _make_df()
        df.loc[0, "vapor_pressure_deficit"] = -50.0
        with pytest.warns(UserWarning, match="vapor_pressure_deficit"):
            ClimateDrivers.from_dataframe(df)

    def test_zero_wspd_warns(self):
        df = _make_df()
        df.loc[0, "wind_speed"] = 0.0
        with pytest.warns(UserWarning, match="wind_speed"):
            ClimateDrivers.from_dataframe(df)

    def test_positive_vpd_and_wspd_no_warning(self, recwarn):
        df = _make_df()
        ClimateDrivers.from_dataframe(df)
        vpd_wspd = [
            w
            for w in recwarn.list
            if "vapor_pressure_deficit" in str(w.message).lower()
            or "wind_speed" in str(w.message).lower()
        ]
        assert len(vpd_wspd) == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_n_timesteps(self):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=12))
        assert cd.n_timesteps == 12

    def test_date_range_single_year(self):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=5, start_doy=100, year=2020))
        (y0, d0), (y1, d1) = cd.date_range
        assert (y0, d0) == (2020, 100)
        assert (y1, d1) == (2020, 104)

    def test_date_range_multi_year(self):
        df1 = _make_df(n_days=3, start_doy=363, year=2019)
        df2 = _make_df(n_days=3, start_doy=1, year=2020)
        df = pd.concat([df1, df2], ignore_index=True)
        cd = ClimateDrivers.from_dataframe(df)
        (y0, d0), (y1, d1) = cd.date_range
        assert (y0, d0) == (2019, 363)
        assert (y1, d1) == (2020, 3)

    def test_repr_contains_key_info(self):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=5, start_doy=100, year=2020))
        r = repr(cd)
        assert "ClimateDrivers" in r
        assert "n_columns=12" in r
        assert "5" in r  # timestep count


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------


def _rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


class TestFileIO:
    """The layout is read from the file, as SIPNET reads it, never stated by the caller."""

    @pytest.mark.parametrize("n_columns", [12, 14])
    def test_roundtrip_in_either_layout(self, tmp_path, n_columns):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=7), n_columns=n_columns)
        path = tmp_path / "test.clim"
        cd.to_file(path)
        cd2 = ClimateDrivers.from_file(path)
        assert cd2.n_columns == n_columns
        pd.testing.assert_frame_equal(
            cd.pandas.reset_index(drop=True),
            cd2.pandas.reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )

    def test_new_drivers_are_written_in_the_standard_layout(self, tmp_path):
        """12 columns, starting with the year: what SIPNET has written since v2.0.0."""
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3, year=2021))
        assert cd.n_columns == 12
        path = tmp_path / "test.clim"
        cd.to_file(path)
        rows = _rows(path)
        assert {len(row) for row in rows} == {12}
        assert rows[0][0] == "2021"

    def test_the_legacy_layout_wraps_the_values_in_loc_and_soil_wetness(self, tmp_path):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3), n_columns=14, loc=42)
        path = tmp_path / "test.clim"
        cd.to_file(path)
        rows = _rows(path)
        assert {len(row) for row in rows} == {14}
        assert {row[0] for row in rows} == {"42"}
        assert ClimateDrivers.from_file(path).loc == 42

    def test_n_rows_matches_n_timesteps(self, tmp_path):
        n = 10
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=n))
        path = tmp_path / "test.clim"
        cd.to_file(path)
        assert len(_rows(path)) == n

    def test_thirteen_columns_are_refused_as_sipnet_refuses_them(self, tmp_path):
        """The legacy layout minus its site column: an earlier SIPNET read it, this one does not."""
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=5), n_columns=14)
        path = tmp_path / "test.clim"
        cd.to_file(path)
        path13 = tmp_path / "test13.clim"
        path13.write_text("\n".join(" ".join(row[1:]) for row in _rows(path)) + "\n")
        with pytest.raises(ValueError, match="13 columns.* drop the trailing soil-wetness"):
            ClimateDrivers.from_file(path13)
        with pytest.raises(ValueError, match="13 columns"):
            ClimateDrivers.from_path(path13)

    @pytest.mark.parametrize("n_fields", [5, 11, 15])
    def test_any_other_column_count_is_refused(self, tmp_path, n_fields):
        path = tmp_path / "bad.clim"
        path.write_text(" ".join(["1"] * n_fields) + "\n")
        with pytest.raises(ValueError, match=f"has {n_fields} columns"):
            ClimateDrivers.from_file(path)

    def test_a_leading_blank_line_is_refused(self, tmp_path):
        """SIPNET reads the layout from the first line, so it cannot skip a blank one."""
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3))
        path = tmp_path / "test.clim"
        cd.to_file(path)
        path.write_text("\n" + path.read_text())
        with pytest.raises(ValueError, match="blank line"):
            ClimateDrivers.from_file(path)

    def test_rows_in_different_layouts_are_refused(self, tmp_path):
        path = tmp_path / "mixed.clim"
        ClimateDrivers.from_dataframe(_make_df(n_days=2)).to_file(path)
        legacy = tmp_path / "legacy.clim"
        ClimateDrivers.from_dataframe(_make_df(n_days=1, start_doy=102), n_columns=14).to_file(
            legacy
        )
        path.write_text(path.read_text() + legacy.read_text())
        with pytest.raises(ValueError):
            ClimateDrivers.from_file(path)

    def test_a_legacy_file_naming_two_sites_is_refused(self, tmp_path):
        """SIPNET runs one site per file and errors if the site column changes."""
        path = tmp_path / "two_sites.clim"
        ClimateDrivers.from_dataframe(_make_df(n_days=4), n_columns=14).to_file(path)
        rows = _rows(path)
        rows[2][0] = "1"
        path.write_text("\n".join(" ".join(row) for row in rows) + "\n")
        with pytest.raises(ValueError, match=r"2 locations in its site column \(0, 1\)"):
            ClimateDrivers.from_file(path)

    def test_only_the_two_sipnet_layouts_can_be_chosen(self):
        with pytest.raises(ValueError, match="12 or 14"):
            ClimateDrivers.from_dataframe(_make_df(), n_columns=13)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# from_path (file-backed / lazy)
# ---------------------------------------------------------------------------


class TestFromPath:
    def _write(self, tmp_path, n_days=5, **kwargs) -> tuple[ClimateDrivers, Path]:
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=n_days, **kwargs))
        path = tmp_path / "test.clim"
        cd.to_file(path)
        return cd, path

    def test_data_not_loaded_on_construction(self, tmp_path):
        _, path = self._write(tmp_path)
        ref = ClimateDrivers.from_path(path)
        assert ref._data is None

    def test_source_path_stored(self, tmp_path):
        _, path = self._write(tmp_path)
        ref = ClimateDrivers.from_path(path)
        assert ref.source_path == path

    def test_n_timesteps_without_load(self, tmp_path):
        _, path = self._write(tmp_path, n_days=7)
        ref = ClimateDrivers.from_path(path)
        assert ref.n_timesteps == 7
        assert ref._data is None

    def test_date_range_without_load(self, tmp_path):
        _, path = self._write(tmp_path, start_doy=100, year=2020)
        ref = ClimateDrivers.from_path(path)
        (y0, d0), (y1, d1) = ref.date_range
        assert (y0, d0) == (2020, 100)
        assert (y1, d1) == (2020, 104)
        assert ref._data is None

    def test_data_lazy_loads_on_first_access(self, tmp_path):
        _, path = self._write(tmp_path)
        ref = ClimateDrivers.from_path(path)
        assert ref._data is None
        _ = ref.pandas
        assert ref._data is not None

    def test_data_cached_after_first_access(self, tmp_path):
        _, path = self._write(tmp_path)
        ref = ClimateDrivers.from_path(path)
        df1 = ref.pandas
        df2 = ref.pandas
        assert df1 is df2

    def test_data_matches_original(self, tmp_path):
        cd, path = self._write(tmp_path)
        ref = ClimateDrivers.from_path(path)
        pd.testing.assert_frame_equal(
            cd.pandas.reset_index(drop=True),
            ref.pandas.reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ClimateDrivers.from_path("/nonexistent/path/missing.clim")

    def test_bad_column_count_raises_at_construction(self, tmp_path):
        """The layout is checked from the first line, so it cannot wait for the lazy read."""
        bad = tmp_path / "bad.clim"
        bad.write_text("1 2 3\n4 5 6\n")
        with pytest.raises(ValueError, match="has 3 columns"):
            ClimateDrivers.from_path(bad)

    @pytest.mark.parametrize("n_columns", [12, 14])
    def test_layout_read_from_the_file(self, tmp_path, n_columns):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3), n_columns=n_columns, loc=9)
        path = tmp_path / "test.clim"
        cd.to_file(path)
        ref = ClimateDrivers.from_path(path)
        assert ref.n_columns == n_columns
        assert ref.n_timesteps == 3
        _ = ref.pandas
        assert ref.loc == (9 if n_columns == 14 else 0)

    def test_interior_blank_lines_do_not_move_the_last_row(self, tmp_path):
        """SIPNET skips them, so the metadata must still describe the last real row."""
        path = tmp_path / "gappy.clim"
        ClimateDrivers.from_dataframe(_make_df(n_days=4, start_doy=10)).to_file(path)
        lines = path.read_text().splitlines()
        path.write_text("\n".join([lines[0], "", lines[1], "", "", *lines[2:]]) + "\n\n")
        ref = ClimateDrivers.from_path(path)
        assert ref.n_timesteps == 4
        assert ref.date_range == ((2020, 10), (2020, 13))

    def test_a_str_path_is_accepted(self, tmp_path):
        path = tmp_path / "test.clim"
        ClimateDrivers.from_dataframe(_make_df(n_days=3)).to_file(path)
        ref = ClimateDrivers(source_path=str(path))
        assert ref.source_path == path
        assert ref.n_timesteps == 3

    def test_loading_refreshes_what_was_peeked(self, tmp_path):
        """A file replaced between construction and load is described as it is read."""
        path = tmp_path / "test.clim"
        ClimateDrivers.from_dataframe(_make_df(n_days=3)).to_file(path)
        ref = ClimateDrivers.from_path(path)
        assert ref.n_columns == 12
        ClimateDrivers.from_dataframe(_make_df(n_days=3), n_columns=14, loc=5).to_file(path)
        _ = ref.pandas
        assert (ref.n_columns, ref.loc) == (14, 5)

    def test_a_file_backed_layout_cannot_be_stated(self, tmp_path):
        path = tmp_path / "test.clim"
        ClimateDrivers.from_dataframe(_make_df(n_days=3)).to_file(path)
        with pytest.raises(ValueError, match="read from the file"):
            ClimateDrivers(source_path=path, n_columns=12)

    def test_repr_does_not_load_data(self, tmp_path):
        _, path = self._write(tmp_path, n_days=5, start_doy=100, year=2021)
        ref = ClimateDrivers.from_path(path)
        r = repr(ref)
        assert "ClimateDrivers" in r
        assert "5" in r
        assert ref._data is None


# ---------------------------------------------------------------------------
# ClimateStaging (runner file staging logic)
# ---------------------------------------------------------------------------


class TestClimateStaging:
    def _runner(self, staging: ClimateStaging) -> SIPNETRunner:
        return SIPNETRunner(flags=ModelFlags.standard(), climate_staging=staging)

    def test_in_memory_writes_file(self, tmp_path):
        runner = self._runner(ClimateStaging.COPY)
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3))
        dest = tmp_path / "out.clim"
        runner._stage_clim_file(cd, dest)
        assert dest.exists()
        lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3

    def test_in_memory_symlink_mode_still_writes(self, tmp_path):
        runner = self._runner(ClimateStaging.SYMLINK)
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3))
        dest = tmp_path / "out.clim"
        runner._stage_clim_file(cd, dest)
        assert dest.exists()

    def test_file_backed_copy(self, tmp_path):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=5))
        src = tmp_path / "src.clim"
        cd.to_file(src)
        ref = ClimateDrivers.from_path(src)

        runner = self._runner(ClimateStaging.COPY)
        dest = tmp_path / "dest.clim"
        runner._stage_clim_file(ref, dest)

        assert dest.exists()
        assert not dest.is_symlink()
        assert dest.read_text() == src.read_text()

    def test_file_backed_copy_does_not_load_data(self, tmp_path):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=5))
        src = tmp_path / "src.clim"
        cd.to_file(src)
        ref = ClimateDrivers.from_path(src)

        runner = self._runner(ClimateStaging.COPY)
        runner._stage_clim_file(ref, tmp_path / "dest.clim")
        assert ref._data is None

    def test_file_backed_symlink(self, tmp_path):
        import sys

        if sys.platform == "win32":
            pytest.skip("Symlinks require elevated privileges on Windows")

        cd = ClimateDrivers.from_dataframe(_make_df(n_days=5))
        src = tmp_path / "src.clim"
        cd.to_file(src)
        ref = ClimateDrivers.from_path(src)

        runner = self._runner(ClimateStaging.SYMLINK)
        dest = tmp_path / "dest.clim"
        runner._stage_clim_file(ref, dest)

        assert dest.is_symlink()
        assert dest.resolve() == src.resolve()

    def test_default_staging_is_copy(self):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        assert runner.climate_staging == ClimateStaging.COPY


# ---------------------------------------------------------------------------
# Registry names, aliases and the xarray view
# ---------------------------------------------------------------------------


class TestClimateRegistry:
    def test_columns_are_the_registry_names(self):
        from pysipnet.variables import CLIMATE_COLUMN_NAMES

        assert CLIMATE_COLUMNS == list(CLIMATE_COLUMN_NAMES)
        assert "air_temperature" in CLIMATE_COLUMNS and "tair" not in CLIMATE_COLUMNS

    def test_from_dataframe_accepts_aliases_and_renames_them(self):
        """A DataFrame using the previous short names or SIPNET's names still loads."""
        df = _make_df().rename(
            columns={
                "day_of_year": "day",
                "hour_of_day": "time",
                "timestep_length": "length",
                "air_temperature": "tair",
                "soil_temperature": "tsoil",
                "photosynthetically_active_radiation": "par",
                "precipitation": "precip",
                "vapor_pressure_deficit": "vpd",
                "soil_vapor_pressure_deficit": "vpdSoil",
                "vapor_pressure": "vPress",
                "wind_speed": "wspd",
            }
        )
        cd = ClimateDrivers.from_dataframe(df)
        assert list(cd.pandas.columns) == CLIMATE_COLUMNS
        pd.testing.assert_frame_equal(cd.pandas, ClimateDrivers.from_dataframe(_make_df()).pandas)

    def test_missing_column_error_names_the_registry_name(self):
        with pytest.raises(ValueError, match="air_temperature"):
            ClimateDrivers.from_dataframe(_make_df().drop(columns=["air_temperature"]))

    def test_dataset_shares_the_output_time_axis(self):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3, start_doy=100, year=2020))
        ds = cd.xarray
        assert dict(ds.sizes) == {"time": 3, "bounds": 2}
        assert ds["air_temperature"].dims == ("time",)
        assert ds["timestep_start"].values[0] == np.datetime64("2020-04-09T00:00")
        assert ds["time"].values[0] == np.datetime64("2020-04-10T00:00")
        assert ds["time"].values[0] == ds["timestep_start"].values[1]
        assert ds["time"].attrs["bounds"] == "time_bounds"
        assert ds["air_temperature"].attrs["units"] == "degC"
        assert ds["precipitation"].attrs["time_reference"] == "total over the timestep"
        assert ds["precipitation"].attrs["sipnet_internal_units"] == "cm"
        assert "timestep_length" in ds.coords and "timestep_length" not in ds.data_vars

    def test_alias_and_canonical_column_together_is_an_error(self):
        df = _make_df()
        df["tair"] = df["air_temperature"] + 1.0
        with pytest.raises(ValueError, match="canonical name"):
            ClimateDrivers.from_dataframe(df)
