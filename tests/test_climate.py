"""Tests for ClimateDrivers: construction, validation, properties, and file IO."""

from __future__ import annotations

import pandas as pd
import pytest

from pysipnet.climate import CLIM_COLUMNS_V1, ClimateDrivers


def _make_df(
    n_days: int = 5,
    start_doy: int = 100,
    year: int = 2020,
) -> pd.DataFrame:
    """Return a minimal valid daily climate DataFrame."""
    return pd.DataFrame({
        "year": year,
        "day": range(start_doy, start_doy + n_days),
        "time": 0.0,
        "length": 1.0,
        "tair": 15.0,
        "tsoil": 10.0,
        "par": 10.0,
        "precip": 2.0,
        "vpd": 800.0,
        "vpd_soil": 400.0,
        "vpress": 1200.0,
        "wspd": 2.5,
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFromDataframe:
    def test_happy_path(self):
        cd = ClimateDrivers.from_dataframe(_make_df())
        assert cd.n_timesteps == 5
        assert list(cd.data.columns) == CLIM_COLUMNS_V1

    def test_extra_columns_ignored(self):
        df = _make_df()
        df["extra"] = 99.0
        cd = ClimateDrivers.from_dataframe(df)
        assert "extra" not in cd.data.columns

    def test_column_order_normalised(self):
        df = _make_df()[list(reversed(CLIM_COLUMNS_V1))]
        cd = ClimateDrivers.from_dataframe(df)
        assert list(cd.data.columns) == CLIM_COLUMNS_V1

    def test_missing_column_raises(self):
        df = _make_df().drop(columns=["par"])
        with pytest.raises(ValueError, match="missing required columns"):
            ClimateDrivers.from_dataframe(df)

    def test_version_stored(self):
        cd = ClimateDrivers.from_dataframe(_make_df(), version="v1", loc=7)
        assert cd.version == "v1"
        assert cd.loc == 7

    def test_data_is_a_copy(self):
        df = _make_df()
        cd = ClimateDrivers.from_dataframe(df)
        df["tair"] = 999.0
        assert (cd.data["tair"] != 999.0).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_null_values_raise(self):
        df = _make_df()
        df.loc[2, "tair"] = float("nan")
        with pytest.raises(ValueError, match="Missing values"):
            ClimateDrivers.from_dataframe(df)

    def test_zero_length_raises(self):
        df = _make_df()
        df.loc[0, "length"] = 0.0
        with pytest.raises(ValueError, match="length"):
            ClimateDrivers.from_dataframe(df)

    def test_negative_length_raises(self):
        df = _make_df()
        df.loc[0, "length"] = -1.0
        with pytest.raises(ValueError, match="length"):
            ClimateDrivers.from_dataframe(df)

    def test_non_monotonic_doy_raises(self):
        df = _make_df(n_days=5)
        df.loc[1, "day"] = 99  # goes backward
        with pytest.raises(ValueError, match="chronological"):
            ClimateDrivers.from_dataframe(df)

    def test_non_monotonic_year_raises(self):
        df = _make_df(n_days=4)
        df.loc[2, "year"] = 2019  # year goes backward
        with pytest.raises(ValueError, match="chronological"):
            ClimateDrivers.from_dataframe(df)

    def test_zero_vpd_warns(self):
        df = _make_df()
        df.loc[0, "vpd"] = 0.0
        with pytest.warns(UserWarning, match="vpd"):
            ClimateDrivers.from_dataframe(df)

    def test_negative_vpd_warns(self):
        df = _make_df()
        df.loc[0, "vpd"] = -50.0
        with pytest.warns(UserWarning, match="vpd"):
            ClimateDrivers.from_dataframe(df)

    def test_zero_wspd_warns(self):
        df = _make_df()
        df.loc[0, "wspd"] = 0.0
        with pytest.warns(UserWarning, match="wspd"):
            ClimateDrivers.from_dataframe(df)

    def test_positive_vpd_and_wspd_no_warning(self, recwarn):
        df = _make_df()
        ClimateDrivers.from_dataframe(df)
        vpd_wspd = [
            w for w in recwarn.list
            if "vpd" in str(w.message).lower() or "wspd" in str(w.message).lower()
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
        assert "v1" in r
        assert "5" in r  # timestep count


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------

class TestFileIO:
    def test_roundtrip_v1(self, tmp_path):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=7))
        path = tmp_path / "test.clim"
        cd.to_file(path)
        cd2 = ClimateDrivers.from_file(path, version="v1")
        pd.testing.assert_frame_equal(
            cd.data.reset_index(drop=True),
            cd2.data.reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )

    def test_v1_file_has_14_columns(self, tmp_path):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3))
        path = tmp_path / "test.clim"
        cd.to_file(path)
        first_line = path.read_text().splitlines()[0]
        assert len(first_line.split()) == 14

    def test_loc_written_to_first_column(self, tmp_path):
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=3), loc=42)
        path = tmp_path / "test.clim"
        cd.to_file(path)
        first_line = path.read_text().splitlines()[0]
        assert first_line.split()[0] == "42"

    def test_n_rows_matches_n_timesteps(self, tmp_path):
        n = 10
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=n))
        path = tmp_path / "test.clim"
        cd.to_file(path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == n

    def test_from_file_13_column_format(self, tmp_path):
        """13-column files (without loc column) are also accepted."""
        cd = ClimateDrivers.from_dataframe(_make_df(n_days=5))
        path = tmp_path / "test.clim"
        cd.to_file(path)

        # Strip the loc column to produce a 13-col file
        lines = path.read_text().splitlines()
        stripped = "\n".join(" ".join(row.split()[1:]) for row in lines) + "\n"
        path13 = tmp_path / "test13.clim"
        path13.write_text(stripped)

        cd13 = ClimateDrivers.from_file(path13, version="v1")
        pd.testing.assert_frame_equal(
            cd.data.reset_index(drop=True),
            cd13.data.reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )

    def test_from_file_wrong_column_count_raises(self, tmp_path):
        path = tmp_path / "bad.clim"
        path.write_text("1 2 3 4 5\n6 7 8 9 10\n")
        with pytest.raises(ValueError, match="Expected 13 or 14 columns"):
            ClimateDrivers.from_file(path, version="v1")
