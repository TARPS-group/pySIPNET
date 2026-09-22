"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sipnet_source_params() -> set[str]:
    """Every parameter name the pinned SIPNET source registers.

    Read straight out of the C source rather than hard-coded, so a test that
    checks our assumptions about SIPNET is checking the SIPNET we actually
    ship, not a list that can quietly go stale.

    SIPNET registers each parameter with a call of the form::

        initializeOneModelParam(modelParams, "aMax", &(params.aMax), 1);
    """
    import re

    source_dir = Path(__file__).parent.parent / "sipnet" / "src"
    if not source_dir.exists():
        pytest.skip("SIPNET submodule not populated; run 'git submodule update --init sipnet'")

    pattern = re.compile(r'initializeOneModelParam\(\s*\w+\s*,\s*"([A-Za-z_0-9]+)"')
    names: set[str] = set()
    for path in source_dir.rglob("*.c"):
        names.update(pattern.findall(path.read_text()))

    if not names:
        pytest.fail(f"No parameter registrations found under {source_dir}; has the C API changed?")
    return names


@pytest.fixture
def reference_fixture_dir() -> Path:
    """Directory holding a known-good sipnet.param and sipnet.clim pair.

    Used by tests that need to run the real binary on realistic inputs. The
    files ship inside the package (``pysipnet/data/niwot/``) so that
    downstream projects get them too; tests read them from there, so there is
    exactly one copy.
    """
    from pysipnet.io.reference import niwot_reference_files

    return niwot_reference_files().directory


@pytest.fixture
def reference_clim_path() -> Path:
    """Path to the bundled reference climate file.

    Tracked by git and shipped in the wheel, so it is there in CI, after a
    fresh clone and in an installed package. Any test needing climate input
    should use this rather than reaching into the gitignored ``data/``
    directory, which only exists on the machine that put it there.
    """
    from pysipnet.io.reference import niwot_reference_files

    return niwot_reference_files().clim


@pytest.fixture
def minimal_params():
    """A minimal but valid SIPNETParameters for testing."""
    from pysipnet.parameters import (
        AllocationParams,
        InitialConditions,
        LeafPhysiologyParams,
        PhenologyParams,
        PhotosynthesisParams,
        RespirationParams,
        SIPNETParameters,
        WaterParams,
    )

    return SIPNETParameters(
        initial_conditions=InitialConditions(
            total_wood_carbon=30000.0,
            leaf_area_index=0.0,
            soil_carbon=10000.0,
            soil_wetness_fraction=0.5,
            snow_water_equivalent=1.0,
            fine_root_fraction=0.05,
            coarse_root_fraction=0.15,
        ),
        photosynthesis=PhotosynthesisParams(
            max_photosynthesis_rate=112.0,
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
            snow_melt_rate=0.15,
            aerodynamic_resistance_constant=100.0,
            soil_resistance_intercept=3.0,
            soil_resistance_slope=2.0,
        ),
        leaf=LeafPhysiologyParams(
            leaf_carbon_per_area=32.0,
            leaf_carbon_fraction=0.45,
        ),
    )
