"""Unit conversion that understands ``constituent``.

The factors are stated here as arithmetic on the molar masses and the unit
prefixes, written out by hand, so a wrong table entry or a wrong route through
it cannot cancel itself out.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pysipnet.parameters.model import PARAMETER_SPECS
from pysipnet.units import (
    AMOUNT_ONLY_CONSTITUENTS,
    ATOMS_PER_MOLECULE,
    CONSTITUENTS,
    DENSITY,
    MOLAR_MASS,
    conversion_factor,
    convert,
)
from pysipnet.variables import CLIMATE_VARIABLES, OUTPUT_VARIABLES


def factor(units: str, constituent: str, to_units: str, to_constituent: str) -> float:
    return conversion_factor(
        units=units, constituent=constituent, to_units=to_units, to_constituent=to_constituent
    )


FACTORS = [
    # Same dimension, same or no constituent: Pint alone.
    ("1", "", "percent", "", 100.0),
    ("Mg ha-1", "", "g m-2", "", 100.0),
    ("g m-2 d-1", "", "g m-2 s-1", "", 1 / 86400),
    ("g m-2", "C", "Mg ha-1", "C", 0.01),
    ("mol m-2 d-1", "photons", "umol m-2 s-1", "photons", 1e6 / 86400),
    ("degC", "", "degC", "", 1.0),
    # Mass to amount through the molar mass.
    ("g m-2", "C", "mol m-2", "C", 1 / 12.011),
    ("mol m-2", "N", "g m-2", "N", 14.007),
    ("nmol g-1 s-1", "CO2", "ug g-1 s-1", "CO2", 44.009e-3),
    # A change of constituent, on an amount basis.
    ("g m-2", "C", "g m-2", "CO2", 44.009 / 12.011),
    ("g m-2 d-1", "C", "umol m-2 s-1", "CO2", 1e6 / 12.011 / 86400),
    ("g m-2", "C", "g m-2", "CH4", 16.043 / 12.011),
    ("g m-2", "N", "g m-2", "N2O", 44.013 / (2 * 14.007)),
    ("mol m-2", "N2O", "mol m-2", "N", 2.0),
    ("nmol g-1 s-1", "CO2", "nmol g-1 s-1", "C", 1.0),
    # Water as a depth, through the density.
    ("cm", "H2O", "kg m-2", "H2O", 10.0),
    ("mm", "H2O", "kg m-2", "H2O", 1.0),
    ("cm d-1", "H2O", "kg m-2 s-1", "H2O", 10 / 86400),
    ("cm", "H2O", "mol m-2", "H2O", 10_000 / 18.015),
]


@pytest.mark.parametrize(
    ("units", "constituent", "to_units", "to_constituent", "expected"), FACTORS
)
def test_factor(units, constituent, to_units, to_constituent, expected):
    assert factor(units, constituent, to_units, to_constituent) == pytest.approx(
        expected, rel=1e-12, abs=1e-9
    )


@pytest.mark.parametrize(
    ("units", "constituent", "to_units", "to_constituent", "expected"),
    [
        ("g m-2 d-1", "", "g m-2 s-1", "", 1.1574074e-05),
        ("g m-2", "C", "mol m-2", "C", 0.0832570144),
        ("g m-2", "C", "g m-2", "CO2", 3.6640579469),
        ("g m-2 d-1", "C", "umol m-2 s-1", "CO2", 0.9636228519),
    ],
)
def test_factor_matches_the_published_value(units, constituent, to_units, to_constituent, expected):
    assert abs(factor(units, constituent, to_units, to_constituent) - expected) < 1e-9


@pytest.mark.parametrize(
    ("units", "constituent", "to_units", "to_constituent", "expected"), FACTORS
)
def test_round_trip_is_identity(units, constituent, to_units, to_constituent, expected):
    there = factor(units, constituent, to_units, to_constituent)
    back = factor(to_units, to_constituent, units, constituent)
    assert there * back == pytest.approx(1.0, rel=1e-12)


REFUSALS = [
    # (units, constituent, to_units, to_constituent, text the message must contain)
    ("g m-2", "C", "m2 m-2", "C", "different dimensions"),
    ("g m-2", "", "m2 m-2", "", "different dimensions"),
    ("g m-2", "", "mol m-2", "", "molar mass needs a substance"),
    ("g m-2", "C", "g m-2", "", "one side only"),
    ("g m-2", "", "g m-2", "C", "one side only"),
    ("g m-2", "Si", "g m-2", "Si", "unknown constituent 'Si'"),
    ("g m-2", "C", "g m-2", "Si", "unknown constituent 'Si'"),
    ("g C m-2", "", "g m-2", "", "substance token"),
    ("g m-2", "C", "g C m-2", "C", "substance token"),
    ("g m-2", "C", "g m-2", "N", "no recorded atom ratio between 'C' and 'N'"),
    ("g m-2", "CO2", "g m-2", "CH4", "no recorded atom ratio between 'CO2' and 'CH4'"),
    ("g m-2", "photons", "mol m-2", "photons", "'photons' has no molar mass"),
    ("m2 m-2", "C", "m2 m-2", "CO2", "neither an amount nor a mass of 'C'"),
    ("g m-2", "C", "cm", "C", "different dimensions"),
    ("degC", "", "K", "", "offset temperature scale"),
]


@pytest.mark.parametrize(("units", "constituent", "to_units", "to_constituent", "text"), REFUSALS)
def test_refusal_names_both_sides(units, constituent, to_units, to_constituent, text):
    with pytest.raises(ValueError) as caught:
        factor(units, constituent, to_units, to_constituent)
    message = str(caught.value)
    assert text in message
    assert repr(units) in message and repr(to_units) in message
    for c in (constituent, to_constituent):
        assert (f"constituent {c!r}" if c else "no constituent") in message


def test_mass_of_an_element_converts_to_mass_of_a_molecule_across_prefixes():
    assert factor("g m-2", "N", "kg m-2", "N2O") == pytest.approx(44.013 / (2 * 14.007) / 1000)


def test_constituent_change_goes_through_amount_not_mass():
    # Applying the one-to-one C:CO2 ratio to grams would give 1, not 44.009/12.011.
    assert factor("g m-2", "C", "g m-2", "CO2") != pytest.approx(1.0)


def test_convert_preserves_numpy_shape():
    values = np.arange(6.0).reshape(2, 3)
    out = convert(values, units="g m-2", constituent="C", to_units="Mg ha-1", to_constituent="C")
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, values * 0.01, rtol=1e-12)


def test_convert_preserves_dataarray_shape_and_coordinates():
    da = xr.DataArray(np.ones((4, 2)), dims=("time", "site"), coords={"site": ["a", "b"]})
    out = convert(
        da, units="g m-2 d-1", constituent="C", to_units="umol m-2 s-1", to_constituent="CO2"
    )
    assert isinstance(out, xr.DataArray)
    assert out.shape == da.shape and list(out["site"].values) == ["a", "b"]
    np.testing.assert_allclose(out.values, 1e6 / 12.011 / 86400, rtol=1e-12)


def test_convert_on_a_scalar():
    assert (
        convert(2.0, units="cm", constituent="H2O", to_units="kg m-2", to_constituent="H2O") == 20.0
    )


def test_tables_are_read_only():
    with pytest.raises(TypeError):
        MOLAR_MASS["Si"] = 28.085  # type: ignore[index]


def test_ratio_and_density_tables_name_known_constituents():
    for element, molecule in ATOMS_PER_MOLECULE:
        assert element in MOLAR_MASS and molecule in MOLAR_MASS
    assert set(DENSITY) <= set(MOLAR_MASS)
    assert not (AMOUNT_ONLY_CONSTITUENTS & set(MOLAR_MASS))


def test_every_registry_constituent_can_convert():
    """A constituent cannot enter a registry without a molar mass, or an explicit exemption.

    The exemption is for photons, which PAR counts in moles and which have no mass.
    """
    declared = {
        spec.constituent
        for spec in (*OUTPUT_VARIABLES, *CLIMATE_VARIABLES, *PARAMETER_SPECS.values())
        if spec.constituent
    }
    assert declared <= CONSTITUENTS, declared - CONSTITUENTS
    assert declared - AMOUNT_ONLY_CONSTITUENTS <= set(MOLAR_MASS)
