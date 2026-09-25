"""Unit conversion that understands ``constituent``.

The factors are stated here as arithmetic on the molar masses and the unit
prefixes, written out by hand, so a wrong table entry or a wrong route through
it cannot cancel itself out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    convert_dataarray_units,
    convert_units,
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
    # The g-1 is leaf mass; the constituent qualifies only the first unit.
    ("nmol g-1 s-1", "CO2", "ug g-1 s-1", "C", 12.011e-3),
    ("mg g-1 kPa", "CO2", "mg g-1 kPa", "C", 12.011 / 44.009),
    # Water as a depth, through the density.
    ("cm", "H2O", "kg m-2", "H2O", 10.0),
    ("mm", "H2O", "kg m-2", "H2O", 1.0),
    ("cm d-1", "H2O", "kg m-2 s-1", "H2O", 10 / 86400),
    ("cm", "H2O", "mol m-2", "H2O", 10_000 / 18.015),
    ("m3 m-2", "H2O", "mm", "H2O", 1000.0),
    # Same dimension needs no bridge, whatever the first unit is.
    ("kPa", "H2O", "Pa", "H2O", 1000.0),
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
    ("g m-2", "C", "m2 m-2", "C", "'m2' in 'm2 m-2' is not an amount or a mass of 'C'"),
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
    ("m2 m-2", "C", "m2 m-2", "CO2", "'m2' in 'm2 m-2' is not an amount or a mass of 'C'"),
    ("g m-2", "C", "cm", "C", "'cm' in 'cm' is not an amount or a mass of 'C'"),
    ("g m-2", "C", "mol m-2 s-1", "C", "does not reconcile them"),
    ("g m-2", "H2O", "m2 m-2", "H2O", "an amount, a mass, a depth or a volume of 'H2O'"),
    # The constituent qualifies the first unit, so a mass inside a pressure or a
    # power is not a mass of the substance.
    ("Pa", "C", "Pa", "CO2", "'Pa' in 'Pa' is not an amount or a mass of 'C'"),
    ("W m-2", "C", "W m-2", "CO2", "'W' in 'W m-2' is not an amount or a mass of 'C'"),
    ("g m-2", "C", "Pa mol g-1", "C", "'Pa' in 'Pa mol g-1' is not an amount or a mass"),
    ("degC", "", "K", "", "offset temperature scale"),
    ("degC", "C", "degC", "CO2", "a temperature has no constituent"),
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


def test_to_constituent_defaults_to_the_source_constituent():
    assert conversion_factor(units="g m-2", constituent="C", to_units="Mg ha-1") == 0.01
    assert conversion_factor(units="g m-2", to_units="Mg ha-1") == 0.01
    with pytest.raises(ValueError, match="one side only"):
        conversion_factor(units="g m-2", constituent="C", to_units="g m-2", to_constituent="")


def test_convert_units_preserves_numpy_shape():
    values = np.arange(6.0).reshape(2, 3)
    out = convert_units(values, units="g m-2", constituent="C", to_units="Mg ha-1")
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, values * 0.01, rtol=1e-12)


def test_convert_units_on_a_scalar():
    assert convert_units(2.0, units="cm", constituent="H2O", to_units="kg m-2") == 20.0


def test_convert_units_on_unlabeled_pandas():
    s = pd.Series([1.0, 2.0], index=["a", "b"])
    out = convert_units(s, units="g m-2", constituent="C", to_units="Mg ha-1")
    assert isinstance(out, pd.Series) and list(out.index) == ["a", "b"]
    np.testing.assert_allclose(out.to_numpy(), [0.01, 0.02], rtol=1e-12)
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    out_df = convert_units(df, units="g m-2", constituent="C", to_units="Mg ha-1")
    assert list(out_df.columns) == ["x", "y"]


@pytest.mark.parametrize(
    "values",
    [
        xr.DataArray([1.0], attrs={"units": "g m-2"}),
        xr.DataArray([1.0]),
        xr.Dataset({"a": ("t", [1.0])}),
        xr.Variable("t", [1.0]),
    ],
    ids=["labeled-dataarray", "unlabeled-dataarray", "dataset", "variable"],
)
def test_convert_units_refuses_xarray(values):
    with pytest.raises(TypeError, match="convert_dataarray_units"):
        convert_units(values, units="g m-2", to_units="Mg ha-1")


def test_convert_units_refuses_pandas_that_carries_units():
    # pandas keeps attrs through multiplication, so the result would still say g m-2.
    s = pd.Series([1.0])
    s.attrs = {"units": "g m-2"}
    with pytest.raises(TypeError, match="attrs\\['units'\\]"):
        convert_units(s, units="g m-2", to_units="Mg ha-1")


def test_convert_dataarray_units_reads_the_source_units_from_attrs():
    da = xr.DataArray([1.0, 2.0], attrs={"units": "kg m-2", "constituent": "C"})
    out = convert_dataarray_units(da, to_units="g m-2")
    np.testing.assert_allclose(out.values, [1000.0, 2000.0], rtol=1e-12)


def test_convert_dataarray_units_preserves_shape_coordinates_and_name():
    da = xr.DataArray(
        np.ones((4, 2)),
        dims=("time", "site"),
        coords={"site": ["a", "b"]},
        name="nee",
        attrs={"units": "g m-2 d-1", "constituent": "C"},
    )
    out = convert_dataarray_units(da, to_units="umol m-2 s-1", to_constituent="CO2")
    assert isinstance(out, xr.DataArray) and out.name == "nee"
    assert out.shape == da.shape and list(out["site"].values) == ["a", "b"]
    np.testing.assert_allclose(out.values, 1e6 / 12.011 / 86400, rtol=1e-12)


def test_convert_dataarray_units_relabels_and_leaves_the_input_alone():
    from pysipnet.io.reference import niwot_reference_output

    wood = niwot_reference_output()["wood_carbon"]
    before = dict(wood.attrs)
    out = convert_dataarray_units(wood, to_units="Mg ha-1")
    np.testing.assert_allclose(out.values, wood.values * 0.01, rtol=1e-12)
    assert out.attrs["units"] == "Mg ha-1" and out.attrs["constituent"] == "C"
    assert "output_decimals" not in out.attrs
    assert out.attrs["long_name"] == before["long_name"]
    assert wood.attrs == before


def test_convert_dataarray_units_relabels_constituent_and_drops_sipnet_conversion_attrs():
    da = xr.DataArray(
        [1.0],
        attrs={
            "units": "cm",
            "constituent": "H2O",
            "sipnet_internal_units": "mm",
            "sipnet_internal_conversion": "x 0.1",
        },
    )
    out = convert_dataarray_units(da, to_units="kg m-2")
    assert out.attrs == {"units": "kg m-2", "constituent": "H2O"}
    carbon = xr.DataArray([1.0], attrs={"units": "g m-2", "constituent": "C"})
    out = convert_dataarray_units(carbon, to_units="g m-2", to_constituent="CO2")
    assert out.attrs["constituent"] == "CO2"


def test_convert_dataarray_units_relabels_even_when_xarray_drops_attrs():
    da = xr.DataArray([1.0], attrs={"units": "g m-2", "long_name": "x"})
    with xr.set_options(keep_attrs=False):
        out = convert_dataarray_units(da, to_units="Mg ha-1")
    assert out.attrs == {"units": "Mg ha-1", "long_name": "x"}


def test_convert_dataarray_units_without_a_constituent_attribute():
    da = xr.DataArray([1.0], attrs={"units": "d"})
    out = convert_dataarray_units(da, to_units="h")
    assert out.values[0] == 24.0 and "constituent" not in out.attrs


def test_convert_dataarray_units_refuses_an_array_with_no_units():
    with pytest.raises(ValueError, match="DataArray 'nee' has no 'units' attribute"):
        convert_dataarray_units(xr.DataArray([1.0], name="nee"), to_units="g m-2")


def test_convert_dataarray_units_names_the_array_in_a_refused_conversion():
    da = xr.DataArray([1.0], name="nee", attrs={"units": "g m-2", "constituent": "C"})
    with pytest.raises(ValueError, match="DataArray 'nee': Cannot convert 'g m-2'"):
        convert_dataarray_units(da, to_units="K")


@pytest.mark.parametrize(
    ("values", "text"),
    [
        (xr.Dataset({"a": ("t", [1.0])}), "convert ds\\[name\\]"),
        (np.ones(2), "convert_units"),
    ],
)
def test_convert_dataarray_units_refuses_anything_but_a_dataarray(values, text):
    with pytest.raises(TypeError, match=text):
        convert_dataarray_units(values, to_units="Mg ha-1")


def test_factor_is_cached():
    conversion_factor.cache_clear()
    factor("g m-2", "C", "Mg ha-1", "C")
    factor("g m-2", "C", "Mg ha-1", "C")
    assert conversion_factor.cache_info().hits == 1


def test_every_convertible_substance_is_refused_inside_a_unit_string():
    from pysipnet.units import validate_units

    for substance in MOLAR_MASS:
        # H2O fails the UDUNITS syntax check first; either way it is refused.
        with pytest.raises(ValueError, match="substance token|UDUNITS syntax"):
            validate_units(f"g {substance} m-2")


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
