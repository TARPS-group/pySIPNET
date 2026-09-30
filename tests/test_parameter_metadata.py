"""The parameter metadata table, and the SIPNET documentation sections it groups by."""

from __future__ import annotations

import pandas as pd
import pytest

from pysipnet.io.param_io import PYTHON_TO_SIPNET, _flatten
from pysipnet.parameters.base import SIPNETDocsSection
from pysipnet.parameters.metadata import parameter_metadata_table
from pysipnet.parameters.model import PARAMETER_SPECS, ModelFlags, SIPNETParameters

DocsSections = tuple[list[str], dict[str, set[str]]]

# Parameters the pinned SIPNET registers but its documentation does not list,
# with the section pySIPNET puts each in. When upstream documents one, the
# test below fails until the entry is deleted.
NOT_IN_SIPNET_DOCS = {"leafOnReallocFrac": SIPNETDocsSection.PHENOLOGY}


class TestSIPNETDocsSections:
    def test_every_parameter_is_in_the_section_sipnet_lists_it_under(
        self, sipnet_docs_sections: DocsSections
    ) -> None:
        _, listed = sipnet_docs_sections
        wrong = {
            spec.sipnet_name: (spec.sipnet_docs_section, listed.get(spec.sipnet_name))
            for spec in PARAMETER_SPECS.values()
            if spec.sipnet_name not in NOT_IN_SIPNET_DOCS
            and listed.get(spec.sipnet_name) != {spec.sipnet_docs_section}
        }
        assert not wrong, f"(ours, SIPNET's docs) disagree: {wrong}"

    def test_the_undocumented_parameters_are_still_undocumented(
        self, sipnet_docs_sections: DocsSections
    ) -> None:
        _, listed = sipnet_docs_sections
        for name, section in NOT_IN_SIPNET_DOCS.items():
            assert name not in listed, f"SIPNET now documents {name}; delete its exception."
            spec = next(s for s in PARAMETER_SPECS.values() if s.sipnet_name == name)
            assert spec.sipnet_docs_section == section

    def test_sections_are_sipnet_headings_in_sipnet_order(
        self, sipnet_docs_sections: DocsSections
    ) -> None:
        headings, _ = sipnet_docs_sections
        ours = [section.value for section in SIPNETDocsSection]
        assert set(ours) <= set(headings)
        assert ours == [heading for heading in headings if heading in ours]

    def test_every_section_holds_a_parameter(self) -> None:
        used = {spec.sipnet_docs_section for spec in PARAMETER_SPECS.values()}
        assert used == set(SIPNETDocsSection)


class TestFlatValues:
    def test_is_what_the_writer_writes(self, minimal_params: SIPNETParameters) -> None:
        values = minimal_params.flat_values()
        assert _flatten(minimal_params) == {PYTHON_TO_SIPNET[p]: v for p, v in values.items()}

    def test_omits_unset_parameters_and_keeps_the_rest(
        self, minimal_params: SIPNETParameters
    ) -> None:
        values = minimal_params.flat_values()
        assert "phenology.leaf_on_day" not in values
        assert values["photosynthesis.max_photosynthesis_rate"] == (
            minimal_params.photosynthesis.max_photosynthesis_rate
        )
        assert list(values) == [path for path in PARAMETER_SPECS if path in values]


class TestParameterMetadataTable:
    def test_without_parameters_every_parameter_is_a_row_and_there_are_no_values(self) -> None:
        table = parameter_metadata_table()
        assert set(table.index) == set(PARAMETER_SPECS)
        assert "value" not in table.columns

    def test_rows_are_ordered_by_section_then_declaration(self) -> None:
        table = parameter_metadata_table()
        order = list(PARAMETER_SPECS)
        expected = sorted(
            order,
            key=lambda p: (
                list(SIPNETDocsSection).index(PARAMETER_SPECS[p].sipnet_docs_section),
                order.index(p),
            ),
        )
        assert list(table.index) == expected

    def test_section_is_an_ordered_categorical_in_sipnet_order(self) -> None:
        section = parameter_metadata_table()["section"]
        assert isinstance(section.dtype, pd.CategoricalDtype)
        assert section.cat.ordered
        assert list(section.cat.categories) == [s.value for s in SIPNETDocsSection]

    def test_columns_are_read_from_the_specs(self) -> None:
        table = parameter_metadata_table(units_style="latex")
        for path, row in table.iterrows():
            spec = PARAMETER_SPECS[path]
            assert row["name"] == path.split(".", 1)[1]
            assert row["sipnet_name"] == spec.sipnet_name
            assert row["units"] == spec.formatted_units("latex")
            assert row["per_year"] == spec.per_year
            assert row["label"] == spec.long_label
            assert row["description"] == spec.description
            assert row["domain"] == spec.domain.value
            assert row["required_when"] == spec.requirement_description()

    def test_values_are_the_parameters_that_are_set(self, minimal_params: SIPNETParameters) -> None:
        table = parameter_metadata_table(minimal_params)
        values = minimal_params.flat_values()
        assert set(table.index) == set(values)
        assert table["value"].dtype == float
        assert all(table.loc[path, "value"] == value for path, value in values.items())

    def test_required_is_added_by_flags(self, minimal_params: SIPNETParameters) -> None:
        flags = ModelFlags.standard()
        table = parameter_metadata_table(minimal_params, flags=flags)
        for path, required in table["required"].items():
            assert required == PARAMETER_SPECS[path].required_under(flags)
        assert "required" not in parameter_metadata_table(minimal_params).columns

    def test_flags_without_parameters_describe_the_whole_model(self) -> None:
        table = parameter_metadata_table(flags=ModelFlags.standard())
        assert table.loc["water.snow_melt_rate", "required"]
        assert not table.loc["phenology.leaf_on_day", "required"]

    def test_a_set_the_flags_would_refuse_is_refused(
        self, minimal_params: SIPNETParameters
    ) -> None:
        unset = minimal_params.model_copy(
            update={"water": minimal_params.water.model_copy(update={"snow_melt_rate": None})}
        )
        with pytest.raises(ValueError, match="snow_melt_rate is required"):
            parameter_metadata_table(unset, flags=ModelFlags.standard())

    def test_tags_accept_any_spelling_and_leave_the_rest_missing(
        self, minimal_params: SIPNETParameters
    ) -> None:
        table = parameter_metadata_table(
            minimal_params, tags={"aMax": "calibrated", "soil_whc": "calibrated"}
        )
        tagged = table["tag"].dropna()
        assert set(tagged.index) == {
            "photosynthesis.max_photosynthesis_rate",
            "water.soil_water_holding_capacity",
        }
        assert set(tagged) == {"calibrated"}
        assert table["tag"].isna().sum() == len(table) - 2

    def test_a_misspelled_tag_is_refused(self, minimal_params: SIPNETParameters) -> None:
        with pytest.raises(KeyError, match="aMaks"):
            parameter_metadata_table(minimal_params, tags={"aMaks": "calibrated"})

    def test_one_parameter_tagged_two_ways_is_refused(
        self, minimal_params: SIPNETParameters
    ) -> None:
        with pytest.raises(ValueError, match="tagged both"):
            parameter_metadata_table(minimal_params, tags={"aMax": "calibrated", "a_max": "fixed"})

    def test_tagging_a_parameter_that_is_not_set_is_refused(
        self, minimal_params: SIPNETParameters
    ) -> None:
        with pytest.raises(ValueError, match="phenology.leaf_on_day"):
            parameter_metadata_table(minimal_params, tags={"leafOnDay": "calibrated"})
