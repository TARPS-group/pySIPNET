"""Tests for EventSequence and all individual event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pysipnet.events import (
    EventSequence,
    FertilizationEvent,
    HarvestEvent,
    IrrigationEvent,
    IrrigationMethod,
    PlantingEvent,
    TillageEvent,
)


# ---------------------------------------------------------------------------
# HarvestEvent
# ---------------------------------------------------------------------------

class TestHarvestEvent:
    def test_valid(self):
        e = HarvestEvent(year=2020, day=100,
                         fraction_removed_above=0.4, fraction_removed_below=0.1,
                         fraction_transferred_above=0.2, fraction_transferred_below=0.3)
        assert e.year == 2020
        assert e.type == "harvest"

    def test_above_fractions_exceed_one_raises(self):
        with pytest.raises(ValidationError, match="fraction_removed_above"):
            HarvestEvent(year=2020, day=100,
                         fraction_removed_above=0.8, fraction_removed_below=0.0,
                         fraction_transferred_above=0.5, fraction_transferred_below=0.0)

    def test_below_fractions_exceed_one_raises(self):
        with pytest.raises(ValidationError, match="fraction_removed_below"):
            HarvestEvent(year=2020, day=100,
                         fraction_removed_above=0.0, fraction_removed_below=0.7,
                         fraction_transferred_above=0.0, fraction_transferred_below=0.6)

    def test_fraction_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            HarvestEvent(year=2020, day=100,
                         fraction_removed_above=1.5, fraction_removed_below=0.0,
                         fraction_transferred_above=0.0, fraction_transferred_below=0.0)

    def test_to_line_format(self):
        e = HarvestEvent(year=2024, day=70,
                         fraction_removed_above=0.1, fraction_removed_below=0.2,
                         fraction_transferred_above=0.3, fraction_transferred_below=0.4)
        line = e._to_line()
        tokens = line.split()
        assert tokens[0] == "2024"
        assert tokens[1] == "70"
        assert tokens[2] == "harv"
        assert float(tokens[3]) == pytest.approx(0.1)
        assert float(tokens[6]) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# IrrigationEvent
# ---------------------------------------------------------------------------

class TestIrrigationEvent:
    def test_valid_canopy(self):
        e = IrrigationEvent(year=2020, day=150, amount=5.0, method=IrrigationMethod.CANOPY)
        assert e.method == IrrigationMethod.CANOPY

    def test_valid_soil(self):
        e = IrrigationEvent(year=2020, day=150, amount=3.0, method=IrrigationMethod.SOIL)
        assert e.method == IrrigationMethod.SOIL

    def test_zero_amount_raises(self):
        with pytest.raises(ValidationError):
            IrrigationEvent(year=2020, day=150, amount=0.0, method=IrrigationMethod.SOIL)

    def test_to_line_writes_method_as_int(self):
        e = IrrigationEvent(year=2024, day=70, amount=3.0, method=IrrigationMethod.SOIL)
        tokens = e._to_line().split()
        assert tokens[2] == "irrig"
        assert tokens[3] == "3.0"
        assert tokens[4] == "1"  # SOIL = 1


# ---------------------------------------------------------------------------
# FertilizationEvent
# ---------------------------------------------------------------------------

class TestFertilizationEvent:
    def test_valid(self):
        e = FertilizationEvent(year=2020, day=90, org_n=15.0, org_c=5.0, min_n=10.0)
        assert e.org_n == 15.0

    def test_negative_org_n_raises(self):
        with pytest.raises(ValidationError):
            FertilizationEvent(year=2020, day=90, org_n=-1.0, org_c=5.0, min_n=0.0)

    def test_to_line(self):
        e = FertilizationEvent(year=2022, day=40, org_n=15.0, org_c=5.0, min_n=10.0)
        tokens = e._to_line().split()
        assert tokens[2] == "fert"
        assert float(tokens[3]) == pytest.approx(15.0)
        assert float(tokens[5]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# PlantingEvent
# ---------------------------------------------------------------------------

class TestPlantingEvent:
    def test_valid(self):
        e = PlantingEvent(year=2020, day=90, leaf_c=10.0, wood_c=5.0,
                          fine_root_c=4.0, coarse_root_c=3.0)
        assert e.leaf_c == 10.0

    def test_negative_value_raises(self):
        with pytest.raises(ValidationError):
            PlantingEvent(year=2020, day=90, leaf_c=-1.0, wood_c=5.0,
                          fine_root_c=4.0, coarse_root_c=3.0)

    def test_to_line(self):
        e = PlantingEvent(year=2024, day=70, leaf_c=10.0, wood_c=5.0,
                          fine_root_c=4.0, coarse_root_c=3.0)
        tokens = e._to_line().split()
        assert tokens[2] == "plant"
        assert len(tokens) == 7  # year day plant leafC woodC fineRootC coarseRootC


# ---------------------------------------------------------------------------
# TillageEvent
# ---------------------------------------------------------------------------

class TestTillageEvent:
    def test_valid(self):
        e = TillageEvent(year=2020, day=100, fraction_litter_transferred=0.1,
                         som_decomp_modifier=1.2, litter_decomp_modifier=1.3)
        assert e.som_decomp_modifier == pytest.approx(1.2)

    def test_fraction_above_one_raises(self):
        with pytest.raises(ValidationError):
            TillageEvent(year=2020, day=100, fraction_litter_transferred=1.5,
                         som_decomp_modifier=1.0, litter_decomp_modifier=1.0)

    def test_to_line(self):
        e = TillageEvent(year=2022, day=45, fraction_litter_transferred=0.1,
                         som_decomp_modifier=0.2, litter_decomp_modifier=0.3)
        tokens = e._to_line().split()
        assert tokens[2] == "till"
        assert len(tokens) == 6


# ---------------------------------------------------------------------------
# EventSequence
# ---------------------------------------------------------------------------

class TestEventSequence:
    def _simple_sequence(self):
        return EventSequence(events=[
            IrrigationEvent(year=2020, day=100, amount=5.0, method=IrrigationMethod.SOIL),
            HarvestEvent(year=2020, day=270,
                         fraction_removed_above=0.4, fraction_removed_below=0.0,
                         fraction_transferred_above=0.2, fraction_transferred_below=0.0),
        ])

    def test_empty_sequence(self):
        seq = EventSequence()
        assert len(seq) == 0

    def test_len(self):
        seq = self._simple_sequence()
        assert len(seq) == 2

    def test_out_of_order_raises(self):
        with pytest.raises(ValidationError, match="chronological"):
            EventSequence(events=[
                HarvestEvent(year=2020, day=270,
                             fraction_removed_above=0.4, fraction_removed_below=0.0,
                             fraction_transferred_above=0.2, fraction_transferred_below=0.0),
                IrrigationEvent(year=2020, day=100, amount=5.0,
                                method=IrrigationMethod.SOIL),
            ])

    def test_same_day_allowed(self):
        seq = EventSequence(events=[
            IrrigationEvent(year=2020, day=100, amount=3.0, method=IrrigationMethod.CANOPY),
            FertilizationEvent(year=2020, day=100, org_n=15.0, org_c=5.0, min_n=10.0),
        ])
        assert len(seq) == 2

    def test_repr_contains_counts(self):
        seq = self._simple_sequence()
        r = repr(seq)
        assert "EventSequence" in r
        assert "2" in r

    # ── File IO ────────────────────────────────────────────────────────────

    def test_to_file_writes_correct_line_count(self, tmp_path):
        seq = self._simple_sequence()
        path = tmp_path / "events.in"
        seq.to_file(path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_empty_sequence_writes_empty_file(self, tmp_path):
        path = tmp_path / "events.in"
        EventSequence().to_file(path)
        assert path.read_text() == ""

    def test_roundtrip(self, tmp_path):
        seq = EventSequence(events=[
            IrrigationEvent(year=2022, day=40, amount=5.0, method=IrrigationMethod.CANOPY),
            FertilizationEvent(year=2022, day=40, org_n=15.0, org_c=5.0, min_n=10.0),
            TillageEvent(year=2022, day=45, fraction_litter_transferred=0.1,
                         som_decomp_modifier=0.2, litter_decomp_modifier=0.3),
            PlantingEvent(year=2022, day=46, leaf_c=10.0, wood_c=5.0,
                          fine_root_c=4.0, coarse_root_c=3.0),
            HarvestEvent(year=2022, day=250,
                         fraction_removed_above=0.4, fraction_removed_below=0.1,
                         fraction_transferred_above=0.2, fraction_transferred_below=0.3),
        ])
        path = tmp_path / "events.in"
        seq.to_file(path)
        seq2 = EventSequence.from_file(path)

        assert len(seq2) == 5
        assert seq2.events[0].type == "irrigation"
        assert seq2.events[1].type == "fertilization"
        assert seq2.events[2].type == "tillage"
        assert seq2.events[3].type == "planting"
        assert seq2.events[4].type == "harvest"

    def test_from_file_ignores_comments(self, tmp_path):
        content = (
            "2022  40  irrig  5  0  # canopy irrigation\n"
            "# this is a full-line comment\n"
            "\n"
            "2022  250 harv  0.4 0.1 0.2 0.3\n"
        )
        path = tmp_path / "events.in"
        path.write_text(content)
        seq = EventSequence.from_file(path)
        assert len(seq) == 2

    def test_from_file_real_sipnet_example(self, tmp_path):
        """Verify parsing of the actual SIPNET test fixture format."""
        content = (
            "2022  40  irrig  5   0\n"
            "2022  40  fert   15 5 10\n"
            "2022  45  till   0.1 0.2 0.3\n"
            "2022  46  plant  10 5 4 3\n"
            "2022  250 harv   0.4 0.1 0.2 0.3\n"
        )
        path = tmp_path / "events.in"
        path.write_text(content)
        seq = EventSequence.from_file(path)
        assert len(seq) == 5

        irr = seq.events[0]
        assert isinstance(irr, IrrigationEvent)
        assert irr.amount == pytest.approx(5.0)
        assert irr.method == IrrigationMethod.CANOPY

        harv = seq.events[4]
        assert isinstance(harv, HarvestEvent)
        assert harv.fraction_removed_above == pytest.approx(0.4)

    def test_from_file_unknown_type_raises(self, tmp_path):
        path = tmp_path / "events.in"
        path.write_text("2022 100 mow 0.5\n")
        with pytest.raises(ValueError, match="Unknown event type token"):
            EventSequence.from_file(path)

    def test_serialisation_roundtrip(self):
        """EventSequence survives model_dump / model_validate."""
        seq = self._simple_sequence()
        dumped = seq.model_dump()
        seq2 = EventSequence.model_validate(dumped)
        assert len(seq2) == 2
        assert seq2.events[0].type == seq.events[0].type
