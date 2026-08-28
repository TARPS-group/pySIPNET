"""Tests that the ``events.in`` file pySIPNET writes says what it means.

The third input contract, alongside ``test_sipnet_in.py`` and
``test_param_file_contract.py``. It exists because the events file had no
contract test and a real bug hid there: pySIPNET wrote three values for a
tillage event, SIPNET v2.1.0 reads one, and ``sscanf`` stops once it has filled
its arguments. So a litter fraction was silently read as a decomposition-rate
boost and two values were dropped. No error, no warning, plausible output.

The check that catches that class of bug is not "does the run succeed" — it did
— but "does SIPNET record the values we meant". SIPNET writes every event it
applied to ``events.out``, so these tests read that back and compare.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from pysipnet.build import binary_path
from pysipnet.events import (
    EVENT_ARITY,
    EventSequence,
    FertilizationEvent,
    HarvestEvent,
    IrrigationEvent,
    PlantingEvent,
    TillageEvent,
)
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import _render_sipnet_in

requires_binary = pytest.mark.skipif(
    not binary_path().exists(),
    reason="SIPNET binary not built; run 'make sipnet'",
)

# A year and day that fall inside the reference climate record.
YEAR, DAY = 1998, 310


def _run_with_events(tmp_path, fixture_dir, events: EventSequence):
    """Write a full run directory including events.in, and run SIPNET."""
    for name in ("sipnet.param", "sipnet.clim"):
        shutil.copy(fixture_dir / name, tmp_path / name)
    events.to_file(tmp_path / "events.in")
    (tmp_path / "sipnet.in").write_text(
        _render_sipnet_in(ModelFlags.standard(), events_enabled=True)
    )
    proc = subprocess.run(
        [str(binary_path()), "-i", "sipnet.in"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    events_out = tmp_path / "events.out"
    return proc, (events_out.read_text() if events_out.exists() else "")


class TestArityMatchesSipnet:
    """Our idea of how many values each event carries must match SIPNET's.

    Read out of ``events.h`` rather than hard-coded, so this cannot drift when
    the pin moves. This is the test that would have caught the tillage bug
    without needing to run anything.
    """

    def test_every_arity_matches_the_sipnet_header(self):
        import re
        from pathlib import Path

        header = Path(__file__).parent.parent / "sipnet" / "src" / "sipnet" / "events.h"
        if not header.exists():
            pytest.skip("SIPNET submodule not populated")

        text = header.read_text()
        # e.g. "#define NUM_TILLAGE_PARAMS 1"
        found = {
            name.lower(): int(count)
            for name, count in re.findall(r"#define NUM_(\w+)_PARAMS\s+(\d+)", text)
        }

        for event_type, arity in EVENT_ARITY.items():
            key = {"fertilization": "fertilization", "irrigation": "irrigation"}.get(
                event_type, event_type
            )
            assert key in found, f"SIPNET defines no NUM_{key.upper()}_PARAMS"
            assert arity == found[key], (
                f"{event_type}: pySIPNET writes {arity} value(s), "
                f"SIPNET v2.1.0 reads {found[key]}. "
                "A mismatch is silent — sscanf discards the surplus."
            )

    def test_we_only_write_event_types_sipnet_accepts(self):
        """Writing a keyword SIPNET does not know is a hard error at its end.

        The reverse — SIPNET knowing a type we do not write — is a missing
        feature rather than a bug, so it is listed rather than asserted away.
        """
        import re
        from pathlib import Path

        events_c = Path(__file__).parent.parent / "sipnet" / "src" / "sipnet" / "events.c"
        if not events_c.exists():
            pytest.skip("SIPNET submodule not populated")

        accepted = set(re.findall(r'strcmp\(eventTypeStr,\s*"(\w+)"\)', events_c.read_text()))
        ours = {"harv", "irrig", "fert", "plant", "till"}

        unknown = ours - accepted
        assert not unknown, (
            f"pySIPNET writes event types SIPNET does not accept: {sorted(unknown)}. "
            "SIPNET exits with EXIT_CODE_UNKNOWN_EVENT_TYPE_OR_PARAM on these."
        )

    def test_unmodelled_event_types_are_the_ones_we_expect(self):
        """Pin the known gaps so a new upstream type shows up as a change.

        ``leafon`` and ``leafoff`` prescribe leaf-out and leaf-fall timing from
        observed dates instead of a fitted parameter, and ``plantdeath`` is
        emitted by SIPNET rather than read. Adding the leaf events is tracked
        in issue #25; this test exists so a *fourth* new type cannot appear
        unnoticed.
        """
        import re
        from pathlib import Path

        events_c = Path(__file__).parent.parent / "sipnet" / "src" / "sipnet" / "events.c"
        if not events_c.exists():
            pytest.skip("SIPNET submodule not populated")

        accepted = set(re.findall(r'strcmp\(eventTypeStr,\s*"(\w+)"\)', events_c.read_text()))
        ours = {"harv", "irrig", "fert", "plant", "till"}

        assert accepted - ours == {"leafon", "leafoff", "plantdeath"}, (
            f"the set of event types pySIPNET does not model has changed: "
            f"{sorted(accepted - ours)}. If a new type appeared upstream, decide "
            "whether to model it and update this test."
        )


@requires_binary
class TestSipnetReceivesWhatWeWrote:
    def test_tillage_effect_reaches_sipnet_unchanged(self, tmp_path, reference_fixture_dir):
        """The regression. SIPNET echoes the value it applied; it must be ours."""
        events = EventSequence(events=[TillageEvent(year=YEAR, day=DAY, tillage_effect=0.35)])
        proc, events_out = _run_with_events(tmp_path, reference_fixture_dir, events)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "till" in events_out, f"SIPNET recorded no tillage event:\n{events_out}"
        assert "0.35" in events_out, (
            f"SIPNET did not apply the tillage effect we asked for. events.out says:\n{events_out}"
        )

    @pytest.mark.parametrize(
        "event",
        [
            pytest.param(TillageEvent(year=YEAR, day=DAY, tillage_effect=0.25), id="till"),
            pytest.param(IrrigationEvent(year=YEAR, day=DAY, amount=1.5, method=1), id="irrig"),
            pytest.param(
                FertilizationEvent(year=YEAR, day=DAY, org_n=1.0, org_c=2.0, min_n=3.0),
                id="fert",
            ),
            pytest.param(
                PlantingEvent(
                    year=YEAR, day=DAY, leaf_c=1.0, wood_c=2.0, fine_root_c=3.0, coarse_root_c=4.0
                ),
                id="plant",
            ),
            pytest.param(
                HarvestEvent(
                    year=YEAR,
                    day=DAY,
                    fraction_removed_above=0.5,
                    fraction_removed_below=0.1,
                    fraction_transferred_above=0.2,
                    fraction_transferred_below=0.1,
                ),
                id="harv",
            ),
        ],
    )
    def test_every_event_type_is_accepted_and_applied(self, tmp_path, reference_fixture_dir, event):
        """Each type must run cleanly and appear in SIPNET's own event log."""
        proc, events_out = _run_with_events(
            tmp_path, reference_fixture_dir, EventSequence(events=[event])
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        combined = proc.stdout + proc.stderr
        assert "error" not in combined.lower(), combined
        assert events_out.strip(), f"SIPNET applied no event for {event.type}"

    def test_a_surplus_value_would_be_silently_swallowed(self, tmp_path, reference_fixture_dir):
        """Document the behaviour that made the tillage bug invisible.

        Writing an extra value by hand does not fail. Nothing warns. This is
        why the arity is asserted against the SIPNET header rather than
        inferred from whether a run succeeds.
        """
        for name in ("sipnet.param", "sipnet.clim"):
            shutil.copy(reference_fixture_dir / name, tmp_path / name)
        (tmp_path / "events.in").write_text(f"{YEAR}  {DAY}  till  0.35  9.99  8.88\n")
        (tmp_path / "sipnet.in").write_text(
            _render_sipnet_in(ModelFlags.standard(), events_enabled=True)
        )
        proc = subprocess.run(
            [str(binary_path()), "-i", "sipnet.in"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, "expected SIPNET to accept the surplus silently"
        events_out = (tmp_path / "events.out").read_text()
        assert "0.35" in events_out
        assert "9.99" not in events_out and "8.88" not in events_out


class TestEventFileParsing:
    """Reading back must be as strict as writing, or a bad file loads quietly."""

    def test_wrong_parameter_count_is_refused(self, tmp_path):
        path = tmp_path / "events.in"
        path.write_text(f"{YEAR}  {DAY}  till  0.30  0.77  0.88\n")
        with pytest.raises(ValueError, match="has 3 parameter"):
            EventSequence.from_file(path)

    def test_error_names_the_expected_count_and_the_values_found(self, tmp_path):
        path = tmp_path / "events.in"
        path.write_text(f"{YEAR}  {DAY}  till  0.30  0.77  0.88\n")
        with pytest.raises(ValueError) as exc:
            EventSequence.from_file(path)
        message = str(exc.value)
        assert "expected 1" in message
        assert "0.30" in message

    @pytest.mark.parametrize(("event_type", "arity"), sorted(EVENT_ARITY.items()))
    def test_too_few_parameters_is_refused(self, tmp_path, event_type, arity):
        keyword = {
            "harvest": "harv",
            "irrigation": "irrig",
            "fertilization": "fert",
            "planting": "plant",
            "tillage": "till",
        }[event_type]
        path = tmp_path / "events.in"
        path.write_text(f"{YEAR}  {DAY}  {keyword}  " + "  ".join(["1.0"] * (arity - 1)) + "\n")
        with pytest.raises(ValueError):
            EventSequence.from_file(path)

    def test_round_trip_preserves_the_value(self, tmp_path):
        path = tmp_path / "events.in"
        original = EventSequence(events=[TillageEvent(year=YEAR, day=DAY, tillage_effect=0.42)])
        original.to_file(path)
        assert EventSequence.from_file(path).events[0].tillage_effect == pytest.approx(0.42)
