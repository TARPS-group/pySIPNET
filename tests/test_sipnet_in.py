"""Tests for the ``sipnet.in`` config file pySIPNET writes for each run.

This file is the whole of pySIPNET's control over how SIPNET behaves, so it is
worth testing on its own. Two things can go wrong and neither is loud:

- We write a key SIPNET does not recognise. SIPNET logs a message and carries
  on with its own default, so the run succeeds while silently ignoring what we
  asked for.
- We leave a key out. SIPNET falls back to a built-in default that may not
  match what the caller requested.

The unit tests below check the text we generate. :class:`TestSipnetAcceptsOurConfig`
goes further and hands the file to the real binary, then reads SIPNET's own log
output back to confirm it did not ignore anything.
"""

from __future__ import annotations

import subprocess

import pytest

from pysipnet.build import binary_path
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import _render_sipnet_in

requires_binary = pytest.mark.skipif(
    not binary_path().exists(),
    reason="SIPNET binary not built; run 'make sipnet'",
)


def _parse(text: str) -> dict[str, str]:
    """Read rendered config text back into a ``{key: value}`` dict."""
    settings = {}
    for line in text.splitlines():
        line = line.split("!")[0].strip()
        if not line:
            continue
        key, _, value = line.partition("=")
        settings[key.strip()] = value.strip()
    return settings


class TestRenderedContent:
    def test_points_sipnet_at_our_filenames(self):
        settings = _parse(_render_sipnet_in(ModelFlags(), events_enabled=False))
        assert settings["FILE_NAME"] == "sipnet"

    def test_always_requests_the_header_row(self):
        """The output reader matches columns by name, so the header must be on."""
        settings = _parse(_render_sipnet_in(ModelFlags(), events_enabled=False))
        assert settings["PRINT_HEADER"] == "1"

    def test_events_off_is_written_explicitly(self):
        """Explicitly disabling events stops a stale events.in being picked up."""
        settings = _parse(_render_sipnet_in(ModelFlags(), events_enabled=False))
        assert settings["EVENTS"] == "0"

    def test_events_on_when_events_were_supplied(self):
        settings = _parse(_render_sipnet_in(ModelFlags(), events_enabled=True))
        assert settings["EVENTS"] == "1"

    def test_includes_every_model_flag(self):
        settings = _parse(_render_sipnet_in(ModelFlags(), events_enabled=False))
        for key in ModelFlags().to_config_keys():
            assert key in settings, f"{key} missing from sipnet.in"

    def test_flag_values_are_carried_through(self):
        on = _parse(_render_sipnet_in(ModelFlags.forest(), events_enabled=False))
        off = _parse(_render_sipnet_in(ModelFlags.standard(), events_enabled=False))
        assert on["LITTER_POOL"] == "1"
        assert off["LITTER_POOL"] == "0"

    def test_starts_with_a_comment_identifying_the_writer(self):
        text = _render_sipnet_in(ModelFlags(), events_enabled=False)
        assert text.startswith("!")
        assert "pySIPNET" in text.splitlines()[0]

    def test_ends_with_a_newline(self):
        """SIPNET reads line by line; a missing final newline risks a dropped key."""
        assert _render_sipnet_in(ModelFlags(), events_enabled=False).endswith("\n")

    def test_the_label_is_not_written(self):
        text = _render_sipnet_in(ModelFlags(name="niwot"), events_enabled=False)
        assert "niwot" not in text


@requires_binary
class TestSipnetAcceptsOurConfig:
    """Hand the generated file to SIPNET and check it understood every key.

    SIPNET reports an unrecognised config key as "ignoring input file
    parameter <KEY>". Because that is only a log line, a typo or a renamed key
    would otherwise pass every test we have while quietly changing the model
    configuration. These tests read that log output and fail on it.
    """

    @staticmethod
    def _run(tmp_path, flags, fixture_dir):
        import shutil

        for name in ("sipnet.param", "sipnet.clim"):
            shutil.copy(fixture_dir / name, tmp_path / name)
        (tmp_path / "sipnet.in").write_text(_render_sipnet_in(flags, events_enabled=False))
        return subprocess.run(
            [str(binary_path()), "-i", "sipnet.in"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_no_key_is_ignored(self, tmp_path, reference_fixture_dir):
        proc = self._run(tmp_path, ModelFlags.standard(), reference_fixture_dir)
        combined = proc.stdout + proc.stderr
        ignored = [line for line in combined.splitlines() if "ignoring input file parameter" in line]
        assert not ignored, "SIPNET did not recognise these keys:\n" + "\n".join(ignored)

    def test_run_succeeds(self, tmp_path, reference_fixture_dir):
        proc = self._run(tmp_path, ModelFlags.standard(), reference_fixture_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_forest_flags_are_accepted(self, tmp_path, reference_fixture_dir):
        """The litter pool could not even be compiled before the v2.1.0 pin."""
        proc = self._run(tmp_path, ModelFlags.forest(), reference_fixture_dir)
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 0, combined
        assert "ignoring input file parameter" not in combined

    def test_sipnet_confirms_the_flags_we_asked_for(self, tmp_path, reference_fixture_dir):
        """Ask SIPNET to dump its final configuration and compare it to ours.

        This is the strongest check available: SIPNET writes out the settings
        it actually resolved, so agreement means our file was both understood
        and applied, not merely tolerated.
        """
        import shutil

        for name in ("sipnet.param", "sipnet.clim"):
            shutil.copy(reference_fixture_dir / name, tmp_path / name)
        flags = ModelFlags.forest()
        text = _render_sipnet_in(flags, events_enabled=False)
        (tmp_path / "sipnet.in").write_text(text.replace("PRINT_HEADER = 1", "PRINT_HEADER = 1\nDUMP_CONFIG = 1"))

        proc = subprocess.run(
            [str(binary_path()), "-i", "sipnet.in"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

        # Each dump line is "<KEY> <SOURCE> <VALUE>", where SOURCE says where
        # SIPNET got the value from.
        resolved = {}
        for line in (tmp_path / "sipnet.config").read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                resolved[parts[0]] = (parts[1], parts[2])

        for key, expected in flags.to_config_keys().items():
            assert key in resolved, f"{key} absent from SIPNET's own config dump"
            source, value = resolved[key]
            assert value == str(expected), (
                f"we asked for {key}={expected} but SIPNET resolved it to {value}"
            )
            # A source of DEFAULT would mean SIPNET never saw our setting and
            # happened to agree with us, which is not the same as obeying it.
            assert source == "INPUT_FILE", (
                f"{key} came from {source}, not from the config file we wrote"
            )
