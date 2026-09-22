"""The ``pysipnet`` command: the one thing a non-developer runs after ``pip install``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysipnet.build import BINARY_NAME, install_target, user_cache_dir
from pysipnet.cli import build_parser, main, staged_bundle_path
from pysipnet.version import SIPNET_NUMERIC_VERSION, SIPNET_PINNED_TAG
from tests.helpers import fake_sipnet_binary as _fake_binary


@pytest.fixture
def isolated(isolated_binary_locations: dict[str, Path], tmp_path: Path) -> Path:
    return tmp_path


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


class TestInfo:
    def test_reports_the_pin_and_where_it_looked(self, isolated, capsys):
        code = main(["info"])
        out = capsys.readouterr().out
        assert code == 1, "no binary anywhere is a non-zero exit, so a setup script can tell"
        assert SIPNET_PINNED_TAG in out
        assert "user cache" in out and "bundled" in out
        assert "pysipnet install-sipnet" in out

    def test_reports_the_binary_in_use(self, isolated, capsys):
        planted = _fake_binary(user_cache_dir() / BINARY_NAME)
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        assert f"using: {planted} (user cache)" in out

    def test_a_mismatched_binary_is_reported_and_fails(self, isolated, capsys):
        _fake_binary(user_cache_dir() / BINARY_NAME, "SIPNET version 2.1.0 (v2.1.0)")
        assert main(["info"]) == 1
        assert "v2.1.0" in capsys.readouterr().out


class TestInstall:
    def test_installs_and_prints_the_version(self, isolated, monkeypatch, capsys):
        planted = _fake_binary(isolated / "new" / BINARY_NAME)
        monkeypatch.setattr("pysipnet.cli.install_sipnet", lambda **kw: planted)
        assert main(["install-sipnet"]) == 0
        assert f"SIPNET {SIPNET_NUMERIC_VERSION} ({SIPNET_PINNED_TAG}) at {planted}" in (
            capsys.readouterr().out
        )

    def test_passes_method_and_force_through(self, isolated, monkeypatch):
        seen = {}
        planted = _fake_binary(isolated / "new" / BINARY_NAME)

        def fake(**kw):
            seen.update(kw)
            return planted

        monkeypatch.setattr("pysipnet.cli.install_sipnet", fake)
        assert main(["install-sipnet", "--method", "compile", "--force"]) == 0
        assert seen == {"method": "compile", "force": True}

    def test_an_existing_wrong_binary_is_one_line_not_a_traceback(
        self, isolated, monkeypatch, capsys
    ):
        """install-sipnet when the binary already at the install target is the wrong release.

        ``--method download`` without ``--force`` keeps an existing target, so
        the CLI's own check is what catches it, and that failure must come out
        as the one-line error rather than a traceback.
        """
        _fake_binary(install_target(), "SIPNET version 2.1.0 (v2.1.0)")
        monkeypatch.setattr(
            "pysipnet.build._open_url", lambda *a, **kw: pytest.fail("must not download")
        )
        assert main(["install-sipnet", "--method", "download"]) == 1
        err = capsys.readouterr().err
        assert err.startswith("error: ") and "v2.1.0" in err and "Traceback" not in err

    def test_a_failure_is_one_line_on_stderr_and_exit_1(self, isolated, monkeypatch, capsys):
        from pysipnet.build import DownloadError

        def fail(**kw):
            raise DownloadError("checksum mismatch")

        monkeypatch.setattr("pysipnet.cli.install_sipnet", fail)
        assert main(["install-sipnet"]) == 1
        assert "error: checksum mismatch" in capsys.readouterr().err

    def test_method_choices_are_closed(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["install-sipnet", "--method", "wish"])


class TestStageBundle:
    def test_fetches_the_named_platform_into_the_bundle_dir(self, isolated, monkeypatch, capsys):
        calls = []

        def fake_fetch(key, target, **kw):
            calls.append((key, target))
            return target

        monkeypatch.setattr("pysipnet.cli.fetch_release_binary", fake_fetch)
        assert main(["stage-bundle", "linux-x86_64"]) == 0
        assert calls == [("linux-x86_64", staged_bundle_path("linux-x86_64"))]
        assert staged_bundle_path("linux-x86_64").parent.name == "linux-x86_64"
        assert "staged linux-x86_64" in capsys.readouterr().out

    def test_only_published_platforms_are_accepted(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["stage-bundle", "windows-x86_64"])
