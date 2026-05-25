"""Tests for pysipnet.build — binary cache management and build helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysipnet.build import (
    _CACHE_DIR,
    _PATCH_SCRIPT,
    _REPO_ROOT,
    _SIPNET_DIR,
    apply_patch,
    binary_sha256,
    build_preset,
    ensure_binary,
    init_submodule,
)
from pysipnet.runner import ModelPreset

_STANDARD_BINARY = _CACHE_DIR / ModelPreset.STANDARD.binary_name


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_repo_root_has_makefile(self):
        assert (_REPO_ROOT / "Makefile").exists()

    def test_sipnet_dir_exists(self):
        assert _SIPNET_DIR.exists()

    def test_patch_script_exists(self):
        assert _PATCH_SCRIPT.exists()


# ---------------------------------------------------------------------------
# ensure_binary
# ---------------------------------------------------------------------------

class TestEnsureBinary:
    def test_raises_if_binary_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="sipnet_standard"):
            ensure_binary(ModelPreset.STANDARD)

    def test_error_message_includes_make_command(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="make sipnet-standard"):
            ensure_binary(ModelPreset.STANDARD)

    @pytest.mark.skipif(
        not _STANDARD_BINARY.exists(),
        reason="Binary not built; run 'make sipnet-standard'",
    )
    def test_returns_path_when_binary_exists(self):
        path = ensure_binary(ModelPreset.STANDARD)
        assert path.exists()
        assert path.name == "sipnet_standard"


# ---------------------------------------------------------------------------
# binary_sha256
# ---------------------------------------------------------------------------

class TestBinarySha256:
    @pytest.mark.skipif(
        not _STANDARD_BINARY.exists(),
        reason="Binary not built; run 'make sipnet-standard'",
    )
    def test_returns_64_char_hex(self):
        digest = binary_sha256(ModelPreset.STANDARD)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    @pytest.mark.skipif(
        not _STANDARD_BINARY.exists(),
        reason="Binary not built; run 'make sipnet-standard'",
    )
    def test_deterministic(self):
        assert binary_sha256(ModelPreset.STANDARD) == binary_sha256(ModelPreset.STANDARD)

    def test_raises_if_binary_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            binary_sha256(ModelPreset.STANDARD)

    def test_content_sensitive(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        binary_a = tmp_path / "sipnet_standard"
        binary_b = tmp_path / "sipnet_forest"
        binary_a.write_bytes(b"content_a")
        binary_b.write_bytes(b"content_b")
        assert binary_sha256(ModelPreset.STANDARD) != binary_sha256(ModelPreset.FOREST)


# ---------------------------------------------------------------------------
# build_preset (no-op path only — avoids running make in unit tests)
# ---------------------------------------------------------------------------

class TestBuildPreset:
    @pytest.mark.skipif(
        not _STANDARD_BINARY.exists(),
        reason="Binary not built; run 'make sipnet-standard'",
    )
    def test_noop_if_binary_exists(self, monkeypatch):
        """build_preset(force=False) returns the existing path without calling make."""
        called = []

        def fake_run(args, **kwargs):
            called.append(args)

        monkeypatch.setattr("pysipnet.build.subprocess.run", fake_run)
        path = build_preset(ModelPreset.STANDARD, force=False)
        assert path.exists()
        assert called == [], "subprocess.run should not have been called"

    @pytest.mark.skipif(
        not _STANDARD_BINARY.exists(),
        reason="Binary not built; run 'make sipnet-standard'",
    )
    def test_returns_correct_path(self):
        path = build_preset(ModelPreset.STANDARD, force=False)
        assert path == _CACHE_DIR / "sipnet_standard"


# ---------------------------------------------------------------------------
# init_submodule
# ---------------------------------------------------------------------------

class TestInitSubmodule:
    def test_noop_if_makefile_exists(self, monkeypatch):
        """init_submodule does nothing if sipnet/Makefile is present."""
        called = []
        monkeypatch.setattr("pysipnet.build.subprocess.run", lambda *a, **kw: called.append(a))
        init_submodule()
        assert called == [], "subprocess.run should not be called when submodule is present"
