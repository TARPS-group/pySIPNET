"""Tests for :mod:`pysipnet.build` — compiling and locating the SIPNET binary.

Two groups of tests here do different jobs.

The unit tests use a temporary directory in place of the real binary cache, so
they run anywhere and never invoke the compiler.

The tests marked ``requires_binary`` check the actual compiled binary, and are
skipped when it has not been built. Among them are two checks that the binary
and the pin agree with each other: a mismatch there means pySIPNET would be
driving a different model than the one this release claims to support, which
would not otherwise announce itself.
"""

from __future__ import annotations

import subprocess

import pytest

from pysipnet.build import (
    _CACHE_DIR,
    _REPO_ROOT,
    _SIPNET_DIR,
    BINARY_NAME,
    binary_path,
    binary_sha256,
    build_sipnet,
    ensure_binary,
    init_submodule,
    sipnet_build_tag,
    sipnet_version,
)
from pysipnet.version import (
    SIPNET_NUMERIC_VERSION,
    SIPNET_PINNED_COMMIT,
    SIPNET_PINNED_TAG,
)

# Skip marker for tests that need a compiled binary present.
requires_binary = pytest.mark.skipif(
    not binary_path().exists(),
    reason="SIPNET binary not built; run 'make sipnet'",
)


class TestPaths:
    """The module-level paths should point at real things in this repository."""

    def test_repo_root_has_makefile(self):
        assert (_REPO_ROOT / "Makefile").exists()

    def test_sipnet_submodule_is_populated(self):
        assert (_SIPNET_DIR / "Makefile").exists(), (
            "sipnet/ submodule looks empty; run 'git submodule update --init sipnet'"
        )

    def test_binary_path_is_inside_cache_dir(self):
        assert binary_path() == _CACHE_DIR / BINARY_NAME

    def test_binary_path_does_not_touch_the_filesystem(self, tmp_path, monkeypatch):
        """binary_path only computes a path, so it works before anything is built."""
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        assert binary_path() == tmp_path / BINARY_NAME


class TestEnsureBinary:
    def test_raises_when_binary_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            ensure_binary()

    def test_error_message_says_how_to_fix_it(self, tmp_path, monkeypatch):
        """A missing binary is a setup problem, so the message must be actionable."""
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="make sipnet"):
            ensure_binary()

    def test_never_compiles_anything(self, tmp_path, monkeypatch):
        """ensure_binary is a check, not a build step."""
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run",
            lambda *a, **kw: pytest.fail("ensure_binary must not run subprocesses"),
        )
        with pytest.raises(FileNotFoundError):
            ensure_binary()

    @requires_binary
    def test_returns_the_binary_path(self):
        assert ensure_binary() == binary_path()

    @requires_binary
    def test_returned_path_is_executable(self):
        import os

        assert os.access(ensure_binary(), os.X_OK)


class TestBinarySha256:
    @requires_binary
    def test_is_64_hex_characters(self):
        digest = binary_sha256()
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    @requires_binary
    def test_is_stable_across_calls(self):
        assert binary_sha256() == binary_sha256()

    def test_changes_when_the_binary_changes(self, tmp_path, monkeypatch):
        """The digest identifies a specific build, so different bytes must differ."""
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        target = tmp_path / BINARY_NAME

        target.write_bytes(b"one build")
        first = binary_sha256()

        target.write_bytes(b"a different build")
        assert binary_sha256() != first

    def test_raises_when_binary_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            binary_sha256()


class TestBuild:
    def test_skips_compiling_when_a_binary_is_already_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        (tmp_path / BINARY_NAME).write_bytes(b"pretend binary")
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run",
            lambda *a, **kw: pytest.fail("build_sipnet() should not have compiled"),
        )
        assert build_sipnet() == tmp_path / BINARY_NAME

    def test_force_compiles_even_when_a_binary_is_present(self, tmp_path, monkeypatch):
        """force=True is the escape hatch after changing the pinned SIPNET version."""
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        (tmp_path / BINARY_NAME).write_bytes(b"stale binary")

        commands = []
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run", lambda args, **kw: commands.append(args)
        )

        build_sipnet(force=True)
        assert ["make", "sipnet"] in commands

    def test_compiles_when_no_binary_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        commands = []
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run", lambda args, **kw: commands.append(args)
        )

        build_sipnet()
        assert ["make", "sipnet"] in commands


class TestInitSubmodule:
    def test_does_nothing_when_the_submodule_is_populated(self, monkeypatch):
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run",
            lambda *a, **kw: pytest.fail("submodule is already present"),
        )
        init_submodule()

    def test_fetches_when_the_submodule_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._SIPNET_DIR", tmp_path / "empty")
        commands = []
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run", lambda args, **kw: commands.append(args)
        )

        init_submodule()
        assert commands == [["git", "submodule", "update", "--init", "sipnet"]]


class TestBinaryMatchesThePin:
    """The compiled binary and the recorded pin must describe the same SIPNET.

    These are the tests that catch a stale binary or a half-finished version
    bump. Without them, pySIPNET would happily drive a binary built from
    different source than the version constants advertise, and the only symptom
    would be quietly wrong model output.
    """

    def test_submodule_is_checked_out_at_the_pinned_commit(self):
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_SIPNET_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == SIPNET_PINNED_COMMIT, (
            "sipnet/ is not at the commit recorded in pysipnet.version. "
            "Run 'git submodule update --init sipnet'."
        )

    @requires_binary
    def test_binary_was_built_from_the_pinned_tag(self):
        """A binary left over from a previous pin would fail here.

        Checks the ``git describe`` tag, not the numeric version. SIPNET's
        version.h lags pre-release tags — at v2.2.0-alpha.1 it still reads
        2.1.0 — so a numeric check would accept a binary from the wrong
        release and report success.
        """
        tag = sipnet_build_tag()
        assert tag, (
            "the binary carries no build tag, so which SIPNET source it came from "
            "cannot be verified. SIPNET's Makefile stamps in `git describe --tags`, "
            "which is empty when the submodule has no tags — run "
            "`git -C sipnet fetch --tags` and rebuild with `make sipnet`. "
            f"(Full version string: {sipnet_version()!r}.)"
        )
        assert tag == SIPNET_PINNED_TAG, (
            f"binary was built from {tag!r} but this release pins "
            f"{SIPNET_PINNED_TAG!r}. Rebuild with 'make sipnet'. "
            f"(Full version string: {sipnet_version()!r}.)"
        )

    @requires_binary
    def test_numeric_version_is_recorded_accurately(self):
        """The numeric version is reported, not used for identity.

        Recorded so that a change to it is noticed, and so the gap between it
        and the tag stays visible rather than becoming a surprise.
        """
        assert sipnet_version().startswith(SIPNET_NUMERIC_VERSION)

    @requires_binary
    def test_the_numeric_version_alone_would_not_identify_the_pin(self):
        """Documents why the identity check uses the tag.

        If these two ever agree, the distinction still holds in principle but
        this test stops being informative — so it asserts the mismatch that
        motivates the design.
        """
        assert SIPNET_NUMERIC_VERSION != SIPNET_PINNED_TAG.removeprefix("v"), (
            "numeric version now matches the tag; the identity check can stay "
            "as it is, but this test no longer demonstrates why it exists"
        )

    @requires_binary
    def test_version_string_is_not_empty(self):
        assert sipnet_version().strip()
