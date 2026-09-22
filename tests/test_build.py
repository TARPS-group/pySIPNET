"""Tests for :mod:`pysipnet.build` — finding, compiling and verifying the SIPNET binary.

Three groups.

The unit tests point every candidate location at a temporary directory, so
they run anywhere, never touch the network and never invoke a compiler. They
pin the *search order*: which binary wins when several exist, and what a user
is told when none does.

The tests on :func:`verify_binary_matches_pin` use small shell scripts that
print a chosen version string, because the property under test is what the
runner does with the answer, not what the real binary answers.

The tests marked ``requires_binary`` check the actual compiled binary and are
skipped when it has not been built. Among them are the checks that the binary
and the pin agree: a mismatch there means pySIPNET would be driving a different
model than the one this release claims to support.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from pysipnet.build import (
    _REPO_ROOT,
    _SIPNET_DIR,
    BINARY_ENV_VAR,
    BINARY_NAME,
    CACHE_DIR_ENV_VAR,
    PINNED_CACHE_SUBDIR,
    BinaryVersionError,
    BuildError,
    binary_candidates,
    binary_path,
    binary_sha256,
    build_sipnet,
    checkout_cache_dir,
    describe_binary_search,
    ensure_binary,
    find_binary,
    in_source_tree,
    init_submodule,
    install_sipnet,
    install_target,
    sipnet_build_tag,
    sipnet_version,
    user_cache_dir,
    verify_binary_matches_pin,
)
from pysipnet.version import (
    SIPNET_NUMERIC_VERSION,
    SIPNET_PINNED_COMMIT,
    SIPNET_PINNED_TAG,
)
from tests.helpers import PINNED_VERSION_LINE
from tests.helpers import fake_sipnet_binary as _fake_binary

# Skip marker for tests that need a compiled binary present.
requires_binary = pytest.mark.skipif(
    find_binary() is None,
    reason="no SIPNET binary; run 'pysipnet install-sipnet' (or 'make sipnet' in a checkout)",
)

isolated = pytest.fixture(lambda isolated_binary_locations: isolated_binary_locations)


class TestPaths:
    """The module-level paths should point at real things in this repository."""

    def test_repo_root_has_makefile(self):
        assert (_REPO_ROOT / "Makefile").exists()

    def test_sipnet_submodule_is_populated(self):
        assert (_SIPNET_DIR / "Makefile").exists(), (
            "sipnet/ submodule looks empty; run 'git submodule update --init sipnet'"
        )

    def test_the_test_suite_runs_in_a_source_tree(self):
        assert in_source_tree()

    def test_install_target_in_a_source_tree_is_the_checkout_cache(self):
        assert install_target() == checkout_cache_dir() / BINARY_NAME

    def test_checkout_cache_is_named_by_the_pinned_commit_like_the_makefile(self):
        """make sipnet reads the commit from version.py; both must land in the same directory."""
        assert checkout_cache_dir().name == PINNED_CACHE_SUBDIR == SIPNET_PINNED_COMMIT[:12]
        dry_run = subprocess.run(
            ["make", "-n", "sipnet"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        )
        assert f".sipnet_cache/{PINNED_CACHE_SUBDIR}/sipnet" in dry_run.stdout

    def test_install_target_outside_a_source_tree_is_the_user_cache(self, isolated, monkeypatch):
        monkeypatch.setattr("pysipnet.build.in_source_tree", lambda: False)
        assert install_target() == user_cache_dir() / BINARY_NAME

    def test_user_cache_is_named_by_the_pinned_commit(self, isolated):
        """A pin bump must look in a fresh directory, never at the previous version's binary."""
        assert user_cache_dir() == isolated["user cache"]
        assert user_cache_dir().name == SIPNET_PINNED_COMMIT[:12]

    def test_user_cache_defaults_to_platformdirs(self, isolated, monkeypatch):
        monkeypatch.delenv(CACHE_DIR_ENV_VAR)
        monkeypatch.setattr("pysipnet.build.platformdirs.user_cache_dir", lambda app: "/pd/" + app)
        assert user_cache_dir() == Path("/pd/pysipnet/sipnet") / SIPNET_PINNED_COMMIT[:12]


class TestSearchOrder:
    def test_order_is_environment_bundled_checkout_user(self, isolated, monkeypatch):
        monkeypatch.setenv(BINARY_ENV_VAR, "/explicit/sipnet")
        assert [c.source for c in binary_candidates()] == [
            "environment",
            "bundled",
            "source tree",
            "user cache",
        ]

    def test_environment_candidate_only_when_set(self, isolated):
        assert [c.source for c in binary_candidates()][0] == "bundled"

    def test_checkout_candidate_only_in_a_source_tree(self, isolated, monkeypatch):
        monkeypatch.setattr("pysipnet.build.in_source_tree", lambda: False)
        assert "source tree" not in [c.source for c in binary_candidates()]

    def test_nothing_found_when_nothing_installed(self, isolated):
        assert find_binary() is None

    def test_user_cache_is_found_when_it_is_all_there_is(self, isolated):
        target = user_cache_dir() / BINARY_NAME
        _fake_binary(target, PINNED_VERSION_LINE)
        found = find_binary()
        assert found is not None and (found.source, found.path) == ("user cache", target)

    def test_checkout_cache_beats_user_cache(self, isolated):
        _fake_binary(user_cache_dir() / BINARY_NAME, PINNED_VERSION_LINE)
        checkout = _fake_binary(isolated["source tree"] / BINARY_NAME, PINNED_VERSION_LINE)
        found = find_binary()
        assert found is not None and found.path == checkout

    def test_bundled_beats_checkout_cache(self, isolated):
        _fake_binary(isolated["source tree"] / BINARY_NAME, PINNED_VERSION_LINE)
        bundled = _fake_binary(isolated["bundled"] / BINARY_NAME, PINNED_VERSION_LINE)
        found = find_binary()
        assert found is not None and found.path == bundled

    def test_environment_beats_everything(self, isolated, monkeypatch, tmp_path):
        _fake_binary(isolated["bundled"] / BINARY_NAME, PINNED_VERSION_LINE)
        explicit = _fake_binary(tmp_path / "elsewhere" / "sipnet", PINNED_VERSION_LINE)
        monkeypatch.setenv(BINARY_ENV_VAR, str(explicit))
        found = find_binary()
        assert found is not None and (found.source, found.path) == ("environment", explicit)

    def test_environment_path_that_does_not_exist_is_skipped_not_trusted(
        self, isolated, monkeypatch
    ):
        """A wrong $PYSIPNET_BINARY falls through to the next candidate, and is reported."""
        monkeypatch.setenv(BINARY_ENV_VAR, str(isolated["bundled"] / "missing"))
        real = _fake_binary(user_cache_dir() / BINARY_NAME, PINNED_VERSION_LINE)
        found = find_binary()
        assert found is not None and found.path == real
        assert "environment" in describe_binary_search()
        assert "(not found)" in describe_binary_search().splitlines()[0]

    def test_relative_environment_path_is_resolved(self, isolated, monkeypatch, tmp_path):
        """The binary runs with the workdir as cwd, where a relative path would point elsewhere."""
        _fake_binary(tmp_path / "here" / "sipnet", PINNED_VERSION_LINE)
        monkeypatch.chdir(tmp_path / "here")
        monkeypatch.setenv(BINARY_ENV_VAR, "./sipnet")
        found = find_binary()
        assert found is not None and found.path.is_absolute()
        assert found.path == (tmp_path / "here" / "sipnet").resolve()

    def test_binary_path_is_the_found_binary_or_the_install_target(self, isolated):
        assert binary_path() == install_target()
        planted = _fake_binary(user_cache_dir() / BINARY_NAME, PINNED_VERSION_LINE)
        assert binary_path() == planted


class TestEnsureBinary:
    def test_raises_when_binary_is_missing(self, isolated):
        with pytest.raises(FileNotFoundError):
            ensure_binary()

    def test_error_lists_every_place_it_looked_and_how_to_fix_it(self, isolated):
        """A missing binary is a setup problem, so the message must be actionable."""
        with pytest.raises(FileNotFoundError) as excinfo:
            ensure_binary()
        message = str(excinfo.value)
        assert "pysipnet install-sipnet" in message
        assert BINARY_ENV_VAR in message
        for source in ("bundled", "source tree", "user cache"):
            assert source in message

    def test_never_compiles_anything(self, isolated, monkeypatch):
        """ensure_binary is a check, not a build step."""
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

    def test_changes_when_the_binary_changes(self, isolated):
        """The digest identifies a specific build, so different bytes must differ."""
        target = user_cache_dir() / BINARY_NAME
        target.parent.mkdir(parents=True)
        target.write_bytes(b"one build")
        first = binary_sha256()
        target.write_bytes(b"a different build")
        assert binary_sha256() != first

    def test_raises_when_binary_is_missing(self, isolated):
        with pytest.raises(FileNotFoundError):
            binary_sha256()


class TestBuild:
    def test_skips_compiling_when_a_binary_is_at_the_install_target(self, isolated, monkeypatch):
        isolated["source tree"].mkdir(parents=True)
        (isolated["source tree"] / BINARY_NAME).write_bytes(b"pretend binary")
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run",
            lambda *a, **kw: pytest.fail("build_sipnet() should not have compiled"),
        )
        assert build_sipnet() == isolated["source tree"] / BINARY_NAME

    def test_force_compiles_even_when_a_binary_is_present(self, isolated, monkeypatch):
        """force=True is the escape hatch after changing the pinned SIPNET version."""
        isolated["source tree"].mkdir(parents=True)
        (isolated["source tree"] / BINARY_NAME).write_bytes(b"stale binary")
        commands = []
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run", lambda args, **kw: commands.append(args)
        )
        build_sipnet(force=True)
        assert ["make", "sipnet"] in commands

    def test_in_a_source_tree_compiles_with_make(self, isolated, monkeypatch):
        commands = []
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run", lambda args, **kw: commands.append(args)
        )
        build_sipnet()
        assert ["make", "sipnet"] in commands

    def test_outside_a_source_tree_fetches_the_pinned_commit_and_stamps_the_tag(
        self, isolated, monkeypatch
    ):
        """Without a checkout, the source is fetched by commit and built with SIPNET's Makefile."""
        monkeypatch.setattr("pysipnet.build.in_source_tree", lambda: False)
        monkeypatch.setattr("pysipnet.build.shutil.which", lambda name: f"/usr/bin/{name}")
        commands: list[list[str]] = []
        target = install_target()

        def fake_run(args, **kw):
            commands.append(list(args))
            cwd = Path(kw.get("cwd", "."))

            class Result:
                stdout = SIPNET_PINNED_COMMIT + "\n"
                returncode = 0

            if args[:2] == ["make", "sipnet"]:
                _fake_binary(cwd / BINARY_NAME, PINNED_VERSION_LINE)
            if args[1:] == ["--version"]:
                Result.stdout = PINNED_VERSION_LINE
            return Result()

        monkeypatch.setattr("pysipnet.build.subprocess.run", fake_run)
        assert build_sipnet() == target
        assert target.is_file()
        assert ["git", "fetch", "-q", "--depth", "1", "origin", SIPNET_PINNED_COMMIT] in commands
        assert ["make", "sipnet", f"GIT_HASH={SIPNET_PINNED_TAG}"] in commands

    def test_outside_a_source_tree_names_the_missing_tool(self, isolated, monkeypatch):
        monkeypatch.setattr("pysipnet.build.in_source_tree", lambda: False)
        monkeypatch.setattr("pysipnet.build.shutil.which", lambda name: None)
        with pytest.raises(BuildError, match="'git' was not found"):
            build_sipnet()

    def test_pre_check_asks_for_the_compiler_the_makefile_uses(self, isolated, monkeypatch):
        """SIPNET's Makefile hard-codes CC=gcc, so gcc, not cc, is the name that must resolve."""
        assert (_SIPNET_DIR / "Makefile").read_text().splitlines()[0] == "CC=gcc"
        monkeypatch.setattr("pysipnet.build.in_source_tree", lambda: False)
        monkeypatch.setattr(
            "pysipnet.build.shutil.which",
            lambda name: None if name == "gcc" else f"/usr/bin/{name}",
        )
        with pytest.raises(BuildError, match="'gcc' was not found"):
            build_sipnet()

    def test_a_wrong_tag_from_the_compile_route_is_a_build_error(self, isolated, monkeypatch):
        """The compile route must not blame the download digest for a mismatched binary."""
        monkeypatch.setattr("pysipnet.build.in_source_tree", lambda: False)
        monkeypatch.setattr("pysipnet.build.shutil.which", lambda name: f"/usr/bin/{name}")

        def fake_run(args, **kw):
            class Result:
                stdout = SIPNET_PINNED_COMMIT + "\n"

            if args[:2] == ["make", "sipnet"]:
                _fake_binary(Path(kw["cwd"]) / BINARY_NAME, "SIPNET version 2.1.0 (v2.1.0)")
            if args[1:] == ["--version"]:
                Result.stdout = "SIPNET version 2.1.0 (v2.1.0)"
            return Result()

        monkeypatch.setattr("pysipnet.build.subprocess.run", fake_run)
        with pytest.raises(BuildError, match="v2.1.0.*It was not installed"):
            build_sipnet()
        assert not install_target().exists()
        assert not list(install_target().parent.glob(".sipnet.*")), "staging file left behind"


@pytest.mark.network
class TestCompileFromGitForReal:
    """The fetch-and-compile route, run for real: git, make and gcc against upstream."""

    def test_compiles_the_pinned_commit_into_the_target(self, tmp_path):
        import shutil

        from pysipnet.build import _compile_pinned_source

        missing = [tool for tool in ("git", "make", "gcc") if shutil.which(tool) is None]
        if missing:
            pytest.skip(f"toolchain missing: {missing}")
        target = tmp_path / "cache" / BINARY_NAME
        _compile_pinned_source(target)
        assert target.is_file() and os.access(target, os.X_OK)
        assert (
            verify_binary_matches_pin(target) == f"{SIPNET_NUMERIC_VERSION} ({SIPNET_PINNED_TAG})"
        )
        assert not (target.parent / "src").exists(), "source tree should be cleaned up"


class TestInstallSipnet:
    def test_returns_an_existing_verified_binary_without_installing(self, isolated, monkeypatch):
        planted = _fake_binary(user_cache_dir() / BINARY_NAME, PINNED_VERSION_LINE)
        monkeypatch.setattr(
            "pysipnet.build.download_sipnet", lambda **kw: pytest.fail("must not download")
        )
        monkeypatch.setattr(
            "pysipnet.build.build_sipnet", lambda **kw: pytest.fail("must not compile")
        )
        assert install_sipnet() == planted

    def test_never_hands_back_a_binary_from_the_wrong_release(self, isolated, monkeypatch):
        """An existing binary is only "already installed" if it is the pinned SIPNET."""
        stale = _fake_binary(install_target(), "SIPNET version 2.1.0 (v2.1.0)")
        monkeypatch.setattr("pysipnet.build.prebuilt_unavailable_reason", lambda key=None: None)
        seen = {}

        def fake_download(*, force):
            seen["force"] = force
            return _fake_binary(stale, PINNED_VERSION_LINE)

        monkeypatch.setattr("pysipnet.build.download_sipnet", fake_download)
        assert install_sipnet() == stale
        assert seen == {"force": True}, "the stale file at the target must be replaced"

    def test_a_wrong_environment_binary_is_an_error_not_a_reinstall(
        self, isolated, monkeypatch, tmp_path
    ):
        wrong = _fake_binary(tmp_path / "module" / "sipnet", "SIPNET version 2.1.0 (v2.1.0)")
        monkeypatch.setenv(BINARY_ENV_VAR, str(wrong))
        monkeypatch.setattr(
            "pysipnet.build.download_sipnet", lambda **kw: pytest.fail("must not download")
        )
        with pytest.raises(BinaryVersionError, match="unset it first"):
            install_sipnet()

    def test_auto_downloads_when_a_usable_prebuilt_exists(self, isolated, monkeypatch):
        monkeypatch.setattr("pysipnet.build.prebuilt_unavailable_reason", lambda key=None: None)
        monkeypatch.setattr("pysipnet.build.download_sipnet", lambda **kw: Path("/downloaded"))
        assert install_sipnet() == Path("/downloaded")

    def test_auto_compiles_when_no_usable_prebuilt_exists(self, isolated, monkeypatch):
        monkeypatch.setattr(
            "pysipnet.build.prebuilt_unavailable_reason", lambda key=None: "too old"
        )
        monkeypatch.setattr("pysipnet.build.build_sipnet", lambda **kw: Path("/compiled"))
        assert install_sipnet() == Path("/compiled")

    def test_explicit_method_is_obeyed(self, isolated, monkeypatch):
        monkeypatch.setattr("pysipnet.build.prebuilt_unavailable_reason", lambda key=None: None)
        monkeypatch.setattr("pysipnet.build.build_sipnet", lambda **kw: Path("/compiled"))
        assert install_sipnet(method="compile") == Path("/compiled")


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


class TestVerifyBinaryMatchesPin:
    """The runtime check that a binary is the pinned SIPNET, not merely *a* SIPNET."""

    def test_accepts_the_pinned_tag_and_returns_the_version(self, tmp_path):
        binary = _fake_binary(tmp_path / "sipnet", PINNED_VERSION_LINE)
        assert (
            verify_binary_matches_pin(binary) == f"{SIPNET_NUMERIC_VERSION} ({SIPNET_PINNED_TAG})"
        )

    def test_refuses_another_tag(self, tmp_path):
        binary = _fake_binary(tmp_path / "sipnet", "SIPNET version 2.1.0 (v2.1.0)")
        with pytest.raises(BinaryVersionError, match="v2.1.0"):
            verify_binary_matches_pin(binary)

    def test_refuses_an_untagged_build(self, tmp_path):
        binary = _fake_binary(tmp_path / "sipnet", "SIPNET version 2.1.0 ()")
        with pytest.raises(BinaryVersionError, match="untagged"):
            verify_binary_matches_pin(binary)

    def test_numeric_version_alone_does_not_pass(self, tmp_path):
        """The numeric version lags pre-release tags; only the tag identifies the pin."""
        binary = _fake_binary(tmp_path / "sipnet", f"SIPNET version {SIPNET_NUMERIC_VERSION}")
        with pytest.raises(BinaryVersionError):
            verify_binary_matches_pin(binary)

    def test_a_freshly_installed_binary_is_not_run_twice(self, isolated, monkeypatch):
        """Installing checks the binary; the CLI's verify afterwards must be a cache hit."""
        from pysipnet.build import _install_binary

        source = _fake_binary(isolated["bundled"] / "src" / "sipnet", PINNED_VERSION_LINE)
        target = install_target()
        _install_binary(source, target)
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run",
            lambda *a, **kw: pytest.fail("verify after install must hit the cache"),
        )
        assert (
            verify_binary_matches_pin(target) == f"{SIPNET_NUMERIC_VERSION} ({SIPNET_PINNED_TAG})"
        )

    def test_a_binary_that_will_not_run_is_reported_with_its_stderr(self, tmp_path):
        binary = tmp_path / "sipnet"
        binary.write_text("#!/bin/sh\necho 'dyld: incompatible' >&2\nexit 1\n")
        binary.chmod(0o755)
        with pytest.raises(BinaryVersionError, match="would not run.*incompatible"):
            verify_binary_matches_pin(binary)

    def test_result_is_cached_per_file(self, tmp_path, monkeypatch):
        """An ensemble pays one --version call per process, not one per run."""
        binary = _fake_binary(tmp_path / "sipnet", PINNED_VERSION_LINE)
        verify_binary_matches_pin(binary)
        monkeypatch.setattr(
            "pysipnet.build.subprocess.run",
            lambda *a, **kw: pytest.fail("second call must hit the cache"),
        )
        verify_binary_matches_pin(binary)

    def test_cache_notices_a_replaced_file(self, tmp_path):
        binary = _fake_binary(tmp_path / "sipnet", PINNED_VERSION_LINE)
        verify_binary_matches_pin(binary)
        _fake_binary(binary, "SIPNET version 9.9.9 (v9.9.9) padded so the size differs")
        with pytest.raises(BinaryVersionError):
            verify_binary_matches_pin(binary)


class TestRunnerRefusesTheWrongBinary:
    """The check is wired into the runner, so a run cannot start on a mismatched binary."""

    def test_run_raises_before_writing_anything(self, tmp_path, minimal_params):
        from pysipnet.climate import ClimateDrivers
        from pysipnet.io.reference import niwot_reference_climate
        from pysipnet.runner import SIPNETRunner

        wrong = _fake_binary(tmp_path / "sipnet", "SIPNET version 2.1.0 (v2.1.0)")
        climate = ClimateDrivers.from_dataframe(
            niwot_reference_climate().pandas.head(4).copy(), n_columns=14
        )
        runner = SIPNETRunner(binary=wrong, workdir_base=tmp_path / "work")
        with pytest.raises(BinaryVersionError, match="v2.1.0"):
            runner.run(minimal_params, climate)
        assert not (tmp_path / "work").exists()

    def test_verify_binary_false_lets_it_through(self, tmp_path):
        from pysipnet.runner import SIPNETRunner

        wrong = _fake_binary(tmp_path / "sipnet", "SIPNET version 2.1.0 (v2.1.0)")
        SIPNETRunner(binary=wrong, verify_binary=False)._check_binary()

    def test_explicit_binary_wins_over_the_search(self, tmp_path, isolated):
        from pysipnet.runner import SIPNETRunner

        _fake_binary(isolated["bundled"] / BINARY_NAME, PINNED_VERSION_LINE)
        explicit = _fake_binary(tmp_path / "mine" / "sipnet", PINNED_VERSION_LINE)
        assert SIPNETRunner(binary=explicit).binary_path == explicit.resolve()
        assert SIPNETRunner(cache_dir=tmp_path / "mine").binary_path == explicit.resolve()

    def test_relative_binary_is_resolved_so_the_check_and_the_run_agree(
        self, tmp_path, monkeypatch
    ):
        """Path("./sipnet") is "sipnet", which --version would look up on PATH and the run
        would look for inside the temporary working directory."""
        from pysipnet.runner import SIPNETRunner

        real = _fake_binary(tmp_path / "here" / "sipnet", PINNED_VERSION_LINE)
        monkeypatch.chdir(tmp_path / "here")
        runner = SIPNETRunner(binary="./sipnet")
        assert runner.binary_path == real.resolve()
        assert runner._check_binary() == real.resolve()

    def test_default_runner_follows_the_search(self, isolated):
        from pysipnet.runner import SIPNETRunner

        planted = _fake_binary(user_cache_dir() / BINARY_NAME, PINNED_VERSION_LINE)
        assert SIPNETRunner().binary_path == planted

    def test_missing_binary_message_names_the_command(self, isolated):
        from pysipnet.runner import SIPNETRunner

        with pytest.raises(FileNotFoundError, match="pysipnet install-sipnet"):
            SIPNETRunner()._check_binary()


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
    def test_the_runtime_check_accepts_the_real_binary(self):
        assert verify_binary_matches_pin(ensure_binary()) == sipnet_version()

    @requires_binary
    def test_numeric_version_is_recorded_accurately(self):
        """The numeric version is reported, not used for identity."""
        assert sipnet_version().startswith(SIPNET_NUMERIC_VERSION)

    @requires_binary
    def test_the_numeric_version_alone_would_not_identify_the_pin(self):
        """Documents why the identity check uses the tag."""
        assert SIPNET_NUMERIC_VERSION != SIPNET_PINNED_TAG.removeprefix("v"), (
            "numeric version now matches the tag; the identity check can stay "
            "as it is, but this test no longer demonstrates why it exists"
        )
