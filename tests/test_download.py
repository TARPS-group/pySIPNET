"""Tests for fetching a prebuilt SIPNET binary.

What arrives over the network gets executed, so most of these tests are about
refusing bad input rather than about the happy path. They build their own
archives in a temporary directory and serve them from a stubbed opener, so the
whole file runs offline and deterministically.

The one test that really does reach GitHub is marked ``network`` and is
deselected by default; see the bottom of this file.
"""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from pysipnet.build import (
    BINARY_NAME,
    DownloadError,
    _check_archive_members,
    _find_binary,
    _sha256_of,
    download_sipnet,
    platform_key,
    release_asset,
    release_url,
)
from pysipnet.version import (
    SIPNET_RELEASE_ASSETS,
    SIPNET_RELEASE_REPO,
    SIPNET_RELEASE_TAG,
    SIPNET_TARGET_VERSION,
)

# A payload that behaves like the real binary for the one thing the installer
# checks after unpacking: that `--version` reports the targeted release.
FAKE_BINARY = f"""#!/bin/sh
echo "SIPNET version {SIPNET_TARGET_VERSION.lstrip("v")} ({SIPNET_TARGET_VERSION})"
""".encode()


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    """Build a gzipped tar in memory from ``{path: contents}``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def served(monkeypatch, tmp_path):
    """Serve chosen bytes from the download URL, and install into tmp_path.

    Returns a function that takes the bytes to serve and pins the matching
    checksum, so a test can choose whether the two agree.
    """
    monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)

    def _serve(payload: bytes, *, pinned_sha256: str | None = None):
        class _Response:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("pysipnet.build.urllib.request.urlopen", lambda *a, **kw: _Response())
        digest = pinned_sha256 or hashlib.sha256(payload).hexdigest()
        monkeypatch.setattr(
            "pysipnet.build.release_asset", lambda key=None: ("sipnet-test.tar.gz", digest)
        )

    return _serve


class TestPlatformKey:
    def test_returns_system_and_architecture(self):
        assert "-" in platform_key()

    @pytest.mark.parametrize(
        ("machine", "expected"),
        [("arm64", "arm64"), ("aarch64", "arm64"), ("x86_64", "x86_64"), ("AMD64", "x86_64")],
    )
    def test_architecture_aliases_are_normalised(self, monkeypatch, machine, expected):
        """The same architecture goes by several names depending on the OS."""
        monkeypatch.setattr("pysipnet.build.platform.system", lambda: "Linux")
        monkeypatch.setattr("pysipnet.build.platform.machine", lambda: machine)
        assert platform_key() == f"linux-{expected}"

    def test_unknown_architecture_passes_through(self, monkeypatch):
        """An unrecognised machine still produces a key, so the error can name it."""
        monkeypatch.setattr("pysipnet.build.platform.system", lambda: "Linux")
        monkeypatch.setattr("pysipnet.build.platform.machine", lambda: "riscv64")
        assert platform_key() == "linux-riscv64"


class TestReleaseAsset:
    def test_every_pinned_platform_resolves(self):
        for key in SIPNET_RELEASE_ASSETS:
            filename, digest = release_asset(key)
            assert filename
            assert len(digest) == 64

    def test_unsupported_platform_raises(self):
        with pytest.raises(DownloadError, match="No prebuilt SIPNET binary is published"):
            release_asset("plan9-vax")

    def test_error_lists_the_platforms_that_do_work(self):
        with pytest.raises(DownloadError) as exc:
            release_asset("plan9-vax")
        for key in SIPNET_RELEASE_ASSETS:
            assert key in str(exc.value)

    def test_error_points_at_building_from_source(self):
        """Compiling always works, so an unsupported platform is not a dead end."""
        with pytest.raises(DownloadError, match="make sipnet"):
            release_asset("plan9-vax")

    def test_pinned_digests_look_like_sha256(self):
        for _, digest in SIPNET_RELEASE_ASSETS.values():
            assert len(digest) == 64
            assert all(c in "0123456789abcdef" for c in digest)

    def test_pinned_filenames_carry_the_pinned_tag(self):
        """A filename from another release is the easiest bump mistake to make."""
        for filename, _ in SIPNET_RELEASE_ASSETS.values():
            assert SIPNET_RELEASE_TAG in filename, (
                f"{filename} does not mention {SIPNET_RELEASE_TAG}; is the pin half-updated?"
            )


class TestReleaseUrl:
    def test_points_at_the_pinned_repo_and_tag(self):
        url = release_url("x.tar.gz")
        assert SIPNET_RELEASE_REPO in url
        assert f"/{SIPNET_RELEASE_TAG}/" in url
        assert url.startswith("https://")


class TestChecksum:
    def test_matches_hashlib(self, tmp_path):
        path = tmp_path / "f"
        path.write_bytes(b"some bytes")
        assert _sha256_of(path) == hashlib.sha256(b"some bytes").hexdigest()

    def test_handles_a_file_larger_than_one_chunk(self, tmp_path):
        """Read in 64 KiB chunks, so a multi-chunk file is the case worth testing."""
        payload = b"x" * (65536 * 3 + 17)
        path = tmp_path / "big"
        path.write_bytes(payload)
        assert _sha256_of(path) == hashlib.sha256(payload).hexdigest()


class TestArchiveSafety:
    """An archive member may name any path, including one outside the target.

    The pinned checksum makes a hostile archive unlikely, but these checks do
    not depend on the checksum being correct, which is the point of them.
    """

    def _tar_with(self, tmp_path, info_builder) -> tarfile.TarFile:
        path = tmp_path / "a.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            info_builder(tar)
        return tarfile.open(path, "r:gz")

    def test_accepts_an_ordinary_archive(self, tmp_path):
        def build(tar):
            info = tarfile.TarInfo("sipnet")
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))

        with self._tar_with(tmp_path, build) as tar:
            _check_archive_members(tar)  # must not raise

    def test_rejects_an_absolute_path(self, tmp_path):
        def build(tar):
            info = tarfile.TarInfo("/etc/passwd")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))

        with self._tar_with(tmp_path, build) as tar:
            with pytest.raises(DownloadError, match="outside the destination"):
                _check_archive_members(tar)

    def test_rejects_a_parent_directory_escape(self, tmp_path):
        def build(tar):
            info = tarfile.TarInfo("../../evil")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))

        with self._tar_with(tmp_path, build) as tar:
            with pytest.raises(DownloadError, match="outside the destination"):
                _check_archive_members(tar)

    def test_rejects_a_symlink(self, tmp_path):
        """A symlink member can redirect a later write anywhere on disk."""

        def build(tar):
            info = tarfile.TarInfo("sipnet")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)

        with self._tar_with(tmp_path, build) as tar:
            with pytest.raises(DownloadError, match="is a link"):
                _check_archive_members(tar)

    def test_rejects_a_device_node(self, tmp_path):
        def build(tar):
            info = tarfile.TarInfo("weird")
            info.type = tarfile.CHRTYPE
            tar.addfile(info)

        with self._tar_with(tmp_path, build) as tar:
            with pytest.raises(DownloadError, match="neither a regular file nor a directory"):
                _check_archive_members(tar)


class TestFindBinary:
    def test_finds_a_binary_at_the_root(self, tmp_path):
        (tmp_path / BINARY_NAME).write_bytes(b"x")
        assert _find_binary(tmp_path).name == BINARY_NAME

    def test_finds_a_binary_in_a_subdirectory(self, tmp_path):
        """The archive layout is upstream's to change; do not assume a path."""
        nested = tmp_path / "sipnet-v2.1.0" / "bin"
        nested.mkdir(parents=True)
        (nested / BINARY_NAME).write_bytes(b"x")
        assert _find_binary(tmp_path).parent == nested

    def test_prefers_the_shallowest_match(self, tmp_path):
        (tmp_path / BINARY_NAME).write_bytes(b"top")
        nested = tmp_path / "extras"
        nested.mkdir()
        (nested / BINARY_NAME).write_bytes(b"nested")
        assert _find_binary(tmp_path).read_bytes() == b"top"

    def test_raises_when_absent_and_says_what_was_there(self, tmp_path):
        (tmp_path / "README").write_bytes(b"x")
        with pytest.raises(DownloadError, match="README"):
            _find_binary(tmp_path)


class TestDownloadSipnet:
    def test_installs_and_returns_the_binary(self, served, tmp_path):
        served(_tar_bytes({BINARY_NAME: FAKE_BINARY}))
        path = download_sipnet()
        assert path == tmp_path / BINARY_NAME
        assert path.exists()

    def test_installed_binary_is_executable(self, served, tmp_path):
        import os

        served(_tar_bytes({BINARY_NAME: FAKE_BINARY}))
        assert os.access(download_sipnet(), os.X_OK)

    def test_finds_the_binary_inside_a_directory(self, served):
        served(_tar_bytes({"sipnet-v2.1.0/" + BINARY_NAME: FAKE_BINARY}))
        assert download_sipnet().exists()

    def test_existing_binary_is_kept_without_force(self, served, tmp_path, monkeypatch):
        (tmp_path / BINARY_NAME).write_bytes(b"already here")
        monkeypatch.setattr(
            "pysipnet.build.urllib.request.urlopen",
            lambda *a, **kw: pytest.fail("must not download when a binary exists"),
        )
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        assert download_sipnet().read_bytes() == b"already here"

    def test_force_replaces_an_existing_binary(self, served, tmp_path):
        (tmp_path / BINARY_NAME).write_bytes(b"stale")
        served(_tar_bytes({BINARY_NAME: FAKE_BINARY}))
        assert download_sipnet(force=True).read_bytes() != b"stale"

    def test_checksum_mismatch_is_refused(self, served, tmp_path):
        """The whole point of pinning: mismatched bytes must not be installed."""
        served(_tar_bytes({BINARY_NAME: FAKE_BINARY}), pinned_sha256="00" * 32)
        with pytest.raises(DownloadError, match="Checksum mismatch"):
            download_sipnet()
        assert not (tmp_path / BINARY_NAME).exists(), "a mismatched download was installed"

    def test_checksum_error_shows_both_digests(self, served):
        served(_tar_bytes({BINARY_NAME: FAKE_BINARY}), pinned_sha256="00" * 32)
        with pytest.raises(DownloadError) as exc:
            download_sipnet()
        assert "expected" in str(exc.value) and "actual" in str(exc.value)

    def test_checksum_is_verified_before_extracting(self, served, tmp_path, monkeypatch):
        """Never unpack bytes that have not been verified.

        Build the archive first: patching tarfile.open patches it for this
        module too, so constructing the payload afterwards would trip the
        guard rather than the code under test.
        """
        payload = _tar_bytes({BINARY_NAME: FAKE_BINARY})
        served(payload, pinned_sha256="11" * 32)
        monkeypatch.setattr(
            "pysipnet.build.tarfile.open",
            lambda *a, **kw: pytest.fail("extracted before verifying the checksum"),
        )
        with pytest.raises(DownloadError, match="Checksum mismatch"):
            download_sipnet()

    def test_unsafe_archive_is_refused(self, served, tmp_path):
        payload = _tar_bytes({"../escape": b"x", BINARY_NAME: FAKE_BINARY})
        served(payload)
        with pytest.raises(DownloadError, match="outside the destination"):
            download_sipnet()
        assert not (tmp_path / BINARY_NAME).exists()

    def test_archive_without_a_binary_is_refused(self, served):
        served(_tar_bytes({"README": b"no binary here"}))
        with pytest.raises(DownloadError, match=f"No file named {BINARY_NAME!r}"):
            download_sipnet()

    def test_network_failure_is_reported_clearly(self, monkeypatch, tmp_path):
        import urllib.error

        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)

        def _boom(*a, **kw):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr("pysipnet.build.urllib.request.urlopen", _boom)
        with pytest.raises(DownloadError, match="Could not download"):
            download_sipnet()

    def test_network_failure_suggests_building_from_source(self, monkeypatch, tmp_path):
        import urllib.error

        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "pysipnet.build.urllib.request.urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("offline")),
        )
        with pytest.raises(DownloadError, match="make sipnet"):
            download_sipnet()

    def test_binary_reporting_the_wrong_version_is_removed(self, served, tmp_path):
        """A binary from a different release must not be left installed.

        This is the check that catches a half-finished pin bump, where the
        asset was updated but SIPNET_TARGET_VERSION was not, or vice versa.
        """
        wrong = b'#!/bin/sh\necho "SIPNET version 1.2.3 (v1.2.3)"\n'
        served(_tar_bytes({BINARY_NAME: wrong}))
        with pytest.raises(DownloadError, match="reports version"):
            download_sipnet()
        assert not (tmp_path / BINARY_NAME).exists(), "a wrong-version binary was left behind"

    def test_binary_that_will_not_run_is_removed(self, served, tmp_path):
        served(_tar_bytes({BINARY_NAME: b"not an executable at all"}))
        with pytest.raises(DownloadError):
            download_sipnet()
        assert not (tmp_path / BINARY_NAME).exists()


@pytest.mark.network
class TestAgainstTheRealRelease:
    """Actually fetch from GitHub. Deselected unless -m network is passed.

    Everything above uses synthetic archives, which proves the logic but not
    that the pinned filenames and digests match what upstream published. This
    is the test that proves that, so it is worth running after any pin bump::

        uv run pytest -m network
    """

    def test_pinned_digest_matches_the_published_release(self, tmp_path):
        import json
        import urllib.request

        api = (
            f"https://api.github.com/repos/{SIPNET_RELEASE_REPO}/releases/tags/{SIPNET_RELEASE_TAG}"
        )
        with urllib.request.urlopen(api, timeout=60) as response:  # noqa: S310
            release = json.load(response)

        published = {a["name"]: a.get("digest") for a in release.get("assets", [])}
        for filename, digest in SIPNET_RELEASE_ASSETS.values():
            assert filename in published, f"{filename} is not published on {SIPNET_RELEASE_TAG}"
            if published[filename]:
                assert published[filename] == f"sha256:{digest}", (
                    f"pinned digest for {filename} does not match the published one"
                )

    def test_download_installs_a_working_binary(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pysipnet.build._CACHE_DIR", tmp_path)
        path = download_sipnet(force=True)
        assert path.exists()
