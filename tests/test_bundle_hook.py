"""Platform wheels that bundle SIPNET: the hatch hook, and the facts it tags them with.

``hatch_build.py`` runs inside hatchling's build, not inside this package, so
the only honest test builds a wheel and reads what came out. The binary staged
for these tests is a stand-in file, because the hook's job is packaging:
include what is at ``pysipnet/bin/<key>/sipnet`` as ``pysipnet/bin/sipnet``
and tag the wheel for the platform named in ``$PYSIPNET_BUNDLE_SIPNET``. What
the real binaries need from a machine is checked separately, against the
published archives, under the ``network`` marker.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from pysipnet.build import describe_wheel_tag
from pysipnet.cli import staged_bundle_path
from pysipnet.version import SIPNET_RELEASE_ASSETS, SIPNET_WHEEL_PLATFORM_TAGS

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO_ROOT / "pysipnet" / "bin"
BUNDLE_ENV_VAR = "PYSIPNET_BUNDLE_SIPNET"


def _build_wheel(out: Path, env: dict[str, str]) -> Path:
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(out)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        env={**os.environ, **env},
    )
    (wheel,) = out.glob("pysipnet-*.whl")
    return wheel


@pytest.fixture
def staged_stand_ins():
    """Put a distinct stand-in binary under pysipnet/bin/<key>/ per platform, then remove them.

    Skips rather than clobbers if something is already staged there, since
    that would be a real binary someone is about to build a wheel from.
    """
    if BUNDLE_DIR.exists():
        pytest.skip(f"{BUNDLE_DIR} exists; a real bundle is staged")
    binaries = {}
    for key in SIPNET_RELEASE_ASSETS:
        binary = staged_bundle_path(key)
        assert binary.is_relative_to(BUNDLE_DIR)
        binary.parent.mkdir(parents=True)
        binary.write_bytes(f"#!/bin/sh\necho stand-in for {key}\n".encode())
        binary.chmod(0o755)
        binaries[key] = binary
    try:
        yield binaries
    finally:
        shutil.rmtree(BUNDLE_DIR, ignore_errors=True)


@pytest.mark.slow
class TestHook:
    @pytest.mark.parametrize("key", sorted(SIPNET_RELEASE_ASSETS))
    def test_bundled_wheel_carries_its_own_platforms_binary_and_tag(
        self, key, staged_stand_ins, tmp_path
    ):
        """With both platforms staged, the wheel for one must carry that one's binary."""
        wheel = _build_wheel(tmp_path, {BUNDLE_ENV_VAR: key})
        tag = SIPNET_WHEEL_PLATFORM_TAGS[key]
        assert wheel.name.endswith(f"-py3-none-{tag}.whl"), wheel.name
        with zipfile.ZipFile(wheel) as zf:
            info = zf.getinfo("pysipnet/bin/sipnet")
            assert zf.read(info) == staged_stand_ins[key].read_bytes()
            bin_members = [n for n in zf.namelist() if n.startswith("pysipnet/bin/")]
            assert bin_members == ["pysipnet/bin/sipnet"], bin_members
            mode = info.external_attr >> 16
            assert mode & stat.S_IXUSR, "the bundled binary must stay executable"
            (wheel_meta,) = (n for n in zf.namelist() if n.endswith(".dist-info/WHEEL"))
            text = zf.read(wheel_meta).decode()
        assert "Root-Is-Purelib: false" in text
        assert f"Tag: py3-none-{tag}" in text

    def test_pure_wheel_leaves_a_staged_binary_out(self, staged_stand_ins, tmp_path):
        """Without the variable the wheel is the ordinary pure one, whatever is staged."""
        wheel = _build_wheel(tmp_path, {BUNDLE_ENV_VAR: ""})
        assert wheel.name.endswith("-py3-none-any.whl")
        with zipfile.ZipFile(wheel) as zf:
            assert not [n for n in zf.namelist() if n.startswith("pysipnet/bin/")]

    def test_refuses_to_bundle_when_nothing_is_staged(self, tmp_path):
        if BUNDLE_DIR.exists():
            pytest.skip(f"{BUNDLE_DIR} exists; a real bundle is staged")
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            _build_wheel(tmp_path, {BUNDLE_ENV_VAR: "linux-x86_64"})
        assert b"stage-bundle linux-x86_64" in excinfo.value.stderr

    def test_refuses_an_unknown_platform(self, staged_stand_ins, tmp_path):
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            _build_wheel(tmp_path, {BUNDLE_ENV_VAR: "windows-x86_64"})
        assert b"not a platform" in excinfo.value.stderr


class TestWheelTagTable:
    def test_every_published_platform_has_a_wheel_tag(self):
        assert set(SIPNET_WHEEL_PLATFORM_TAGS) == set(SIPNET_RELEASE_ASSETS)

    def test_wheel_tags_are_valid_platform_tags(self):
        for key, tag in SIPNET_WHEEL_PLATFORM_TAGS.items():
            assert re.fullmatch(r"(macosx_\d+_\d+_arm64|manylinux_\d+_\d+_x86_64)", tag), key

    def test_tags_are_described_in_words(self):
        assert describe_wheel_tag("macosx_26_0_arm64") == "arm64 macOS 26.0 or newer"
        assert (
            describe_wheel_tag("manylinux_2_34_x86_64") == "x86_64 Linux with glibc 2.34 or newer"
        )

    def test_version_module_stays_loadable_without_the_package(self):
        """hatch_build.py loads version.py by path in an environment with only hatchling."""
        text = (REPO_ROOT / "pysipnet" / "version.py").read_text()
        assert not re.search(r"^(import|from) ", text, re.MULTILINE), (
            "version.py must stay import-free; the build hook loads it by path"
        )


def _minimum_from_tag(tag: str) -> str:
    """``macosx_26_0_arm64`` -> ``26.0``; ``manylinux_2_34_x86_64`` -> ``2.34``."""
    match = re.fullmatch(r"\w+?_(\d+)_(\d+)_\w+", tag)
    assert match, tag
    return f"{match[1]}.{match[2]}"


def _macho_minos(data: bytes) -> str:
    """Minimum macOS from the LC_BUILD_VERSION load command of a 64-bit Mach-O."""
    _, _, _, _, ncmds, _, _, _ = struct.unpack("<IiiIIIII", data[:32])
    offset = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack("<II", data[offset : offset + 8])
        if cmd == 0x32:
            _, minos, _ = struct.unpack("<III", data[offset + 8 : offset + 20])
            return f"{minos >> 16}.{(minos >> 8) & 0xFF}"
        offset += size
    raise AssertionError("no LC_BUILD_VERSION load command")


def _max_glibc_symbol_version(data: bytes) -> str:
    versions = {
        tuple(int(x) for x in v.split(b".")) for v in re.findall(rb"GLIBC_(\d+\.\d+)", data)
    }
    return ".".join(str(x) for x in max(versions))


@pytest.mark.network
class TestRequirementsMatchThePublishedBinaries:
    """Re-derive the requirements table from the archives upstream actually published."""

    @pytest.fixture(scope="class")
    def binaries(self, tmp_path_factory) -> dict[str, bytes]:
        from pysipnet.build import fetch_release_binary

        out = {}
        for key in SIPNET_RELEASE_ASSETS:
            target = tmp_path_factory.mktemp(key) / "sipnet"
            fetch_release_binary(key, target, verify_runs=False)
            out[key] = target.read_bytes()
        return out

    def test_macos_binary_minimum_os(self, binaries):
        assert _macho_minos(binaries["darwin-arm64"]) == _minimum_from_tag(
            SIPNET_WHEEL_PLATFORM_TAGS["darwin-arm64"]
        )

    def test_linux_binary_glibc_requirement(self, binaries):
        assert _max_glibc_symbol_version(binaries["linux-x86_64"]) == _minimum_from_tag(
            SIPNET_WHEEL_PLATFORM_TAGS["linux-x86_64"]
        )
