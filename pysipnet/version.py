"""Which version of SIPNET this release of pySIPNET drives.

pySIPNET is pinned to one exact SIPNET commit, recorded here and in the
``sipnet/`` git submodule. Pinning by commit rather than by branch means that
anyone who clones this repository compiles the same model source, so results
are reproducible and a version change is an explicit, reviewable edit.

Anything that differs between SIPNET versions — the climate file layout, which
parameters exist, the output columns — is keyed off :data:`SIPNET_TARGET_VERSION`
so that the rest of the package does not need to know the version at all.
"""

PYSIPNET_VERSION: str = "0.1.0.dev0"

# ── SIPNET source pinning ─────────────────────────────────────────────────────
# The sipnet/ submodule is checked out to this commit, which is the v2.1.0
# release tag ("Nitrogen Cycle, Methane, and Restart").
#
# Why a v2 release rather than something older: from v2.0.0 onward SIPNET
# chooses its model options at run time, so pySIPNET builds one binary and
# writes the options into sipnet.in. Before that, options were compile-time
# #define switches, which forced one binary per combination of options and a
# patch to the SIPNET source to make those switches overridable at all.
SIPNET_PINNED_COMMIT: str = "1bd16b782c9941c98abcb9615e8626f1fd78c309"
SIPNET_TARGET_VERSION: str = "v2.1.0"

# ── Climate file column counts ────────────────────────────────────────────────
# SIPNET accepts the climate file in two layouts, and detects which one it has
# been given by counting the columns on the first line.
#
# 12 columns (current):
#     year day time length tair tsoil par precip vpd vpdSoil vPress wspd
#
# 14 columns (older files, still accepted): the same 12 values wrapped in two
# columns SIPNET no longer uses — a leading site identifier and a trailing soil
# wetness value. SIPNET reads both, ignores both, and logs that it did so.
CLIM_COLS_12: int = 12
CLIM_COLS_14: int = 14

# ── Prebuilt binaries ─────────────────────────────────────────────────────────
# From v2.0.0 onward the SIPNET project publishes compiled binaries with each
# release, so pySIPNET can fetch one instead of compiling. Downloading is never
# automatic: call pysipnet.build.download_sipnet() or run `make sipnet-download`.
#
# The SHA-256 of each archive is pinned here, exactly as the pinned commit is,
# and is checked before anything is extracted. The digests come from GitHub's
# own release metadata (`gh release view v2.1.0 --repo PecanProject/sipnet
# --json assets`). Pinning them means the identity of a downloaded binary is a
# reviewable part of this file rather than whatever the network returned.
#
# When bumping the pin, refresh these together with SIPNET_PINNED_COMMIT: a
# stale digest fails the download loudly, which is the intended behaviour.

SIPNET_RELEASE_REPO: str = "PecanProject/sipnet"
SIPNET_RELEASE_TAG: str = SIPNET_TARGET_VERSION

SIPNET_RELEASE_ASSETS: dict[str, tuple[str, str]] = {
    # platform key -> (archive filename, SHA-256 of the archive)
    "darwin-arm64": (
        "sipnet-macos-arm64-v2.1.0.tar.gz",
        "3a90c6068c4d82dd1b667f99d597c665a2663ee606b748a711a880e0ac38a4f7",
    ),
    "linux-x86_64": (
        "sipnet-linux-x86_64-v2.1.0.tar.gz",
        "d9b1f457d3349c52d360347b1f22331ecc61e0efab7f7af0575f766d0a1213e1",
    ),
}
"""Archives published for the pinned release, keyed by platform.

Only the platforms upstream builds for appear here. Anywhere else — Intel
macOS, Windows, ARM Linux — has to compile from source, which always works.
"""
