"""Patch SIPNET v1 source files to support compile-time flag overrides.

The original SIPNET v1 source uses plain ``#define FLAG value`` statements.
These cannot be overridden by ``-DFLAG=value`` on the compiler command line
because C preprocessor ``#define`` in a source file always wins over a
command-line definition.

This script replaces each flag definition with an ``#ifndef`` guard:

    Before:  #define SNOW 1
    After:   #ifndef SNOW
             #define SNOW 1
             #endif

After patching, ``-DSNOW=0`` on the command line takes effect as expected.
The patch is idempotent: re-running it on an already-patched source is a no-op.

Usage::

    python3 patches/apply_flags_patch.py <sipnet-submodule-dir>
"""

import sys
from pathlib import Path

_PATCH_MARKER = "// pySIPNET: #ifndef flags patch v1"

_SIPNET_C_REPLACEMENTS: list[tuple[str, str]] = [
    (
        "#define GROWTH_RESP 0",
        "#ifndef GROWTH_RESP\n#define GROWTH_RESP 0\n#endif",
    ),
    (
        "#define WATER_HRESP 1",
        "#ifndef WATER_HRESP\n#define WATER_HRESP 1\n#endif",
    ),
    (
        "#define LEAF_WATER 0",
        "#ifndef LEAF_WATER\n#define LEAF_WATER 0\n#endif",
    ),
    (
        "#define SNOW 1",
        "#ifndef SNOW\n#define SNOW 1\n#endif",
    ),
    (
        "#define GDD 1",
        "#ifndef GDD\n#define GDD 1\n#endif",
    ),
    (
        "#define SOIL_PHENOL 0 && !GDD",
        "#ifndef SOIL_PHENOL\n#define SOIL_PHENOL 0 && !GDD\n#endif",
    ),
]

_MODEL_STRUCTURES_H_REPLACEMENTS: list[tuple[str, str]] = [
    (
        "#define LITTER_POOL 0",
        "#ifndef LITTER_POOL\n#define LITTER_POOL 0\n#endif",
    ),
    (
        "#define HEADER 0",
        "#ifndef HEADER\n#define HEADER 0\n#endif",
    ),
]


def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    source = path.read_text()
    if _PATCH_MARKER in source:
        print(f"  already patched: {path}")
        return
    for old, new in replacements:
        if old not in source:
            print(f"  WARNING: expected string not found in {path}:\n    {old!r}", file=sys.stderr)
            continue
        source = source.replace(old, new, 1)
    source = _PATCH_MARKER + "\n" + source
    path.write_text(source)
    print(f"  patched: {path}")


def main(sipnet_dir: str) -> None:
    root = Path(sipnet_dir)
    sipnet_c = root / "src" / "sipnet" / "sipnet.c"
    model_h = root / "src" / "sipnet" / "modelStructures.h"
    for p in (sipnet_c, model_h):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)
    print("Applying #ifndef flag patch to SIPNET source:")
    _patch_file(sipnet_c, _SIPNET_C_REPLACEMENTS)
    _patch_file(model_h, _MODEL_STRUCTURES_H_REPLACEMENTS)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <sipnet-dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
