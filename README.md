# pySIPNET

A clean, well-documented Python interface to [SIPNET](https://github.com/PecanProject/sipnet) — the Simplified Photosynthesis and Evapotranspiration Model.

**Status:** Early development (v0.1.0.dev0).  The parameter and climate data models are implemented; the IO layer and runner are stubs.

## Features

- Typed, hierarchical parameter models with units and calibration metadata on every field
- Validated climate driver container (v1 and v2 format support)
- Isolated run execution — each run gets its own working directory, enabling trivial parallelism
- Clean output as labelled DataFrames
- Independent of PEcAn

## Requirements

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), `gcc`/`clang`, `make`

## Quick setup

```bash
git clone --recurse-submodules https://github.com/andrewroberts/pySIPNET.git
cd pySIPNET
uv sync
make sipnet
uv run pytest
```

## Documentation

See [`docs/`](docs/) or build locally:

```bash
uv run mkdocs serve
```

## SIPNET version

Pinned to SIPNET commit `e4abf14f` — the last pre-v2 state (compile-time flags, 14-column climate file).  See [docs/sipnet-version.md](docs/sipnet-version.md).

## License

MIT
