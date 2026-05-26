# Installation

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- C compiler: `gcc` or `clang`
- `make`

## 1. Clone the repository

pySIPNET uses a git **submodule** to track the pinned SIPNET source.  You must
clone with `--recurse-submodules` to get it:

```bash
git clone --recurse-submodules https://github.com/andrewroberts/pySIPNET.git
cd pySIPNET
```

If you already cloned without the flag:

```bash
git submodule update --init sipnet/
```

## 2. Install Python dependencies

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync                              # installs runtime + dev dependencies
uv sync --extra xarray               # + xarray for Dataset output
uv sync --extra viz                  # + plotly for the result dashboard
uv sync --extra ensemble             # + pyens for ensemble runs
```

Using pip:

```bash
pip install -e "."                   # runtime only
pip install -e ".[xarray]"          # + xarray
pip install -e ".[viz]"             # + plotly
pip install -e ".[ensemble]"        # + pyens
pip install -e ".[xarray,viz,ensemble]"  # all optional extras
```

### Optional extras

| Extra | Package | When you need it |
|:------|:--------|:-----------------|
| `xarray` | `xarray>=2023.0` | `SIPNETResult.to_xarray()` |
| `viz` | `plotly>=5.3` | `pysipnet.viz.dashboard()` |
| `ensemble` | `pyens>=0.1` | `pysipnet.ensemble.SIPNETModel` |
| `examples` | matplotlib, jupyter | Running the example notebooks |

## 3. Build the SIPNET binary

The `Makefile` at the repo root compiles SIPNET from the pinned submodule
source.  Binaries are placed in `.sipnet_cache/`.

```bash
make sipnet           # builds all presets (standard + forest)
make sipnet-standard  # build only the standard preset
make sipnet-forest    # build only the forest preset
```

!!! note "What the build does"
    Before compiling, the build applies a small source patch
    (`patches/apply_flags_patch.py`) that wraps SIPNET's compile-time flag
    `#define` statements with `#ifndef` guards.  This enables the preset
    system to override flags via `-D` compiler arguments without modifying
    the SIPNET source permanently.  The patch is idempotent and does not
    change model behaviour.

### Available presets

| Preset       | Binary name          | Active flags                         |
|:-------------|:---------------------|:-------------------------------------|
| `standard`   | `sipnet_standard`    | SNOW=1, GDD=1, WATER_HRESP=1         |
| `forest`     | `sipnet_forest`      | standard + LITTER_POOL=1             |

### Custom presets

To add a new flag combination, extend the `Makefile` with a new target and
register it in `pysipnet/runner.py` (`ModelPreset` enum).  See the existing
`sipnet-forest` target as a template.

## 4. Verify the installation

```bash
uv run python -c "import pysipnet; print(pysipnet.__version__)"
uv run pytest tests/ -m "not integration"   # fast tests (no binary required)
uv run pytest tests/ -m integration         # full tests (requires compiled binary)
```

## Upgrading SIPNET

The SIPNET source is pinned to commit `e4abf14f` (the last pre-v2 commit).
To update the pin:

1. Navigate to the submodule: `cd sipnet/`
2. Check out the new target commit: `git checkout <new-commit>`
3. Return to the repo root and stage the change: `cd .. && git add sipnet/`
4. Commit the update: `git commit -m "chore: update SIPNET pin to <short-hash>"`
5. Update `pysipnet/version.py` (`SIPNET_PINNED_COMMIT`) and this page.
6. Rebuild all binaries: `make clean-sipnet sipnet`
7. Run the full test suite: `uv run pytest`

!!! warning "Regenerate documentation after any version change"
    If file formats or parameters change with the new SIPNET pin, update
    `pysipnet/parameters/v1.py`, `pysipnet/io/`, and `docs/sipnet-version.md`
    accordingly.
