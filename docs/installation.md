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
git clone --recurse-submodules https://github.com/TARPS-group/pySIPNET.git
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
```

Using pip:

```bash
pip install -e "."                   # runtime only
pip install -e ".[xarray]"          # + xarray
pip install -e ".[viz]"             # + plotly
pip install -e ".[xarray,viz]"      # both optional extras
```

### Optional extras

| Extra | Package | When you need it |
|:------|:--------|:-----------------|
| `xarray` | `xarray>=2023.0` | `SIPNETResult.to_xarray()` |
| `viz` | `plotly>=5.3` | `pysipnet.viz.dashboard()` |
| `examples` | matplotlib, jupyter | Running the example notebooks |

### Ensemble module (PyEns)

`pysipnet.ensemble` requires [PyEns](https://github.com/arob5/PyEns), which is
not yet on PyPI.  Install it from source before using the ensemble module:

```bash
pip install git+https://github.com/arob5/PyEns.git
```

or, if you have a local clone:

```bash
pip install -e /path/to/pyens
```

## 3. Build the SIPNET binary

The `Makefile` at the repo root compiles SIPNET from the pinned submodule
source.  Binaries are placed in `.sipnet_cache/`.

```bash
make sipnet           # builds the one SIPNET binary
```

### Without a C compiler

If compiling is inconvenient, fetch the binary the SIPNET project publishes
with each release:

```bash
make sipnet-download
```

Available for `macos-arm64` and `linux-x86_64` only; anywhere else has to
compile, which always works. The archive's SHA-256 is pinned in
`pysipnet/version.py` and verified before anything is unpacked, and the
installed binary is then asked for its version to confirm it is the release
pySIPNET targets. Any mismatch aborts and leaves nothing installed.

Compiling from source stays the default. Downloading never happens on its own —
you have to ask for it.

!!! note "One binary, no compiler flags"
    The build passes no configuration to the compiler. Every model option is
    chosen when a run starts, not when SIPNET is compiled, so a single binary
    covers every configuration pySIPNET can ask for.

### Choosing model options

Options are set per run, through
[`ModelFlags`](api/index.md), and pySIPNET writes them into the
`sipnet.in` file it generates for each run:

```python
from pysipnet import ModelFlags, SIPNETRunner

runner = SIPNETRunner(flags=ModelFlags.forest())          # named starting point
runner = SIPNETRunner(flags=ModelFlags(litter_pool=True))  # or build your own
```

`ModelFlags.standard()` and `ModelFlags.forest()` are conveniences, not a
closed list — any valid combination of flags works without rebuilding.

## 4. Verify the installation

```bash
uv run python -c "import pysipnet; print(pysipnet.__version__)"
uv run pytest tests/ -m "not integration and not network"   # fast tests (no binary required)
uv run pytest tests/ -m integration         # full tests (requires compiled binary)
```

## Upgrading SIPNET

The SIPNET source is pinned to the v2.1.0 release (commit `1bd16b78`).
To update the pin:

1. Navigate to the submodule: `cd sipnet/`
2. Check out the new target commit: `git checkout <new-commit>`
3. Return to the repo root and stage the change: `cd .. && git add sipnet/`
4. Commit the update: `git commit -m "chore: update SIPNET pin to <short-hash>"`
5. Update `pysipnet/version.py` (`SIPNET_PINNED_COMMIT`) and this page.
6. Rebuild the binary: `make clean-sipnet sipnet`
7. Run the full test suite: `uv run pytest`

!!! warning "Regenerate documentation after any version change"
    If file formats or parameters change with the new SIPNET pin, update
    `pysipnet/parameters/model.py`, `pysipnet/io/`, and `docs/sipnet-version.md`
    accordingly.
