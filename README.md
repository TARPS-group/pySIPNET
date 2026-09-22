# pySIPNET

A Python interface to [SIPNET](https://github.com/PecanProject/sipnet) — the Simplified Photosynthesis and Evapotranspiration Model, a lightweight process-based model for coupled carbon, water, nitrogen, and greenhouse-gas dynamics at a single site.

pySIPNET is independent of the [PEcAn](https://github.com/pecanproject) ecosystem and designed from the ground up for ensemble and data-assimilation workflows.

## Features

- Typed, hierarchical parameter models with units and validation on every field
- Validated climate driver container (12- and 14-column SIPNET layouts)
- Isolated run execution — each run gets its own working directory, enabling trivial parallelism
- Clean output as a labeled DataFrame or a self-describing xarray Dataset, every variable named for what it is and carrying its units and time reference
- `SIPNETModel` — a single callable compatible with PyEns, Dask, Parsl, Ray, and any framework that treats the model as `(**inputs) → output`

## Installation

pySIPNET is two things: a Python package, and the SIPNET binary it drives.
Installing the package does not install SIPNET, so setup is two steps on any
machine. Full details, including clusters and offline machines, are in the
[installation guide](docs/installation.md).

### Using pySIPNET

```bash
pip install git+https://github.com/TARPS-group/pySIPNET.git   # 1. the package
pysipnet install-sipnet                                        # 2. the SIPNET binary
pysipnet info                                                  # check: what was found, and where
```

`pysipnet install-sipnet` downloads the binary the SIPNET project publishes
when one exists for your machine (arm64 macOS 26+, x86_64 Linux with glibc 2.34+)
and otherwise compiles the pinned SIPNET source, which needs `git`, `make` and a
C compiler. It goes into a per-user cache named by the pinned SIPNET commit, so a
later pySIPNET that pins a different SIPNET cannot pick it up by mistake.
Release wheels for those two platforms carry the binary inside them, and on
them step 2 is not needed.

Two environment variables cover the cases where you already have a binary or
cannot write to your home directory:

- `PYSIPNET_BINARY=/path/to/sipnet` — use this binary, for example a cluster
  module or a shared build. It is checked before anything else.
- `PYSIPNET_CACHE_DIR=/scratch/me` — put the cache there instead of the
  platform default, for example a filesystem every compute node can see.

Before its first run, pySIPNET asks the binary for its version and refuses one
built from a different SIPNET than it pins, whatever route it arrived by.

### Developing pySIPNET

```bash
git clone --recurse-submodules https://github.com/TARPS-group/pySIPNET.git
cd pySIPNET
uv sync                 # Python dependencies, including the dev tools
make sipnet             # compile the pinned submodule into .sipnet_cache/
uv run pytest
```

In a checkout, `.sipnet_cache/sipnet` is where pySIPNET looks first after
`PYSIPNET_BINARY`, so the binary you built is the one the tests run. Requires
Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), `gcc`/`clang` and `make`.

## Usage

```python
from pysipnet import SIPNETRunner, SIPNETModel, ModelFlags, ClimateDrivers, SIPNETParameters
from pysipnet.parameters import PhotosynthesisParams, RespirationParams

climate    = ClimateDrivers.from_file("site1.clim", n_columns=14)
other_site = ClimateDrivers.from_file("site2.clim", n_columns=14)

params = SIPNETParameters(
    photosynthesis=PhotosynthesisParams(max_photosynthesis_rate=112.0, optimum_photosynthesis_temperature=24.0),
    respiration=RespirationParams(base_wood_respiration_rate=0.02),
    # ... and the other five groups
)

runner = SIPNETRunner(flags=ModelFlags.standard())
model  = SIPNETModel(runner, base_params=params, base_climate=climate)

result = model()                    # baseline run
result = model(max_photosynthesis_rate=140.0)         # override a single parameter
result = model(climate=other_site)  # swap climate drivers

print(result.outputs.pandas[["net_ecosystem_exchange", "gross_primary_production"]].sum())
```

## Documentation

- **[User guide](https://tarps-group.github.io/pySIPNET/user-guide/running-a-model/)** — running a model, inspecting results, ensemble workflows
- **[API reference](https://tarps-group.github.io/pySIPNET/api/)** — full API docs
- **[Tutorial notebook](examples/tutorial.ipynb)** — a guided tour of the interface
- **[MCMC calibration notebook](docs/examples/mcmc_calibration.ipynb)** — parameter calibration on the bundled Niwot Ridge data

Build the docs locally:

```bash
uv run mkdocs serve
```

## SIPNET version

Pinned to the SIPNET **v2.2.0-alpha.1** pre-release (commit `41fa853e`). Model options are
chosen at run time, so there is a single binary and no compiler flags. Every route to a
binary — download, compile, bundled wheel, `PYSIPNET_BINARY` — is checked against that pin
before the first run. See [docs/sipnet-version.md](docs/sipnet-version.md).

## Requirements

Python ≥ 3.11. To compile SIPNET rather than download it: `git`, `make`, `gcc`/`clang`.

## License

MIT
