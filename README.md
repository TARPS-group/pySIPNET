# pySIPNET

A Python interface to [SIPNET](https://github.com/PecanProject/sipnet) — the Simplified Photosynthesis and Evapotranspiration Model, a lightweight process-based model for coupled carbon, water, and greenhouse-gas dynamics at a single site.

pySIPNET is independent of the [PEcAn](https://github.com/pecanproject) ecosystem and designed from the ground up for ensemble and data-assimilation workflows.

## Features

- Typed, hierarchical parameter models with units and validation on every field
- Validated climate driver container (SIPNET v1 format)
- Isolated run execution — each run gets its own working directory, enabling trivial parallelism
- Clean output as labelled DataFrames (optional xarray export)
- `SIPNETModel` — a single callable compatible with PyEns, Dask, Parsl, Ray, and any framework that treats the model as `(**inputs) → output`

## Quick start

```bash
git clone --recurse-submodules https://github.com/TARPS-group/pySIPNET.git
cd pySIPNET
uv sync
make sipnet
uv run pytest
```

## Usage

```python
from pysipnet import SIPNETRunner, SIPNETModel, ModelPreset, ClimateDrivers, SIPNETParametersV1
from pysipnet.parameters import PhotosynthesisParams, RespirationParams  # and others

climate = ClimateDrivers.from_file("data/era5_site1.clim")
params  = SIPNETParametersV1(
    photosynthesis=PhotosynthesisParams(a_max=112.0, psn_t_opt=24.0, ...),
    respiration=RespirationParams(base_veg_resp=0.02, ...),
    # ... remaining sub-models
)

runner = SIPNETRunner(preset=ModelPreset.STANDARD)
model  = SIPNETModel(runner, base_params=params, base_climate=climate)

result = model()                    # baseline run
result = model(a_max=140.0)         # override a single parameter
result = model(climate=other_site)  # swap climate drivers

print(result.outputs[["nee", "gpp"]].sum())
```

## Documentation

- **[User guide](https://tarps-group.github.io/pySIPNET/user-guide/running-a-model/)** — running a model, inspecting results, ensemble workflows
- **[API reference](https://tarps-group.github.io/pySIPNET/api/)** — full API docs

Build the docs locally:

```bash
uv run mkdocs serve
```

## SIPNET version

Pinned to SIPNET commit `e4abf14f` — the last pre-v2 state (compile-time flags, 14-column climate file).  See [docs/sipnet-version.md](docs/sipnet-version.md).

## Requirements

Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), `gcc`/`clang`, `make`

## License

MIT
