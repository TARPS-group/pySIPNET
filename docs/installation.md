# Installation

pySIPNET is two things: a Python package, and the SIPNET binary it drives.
Installing the package does not install SIPNET. Getting a binary is a separate,
explicit step, and pySIPNET never does it on its own: a model run that reached
for the network would be a surprise inside an ensemble and a failure on a
cluster's compute nodes.

There are two ways to set up, depending on whether you are using pySIPNET or
working on it.

## Using pySIPNET

### 1. Install the package

```bash
pip install git+https://github.com/TARPS-group/pySIPNET.git
```

or, in a project managed by [uv](https://docs.astral.sh/uv/), add it as a
dependency with a git source. Requires Python ≥ 3.11.

Release wheels for arm64 macOS 26+ and x86_64 Linux with glibc 2.34+ carry the
SIPNET binary inside them. If you installed one of those, `pysipnet info` will
already show a binary and step 2 is not needed.

### 2. Install the SIPNET binary

```bash
pysipnet install-sipnet
```

This picks a route for your machine:

- **Download**, when the SIPNET project publishes a binary for your platform
  and it can run here. Upstream builds for arm64 macOS and x86_64 Linux, on
  recent systems: the macOS binary needs macOS 26 or newer, the Linux binary
  glibc 2.34 or newer (Ubuntu 22.04, RHEL 9, or later). The archive's SHA-256
  is checked against the digest pinned in `pysipnet/version.py` before anything
  is unpacked.
- **Compile**, everywhere else. The pinned SIPNET commit is fetched with `git`
  and built with SIPNET's own Makefile, so this needs `git`, `make` and a C
  compiler (`gcc` or `clang`). SIPNET is small and compiles in seconds.

`--method download` or `--method compile` insists on one route; `--force`
replaces a binary that is already installed. The binary goes into a per-user
cache directory named by the pinned SIPNET commit, so a later pySIPNET that
pins a different SIPNET looks in a new, empty directory rather than finding
this one.

Whichever route it took, the installed binary is asked for its version and
refused if it was not built from the pinned SIPNET tag.

### 3. Check

```bash
pysipnet info
```

prints the SIPNET version pySIPNET pins, every place it looks for a binary and
whether one was there, and the binary it will use. The exit status is non-zero
when no usable binary was found, so a setup script can test it.

### Clusters, shared installs and offline machines

pySIPNET looks for a binary in this order and uses the first that exists:

| Order | Where | When to use it |
|:--|:--|:--|
| 1 | `$PYSIPNET_BINARY` | You already have a SIPNET binary: a cluster module, a shared build, one you compiled yourself. |
| 2 | `pysipnet/bin/sipnet` inside the installed package | Present only in a platform wheel that bundles the binary. |
| 3 | `.sipnet_cache/sipnet` at the repository root | A source checkout, where `make sipnet` puts it. |
| 4 | The per-user cache, named by the pinned commit | Where `pysipnet install-sipnet` puts a binary outside a checkout. |

`PYSIPNET_CACHE_DIR` replaces the cache root in row 4, for a cluster where
home directories are small or where a scratch filesystem is what every compute
node can see. Set it before `pysipnet install-sipnet` on the login node, and
the compute nodes find the binary without touching the network.

A binary from `$PYSIPNET_BINARY` is checked against the pin like any other.
`SIPNETRunner(binary=...)` names one from Python; `SIPNETRunner(verify_binary=False)`
switches the version check off, for the rare case where running a different
SIPNET is what you mean.

## Developing pySIPNET

### 1. Clone with the submodule

pySIPNET tracks the pinned SIPNET source as a git **submodule**, which several
tests read directly:

```bash
git clone --recurse-submodules https://github.com/TARPS-group/pySIPNET.git
cd pySIPNET
```

If you already cloned without the flag:

```bash
git submodule update --init sipnet/
```

### 2. Install Python dependencies

```bash
uv sync                              # runtime + dev dependencies
uv sync --extra viz                  # + plotly for the result dashboard
```

Using pip: `pip install -e "."` or `pip install -e ".[viz]"`.

| Extra | Package | When you need it |
|:------|:--------|:-----------------|
| `viz` | `plotly>=5.3` | `pysipnet.viz.dashboard()` |
| `examples` | matplotlib, jupyter | Running the example notebooks |

`pysipnet.ensemble` requires [PyEns](https://github.com/arob5/PyEns), which is
not yet on PyPI: `pip install git+https://github.com/arob5/PyEns.git`.

### 3. Build the SIPNET binary

```bash
make sipnet           # compiles the submodule into .sipnet_cache/sipnet
```

In a checkout this is where pySIPNET looks after `$PYSIPNET_BINARY`, so the
binary you built is the one the tests run. `make sipnet-download` fetches the
published binary into the same place instead, when compiling is inconvenient.

!!! note "One binary, no compiler flags"
    The build passes no configuration to the compiler. Every model option is
    chosen when a run starts, through [`ModelFlags`](api/index.md), and
    pySIPNET writes it into the `sipnet.in` file it generates for each run, so
    a single binary covers every configuration.

### 4. Verify

```bash
uv run pysipnet info
uv run pytest tests/ -m "not integration and not network"   # fast tests, no binary required
uv run pytest                                                # everything but network tests
```

## Release wheels

The `Wheels` workflow builds an sdist, the pure-Python wheel, and one wheel per
platform upstream publishes a binary for, with the verified binary bundled and
the wheel tagged for the system that binary needs. It runs on demand and on a
version tag, and on a tag attaches the artifacts to a draft GitHub release.
Locally:

```bash
uv run pysipnet stage-bundle linux-x86_64            # verified download to pysipnet/bin/
PYSIPNET_BUNDLE_SIPNET=linux-x86_64 uv build --wheel  # tagged manylinux_2_34_x86_64
rm -rf pysipnet/bin
```

## Upgrading SIPNET

The SIPNET source is pinned to the v2.2.0-alpha.1 pre-release (commit `41fa853e`).
To update the pin:

1. Navigate to the submodule: `cd sipnet/`
2. Check out the new target commit: `git checkout <new-commit>`
3. Return to the repo root and stage the change: `cd .. && git add sipnet/`
4. Commit the update: `git commit -m "chore: update SIPNET pin to <short-hash>"`
5. Update `pysipnet/version.py` — `SIPNET_PINNED_COMMIT`, `SIPNET_PINNED_TAG`,
   `SIPNET_NUMERIC_VERSION`, `SIPNET_RELEASE_ASSETS` and
   `SIPNET_PREBUILT_REQUIREMENTS` all move together — and this page.
   `pytest -m network` re-derives the digests and the platform requirements
   from the published archives.
6. Rebuild the binary: `make clean-sipnet sipnet`. The user-cache route needs
   nothing: its directory is named by the commit.
7. Run the full test suite: `uv run pytest`

!!! warning "Regenerate documentation after any version change"
    If file formats or parameters change with the new SIPNET pin, update
    `pysipnet/parameters/model.py`, `pysipnet/io/`, and `docs/sipnet-version.md`
    accordingly.
