# Cutting a release

Maintainer notes. Users never need this page; the user-facing setup is in the
README and `docs/installation.md`.

A release is a git tag. Pushing it runs the `Wheels` workflow, which builds the
distributable files, installs each on a machine that can run its binary, runs
SIPNET on the bundled Niwot inputs, and attaches everything to a **draft**
GitHub release. Nothing becomes public until a person publishes the draft.

## Before tagging

1. **Decide the version.** pySIPNET uses PEP 440 versions: `0.1.0`, `0.2.0a1`,
   `0.1.0.post1`. The SIPNET pin is independent of this number; a release
   that bumps the pin should say so in its notes, since parameters calibrated
   against one pin do not transfer to another (see `docs/sipnet-version.md`).

2. **Set it in both places**, which must agree:

   - `pyproject.toml`, `version = "..."`
   - `pysipnet/version.py`, `PYSIPNET_VERSION = "..."`

   `tests/test_version.py` fails if they differ.

3. **Merge that to `main`** through a pull request like any other change, and
   make sure CI is green on the merge commit. The tag must point at a commit on
   `main`.

4. **Run the suite with a binary present**, so `test_fidelity` and
   `test_golden` execute rather than skip:

   ```bash
   make sipnet
   uv run pytest
   uv run pytest -m network      # digests and wheel tags against the published SIPNET release
   ```

## Tag and push

```bash
git checkout main && git pull
git tag -a v0.1.0 -m "pySIPNET 0.1.0"
git push origin v0.1.0
```

The tag name is `v` plus the version. The workflow triggers on any `v*` tag.

## What the workflow does

Watch it with `gh run watch`, or at
https://github.com/TARPS-group/pySIPNET/actions/workflows/wheels.yml.

| job | what it proves |
|:--|:--|
| Build sdist and wheels | The sdist, the pure wheel, and one wheel per platform SIPNET publishes a binary for, with that binary inside and the wheel tagged for the system it needs. The binaries are upstream's release assets, verified against the digests in `pysipnet/version.py`. |
| Install and run (ubuntu-24.04, manylinux_2_34_x86_64) | The Linux wheel installs, the bundled binary is found and is the pinned SIPNET, and it runs the Niwot record. |
| Install and run (macos-26, macosx_26_0_arm64) | The same for the macOS wheel. |
| Install and run (ubuntu-24.04, any) | The pure wheel installs, `pysipnet install-sipnet` fetches a binary into the user cache, and it runs. |
| Attach to a draft release | All four files attached to a draft release named after the tag. Runs only on a tag, only after the smoke jobs pass. |

If a smoke job fails, nothing is attached. Fix on `main`, delete the tag
(`git push --delete origin v0.1.0 && git tag -d v0.1.0`), and tag again.

## Publish

1. Open https://github.com/TARPS-group/pySIPNET/releases. The draft is there
   with the four files.
2. Write the notes. Say which SIPNET the release pins (`SIPNET_PINNED_TAG`) and
   whether that changed; list user-visible changes.
3. Publish.

A user can then install a wheel straight from the release page:

```bash
pip install https://github.com/TARPS-group/pySIPNET/releases/download/v0.1.0/pysipnet-0.1.0-py3-none-any.whl
```

## Not yet done: PyPI

The workflow does not upload to PyPI. When the project is ready, add a job to
`wheels.yml` after `release` that uses `pypa/gh-action-pypi-publish` with
trusted publishing (no API token in the repository), and register the workflow
as a trusted publisher on PyPI first. Until then, `pip install pysipnet` does
not exist; the README installs from git.

## Local dry run

To see exactly what the workflow will build, without tagging:

```bash
uv build -o dist                                    # sdist + pure wheel
uv run pysipnet stage-bundle darwin-arm64
PYSIPNET_BUNDLE_SIPNET=darwin-arm64 uv build --wheel -o dist
uv run pysipnet stage-bundle linux-x86_64
PYSIPNET_BUNDLE_SIPNET=linux-x86_64 uv build --wheel -o dist
```

Then `unzip -l dist/<wheel>` to confirm `pysipnet/bin/sipnet` is in each
platform wheel and absent from the pure one. `pysipnet/bin/` is ignored by git.
The workflow itself can also be run by hand from the Actions page or with
`gh workflow run wheels.yml`; without a tag it builds and smoke-tests but
attaches nothing.
