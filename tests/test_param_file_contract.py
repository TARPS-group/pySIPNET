"""Tests that the ``.param`` file pySIPNET writes is what SIPNET expects.

The parameter file is the other half of pySIPNET's contract with the binary
(the first half being ``sipnet.in``, covered in ``test_sipnet_in.py``). Both
directions of that contract can break quietly:

- **We write a name SIPNET does not know.** SIPNET logs "Unknown param(s)
  found (and ignored)" and runs anyway, using its own default for whatever we
  meant to set. Nothing fails, and the output looks reasonable.
- **We omit something SIPNET requires.** This one is loud — SIPNET exits with
  an error naming the parameter — but only for the flag combination being
  tested, so it is easy to miss a configuration.

These tests read SIPNET's own log output to catch the first case, and run
several flag combinations to catch the second.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from pysipnet.build import binary_path
from pysipnet.io.param_io import write_param_file
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import _render_sipnet_in

requires_binary = pytest.mark.skipif(
    not binary_path().exists(),
    reason="SIPNET binary not built; run 'make sipnet'",
)

# Flag combinations to exercise. Each is a configuration a user could
# reasonably ask for, and each requires a different set of parameters.
FLAG_CASES = [
    pytest.param(ModelFlags.standard(), id="standard"),
    pytest.param(ModelFlags.forest(), id="forest"),
    pytest.param(ModelFlags(gdd=False, soil_phenol=True), id="soil-temperature-phenology"),
    pytest.param(ModelFlags(growth_resp=True), id="explicit-growth-respiration"),
    pytest.param(ModelFlags(snow=False), id="no-snow"),
    pytest.param(ModelFlags(water_hresp=False), id="moisture-insensitive-respiration"),
]


def _run_sipnet(tmp_path, params, flags, clim_source):
    """Write a complete run directory, execute SIPNET, and return the process."""
    shutil.copy(clim_source, tmp_path / "sipnet.clim")
    write_param_file(params, flags, tmp_path / "sipnet.param")
    (tmp_path / "sipnet.in").write_text(_render_sipnet_in(flags, events_enabled=False))
    return subprocess.run(
        [str(binary_path()), "-i", "sipnet.in"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def params_for(minimal_params):
    """Return a parameter set adjusted for the flags under test.

    The shared fixture is built for the default configuration. Configurations
    that need extra parameters get them filled in here, so that a failure means
    a genuine contract problem rather than an incomplete fixture.
    """

    def _build(flags: ModelFlags):
        data = minimal_params.model_dump()
        if flags.soil_phenol:
            data["phenology"]["soil_temp_leaf_on"] = 5.0
            data["phenology"]["gdd_leaf_on"] = None
        if flags.litter_pool:
            data["respiration"]["litter_breakdown_rate"] = 0.5
            data["respiration"]["frac_litter_respired"] = 0.5
        if flags.growth_resp:
            data["respiration"]["growth_resp_frac"] = 0.25
        if flags.leaf_water:
            data["water"]["leaf_pool_depth"] = 0.05
        return type(minimal_params).model_validate(data)

    return _build


@requires_binary
class TestSipnetRecognisesEveryParameter:
    @pytest.mark.parametrize("flags", FLAG_CASES)
    def test_no_unknown_parameters(self, tmp_path, params_for, sample_clim_path, flags):
        """Every name we write must be one SIPNET knows.

        An unknown name is only a log line in SIPNET, so without this test a
        renamed or misspelled parameter would silently stop having any effect.
        """
        proc = _run_sipnet(tmp_path, params_for(flags), flags, sample_clim_path)
        combined = proc.stdout + proc.stderr
        unknown = [ln for ln in combined.splitlines() if "Unknown param" in ln]
        assert not unknown, "SIPNET did not recognise some parameters:\n" + "\n".join(unknown)

    @pytest.mark.parametrize("flags", FLAG_CASES)
    def test_no_required_parameter_is_missing(self, tmp_path, params_for, sample_clim_path, flags):
        """SIPNET must not report a required parameter as absent.

        Complements ``validate_for_flags``: that checks our own idea of what is
        required, this checks SIPNET's.
        """
        proc = _run_sipnet(tmp_path, params_for(flags), flags, sample_clim_path)
        combined = proc.stdout + proc.stderr
        missing = [ln for ln in combined.splitlines() if "required parameter" in ln]
        assert not missing, "SIPNET reported missing parameters:\n" + "\n".join(missing)

    @pytest.mark.parametrize("flags", FLAG_CASES)
    def test_run_succeeds(self, tmp_path, params_for, sample_clim_path, flags):
        proc = _run_sipnet(tmp_path, params_for(flags), flags, sample_clim_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    @pytest.mark.parametrize("flags", FLAG_CASES)
    def test_output_file_is_produced(self, tmp_path, params_for, sample_clim_path, flags):
        _run_sipnet(tmp_path, params_for(flags), flags, sample_clim_path)
        out = tmp_path / "sipnet.out"
        assert out.exists()
        # Header row plus at least one timestep.
        assert len(out.read_text().splitlines()) > 1


@requires_binary
class TestObsoleteParametersAreGone:
    """The nine placeholder parameters must no longer be written.

    Before the v2.1.0 pin, SIPNET required nine parameters that it read and
    then ignored, so pySIPNET appended fixed placeholder values to every file.
    SIPNET has since removed them. Writing them now would produce nine
    "Unknown param" lines per run — harmless, but it would bury a real unknown
    parameter in noise.
    """

    RETIRED = [
        "baseSoilRespCold",
        "soilRespQ10Cold",
        "coldSoilThreshold",
        "E0",
        "T0",
        "litWaterDrainRate",
        "totNitrogen",
        "microbeNC",
        "m_ballBerry",
    ]

    def test_not_written_to_the_param_file(self, tmp_path, minimal_params):
        write_param_file(minimal_params, ModelFlags.standard(), tmp_path / "sipnet.param")
        written = (tmp_path / "sipnet.param").read_text()
        for name in self.RETIRED:
            assert name not in written, f"retired placeholder {name!r} is still being written"

    def test_sipnet_logs_no_unknown_parameters_at_all(
        self, tmp_path, minimal_params, sample_clim_path
    ):
        """The clean-run baseline: zero unknown-parameter lines, not merely few."""
        proc = _run_sipnet(tmp_path, minimal_params, ModelFlags.standard(), sample_clim_path)
        assert "Unknown param" not in proc.stdout + proc.stderr
