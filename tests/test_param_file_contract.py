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

The reading direction has the same shape as the ``.clim`` layout contract
(``test_clim_layout_contract.py``): :func:`~pysipnet.io.param_io.read_parameters`
must accept what SIPNET accepts, read it as SIPNET reads it, and refuse what
SIPNET refuses. The one deliberate difference is a value SIPNET misreads
without a word, which pySIPNET refuses; those tests pin the misreading, so a
fix upstream shows up here.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pysipnet.build import binary_path
from pysipnet.io.param_io import UnknownParameterWarning, read_parameters, write_param_file
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import _render_sipnet_in
from tests.helpers import BareRun, run_sipnet_directly

requires_binary = pytest.mark.skipif(
    not binary_path().exists(),
    reason="SIPNET binary not built; run 'make sipnet'",
)

# Flag combinations to exercise. Each is a configuration a user could
# reasonably ask for, and each requires a different set of parameters.
FLAG_CASES = [
    pytest.param(ModelFlags.standard(), id="standard"),
    pytest.param(ModelFlags(litter_pool=True), id="litter_pool"),
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
            data["phenology"]["leaf_on_soil_temperature"] = 5.0
            data["phenology"]["leaf_on_growing_degree_days"] = None
        if flags.litter_pool:
            data["respiration"]["litter_breakdown_rate"] = 0.5
            data["respiration"]["litter_respired_fraction"] = 0.5
        if flags.growth_resp:
            data["respiration"]["growth_respiration_fraction"] = 0.25
        if flags.leaf_water:
            data["water"]["leaf_water_pool_depth"] = 0.05
        # Left out when its flag is off, so the run shows SIPNET does not need it.
        if not flags.water_hresp:
            data["respiration"]["soil_respiration_moisture_exponent"] = None
        return type(minimal_params).model_validate(data)

    return _build


@requires_binary
class TestSipnetRecognisesEveryParameter:
    @pytest.mark.parametrize("flags", FLAG_CASES)
    def test_no_unknown_parameters(self, tmp_path, params_for, reference_clim_path, flags):
        """Every name we write must be one SIPNET knows.

        An unknown name is only a log line in SIPNET, so without this test a
        renamed or misspelled parameter would silently stop having any effect.
        """
        proc = _run_sipnet(tmp_path, params_for(flags), flags, reference_clim_path)
        combined = proc.stdout + proc.stderr
        unknown = [ln for ln in combined.splitlines() if "Unknown param" in ln]
        assert not unknown, "SIPNET did not recognize some parameters:\n" + "\n".join(unknown)

    @pytest.mark.parametrize("flags", FLAG_CASES)
    def test_no_required_parameter_is_missing(
        self, tmp_path, params_for, reference_clim_path, flags
    ):
        """SIPNET must not report a required parameter as absent.

        Complements ``validate_for_flags``: that checks our own idea of what is
        required, this checks SIPNET's.
        """
        proc = _run_sipnet(tmp_path, params_for(flags), flags, reference_clim_path)
        combined = proc.stdout + proc.stderr
        missing = [ln for ln in combined.splitlines() if "required parameter" in ln]
        assert not missing, "SIPNET reported missing parameters:\n" + "\n".join(missing)

    @pytest.mark.parametrize("flags", FLAG_CASES)
    def test_run_succeeds(self, tmp_path, params_for, reference_clim_path, flags):
        proc = _run_sipnet(tmp_path, params_for(flags), flags, reference_clim_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    @pytest.mark.parametrize("flags", FLAG_CASES)
    def test_output_file_is_produced(self, tmp_path, params_for, reference_clim_path, flags):
        _run_sipnet(tmp_path, params_for(flags), flags, reference_clim_path)
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
        self, tmp_path, minimal_params, reference_clim_path
    ):
        """The clean-run baseline: zero unknown-parameter lines, not merely few."""
        proc = _run_sipnet(tmp_path, minimal_params, ModelFlags.standard(), reference_clim_path)
        assert "Unknown param" not in proc.stdout + proc.stderr


@requires_binary
class TestReaderAgreesWithSipnet:
    """Each file is given to the bare binary and to the reader, and the verdicts compared."""

    @pytest.fixture
    def lines(self, tmp_path, minimal_params) -> list[str]:
        path = tmp_path / "clean.param"
        write_param_file(minimal_params, ModelFlags.standard(), path)
        return [line for line in path.read_text().splitlines() if not line.startswith("!")]

    @pytest.fixture
    def clim(self, tmp_path, reference_clim_path) -> Path:
        path = tmp_path / "short.clim"
        path.write_text("".join(reference_clim_path.read_text().splitlines(True)[:60]))
        return path

    def _both(self, tmp_path, clim, lines, name) -> tuple[BareRun, Path]:
        path = tmp_path / f"{name}.param"
        path.write_text("".join(f"{line}\n" for line in lines))
        return run_sipnet_directly(binary_path(), path, clim), path

    @staticmethod
    def _replace(lines, name, replacement):
        return [replacement if line.split()[0] == name else line for line in lines]

    @staticmethod
    def _assert_same_output(a: BareRun, b: BareRun) -> None:
        assert a.output is not None and b.output is not None
        pd.testing.assert_frame_equal(a.output, b.output, check_exact=True)

    def test_a_clean_file_is_accepted_by_both(self, tmp_path, clim, lines, minimal_params):
        run, path = self._both(tmp_path, clim, lines, "clean")
        assert run.returncode == 0, run.log
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert read_parameters(path) == minimal_params

    def test_names_in_another_case_are_the_same_parameters(
        self, tmp_path, clim, lines, minimal_params
    ):
        clean, _ = self._both(tmp_path, clim, lines, "clean")
        upper, path = self._both(tmp_path, clim, [line.upper() for line in lines], "upper")
        assert upper.returncode == 0, upper.log
        assert "Unknown param" not in upper.log
        self._assert_same_output(clean, upper)
        assert read_parameters(path) == minimal_params

    def test_an_unknown_name_is_ignored_by_both(self, tmp_path, clim, lines, minimal_params):
        clean, _ = self._both(tmp_path, clim, lines, "clean")
        extra, path = self._both(tmp_path, clim, [*lines, "microbeInit 0.5"], "extra")
        assert extra.returncode == 0, extra.log
        assert "microbeInit" in extra.log
        self._assert_same_output(clean, extra)
        with pytest.warns(UnknownParameterWarning, match="microbeInit"):
            assert read_parameters(path) == minimal_params

    def test_an_unknown_name_may_repeat_in_both(self, tmp_path, clim, lines):
        run, path = self._both(tmp_path, clim, [*lines, "E0 1", "E0 2"], "repeat")
        assert run.returncode == 0, run.log
        with pytest.warns(UnknownParameterWarning, match="E0"):
            read_parameters(path)

    @pytest.mark.parametrize(
        ("edit", "sipnet_says", "reader_says"),
        [
            pytest.param(
                lambda ls: [*ls, "AMAX 1"],
                "already been set",
                "given twice",
                id="duplicate",
            ),
            pytest.param(
                lambda ls: TestReaderAgreesWithSipnet._replace(ls, "aMax", "aMax *"),
                "no longer supported",
                "spatially-varying marker",
                id="star",
            ),
            pytest.param(
                lambda ls: [ln for ln in ls if ln.split()[0] != "litterInit"],
                "Did not find required parameter litterInit",
                "litterInit",
                id="missing-despite-default",
            ),
            pytest.param(
                lambda ls: [ln for ln in ls if ln.split()[0] != "aMax"],
                "Did not find required parameter aMax",
                "aMax",
                id="missing",
            ),
        ],
    )
    def test_refused_by_both(self, tmp_path, clim, lines, edit, sipnet_says, reader_says):
        run, path = self._both(tmp_path, clim, edit(lines), "refused")
        assert run.returncode != 0
        assert sipnet_says in run.log
        with pytest.raises(ValueError, match=reader_says):
            read_parameters(path)

    def test_a_name_without_a_value_crashes_sipnet(self, tmp_path, clim, lines):
        run, path = self._both(tmp_path, clim, [*lines, "microbeInit"], "no_value")
        assert run.returncode != 0
        with pytest.raises(ValueError, match="has no value"):
            read_parameters(path)

    @pytest.mark.parametrize(
        ("token", "read_as"),
        [("abc", "0"), ("112x", "112"), ("1_12", "1")],
    )
    def test_sipnet_misreads_a_non_number_silently_and_the_reader_refuses(
        self, tmp_path, clim, lines, token, read_as
    ):
        misread, path = self._both(
            tmp_path, clim, self._replace(lines, "aMax", f"aMax {token}"), "misread"
        )
        assert misread.returncode == 0, misread.log
        as_read, _ = self._both(
            tmp_path, clim, self._replace(lines, "aMax", f"aMax {read_as}"), "as_read"
        )
        self._assert_same_output(misread, as_read)
        with pytest.raises(ValueError, match="not a decimal number"):
            read_parameters(path)

    def test_sipnet_splits_a_long_line_and_the_reader_refuses(self, tmp_path, clim, lines):
        comment = "! " + "word " * 60
        run, path = self._both(tmp_path, clim, [*lines, comment], "long_line")
        assert "Unknown param" in run.log, "SIPNET read the comment's tail as a parameter"
        with pytest.raises(ValueError, match="255-character"):
            read_parameters(path)
