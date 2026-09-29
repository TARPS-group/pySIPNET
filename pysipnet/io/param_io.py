"""Read and write SIPNET ``.param`` files.

SIPNET parameter file format
----------------------------------
Space/tab-delimited, two used columns::

    parameterName  value   [changeable  min  max  sigma ...]

- Comment character: ``!``  (everything after ``!`` on a line is ignored)
- Column order is irrelevant (SIPNET reads by name, not position)
- Columns beyond the second are silently accepted (legacy 5+ column format)

Reading
-------
:func:`read_param_file` parses a file the way SIPNET's ``readModelParams``
(``src/common/modelParams.c``) does, and refuses what SIPNET refuses or
misreads: see its docstring. :func:`read_parameters` builds a
:class:`~pysipnet.parameters.model.SIPNETParameters` from the result, matching
names case-insensitively as SIPNET does, and reports the names it cannot carry
with an :class:`UnknownParameterWarning`.

Python-to-SIPNET name mapping
------------------------------
Python field names follow pySIPNET's naming convention (lower-case words,
no acronyms); SIPNET uses ``camelCase`` abbreviations. Each field's
:class:`~pysipnet.parameters.base.ParameterSpec` records its ``sipnet_name``,
and :data:`PYTHON_TO_SIPNET` is derived from those specs, so the parameter
model is the single statement of the mapping. ``tests/test_param_name_mapping.py``
states it a second time by hand so that a change is deliberate.

Unit contract
~~~~~~~~~~~~~
Values are written as-is.  For parameters with ``per_year=True`` in their
:class:`~pysipnet.parameters.base.ParameterSpec`, the value written is the
per-year rate — matching what SIPNET expects in the param file (SIPNET divides
by 365 internally).
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from pysipnet.parameters.base import get_parameter_specs
from pysipnet.parameters.model import UNSUPPORTED_FLAGS, ModelFlags, SIPNETParameters

# Maps dot-separated Python path → SIPNET param file name, from the specs.
PYTHON_TO_SIPNET: dict[str, str] = {
    path: spec.sipnet_name for path, spec in get_parameter_specs(SIPNETParameters).items()
}

# The inverse, for reading a .param file back.
SIPNET_TO_PYTHON: dict[str, str] = {v: k for k, v in PYTHON_TO_SIPNET.items()}

# SIPNET matches names with strcasecmp, so lookups go through the lower case.
_PYTHON_PATH_BY_KEY: dict[str, str] = {k.lower(): v for k, v in SIPNET_TO_PYTHON.items()}

# Registered by the pinned SIPNET but not modeled, keyed by lower-cased name.
# tests/test_parameters.py asserts these and the modeled names are everything
# SIPNET registers, so the two together say which names SIPNET would read.
_UNMODELED_FLAGS_BY_KEY: dict[str, list[str]] = {}
for _flag, (_, _names) in UNSUPPORTED_FLAGS.items():
    for _name in _names:
        _UNMODELED_FLAGS_BY_KEY.setdefault(_name.lower(), []).append(_flag)

# Fields with a default that SIPNET nevertheless requires in the file. The
# reader must not fill these in, or it would accept a file SIPNET refuses.
# growth_respiration_fraction's default is not here: SIPNET requires it only
# under growth_resp, and otherwise runs on 0, which is the default.
# tests/test_io.py checks this against SIPNET's source.
SIPNET_REQUIRES_DESPITE_DEFAULT: tuple[str, ...] = (
    "initial_conditions.litter_carbon",
    "initial_conditions.snow_water_equivalent",
)

# (group, field) for every parameter a .param file must name, in declaration order.
_REQUIRED_FIELDS: list[tuple[str, str]] = [
    (group, field)
    for group, group_info in SIPNETParameters.model_fields.items()
    for field, field_info in cast("type[BaseModel]", group_info.annotation).model_fields.items()
    if field_info.is_required() or f"{group}.{field}" in SIPNET_REQUIRES_DESPITE_DEFAULT
]

_REGISTERED_KEYS = frozenset(_PYTHON_PATH_BY_KEY) | frozenset(_UNMODELED_FLAGS_BY_KEY)

# Sizes of the buffers readModelParams copies into: fgets(line, 256) and
# strcpy into pName[MODEL_PARAM_MAXNAME = 64] and strValue[32].
_MAX_LINE_CHARS = 255
_MAX_NAME_CHARS = 63
_MAX_VALUE_CHARS = 31

# readModelParams tokenizes on exactly these, where str.split() would also
# split on form feeds, vertical tabs and Unicode spaces.
_SEPARATORS = re.compile(r"[ \t\r]+")

# What strtod reads in full as a decimal number. Anything else it reads only a
# prefix of, or nothing, and SIPNET uses the result without checking.
_DECIMAL = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


class UnknownParameterWarning(UserWarning):
    """A ``.param`` file names a parameter :class:`SIPNETParameters` cannot hold."""


def _flatten(params: SIPNETParameters) -> dict[str, float]:
    """Return a flat ``{sipnet_name: value}`` dict, omitting ``None`` values."""
    result: dict[str, float] = {}
    dump = params.model_dump()
    for python_path, sipnet_name in PYTHON_TO_SIPNET.items():
        group, field = python_path.split(".", 1)
        value = dump[group][field]
        if value is not None:
            result[sipnet_name] = value
    return result


def write_param_file(
    parameters: SIPNETParameters,
    flags: ModelFlags,
    path: Path,
) -> None:
    """Write the ``.param`` file SIPNET reads its parameters from.

    Parameters left as ``None`` are omitted. Those are the ones only some
    model configurations need, so writing them unconditionally would mean
    inventing values SIPNET would then ignore.

    The flags are checked against the parameters first, so a configuration
    asking for a process whose parameters are missing fails here rather than
    inside SIPNET.

    Parameters
    ----------
    parameters:
        The full parameter set.
    flags:
        Model options for this run, used to decide which parameters are
        required.
    path:
        Where to write the file, normally ``<workdir>/sipnet.param``.
    """
    parameters.validate_for_flags(flags)
    flat = _flatten(parameters)
    lines = ["! SIPNET parameter file - generated by pySIPNET\n"]
    for sipnet_name, value in flat.items():
        # Format explicitly rather than with repr(). A numpy scalar reprs as
        # "np.float64(8.3)", which SIPNET's strtod reads as 0 without
        # complaining, and repr() also emits "nan"/"inf", which it accepts.
        # .17g round-trips a double exactly.
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"Refusing to write {sipnet_name} = {value!r}: SIPNET would parse it "
                "and run to completion, producing output that looks plausible."
            )
        lines.append(f"{sipnet_name}\t{numeric:.17g}\n")
    path.write_text("".join(lines))


def read_param_file(path: Path) -> dict[str, float]:
    """Read a SIPNET ``.param`` file and return a ``{name: value}`` dict.

    Names are kept as the file spells them. Comments (``!``), blank lines and
    columns beyond the second are ignored, as SIPNET ignores them.

    Raises ``ValueError``, naming the file and line, for everything SIPNET
    refuses or reads wrongly without saying so:

    - a value of ``*``, the retired spatially-varying marker (SIPNET exits);
    - a parameter SIPNET registers given twice, compared case-insensitively
      (SIPNET exits);
    - a name with no value (SIPNET dereferences a null pointer);
    - a value that is not a decimal number: SIPNET parses it with ``strtod``
      and never checks where parsing stopped, so ``abc`` runs as 0 and
      ``8.3x`` as 8.3;
    - ``nan`` or ``inf``, which ``strtod`` accepts and :func:`write_param_file`
      refuses to write;
    - a line longer than SIPNET's 255-character line buffer, which it would
      read as two lines, or a name or value longer than the buffer SIPNET
      copies it into.

    A name SIPNET does not register may repeat; SIPNET ignores it, and the
    dict keeps the last value.
    """
    result: dict[str, float] = {}
    registered_seen: dict[str, int] = {}
    with path.open(newline="") as handle:
        text = handle.read()
    for lineno, raw in enumerate(text.split("\n"), start=1):
        where = f"{path}:{lineno}"
        if len(raw) > _MAX_LINE_CHARS:
            raise ValueError(
                f"{where}: line is {len(raw)} characters long. SIPNET reads lines into a "
                f"{_MAX_LINE_CHARS}-character buffer and would parse the rest as a separate "
                "line, comment or not."
            )
        parts = [part for part in _SEPARATORS.split(raw.split("!", 1)[0]) if part]
        if not parts:
            continue
        name = parts[0]
        if len(parts) < 2:
            raise ValueError(f"{where}: {name!r} has no value; SIPNET crashes on such a line.")
        token = parts[1]
        if len(name) > _MAX_NAME_CHARS or len(token) > _MAX_VALUE_CHARS:
            raise ValueError(
                f"{where}: SIPNET copies a name into {_MAX_NAME_CHARS + 1} bytes and a value "
                f"into {_MAX_VALUE_CHARS + 1}; {name!r} = {token!r} overflows them."
            )
        if token == "*":
            raise ValueError(
                f"{where}: {name} = '*', the spatially-varying marker, which SIPNET no longer "
                "accepts."
            )
        if not _DECIMAL.fullmatch(token):
            if token.lower().lstrip("+-") in {"nan", "inf", "infinity"}:
                raise ValueError(
                    f"{where}: {name} = {token}. SIPNET would accept it and run to completion; "
                    "a parameter must be finite."
                )
            raise ValueError(
                f"{where}: {name} = {token!r} is not a decimal number. SIPNET parses values "
                "with strtod and does not check where it stopped, so it would run on "
                "whatever prefix it read, or on 0, without saying so."
            )
        key = name.lower()
        if key in _REGISTERED_KEYS:
            if key in registered_seen:
                raise ValueError(
                    f"{where}: {name} was already set on line {registered_seen[key]} "
                    "(names are case-insensitive); SIPNET refuses a parameter given twice."
                )
            registered_seen[key] = lineno
        result[name] = float(token)
    return result


def read_parameters(path: Path) -> SIPNETParameters:
    """Read a SIPNET ``.param`` file into a :class:`SIPNETParameters`.

    The inverse of :func:`write_param_file`: writing a parameter set and
    reading it back gives an equal one. The file is parsed by
    :func:`read_param_file`, with its refusals, and names are matched
    case-insensitively, as SIPNET matches them.

    Names the model has no field for are dropped, with one
    :class:`UnknownParameterWarning` listing them. Dropping them does not
    change the run: some SIPNET does not register and ignores too (older
    files carry many), and the rest belong to processes pySIPNET refuses to
    switch on (:data:`~pysipnet.parameters.model.UNSUPPORTED_FLAGS`), which
    are the only ones that read them. The warning says which is which.

    The flags are not in the file, so nothing here checks that the parameters
    suit them; :func:`write_param_file` does, or call
    :meth:`~pysipnet.parameters.model.SIPNETParameters.validate_for_flags`.

    Raises
    ------
    ValueError
        If the file is malformed (see :func:`read_param_file`), lacks a
        required parameter (all are listed), or holds a value outside a
        parameter's domain. Each parameter is named by SIPNET's name and by
        its field.
    """
    groups: dict[str, dict[str, float]] = {group: {} for group in SIPNETParameters.model_fields}
    ignored: list[str] = []
    unmodeled: list[str] = []
    for name, value in read_param_file(path).items():
        python_path = _PYTHON_PATH_BY_KEY.get(name.lower())
        if python_path is None:
            (unmodeled if name.lower() in _UNMODELED_FLAGS_BY_KEY else ignored).append(name)
            continue
        group, field = python_path.split(".")
        groups[group][field] = value

    # Before the check for missing ones: a misspelled required name shows up
    # in both, and the warning is what explains the error.
    if ignored or unmodeled:
        warnings.warn(
            _unknown_names_message(path, ignored, unmodeled), UnknownParameterWarning, stacklevel=2
        )

    missing = [
        _describe(f"{group}.{field}")
        for group, field in _REQUIRED_FIELDS
        if field not in groups[group]
    ]
    if missing:
        raise ValueError(
            f"{path} lacks {len(missing)} required parameter(s): " + ", ".join(missing) + "."
        )

    try:
        return SIPNETParameters.model_validate(groups)
    except ValidationError as exc:
        problems = [_describe_validation_error(error) for error in exc.errors()]
        raise ValueError(f"{path} holds invalid parameters:\n" + "\n".join(problems)) from exc


def _describe(python_path: str) -> str:
    return f"{PYTHON_TO_SIPNET[python_path]} ({python_path})"


def _describe_validation_error(error: ErrorDetails) -> str:
    python_path = ".".join(str(part) for part in error["loc"])
    if python_path in PYTHON_TO_SIPNET:
        return f"  • {_describe(python_path)} = {error['input']!r}: {error['msg']}"
    return f"  • {error['msg']}"


def _unknown_names_message(path: Path, ignored: list[str], unmodeled: list[str]) -> str:
    parts = [f"{path}: dropped parameters SIPNETParameters has no field for."]
    if ignored:
        parts.append(
            f"SIPNET does not register these either, and ignores them: {', '.join(ignored)}."
        )
    if unmodeled:
        by_flag = ", ".join(
            f"{name} ({' or '.join(_UNMODELED_FLAGS_BY_KEY[name.lower()])})" for name in unmodeled
        )
        parts.append(
            "SIPNET reads these only under flags pySIPNET refuses, so they cannot "
            f"affect a run: {by_flag}."
        )
    return " ".join(parts)
