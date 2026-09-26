"""The variable registry: what every SIPNET output column and climate driver is.

SIPNET writes a ``.out`` file whose header names (``plantWoodC``, ``rSoil``,
``nppStorage``) say little about what the numbers are, and its documentation
does not always agree with its source.  This module records, for each column,
what pySIPNET calls it, what SIPNET calls it, what it measures, in what unit,
and whether the value is a pool at an instant, a total over the timestep, a
mean over the timestep, a rate, or a running total.  Everything here was read
from the SIPNET source at the pinned commit (``outputState()`` and
``updateTrackers()`` in ``src/sipnet/sipnet.c``, and ``src/sipnet/state.h``),
and ``tests/test_variables.py`` checks the registry against the binary.

Naming convention
-----------------
Lower-case words joined by underscores, no acronyms and no truncations:
``net_ecosystem_exchange`` rather than ``nee``, ``soil_respiration`` rather
than ``rSoil``.  The one permitted exception is ``q10``, which is the name of
a quantity rather than an abbreviation of one.  Familiar short forms survive
as *aliases* (``"nee"``, ``"gpp"``, ``"NEE"``) that :func:`resolve_output_variable`
understands, but they are never column names.

Time convention
---------------
SIPNET labels each row with the **start** of the timestep (``year``,
``day_of_year``, ``hour_of_day``), on whatever clock the climate drivers use —
SIPNET has no time zone of its own — but pools are written **after** the step
has been applied, so a ``TIMESTEP_END_STATE`` value is the pool at the *end* of
the step. ``TIMESTEP_TOTAL`` values are accumulated over the step,
``TIMESTEP_MEAN`` values are means over it, the single ``DAILY_RATE`` value is a
per-day rate for the step (the kind's name notwithstanding, a derived rate may
be per hour or per second; its units say which), and ``CUMULATIVE`` values run from the start of the
simulation to the end of the step (and continue across a restart, which carries
them in the checkpoint).

The xarray representation (:mod:`pysipnet.dataset`) therefore puts its ``time``
coordinate at the **end** of the step, with ``time_bounds`` spanning the step:
that is the one labeling under which every variable's CF ``cell_methods`` is
literally true — a pool is ``time: point`` at ``time``, a total is ``time: sum``
over the bounds.  Each spec's :attr:`VariableSpec.time_reference` states the
same thing in words, and that text travels as an attribute so a user never has
to look it up.

Resampling
----------
Which ways of combining consecutive steps are meaningful depends on the kind:
totals add, pools do not; a running total is the last value; means and rates
must be weighted by the step length, because SIPNET steps are not all the same
length.  :data:`RESAMPLING_METHODS_FOR_KIND` records the valid methods for each
kind and :data:`RESAMPLED_KIND` what kind the result is.  There is deliberately
no default: :func:`pysipnet.resample.resample` requires the caller to say which
method they want and refuses one the kind does not support.
:func:`variable_kind` is how it finds the kind: a ``kind`` attribute if there
is one, otherwise the registries, aliases included.

Arithmetic
----------
Multiplying or dividing by a time changes what a value is over its step, and
only two such changes have a kind to name the result: a total over the step
divided by the step's length is a rate, and a rate times it is a total.
:data:`KIND_AFTER_TIME_POWER` records them, and
:mod:`pysipnet.arithmetic` refuses every other change.

Climate drivers
---------------
:data:`CLIMATE_VARIABLES` describes the columns of the ``.clim`` input file the
same way. Each :class:`ClimateVariableSpec` also records the units SIPNET
converts the value to on read (``internal_units``), because SIPNET's own
documentation quotes those rather than the file units. Climate rows follow the
same time convention: ``year`` / ``day_of_year`` / ``hour_of_day`` are the
start of the step on the drivers' clock, which
:class:`~pysipnet.climate.ClimateDrivers` may declare; means are over the step,
and ``photosynthetically_active_radiation`` and ``precipitation`` are totals
over the step.

Precision
---------
SIPNET prints every column with a fixed number of decimals (``%8.3f`` and the
like), recorded in :attr:`VariableSpec.output_decimals`.  Carbon fluxes get
three decimals, so a half-hourly NEE of 0.05 g C m⁻² carries a 1 %
quantization floor.  That matters when the output feeds a likelihood.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

from pysipnet.units import UnitStyle, format_units, validate_units

if TYPE_CHECKING:
    from pysipnet.parameters.model import ModelFlags

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
"""What a pySIPNET variable name must look like."""


class VariableKind(StrEnum):
    """What a value in a column represents, relative to the timestep it is labeled with."""

    TIMESTEP_START_COORDINATE = "timestep_start_coordinate"
    """Identifies the row: the start of the timestep."""

    TIMESTEP_END_STATE = "timestep_end_state"
    """A pool, reported at the end of the timestep."""

    TIMESTEP_TOTAL = "timestep_total"
    """A quantity accumulated over the timestep: a flux integrated over it, or its duration."""

    DAILY_RATE = "daily_rate"
    """A transfer rate that applied during the timestep.

    SIPNET's own rate column is per day, which gives the kind its name; a rate
    derived with :mod:`pysipnet.arithmetic` or converted with
    :func:`~pysipnet.units.convert_dataarray_units` may be per hour or per
    second, and its units say which.
    """

    TIMESTEP_MEAN = "timestep_mean"
    """A quantity averaged over the timestep."""

    CUMULATIVE = "cumulative"
    """A running total from the start of the simulation to the end of the timestep."""


ResamplingMethod = Literal["sum", "mean", "last"]
"""How consecutive steps may be combined: add them, time-weighted-average them, or keep the last."""

RESAMPLING_METHODS_FOR_KIND: dict[VariableKind, frozenset[str]] = {
    VariableKind.TIMESTEP_START_COORDINATE: frozenset(),
    VariableKind.TIMESTEP_END_STATE: frozenset({"last", "mean"}),
    VariableKind.TIMESTEP_TOTAL: frozenset({"sum"}),
    VariableKind.DAILY_RATE: frozenset({"mean"}),
    VariableKind.TIMESTEP_MEAN: frozenset({"mean"}),
    VariableKind.CUMULATIVE: frozenset({"last"}),
}
"""The methods that give a meaningful value when steps of this kind are combined.

A total over a step adds to the next; a pool at the end of a step does not, so
only its last value (still a pool) or its time-weighted mean (now a mean) make
sense; a rate or a mean over a step averages, weighted by step length; a running
total is whatever it had reached by the last step.
"""

RESAMPLED_KIND: dict[tuple[VariableKind, str], VariableKind] = {
    (VariableKind.TIMESTEP_END_STATE, "last"): VariableKind.TIMESTEP_END_STATE,
    (VariableKind.TIMESTEP_END_STATE, "mean"): VariableKind.TIMESTEP_MEAN,
    (VariableKind.TIMESTEP_TOTAL, "sum"): VariableKind.TIMESTEP_TOTAL,
    (VariableKind.DAILY_RATE, "mean"): VariableKind.DAILY_RATE,
    (VariableKind.TIMESTEP_MEAN, "mean"): VariableKind.TIMESTEP_MEAN,
    (VariableKind.CUMULATIVE, "last"): VariableKind.CUMULATIVE,
}
"""What kind a variable of a given kind becomes after each valid resampling method."""

TIME_REFERENCE_FOR_KIND: dict[VariableKind, str] = {
    VariableKind.TIMESTEP_START_COORDINATE: "start of the timestep",
    VariableKind.TIMESTEP_END_STATE: "value at the end of the timestep",
    VariableKind.TIMESTEP_TOTAL: "total over the timestep",
    VariableKind.DAILY_RATE: "rate during the timestep",
    VariableKind.TIMESTEP_MEAN: "mean over the timestep",
    VariableKind.CUMULATIVE: "running total from the start of the run to the end of the timestep",
}

KIND_AFTER_TIME_POWER: dict[tuple[VariableKind, int], VariableKind] = {
    (VariableKind.TIMESTEP_TOTAL, -1): VariableKind.DAILY_RATE,
    (VariableKind.DAILY_RATE, 1): VariableKind.TIMESTEP_TOTAL,
}
"""What a kind becomes when its value is multiplied by a time raised to a power.

The key is ``(kind, power)``: ``-1`` for dividing by a time (or multiplying by
a per-time), ``+1`` for multiplying by a time (or dividing by a per-time).  A
power of zero leaves every kind as it is.  A pair not listed has no kind to
name its result: a pool times a per-day turnover rate is a flux, which SIPNET
reports itself, and a running total per day is not anything a step covers.
"""

# ``cell_methods`` vocabulary from the Climate and Forecast (CF) conventions,
# true under the step-end ``time`` labeling in pysipnet.dataset. A cumulative
# value has none: its accumulation interval is the run so far, not the row's
# bounds, and CF has no way to say so in cell_methods.
CELL_METHODS_FOR_KIND: dict[VariableKind, str | None] = {
    VariableKind.TIMESTEP_START_COORDINATE: None,
    VariableKind.TIMESTEP_END_STATE: "time: point",
    VariableKind.TIMESTEP_TOTAL: "time: sum",
    VariableKind.DAILY_RATE: "time: mean",
    VariableKind.TIMESTEP_MEAN: "time: mean",
    VariableKind.CUMULATIVE: None,
}


@dataclass(frozen=True)
class VariableSpec:
    """Everything a consumer needs to know about one output variable."""

    name: str
    """pySIPNET name; the column name in every DataFrame and Dataset."""

    sipnet_name: str
    """The exact header token SIPNET writes."""

    kind: VariableKind
    """What the value is relative to its timestep; see :class:`VariableKind`."""

    units: str
    """UDUNITS-style unit string, physical units only (see :mod:`pysipnet.units`)."""

    description: str
    """What the variable is, including how SIPNET computes it."""

    long_label: str
    """Plot-ready name without units, e.g. ``"Net ecosystem exchange"``."""

    constituent: str = ""
    """Substance the unit refers to: ``"C"``, ``"N"``, ``"H2O"`` or empty."""

    short_label: str = ""
    """Compact label for legends, e.g. ``"NEE"``; empty means use *long_label*."""

    aliases: tuple[str, ...] = ()
    """Other names :func:`resolve_output_variable` accepts for this variable."""

    requires_flag: str | None = None
    """``ModelFlags`` field without which SIPNET writes the column as constant zero."""

    output_decimals: int | None = None
    """Decimals in SIPNET's fixed-precision output; ``None`` for integers."""

    sign_convention: str = ""
    """Which direction is positive, when that is not obvious."""

    group: str = ""
    """Display grouping: ``"carbon_pools"``, ``"carbon_fluxes"``, ``"water"``, ..."""

    comment: str = ""
    """Free-text CF ``comment`` attribute, for what ``cell_methods`` cannot say."""

    cell_methods_comment: str = ""
    """Qualifies the kind's ``cell_methods``, e.g. how a mean was actually formed."""

    def __post_init__(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"Variable name {self.name!r} violates the naming convention.")
        validate_units(self.units)
        if not self.description or not self.long_label:
            raise ValueError(f"Variable {self.name!r} needs a description and a long_label.")

    @property
    def time_reference(self) -> str:
        """In words, which moment or interval of the timestep the value refers to."""
        return TIME_REFERENCE_FOR_KIND[self.kind]

    @property
    def cell_methods(self) -> str | None:
        """CF ``cell_methods`` for this variable, or ``None`` when CF has no way to say it."""
        base = CELL_METHODS_FOR_KIND[self.kind]
        if base is not None and self.cell_methods_comment:
            return f"{base} (comment: {self.cell_methods_comment})"
        return base

    @property
    def label(self) -> str:
        """*short_label* if set, else *long_label*."""
        return self.short_label or self.long_label

    def formatted_units(self, style: UnitStyle = "unicode") -> str:
        """Units rendered for display, constituent included, e.g. ``"g C m⁻²"``."""
        return format_units(self.units, constituent=self.constituent, style=style)

    def axis_label(self, style: UnitStyle = "unicode") -> str:
        """``"Net ecosystem exchange (g C m⁻²)"``, for a plot axis."""
        return f"{self.long_label} ({self.formatted_units(style)})"

    def xarray_attributes(self) -> dict[str, Any]:
        """Attributes for an :class:`xarray.DataArray` holding this variable.

        Keys follow the Climate and Forecast (CF) metadata conventions where
        one exists (``units``, ``long_name``, ``cell_methods``); the rest are
        pySIPNET's own and are spelled out in full.
        """
        attrs: dict[str, Any] = {
            "units": self.units,
            "long_name": self.long_label,
            "description": self.description,
            "kind": self.kind.value,
            "time_reference": self.time_reference,
            "sipnet_name": self.sipnet_name,
        }
        if self.constituent:
            attrs["constituent"] = self.constituent
        if self.cell_methods is not None:
            attrs["cell_methods"] = self.cell_methods
        if self.comment:
            attrs["comment"] = self.comment
        if self.sign_convention:
            attrs["sign_convention"] = self.sign_convention
        if self.requires_flag is not None:
            attrs["requires_flag"] = self.requires_flag
        if self.output_decimals is not None:
            attrs["output_decimals"] = self.output_decimals
        return attrs

    def to_record(self) -> dict[str, Any]:
        """A plain, JSON-serializable dict of every field plus the derived properties."""
        record = asdict(self)
        record["kind"] = self.kind.value
        record["aliases"] = list(self.aliases)
        record["time_reference"] = self.time_reference
        record["cell_methods"] = self.cell_methods
        record["resampling_methods"] = sorted(RESAMPLING_METHODS_FOR_KIND[self.kind])
        record["formatted_units"] = self.formatted_units()
        return record


def _spec(**kwargs: Any) -> VariableSpec:
    return VariableSpec(**kwargs)


_GC = dict(units="g m-2", constituent="C")
_GN = dict(units="g m-2", constituent="N")

OUTPUT_VARIABLES: tuple[VariableSpec, ...] = (
    # In the order SIPNET writes them (outputHeader() in src/sipnet/sipnet.c);
    # tests/test_variables.py checks this against the binary. Display grouping
    # is the `group` field, not the position here.
    _spec(
        name="year",
        sipnet_name="year",
        kind=VariableKind.TIMESTEP_START_COORDINATE,
        units="1",
        description="Calendar year at the start of the timestep, on the climate drivers' clock.",
        long_label="Year",
        group="time",
    ),
    _spec(
        name="day_of_year",
        sipnet_name="day",
        kind=VariableKind.TIMESTEP_START_COORDINATE,
        units="1",
        description="Day of year at the start of the timestep, on the climate drivers' clock; "
        "1 is January 1st.",
        long_label="Day of year",
        short_label="DOY",
        aliases=("day", "doy"),
        group="time",
    ),
    _spec(
        name="hour_of_day",
        sipnet_name="time",
        kind=VariableKind.TIMESTEP_START_COORDINATE,
        units="h",
        description="Hours after midnight at the start of the timestep, on the climate drivers' "
        "clock; may be fractional.",
        long_label="Hour of day",
        aliases=("time",),
        output_decimals=2,
        group="time",
    ),
    _spec(
        name="wood_carbon",
        sipnet_name="plantWoodC",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GC,
        description=(
            "Total wood carbon: the structural wood pool plus the storage-lag term "
            "reported separately as wood_storage_carbon. Roots are separate pools, so "
            "this is above-ground woody biomass. SIPNET: plantWoodC + plantCAccountingDelta."
        ),
        long_label="Wood carbon",
        short_label="Wood C",
        aliases=("plant_wood_c", "plantWoodC"),
        output_decimals=2,
        group="carbon_pools",
    ),
    _spec(
        name="leaf_carbon",
        sipnet_name="plantLeafC",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GC,
        description=(
            "Carbon in leaves. Leaf area index is leaf_carbon divided by the "
            "leaf carbon per unit area parameter."
        ),
        long_label="Leaf carbon",
        short_label="Leaf C",
        aliases=("plant_leaf_c", "plantLeafC"),
        output_decimals=2,
        group="carbon_pools",
    ),
    _spec(
        name="wood_growth",
        sipnet_name="woodCreation",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description=(
            "Carbon allocated to the wood pool over the timestep, after SIPNET's carbon "
            "and nitrogen limitation adjustments. Excludes carbon added by planting events "
            "and carbon withdrawn from wood for leaf-on."
        ),
        long_label="Wood growth",
        aliases=("wood_creation", "woodCreation"),
        output_decimals=2,
        group="carbon_fluxes",
    ),
    _spec(
        name="soil_carbon",
        sipnet_name="soil",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GC,
        description="Carbon in the single soil organic matter pool.",
        long_label="Soil carbon",
        short_label="Soil C",
        aliases=("soil_c", "soil"),
        output_decimals=2,
        group="carbon_pools",
    ),
    _spec(
        name="coarse_root_carbon",
        sipnet_name="coarseRootC",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GC,
        description="Carbon in coarse roots.",
        long_label="Coarse root carbon",
        short_label="Coarse root C",
        aliases=("coarse_root_c", "coarseRootC"),
        output_decimals=2,
        group="carbon_pools",
    ),
    _spec(
        name="fine_root_carbon",
        sipnet_name="fineRootC",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GC,
        description="Carbon in fine roots.",
        long_label="Fine root carbon",
        short_label="Fine root C",
        aliases=("fine_root_c", "fineRootC"),
        output_decimals=2,
        group="carbon_pools",
    ),
    _spec(
        name="litter_carbon",
        sipnet_name="litter",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GC,
        description="Carbon in the litter pool. Constant zero unless the litter pool is on.",
        long_label="Litter carbon",
        short_label="Litter C",
        aliases=("litter_c", "litter"),
        requires_flag="litter_pool",
        output_decimals=2,
        group="carbon_pools",
    ),
    _spec(
        name="soil_water",
        sipnet_name="soilWater",
        kind=VariableKind.TIMESTEP_END_STATE,
        units="cm",
        constituent="H2O",
        description="Plant-available soil water, as a depth of liquid water.",
        long_label="Soil water",
        aliases=("soilWater",),
        output_decimals=3,
        group="water",
    ),
    _spec(
        name="soil_wetness_fraction",
        sipnet_name="soilWetnessFrac",
        kind=VariableKind.TIMESTEP_MEAN,
        units="1",
        description=(
            "Soil water as a fraction of water holding capacity, averaged over the "
            "timestep as the mean of the start and end values. Diagnostic only: SIPNET's "
            "moisture functions use the instantaneous fraction."
        ),
        long_label="Soil wetness fraction",
        aliases=("soil_wetness_frac", "soilWetnessFrac"),
        cell_methods_comment="mean of the values at the start and end of the timestep",
        output_decimals=3,
        group="water",
    ),
    _spec(
        name="snow_water_equivalent",
        sipnet_name="snow",
        kind=VariableKind.TIMESTEP_END_STATE,
        units="cm",
        constituent="H2O",
        description="Snowpack as a depth of liquid water equivalent. Simulated whether or not "
        "ModelFlags.snow is on: at the pinned SIPNET the flag only decides whether the snow "
        "melt rate parameter is required. If that parameter is supplied it is used either "
        "way; if it is omitted SIPNET leaves it at zero and snow never melts.",
        long_label="Snow water equivalent",
        short_label="SWE",
        aliases=("snow",),
        output_decimals=2,
        group="water",
    ),
    _spec(
        name="net_primary_production",
        sipnet_name="npp",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description="Gross primary production minus autotrophic respiration over the timestep.",
        long_label="Net primary production",
        short_label="NPP",
        aliases=("npp", "NPP"),
        output_decimals=3,
        group="carbon_fluxes",
    ),
    _spec(
        name="net_ecosystem_exchange",
        sipnet_name="nee",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description=(
            "Net carbon flux between ecosystem and atmosphere over the timestep: "
            "ecosystem respiration minus gross primary production. Excludes carbon lost "
            "as methane and carbon removed by harvest events, so it does not by itself "
            "close the carbon balance when those are active."
        ),
        long_label="Net ecosystem exchange",
        short_label="NEE",
        aliases=("nee", "NEE"),
        output_decimals=3,
        sign_convention="positive is a flux from the ecosystem to the atmosphere",
        group="carbon_fluxes",
    ),
    _spec(
        name="cumulative_net_ecosystem_exchange",
        sipnet_name="cumNEE",
        kind=VariableKind.CUMULATIVE,
        **_GC,
        description=(
            "Net ecosystem exchange summed from the start of the simulation to the end of "
            "the timestep. The total is carried in restart checkpoints, so a restarted run "
            "continues it rather than starting again."
        ),
        long_label="Cumulative net ecosystem exchange",
        short_label="Cumulative NEE",
        aliases=("cum_nee", "cumNEE", "NEE_cum"),
        comment=(
            "Accumulated from the start of the run (or of the restart checkpoint it "
            "continues) to the end of the timestep, not over the row's time bounds; "
            "equal to the cumulative sum of net_ecosystem_exchange."
        ),
        output_decimals=3,
        sign_convention="positive is a net loss of carbon from the ecosystem to the atmosphere",
        group="carbon_fluxes",
    ),
    _spec(
        name="gross_primary_production",
        sipnet_name="gpp",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description="Gross photosynthesis over the timestep.",
        long_label="Gross primary production",
        short_label="GPP",
        aliases=("gpp", "GPP"),
        output_decimals=3,
        group="carbon_fluxes",
    ),
    _spec(
        name="above_ground_respiration",
        sipnet_name="rAboveground",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description=(
            "Foliar and wood maintenance respiration over the timestep, plus growth "
            "respiration when that process is on."
        ),
        long_label="Above-ground respiration",
        aliases=("r_aboveground", "rAboveground"),
        output_decimals=3,
        group="respiration",
    ),
    _spec(
        name="soil_respiration",
        sipnet_name="rSoil",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description=(
            "Respiration from below ground over the timestep: root respiration plus "
            "heterotrophic respiration. Not purely heterotrophic, despite SIPNET's "
            "documentation."
        ),
        long_label="Soil respiration",
        aliases=("r_soil", "rSoil"),
        output_decimals=3,
        group="respiration",
    ),
    _spec(
        name="root_respiration",
        sipnet_name="rRoot",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description="Fine plus coarse root respiration over the timestep.",
        long_label="Root respiration",
        aliases=("r_root", "rRoot"),
        output_decimals=3,
        group="respiration",
    ),
    _spec(
        name="autotrophic_respiration",
        sipnet_name="ra",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description="Plant respiration over the timestep: above-ground plus root respiration.",
        long_label="Autotrophic respiration",
        short_label="Rₐ",
        aliases=("ra",),
        output_decimals=3,
        group="respiration",
    ),
    _spec(
        name="heterotrophic_respiration",
        sipnet_name="rh",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description="Microbial respiration from the soil and litter pools over the timestep.",
        long_label="Heterotrophic respiration",
        short_label="Rₕ",
        aliases=("rh",),
        output_decimals=3,
        group="respiration",
    ),
    _spec(
        name="ecosystem_respiration",
        sipnet_name="rtot",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description="Total respiration over the timestep: autotrophic plus heterotrophic.",
        long_label="Ecosystem respiration",
        short_label="Rₑ",
        aliases=("rtot", "total_respiration"),
        output_decimals=3,
        group="respiration",
    ),
    _spec(
        name="evapotranspiration",
        sipnet_name="evapotranspiration",
        kind=VariableKind.TIMESTEP_TOTAL,
        units="cm",
        constituent="H2O",
        description=(
            "Water lost over the timestep as transpiration, soil evaporation, "
            "interception evaporation, sublimation and evaporation of irrigation."
        ),
        long_label="Evapotranspiration",
        short_label="ET",
        aliases=("et", "ET"),
        output_decimals=8,
        group="water",
    ),
    _spec(
        name="transpiration_rate",
        sipnet_name="fluxestranspiration",
        kind=VariableKind.DAILY_RATE,
        units="cm d-1",
        constituent="H2O",
        description=(
            "Transpiration as a per-day rate during the timestep. Unlike every other "
            "flux column this is not multiplied by the timestep length; multiply by the "
            "length in days to get the total over the step."
        ),
        long_label="Transpiration rate",
        aliases=("transpiration", "fluxestranspiration"),
        output_decimals=4,
        group="water",
    ),
    _spec(
        name="mineral_nitrogen",
        sipnet_name="minN",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GN,
        description="Soil mineral nitrogen pool (soil and litter share one pool).",
        long_label="Mineral nitrogen",
        aliases=("mineral_n", "minN"),
        requires_flag="nitrogen_cycle",
        output_decimals=4,
        group="nitrogen",
    ),
    _spec(
        name="soil_organic_nitrogen",
        sipnet_name="soilOrgN",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GN,
        description="Nitrogen in soil organic matter.",
        long_label="Soil organic nitrogen",
        aliases=("soil_organic_n", "soilOrgN"),
        requires_flag="nitrogen_cycle",
        output_decimals=4,
        group="nitrogen",
    ),
    _spec(
        name="litter_nitrogen",
        sipnet_name="litterN",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GN,
        description="Nitrogen in the litter pool.",
        long_label="Litter nitrogen",
        aliases=("litter_n", "litterN"),
        requires_flag="nitrogen_cycle",
        output_decimals=4,
        group="nitrogen",
    ),
    _spec(
        name="plant_nitrogen_storage",
        sipnet_name="plantStorageN",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GN,
        description=(
            "Nitrogen held in plant storage, filled by resorption at leaf-off and drawn "
            "on at leaf-on."
        ),
        long_label="Plant nitrogen storage",
        aliases=("plant_storage_n", "plantStorageN"),
        requires_flag="nitrogen_cycle",
        output_decimals=4,
        group="nitrogen",
    ),
    _spec(
        name="nitrogen_volatilization",
        sipnet_name="n2o",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GN,
        description=(
            "Mineral nitrogen lost to the atmosphere by volatilization over the "
            "timestep. SIPNET labels the column n2o but does not speciate the gas."
        ),
        long_label="Nitrogen volatilization",
        aliases=("n2o",),
        requires_flag="nitrogen_cycle",
        output_decimals=6,
        group="nitrogen",
    ),
    _spec(
        name="nitrogen_leaching",
        sipnet_name="nLeaching",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GN,
        description="Mineral nitrogen lost by leaching over the timestep.",
        long_label="Nitrogen leaching",
        aliases=("n_leaching", "nLeaching"),
        requires_flag="nitrogen_cycle",
        output_decimals=4,
        group="nitrogen",
    ),
    _spec(
        name="nitrogen_fixation",
        sipnet_name="nFixation",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GN,
        description="Plant nitrogen demand met by fixation over the timestep.",
        long_label="Nitrogen fixation",
        aliases=("n_fixation", "nFixation"),
        requires_flag="nitrogen_cycle",
        output_decimals=4,
        group="nitrogen",
    ),
    _spec(
        name="nitrogen_uptake",
        sipnet_name="nUptake",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GN,
        description="Plant nitrogen demand met by uptake from the mineral pool over the timestep.",
        long_label="Nitrogen uptake",
        aliases=("n_uptake", "nUptake"),
        requires_flag="nitrogen_cycle",
        output_decimals=4,
        group="nitrogen",
    ),
    _spec(
        name="methane_production",
        sipnet_name="ch4",
        kind=VariableKind.TIMESTEP_TOTAL,
        **_GC,
        description=(
            "Methane produced from the soil and litter pools over the timestep, "
            "as mass of carbon. Constant zero unless anaerobic processes are on."
        ),
        long_label="Methane production",
        short_label="CH₄",
        aliases=("ch4",),
        requires_flag="anaerobic",
        output_decimals=4,
        group="methane",
    ),
    _spec(
        name="wood_storage_carbon",
        sipnet_name="nppStorage",
        kind=VariableKind.TIMESTEP_END_STATE,
        **_GC,
        description=(
            "Storage-lag component of wood_carbon: the difference between carbon gained "
            "this step (photosynthesis minus autotrophic respiration) and carbon allocated "
            "to growth from the five-day mean NPP, accumulated over the run. Can be negative. "
            "Reset to zero when the plant dies. SIPNET: plantCAccountingDelta."
        ),
        long_label="Wood storage carbon",
        aliases=("npp_storage", "nppStorage"),
        output_decimals=4,
        group="carbon_pools",
    ),
)
"""Every column SIPNET writes at the pinned version, in the order it writes them."""

TIME_COORDINATE_NAMES: tuple[str, ...] = tuple(
    v.name for v in OUTPUT_VARIABLES if v.kind == VariableKind.TIMESTEP_START_COORDINATE
)
"""``("year", "day_of_year", "hour_of_day")``: the columns that identify a row."""

OUTPUT_VARIABLES_BY_NAME: dict[str, VariableSpec] = {v.name: v for v in OUTPUT_VARIABLES}
OUTPUT_VARIABLES_BY_SIPNET_NAME: dict[str, VariableSpec] = {
    v.sipnet_name: v for v in OUTPUT_VARIABLES
}

# Columns written by SIPNET versions before the pin. Not in the registry proper
# because the binary no longer produces them, but mapped so previously saved
# output still reads under convention-following names.
LEGACY_OUTPUT_COLUMNS: dict[str, str] = {
    "bcdeltaC": "carbon_balance_error",
    "bcdeltaN": "nitrogen_balance_error",
    "fPAR": "absorbed_light_fraction",
    "microbeC": "microbe_carbon",
    "litterWater": "litter_water",
}


# ── Climate drivers ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClimateVariableSpec(VariableSpec):
    """A column of the ``.clim`` climate driver file.

    ``units`` are the units **in the file**, which is what a user supplies.
    SIPNET converts some columns on read; ``internal_units`` and
    ``internal_conversion`` record that so nobody has to reconcile the two
    conventions from the SIPNET docs.
    """

    internal_units: str = ""
    """Units SIPNET works in after reading the column, when different from *units*."""

    internal_conversion: str = ""
    """How SIPNET converts the file value, e.g. ``"× 0.1 (mm → cm)"``."""

    def xarray_attributes(self) -> dict[str, Any]:
        attrs = super().xarray_attributes()
        if self.internal_units:
            attrs["sipnet_internal_units"] = self.internal_units
        if self.internal_conversion:
            attrs["sipnet_internal_conversion"] = self.internal_conversion
        return attrs


def _climate(**kwargs: Any) -> ClimateVariableSpec:
    return ClimateVariableSpec(**kwargs)


CLIMATE_VARIABLES: tuple[ClimateVariableSpec, ...] = (
    # In the order of the 12-column .clim layout (readClimData in src/sipnet/sipnet.c).
    _climate(
        name="year",
        sipnet_name="year",
        kind=VariableKind.TIMESTEP_START_COORDINATE,
        units="1",
        description="Calendar year at the start of the timestep, on the climate drivers' clock.",
        long_label="Year",
        group="time",
    ),
    _climate(
        name="day_of_year",
        sipnet_name="day",
        kind=VariableKind.TIMESTEP_START_COORDINATE,
        units="1",
        description="Day of year at the start of the timestep, on the climate drivers' clock; "
        "1 is January 1st.",
        long_label="Day of year",
        short_label="DOY",
        aliases=("day", "doy"),
        group="time",
    ),
    _climate(
        name="hour_of_day",
        sipnet_name="time",
        kind=VariableKind.TIMESTEP_START_COORDINATE,
        units="h",
        description="Hours after midnight at the start of the timestep, on the climate drivers' "
        "clock; may be fractional.",
        long_label="Hour of day",
        aliases=("time",),
        group="time",
    ),
    _climate(
        name="time_step_length",
        sipnet_name="length",
        kind=VariableKind.TIMESTEP_TOTAL,
        units="d",
        description="Duration of the timestep in days. SIPNET also accepts a negative value "
        "meaning seconds; pySIPNET writes days only.",
        long_label="Timestep length",
        aliases=("length",),
        group="time",
    ),
    _climate(
        name="air_temperature",
        sipnet_name="tair",
        kind=VariableKind.TIMESTEP_MEAN,
        units="degC",
        description="Mean air temperature over the timestep.",
        long_label="Air temperature",
        aliases=("tair",),
        group="meteorology",
    ),
    _climate(
        name="soil_temperature",
        sipnet_name="tsoil",
        kind=VariableKind.TIMESTEP_MEAN,
        units="degC",
        description="Mean soil temperature over the timestep.",
        long_label="Soil temperature",
        aliases=("tsoil",),
        group="meteorology",
    ),
    _climate(
        name="photosynthetically_active_radiation",
        sipnet_name="par",
        kind=VariableKind.TIMESTEP_TOTAL,
        units="mol m-2",
        constituent="photons",
        description="Photosynthetically active radiation summed over the timestep, as moles "
        "of photons per square meter of ground (1 Einstein = 1 mol). To convert an "
        "instantaneous flux in µmol m⁻² s⁻¹, multiply by the timestep length in seconds "
        "and divide by 1e6.",
        long_label="Photosynthetically active radiation",
        short_label="PAR",
        aliases=("par",),
        internal_units="mol m-2 d-1",
        internal_conversion="÷ time_step_length (total over step → per day)",
        group="meteorology",
    ),
    _climate(
        name="precipitation",
        sipnet_name="precip",
        kind=VariableKind.TIMESTEP_TOTAL,
        units="mm",
        constituent="H2O",
        description="Total precipitation over the timestep as a depth of liquid water "
        "equivalent; rain or snow.",
        long_label="Precipitation",
        aliases=("precip",),
        internal_units="cm",
        internal_conversion="× 0.1 (mm → cm)",
        group="meteorology",
    ),
    _climate(
        name="vapor_pressure_deficit",
        sipnet_name="vpd",
        kind=VariableKind.TIMESTEP_MEAN,
        units="Pa",
        description="Mean vapor pressure deficit of the air over the timestep. SIPNET clamps "
        "values below a tiny positive number up to it.",
        long_label="Vapor pressure deficit",
        short_label="VPD",
        aliases=("vpd", "vapour_pressure_deficit"),
        internal_units="kPa",
        internal_conversion="× 0.001 (Pa → kPa), clamped ≥ 1e-6",
        group="meteorology",
    ),
    _climate(
        name="soil_vapor_pressure_deficit",
        sipnet_name="vpdSoil",
        kind=VariableKind.TIMESTEP_MEAN,
        units="Pa",
        description="Mean vapor pressure deficit between the soil and the air over the "
        "timestep, using saturation vapor pressure at the soil temperature.",
        long_label="Soil vapor pressure deficit",
        aliases=("vpd_soil", "vpdSoil", "soil_vapour_pressure_deficit"),
        internal_units="kPa",
        internal_conversion="× 0.001 (Pa → kPa)",
        group="meteorology",
    ),
    _climate(
        name="vapor_pressure",
        sipnet_name="vPress",
        kind=VariableKind.TIMESTEP_MEAN,
        units="Pa",
        description="Mean vapor pressure in the canopy airspace over the timestep.",
        long_label="Vapor pressure",
        aliases=("vpress", "vPress", "vapour_pressure"),
        internal_units="kPa",
        internal_conversion="× 0.001 (Pa → kPa)",
        group="meteorology",
    ),
    _climate(
        name="wind_speed",
        sipnet_name="wspd",
        kind=VariableKind.TIMESTEP_MEAN,
        units="m s-1",
        description="Mean wind speed over the timestep. SIPNET clamps values below a tiny "
        "positive number up to it.",
        long_label="Wind speed",
        aliases=("wspd",),
        internal_conversion="clamped ≥ 1e-6",
        group="meteorology",
    ),
)
"""Every column of the 12-value climate record, in file order."""

CLIMATE_COLUMN_NAMES: tuple[str, ...] = tuple(v.name for v in CLIMATE_VARIABLES)
CLIMATE_VARIABLES_BY_NAME: dict[str, ClimateVariableSpec] = {v.name: v for v in CLIMATE_VARIABLES}


def _build_alias_index_for(
    specs: tuple[VariableSpec, ...],
) -> dict[str, VariableSpec]:
    index: dict[str, VariableSpec] = {}
    for spec in specs:
        for key in (spec.name, spec.sipnet_name, *spec.aliases):
            existing = index.get(key)
            if existing is not None and existing is not spec:
                raise ValueError(
                    f"Alias {key!r} is claimed by both {existing.name!r} and {spec.name!r}."
                )
            index[key] = spec
    return index


_CLIMATE_ALIAS_INDEX: dict[str, VariableSpec] = _build_alias_index_for(CLIMATE_VARIABLES)


def resolve_climate_variable(name: str) -> ClimateVariableSpec:
    """Return the climate spec for a column name, alias or SIPNET column name."""
    spec = _CLIMATE_ALIAS_INDEX.get(name)
    if spec is None:
        raise KeyError(
            f"{name!r} is not a SIPNET climate driver or alias. "
            f"Expected one of {list(CLIMATE_COLUMN_NAMES)}."
        )
    return cast(ClimateVariableSpec, spec)


def climate_variable_records() -> list[dict[str, Any]]:
    """The climate registry as JSON-serializable dicts, for documentation and export."""
    return [spec.to_record() for spec in CLIMATE_VARIABLES]


# ── Lookups ────────────────────────────────────────────────────────────────────


def _build_alias_index() -> dict[str, VariableSpec]:
    index: dict[str, VariableSpec] = {}
    for spec in OUTPUT_VARIABLES:
        for key in (spec.name, spec.sipnet_name, *spec.aliases):
            existing = index.get(key)
            if existing is not None and existing is not spec:
                raise ValueError(
                    f"Alias {key!r} is claimed by both {existing.name!r} and {spec.name!r}."
                )
            index[key] = spec
    return index


_ALIAS_INDEX: dict[str, VariableSpec] = _build_alias_index()


def resolve_output_variable(name: str) -> VariableSpec:
    """Return the spec for a variable given its name, an alias, or SIPNET's header token.

    Raises ``KeyError`` naming the nearest matches when nothing fits.
    """
    spec = _ALIAS_INDEX.get(name)
    if spec is not None:
        return spec
    lowered = name.lower()
    near = sorted(
        {s.name for k, s in _ALIAS_INDEX.items() if lowered in k.lower() or k.lower() in lowered}
    )
    hint = f" Did you mean one of {near}?" if near else ""
    raise KeyError(f"{name!r} is not a SIPNET output variable or alias.{hint}")


def resolve_output_variable_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Map names or aliases to canonical variable names, preserving order and dropping repeats.

    Columns written only by older SIPNET versions (:data:`LEGACY_OUTPUT_COLUMNS`) are
    accepted under either their SIPNET or their pySIPNET name.
    """
    resolved: list[str] = []
    for name in names:
        if name in LEGACY_OUTPUT_COLUMNS:
            canonical = LEGACY_OUTPUT_COLUMNS[name]
        elif name in LEGACY_OUTPUT_COLUMNS.values():
            canonical = name
        else:
            canonical = resolve_output_variable(name).name
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def parse_variable_kind(value: VariableKind | str, *, name: str) -> VariableKind:
    """*value* as a :class:`VariableKind`, refusing a string that names none.

    *name* is the variable the message names.  For reading a declared
    ``kind`` attribute; :func:`variable_kind` also falls back on the
    registries.
    """
    try:
        return VariableKind(value)
    except ValueError:
        raise ValueError(
            f"{name!r} has kind {value!r}, which is not one of {[k.value for k in VariableKind]}."
        ) from None


def variable_label(array: Any) -> str:
    """The name a DataArray goes by in pySIPNET's messages.

    Its name, or for an unnamed arithmetic result its ``derivation``
    (:mod:`pysipnet.arithmetic`), or ``"array"``.
    """
    if array.name is not None:
        return str(array.name)
    derivation = array.attrs.get("derivation")
    return derivation if isinstance(derivation, str) and derivation else "array"


_T = TypeVar("_T")
_NO_DEFAULT: Any = object()


@overload
def variable_kind(data: Any) -> VariableKind: ...


@overload
def variable_kind(data: Any, *, default: _T) -> VariableKind | _T: ...


def variable_kind(data: Any, *, default: Any = _NO_DEFAULT) -> Any:
    """The :class:`VariableKind` of a DataArray, or of a variable given by name.

    A DataArray's ``kind`` attribute wins.  Without one, its name is looked
    up, as a string is: in the output registry and then the climate registry,
    by registry name, alias or SIPNET name, so ``"nee"``, ``"NEE"`` and
    ``"net_ecosystem_exchange"`` all give ``timestep_total``.  The two
    registries share only the time columns, which have the same kind in both.
    A datetime array is never looked up: the registries' time columns are
    numbers, and ``"time"`` there is SIPNET's hour-of-day column, not
    pySIPNET's step-end ``time`` coordinate.

    Parameters
    ----------
    data:
        A DataArray, or anything with ``attrs`` and ``name``; or a variable
        name.
    default:
        Returned when neither the attributes nor the registries say what the
        variable is.  Without it, that raises.  A ``kind`` attribute that
        names no kind raises either way: it is a mistake, not an absence.

    Raises
    ------
    ValueError
        If the ``kind`` attribute is not a :class:`VariableKind` value, or,
        when no *default* is given, if the kind cannot be determined.
    """
    if isinstance(data, str):
        name: str | None = data
    else:
        name = None if data.name is None else str(data.name)
        declared = data.attrs.get("kind")
        if declared is not None:
            return parse_variable_kind(declared, name=variable_label(data))
    # The registries' time columns are numbers; a datetime array named "time" is
    # pySIPNET's step-end coordinate, not SIPNET's hour-of-day column.
    if name is not None and getattr(getattr(data, "dtype", None), "kind", None) != "M":
        for resolve in (resolve_output_variable, resolve_climate_variable):
            try:
                return resolve(name).kind
            except KeyError:
                continue
    if default is not _NO_DEFAULT:
        return default
    what = repr(data if isinstance(data, str) else variable_label(data))
    raise ValueError(
        f"Cannot tell what kind of quantity {what} is: it carries no 'kind' attribute and "
        "is not a SIPNET output or climate variable or alias. Set attrs['kind'] to one of "
        f"{[k.value for k in VariableKind]}."
    )


def check_variable_is_written(name: str, flags: ModelFlags) -> None:
    """Raise if SIPNET leaves output variable *name* as constant zero under *flags*.

    Some columns are filled only when a model option is on
    (:attr:`VariableSpec.requires_flag`): ``litter_carbon`` needs
    ``litter_pool``, the nitrogen group ``nitrogen_cycle``.  With the option
    off SIPNET still writes the column, as zeros, and a likelihood would
    consume them without complaint.  :class:`~pysipnet.output.SIPNETOutput`
    applies this to every selection; call it directly to refuse such a
    variable before any run exists.

    *name* may be a registry name, an alias or a SIPNET header token.  Columns
    only older SIPNET versions wrote (:data:`LEGACY_OUTPUT_COLUMNS`) pass,
    since no flag of the pinned version governs them.

    Raises
    ------
    KeyError
        If *name* is not an output variable or alias.
    ValueError
        If the variable needs a flag that *flags* leave off.
    """
    if name in LEGACY_OUTPUT_COLUMNS or name in LEGACY_OUTPUT_COLUMNS.values():
        return
    spec = resolve_output_variable(name)
    flag = spec.requires_flag
    if flag is not None and not getattr(flags, flag):
        raise ValueError(
            f"{spec.name!r} is constant zero with these model flags: SIPNET only fills it "
            f"when the {flag!r} flag is on, and it is off. Run with ModelFlags(..., "
            f"{flag}=True), or, if the zeros are genuinely what you want, read the raw "
            "column from the output's .pandas view."
        )


def output_variable_records() -> list[dict[str, Any]]:
    """The whole registry as JSON-serializable dicts, for documentation and export."""
    return [spec.to_record() for spec in OUTPUT_VARIABLES]


__all__ = [
    "CELL_METHODS_FOR_KIND",
    "CLIMATE_COLUMN_NAMES",
    "CLIMATE_VARIABLES",
    "CLIMATE_VARIABLES_BY_NAME",
    "ClimateVariableSpec",
    "climate_variable_records",
    "resolve_climate_variable",
    "KIND_AFTER_TIME_POWER",
    "LEGACY_OUTPUT_COLUMNS",
    "NAME_PATTERN",
    "OUTPUT_VARIABLES",
    "OUTPUT_VARIABLES_BY_NAME",
    "OUTPUT_VARIABLES_BY_SIPNET_NAME",
    "RESAMPLED_KIND",
    "RESAMPLING_METHODS_FOR_KIND",
    "ResamplingMethod",
    "TIME_COORDINATE_NAMES",
    "TIME_REFERENCE_FOR_KIND",
    "VariableKind",
    "VariableSpec",
    "check_variable_is_written",
    "output_variable_records",
    "parse_variable_kind",
    "resolve_output_variable",
    "resolve_output_variable_names",
    "variable_kind",
    "variable_label",
]
