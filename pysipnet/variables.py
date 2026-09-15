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
``day_of_year``, ``hour_of_day``), but pools are written **after** the step
has been applied, so a ``STATE`` value is the pool at the *end* of the step.
``FLUX`` values are integrals over the step, ``MEAN`` values are means over
it, the single ``RATE`` value is a per-day rate for the step, and
``CUMULATIVE`` values run from the start of the simulation to the end of the
step (and continue across a restart, which carries them in the checkpoint).
Each spec's :attr:`VariableSpec.time_reference` states this in words, and the
same text travels as an attribute on the xarray representation so a user never
has to look it up.

Climate drivers
---------------
:data:`CLIMATE_VARIABLES` describes the columns of the ``.clim`` input file the
same way. Each :class:`ClimateVariableSpec` also records the units SIPNET
converts the value to on read (``internal_units``), because SIPNET's own
documentation quotes those rather than the file units. Climate rows follow the
same time convention: ``year`` / ``day_of_year`` / ``hour_of_day`` are the
start of the step, means are over the step, and ``photosynthetically_active_radiation``
and ``precipitation`` are totals over the step.

Precision
---------
SIPNET prints every column with a fixed number of decimals (``%8.3f`` and the
like), recorded in :attr:`VariableSpec.output_decimals`.  Carbon fluxes get
three decimals, so a half-hourly NEE of 0.05 g C m⁻² carries a 1 %
quantisation floor.  That matters when the output feeds a likelihood.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

from pysipnet.units import UnitStyle, format_units, validate_units

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
"""What a pySIPNET variable name must look like."""


class VariableKind(StrEnum):
    """What a value in a column represents, relative to the timestep it is labelled with."""

    COORDINATE = "coordinate"
    """Identifies the row: the start of the timestep."""

    STATE = "state"
    """A pool, reported at the end of the timestep."""

    FLUX = "flux"
    """A transfer integrated over the timestep."""

    RATE = "rate"
    """A per-day transfer rate that applied during the timestep."""

    MEAN = "mean"
    """A quantity averaged over the timestep."""

    CUMULATIVE = "cumulative"
    """A running total from the start of the simulation to the end of the timestep."""


class Aggregation(StrEnum):
    """How to combine consecutive values when resampling to a coarser timestep."""

    SUM = "sum"
    MEAN = "mean"
    LAST = "last"
    NONE = "none"


_AGGREGATION_FOR_KIND: dict[VariableKind, Aggregation] = {
    VariableKind.COORDINATE: Aggregation.NONE,
    VariableKind.STATE: Aggregation.MEAN,
    VariableKind.FLUX: Aggregation.SUM,
    VariableKind.RATE: Aggregation.MEAN,
    VariableKind.MEAN: Aggregation.MEAN,
    VariableKind.CUMULATIVE: Aggregation.LAST,
}

_TIME_REFERENCE_FOR_KIND: dict[VariableKind, str] = {
    VariableKind.COORDINATE: "start of the timestep",
    VariableKind.STATE: "value at the end of the timestep",
    VariableKind.FLUX: "total over the timestep",
    VariableKind.RATE: "per-day rate during the timestep",
    VariableKind.MEAN: "mean over the timestep",
    VariableKind.CUMULATIVE: "running total from the start of the run to the end of the timestep",
}

# ``cell_methods`` vocabulary from the Climate and Forecast (CF) conventions.
_CELL_METHODS_FOR_KIND: dict[VariableKind, str | None] = {
    VariableKind.COORDINATE: None,
    VariableKind.STATE: "time: point",
    VariableKind.FLUX: "time: sum",
    VariableKind.RATE: "time: mean",
    VariableKind.MEAN: "time: mean",
    VariableKind.CUMULATIVE: "time: sum",
}


@dataclass(frozen=True)
class VariableSpec:
    """Everything a consumer needs to know about one output variable."""

    name: str
    """pySIPNET name; the column name in every DataFrame and Dataset."""

    sipnet_name: str
    """The exact header token SIPNET writes."""

    kind: VariableKind
    """Pool, flux, rate, mean, cumulative total or coordinate."""

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

    def __post_init__(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"Variable name {self.name!r} violates the naming convention.")
        validate_units(self.units)
        if not self.description or not self.long_label:
            raise ValueError(f"Variable {self.name!r} needs a description and a long_label.")

    @property
    def aggregation(self) -> Aggregation:
        """Default rule for resampling this variable to a coarser timestep."""
        return _AGGREGATION_FOR_KIND[self.kind]

    @property
    def time_reference(self) -> str:
        """In words, which moment or interval of the timestep the value refers to."""
        return _TIME_REFERENCE_FOR_KIND[self.kind]

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
            "aggregation": self.aggregation.value,
            "sipnet_name": self.sipnet_name,
        }
        if self.constituent:
            attrs["constituent"] = self.constituent
        cell_methods = _CELL_METHODS_FOR_KIND[self.kind]
        if cell_methods is not None:
            attrs["cell_methods"] = cell_methods
        if self.sign_convention:
            attrs["sign_convention"] = self.sign_convention
        if self.requires_flag is not None:
            attrs["requires_flag"] = self.requires_flag
        if self.output_decimals is not None:
            attrs["output_decimals"] = self.output_decimals
        return attrs

    def to_record(self) -> dict[str, Any]:
        """A plain, JSON-serialisable dict of every field plus the derived properties."""
        record = asdict(self)
        record["kind"] = self.kind.value
        record["aliases"] = list(self.aliases)
        record["aggregation"] = self.aggregation.value
        record["time_reference"] = self.time_reference
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
        kind=VariableKind.COORDINATE,
        units="1",
        description="Calendar year at the start of the timestep.",
        long_label="Year",
        group="time",
    ),
    _spec(
        name="day_of_year",
        sipnet_name="day",
        kind=VariableKind.COORDINATE,
        units="1",
        description="Day of year at the start of the timestep; 1 is January 1st.",
        long_label="Day of year",
        short_label="DOY",
        aliases=("day", "doy"),
        group="time",
    ),
    _spec(
        name="hour_of_day",
        sipnet_name="time",
        kind=VariableKind.COORDINATE,
        units="h",
        description="Hours after midnight at the start of the timestep; may be fractional.",
        long_label="Hour of day",
        aliases=("time",),
        output_decimals=2,
        group="time",
    ),
    _spec(
        name="wood_carbon",
        sipnet_name="plantWoodC",
        kind=VariableKind.STATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.FLUX,
        **_GC,
        description="Carbon allocated to the wood pool over the timestep.",
        long_label="Wood growth",
        aliases=("wood_creation", "woodCreation"),
        output_decimals=2,
        group="carbon_fluxes",
    ),
    _spec(
        name="soil_carbon",
        sipnet_name="soil",
        kind=VariableKind.STATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.MEAN,
        units="1",
        description=(
            "Soil water as a fraction of water holding capacity, averaged over the "
            "timestep as the mean of the start and end values. Diagnostic only: SIPNET's "
            "moisture functions use the instantaneous fraction."
        ),
        long_label="Soil wetness fraction",
        aliases=("soil_wetness_frac", "soilWetnessFrac"),
        output_decimals=3,
        group="water",
    ),
    _spec(
        name="snow_water_equivalent",
        sipnet_name="snow",
        kind=VariableKind.STATE,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
        **_GC,
        description=(
            "Net carbon flux between ecosystem and atmosphere over the timestep: "
            "ecosystem respiration minus gross primary production."
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
        output_decimals=3,
        sign_convention="positive is a net loss of carbon from the ecosystem to the atmosphere",
        group="carbon_fluxes",
    ),
    _spec(
        name="gross_primary_production",
        sipnet_name="gpp",
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.RATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.STATE,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.FLUX,
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
        kind=VariableKind.STATE,
        **_GC,
        description=(
            "Storage-lag component of wood_carbon: the difference between carbon gained "
            "this step (photosynthesis minus autotrophic respiration) and carbon allocated "
            "to growth from the five-day mean NPP, accumulated over the run. Can be negative. "
            "SIPNET: plantCAccountingDelta."
        ),
        long_label="Wood storage carbon",
        aliases=("npp_storage", "nppStorage"),
        output_decimals=4,
        group="carbon_pools",
    ),
)
"""Every column SIPNET writes at the pinned version, in the order it writes them."""

TIME_COORDINATE_NAMES: tuple[str, ...] = tuple(
    v.name for v in OUTPUT_VARIABLES if v.kind == VariableKind.COORDINATE
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
        kind=VariableKind.COORDINATE,
        units="1",
        description="Calendar year at the start of the timestep.",
        long_label="Year",
        group="time",
    ),
    _climate(
        name="day_of_year",
        sipnet_name="day",
        kind=VariableKind.COORDINATE,
        units="1",
        description="Day of year at the start of the timestep; 1 is January 1st.",
        long_label="Day of year",
        short_label="DOY",
        aliases=("day", "doy"),
        group="time",
    ),
    _climate(
        name="hour_of_day",
        sipnet_name="time",
        kind=VariableKind.COORDINATE,
        units="h",
        description="Hours after midnight at the start of the timestep; may be fractional.",
        long_label="Hour of day",
        aliases=("time",),
        group="time",
    ),
    _climate(
        name="time_step_length",
        sipnet_name="length",
        kind=VariableKind.FLUX,
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
        kind=VariableKind.MEAN,
        units="degC",
        description="Mean air temperature over the timestep.",
        long_label="Air temperature",
        aliases=("tair",),
        group="meteorology",
    ),
    _climate(
        name="soil_temperature",
        sipnet_name="tsoil",
        kind=VariableKind.MEAN,
        units="degC",
        description="Mean soil temperature over the timestep.",
        long_label="Soil temperature",
        aliases=("tsoil",),
        group="meteorology",
    ),
    _climate(
        name="photosynthetically_active_radiation",
        sipnet_name="par",
        kind=VariableKind.FLUX,
        units="mol m-2",
        constituent="photons",
        description="Photosynthetically active radiation summed over the timestep, as moles "
        "of photons per square metre of ground (1 Einstein = 1 mol). To convert an "
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
        kind=VariableKind.FLUX,
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
        name="vapour_pressure_deficit",
        sipnet_name="vpd",
        kind=VariableKind.MEAN,
        units="Pa",
        description="Mean vapour pressure deficit of the air over the timestep. SIPNET clamps "
        "values below a tiny positive number up to it.",
        long_label="Vapour pressure deficit",
        short_label="VPD",
        aliases=("vpd",),
        internal_units="kPa",
        internal_conversion="× 0.001 (Pa → kPa), clamped ≥ 1e-6",
        group="meteorology",
    ),
    _climate(
        name="soil_vapour_pressure_deficit",
        sipnet_name="vpdSoil",
        kind=VariableKind.MEAN,
        units="Pa",
        description="Mean vapour pressure deficit between the soil and the air over the "
        "timestep, using saturation vapour pressure at the soil temperature.",
        long_label="Soil vapour pressure deficit",
        aliases=("vpd_soil", "vpdSoil"),
        internal_units="kPa",
        internal_conversion="× 0.001 (Pa → kPa)",
        group="meteorology",
    ),
    _climate(
        name="vapour_pressure",
        sipnet_name="vPress",
        kind=VariableKind.MEAN,
        units="Pa",
        description="Mean vapour pressure in the canopy airspace over the timestep.",
        long_label="Vapour pressure",
        aliases=("vpress", "vPress"),
        internal_units="kPa",
        internal_conversion="× 0.001 (Pa → kPa)",
        group="meteorology",
    ),
    _climate(
        name="wind_speed",
        sipnet_name="wspd",
        kind=VariableKind.MEAN,
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
    """The climate registry as JSON-serialisable dicts, for documentation and export."""
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


def output_variable_records() -> list[dict[str, Any]]:
    """The whole registry as JSON-serialisable dicts, for documentation and export."""
    return [spec.to_record() for spec in OUTPUT_VARIABLES]


__all__ = [
    "Aggregation",
    "CLIMATE_COLUMN_NAMES",
    "CLIMATE_VARIABLES",
    "CLIMATE_VARIABLES_BY_NAME",
    "ClimateVariableSpec",
    "climate_variable_records",
    "resolve_climate_variable",
    "LEGACY_OUTPUT_COLUMNS",
    "NAME_PATTERN",
    "OUTPUT_VARIABLES",
    "OUTPUT_VARIABLES_BY_NAME",
    "OUTPUT_VARIABLES_BY_SIPNET_NAME",
    "TIME_COORDINATE_NAMES",
    "VariableKind",
    "VariableSpec",
    "output_variable_records",
    "resolve_output_variable",
    "resolve_output_variable_names",
]
