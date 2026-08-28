"""Agronomic event data structures and ``events.in`` file I/O.

pySIPNET writes five of the event types SIPNET accepts that alter model state at a specified
(year, day) during a simulation run.

Event file format
-----------------
``events.in`` is space-delimited, one event per line::

    year  day  type  param1 [param2 ...]

The event type token and its parameters:

+---------------+--------+----------------------------------------------------+
| Type token    | Params | Description                                        |
+===============+========+====================================================+
| ``harv``      | fRA fRB fTA fTB | Harvest: fractions removed/transferred    |
+---------------+--------+----------------------------------------------------+
| ``irrig``     | amount method   | Irrigation: cm water, method int (0–1)    |
+---------------+--------+----------------------------------------------------+
| ``fert``      | orgN orgC minN  | Fertilization: g m⁻²                      |
+---------------+--------+----------------------------------------------------+
| ``plant``     | leafC woodC fineRootC coarseRootC | Planting: g C m⁻²    |
+---------------+--------+----------------------------------------------------+
| ``till``      | tillageEffect            | Tillage: decomposition-rate boost      |
+---------------+--------+----------------------------------------------------+

Events must be listed in chronological order (SIPNET errors otherwise).

Inline comments using ``#`` are permitted — ``sscanf`` stops reading at
non-numeric characters, so trailing ``# text`` is silently ignored.

Serialisation
-------------
All event classes are frozen Pydantic models, so a complete event sequence
can be round-tripped through ``model_dump`` / ``model_validate`` for storage
as JSON or YAML alongside other run metadata.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Irrigation method enum
# ---------------------------------------------------------------------------


class IrrigationMethod(IntEnum):
    """Water delivery method for irrigation events.

    Integer values match SIPNET's internal encoding.
    """

    CANOPY = 0
    """Water applied to the canopy (intercepted before reaching soil)."""

    SOIL = 1
    """Water applied directly to the soil surface."""


# ---------------------------------------------------------------------------
# Individual event models
# ---------------------------------------------------------------------------


class HarvestEvent(BaseModel):
    """Remove and/or redistribute aboveground and belowground biomass."""

    model_config = ConfigDict(frozen=True)

    type: Literal["harvest"] = "harvest"
    year: int = Field(gt=0)
    day: int = Field(ge=1, le=366)
    fraction_removed_above: float = Field(
        ge=0,
        le=1,
        description="Fraction of aboveground C removed from the system.",
    )
    fraction_removed_below: float = Field(
        ge=0,
        le=1,
        description="Fraction of belowground C removed from the system.",
    )
    fraction_transferred_above: float = Field(
        ge=0,
        le=1,
        description="Fraction of aboveground C transferred to the surface litter pool.",
    )
    fraction_transferred_below: float = Field(
        ge=0,
        le=1,
        description="Fraction of belowground C transferred to the soil litter pool.",
    )

    @model_validator(mode="after")
    def _check_fractions(self) -> HarvestEvent:
        if self.fraction_removed_above + self.fraction_transferred_above > 1:
            raise ValueError("fraction_removed_above + fraction_transferred_above must be ≤ 1")
        if self.fraction_removed_below + self.fraction_transferred_below > 1:
            raise ValueError("fraction_removed_below + fraction_transferred_below must be ≤ 1")
        return self

    def _to_line(self) -> str:
        return (
            f"{self.year}  {self.day}  harv"
            f"  {self.fraction_removed_above}"
            f"  {self.fraction_removed_below}"
            f"  {self.fraction_transferred_above}"
            f"  {self.fraction_transferred_below}"
        )


class IrrigationEvent(BaseModel):
    """Add water to the soil or canopy."""

    model_config = ConfigDict(frozen=True)

    type: Literal["irrigation"] = "irrigation"
    year: int = Field(gt=0)
    day: int = Field(ge=1, le=366)
    amount: float = Field(
        gt=0,
        description="Water added (cm).",
    )
    method: IrrigationMethod = Field(
        description="Delivery method: CANOPY (0) or SOIL (1).",
    )

    def _to_line(self) -> str:
        return f"{self.year}  {self.day}  irrig  {self.amount}  {self.method.value}"


class FertilizationEvent(BaseModel):
    """Add organic and/or mineral nitrogen to the system."""

    model_config = ConfigDict(frozen=True)

    type: Literal["fertilization"] = "fertilization"
    year: int = Field(gt=0)
    day: int = Field(ge=1, le=366)
    org_n: float = Field(ge=0, description="Organic N added (g N m⁻²).")
    org_c: float = Field(ge=0, description="Organic C added (g C m⁻²).")
    min_n: float = Field(ge=0, description="Mineral N added (g N m⁻²).")

    def _to_line(self) -> str:
        return f"{self.year}  {self.day}  fert  {self.org_n}  {self.org_c}  {self.min_n}"


class PlantingEvent(BaseModel):
    """Introduce plant biomass (crop emergence or transplanting)."""

    model_config = ConfigDict(frozen=True)

    type: Literal["planting"] = "planting"
    year: int = Field(gt=0)
    day: int = Field(ge=1, le=366)
    leaf_c: float = Field(ge=0, description="Leaf C added (g C m⁻²).")
    wood_c: float = Field(ge=0, description="Wood C added (g C m⁻²).")
    fine_root_c: float = Field(ge=0, description="Fine root C added (g C m⁻²).")
    coarse_root_c: float = Field(ge=0, description="Coarse root C added (g C m⁻²).")

    def _to_line(self) -> str:
        return (
            f"{self.year}  {self.day}  plant"
            f"  {self.leaf_c}  {self.wood_c}"
            f"  {self.fine_root_c}  {self.coarse_root_c}"
        )


class TillageEvent(BaseModel):
    """Disturb the soil, temporarily speeding up decomposition.

    Tillage raises the heterotrophic respiration rate by ``tillage_effect``,
    and that boost then decays exponentially back to nothing. Several tillage
    events add together.

    Takes a single value. Earlier SIPNET versions took three
    (``fractionLitterTransferred``, ``somDecompModifier``,
    ``litterDecompModifier``); v2.1.0 replaced them with this one, and
    SIPNET reads only the first value on the line, so writing the old three
    silently fed a litter fraction in as the decomposition boost.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tillage"] = "tillage"
    year: int = Field(gt=0)
    day: int = Field(ge=1, le=366)
    tillage_effect: float = Field(
        ge=0,
        description="How much this tillage raises the decomposition rate "
        "(SIPNET param: tillageEffect). The rate is scaled by "
        "1 + the accumulated effect, so 0 means no change. The boost decays "
        "exponentially and is dropped once it falls below 0.01.",
    )

    def _to_line(self) -> str:
        return f"{self.year}  {self.day}  till  {self.tillage_effect}"


# ---------------------------------------------------------------------------
# Union type alias
# ---------------------------------------------------------------------------

AnyEvent = Annotated[
    HarvestEvent | IrrigationEvent | FertilizationEvent | PlantingEvent | TillageEvent,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Event sequence container
# ---------------------------------------------------------------------------

_TOKEN_TO_TYPE = {
    "harv": "harvest",
    "irrig": "irrigation",
    "fert": "fertilization",
    "plant": "planting",
    "till": "tillage",
}

_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "harvest": HarvestEvent,
    "irrigation": IrrigationEvent,
    "fertilization": FertilizationEvent,
    "planting": PlantingEvent,
    "tillage": TillageEvent,
}


class EventSequence(BaseModel):
    """An ordered sequence of agronomic events for a single SIPNET run.

    Events must be in chronological order — SIPNET errors at runtime if they
    are not.  :class:`EventSequence` enforces this at construction time.

    Example::

        from pysipnet.events import EventSequence, IrrigationEvent, IrrigationMethod

        events = EventSequence(events=[
            IrrigationEvent(year=2020, day=150, amount=5.0,
                            method=IrrigationMethod.SOIL),
        ])
        events.to_file("events.in")
    """

    events: list[AnyEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_chronological(self) -> EventSequence:
        for i in range(1, len(self.events)):
            prev = self.events[i - 1]
            curr = self.events[i]
            if (curr.year, curr.day) < (prev.year, prev.day):
                raise ValueError(
                    f"Events must be in chronological order. "
                    f"Event {i} ({curr.year}, {curr.day}) precedes "
                    f"event {i - 1} ({prev.year}, {prev.day})."
                )
        return self

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_file(self, path: str | Path) -> None:
        """Write the event sequence to a SIPNET ``events.in`` file."""
        lines = [e._to_line() for e in self.events]
        Path(path).write_text("\n".join(lines) + "\n" if lines else "")

    @classmethod
    def from_file(cls, path: str | Path) -> EventSequence:
        """Read a SIPNET ``events.in`` file into an :class:`EventSequence`.

        Lines starting with ``#`` and blank lines are skipped.
        Inline comments after the parameter values are silently ignored.
        """
        events: list[AnyEvent] = []
        for raw_line in Path(path).read_text().splitlines():
            line = raw_line.split("#")[0].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 3:
                raise ValueError(f"Malformed event line: {raw_line!r}")
            year, day, type_token = int(tokens[0]), int(tokens[1]), tokens[2]
            event_type = _TOKEN_TO_TYPE.get(type_token)
            if event_type is None:
                raise ValueError(f"Unknown event type token {type_token!r} on line: {raw_line!r}")
            model_cls = _TYPE_TO_MODEL[event_type]
            params = _parse_params(event_type, year, day, tokens[3:])
            events.append(cast("AnyEvent", model_cls.model_validate(params)))
        return cls(events=events)

    # ── Properties ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.type] = counts.get(e.type, 0) + 1
        summary = ", ".join(f"{v}×{k}" for k, v in counts.items())
        return f"EventSequence({len(self.events)} events: {summary})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# How many values each event type carries after "year day <type>".
#
# SIPNET checks these too, but only that it read *at least* enough: sscanf
# stops when it has filled its arguments, so a line with extra values is
# accepted and the surplus discarded. Reading one here without checking would
# lose the same values just as quietly, so the count is checked exactly.
#
# These match NUM_*_PARAMS in sipnet/src/sipnet/events.h at the pinned version.
EVENT_ARITY: dict[str, int] = {
    "harvest": 4,
    "irrigation": 2,
    "fertilization": 3,
    "planting": 4,
    "tillage": 1,
}


def _parse_params(event_type: str, year: int, day: int, tokens: list[str]) -> dict[str, object]:
    """Convert raw token list to a kwarg dict for the appropriate event model."""
    base = {"year": year, "day": day}

    expected = EVENT_ARITY.get(event_type)
    if expected is not None and len(tokens) != expected:
        raise ValueError(
            f"{event_type} event at ({year}, {day}) has {len(tokens)} parameter(s), "
            f"expected {expected}. SIPNET would read the first {expected} and "
            f"discard the rest without complaining, so this is rejected here instead. "
            f"Values found: {' '.join(tokens) or '(none)'}"
        )
    try:
        if event_type == "harvest":
            fRA, fRB, fTA, fTB = (float(t) for t in tokens[:4])
            return {
                **base,
                "fraction_removed_above": fRA,
                "fraction_removed_below": fRB,
                "fraction_transferred_above": fTA,
                "fraction_transferred_below": fTB,
            }
        if event_type == "irrigation":
            return {**base, "amount": float(tokens[0]), "method": int(tokens[1])}
        if event_type == "fertilization":
            return {
                **base,
                "org_n": float(tokens[0]),
                "org_c": float(tokens[1]),
                "min_n": float(tokens[2]),
            }
        if event_type == "planting":
            return {
                **base,
                "leaf_c": float(tokens[0]),
                "wood_c": float(tokens[1]),
                "fine_root_c": float(tokens[2]),
                "coarse_root_c": float(tokens[3]),
            }
        if event_type == "tillage":
            return {
                **base,
                "tillage_effect": float(tokens[0]),
            }
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Could not parse params for {event_type} event at ({year}, {day}): {exc}"
        ) from exc
    raise ValueError(f"Unknown event type: {event_type!r}")  # unreachable
