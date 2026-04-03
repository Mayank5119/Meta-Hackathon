"""
Disruption definitions and scripted disruption sequences per task level.
Disruptions are pre-scripted (not random) for deterministic reproducibility.
Seed-based variation available for secondary runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from env.models import DisruptionType


@dataclass
class DisruptionEvent:
    """A disruption that fires on a specific simulation day."""

    id: str
    type: DisruptionType
    affected_task_id: str
    delay_days: int
    fire_on_day: int           # day ≥ this triggers the event
    description: str
    resolved: bool = False
    active: bool = False


DISRUPTION_DESCRIPTIONS = {
    DisruptionType.WEATHER: "Heavy rainfall halts outdoor work.",
    DisruptionType.MATERIAL_SHORTAGE: "Delayed material delivery stalls the crew.",
    DisruptionType.EQUIPMENT_FAILURE: "Critical equipment malfunction requires repair.",
    DisruptionType.LABOR_SHORTAGE: "Crew absence due to illness or strike.",
}


def easy_disruptions() -> List[DisruptionEvent]:
    """
    1 weather event hitting structural framing.
    Optimal response: expedite T3 (framing) to absorb delay.
    """
    return [
        DisruptionEvent(
            id="D1",
            type=DisruptionType.WEATHER,
            affected_task_id="T3",
            delay_days=3,
            fire_on_day=8,
            description="Heavy rain delays structural framing by 3 days.",
        )
    ]


def medium_disruptions() -> List[DisruptionEvent]:
    """
    3 disruptions across critical and non-critical paths.
    Requires prioritisation: identify which delays cascade.
    """
    return [
        DisruptionEvent(
            id="D1",
            type=DisruptionType.WEATHER,
            affected_task_id="T3",
            delay_days=3,
            fire_on_day=8,
            description="Rainfall halts structural framing (critical path).",
        ),
        DisruptionEvent(
            id="D2",
            type=DisruptionType.MATERIAL_SHORTAGE,
            affected_task_id="T5",
            delay_days=4,
            fire_on_day=17,
            description="Electrical conduit delivery delayed — rough-in stalled.",
        ),
        DisruptionEvent(
            id="D3",
            type=DisruptionType.EQUIPMENT_FAILURE,
            affected_task_id="T6",
            delay_days=2,
            fire_on_day=19,
            description="Pipe threading machine failure delays plumbing rough-in.",
        ),
    ]


def hard_disruptions() -> List[DisruptionEvent]:
    """
    5 disruptions with overlapping effects, budget pressure.
    Critical path disruptions AND non-critical that become critical after delay.
    """
    return [
        DisruptionEvent(
            id="D1",
            type=DisruptionType.WEATHER,
            affected_task_id="T2",
            delay_days=4,
            fire_on_day=5,
            description="Severe storm delays foundation pouring (critical path).",
        ),
        DisruptionEvent(
            id="D2",
            type=DisruptionType.EQUIPMENT_FAILURE,
            affected_task_id="T3",
            delay_days=3,
            fire_on_day=16,
            description="Crane malfunction halts structural framing.",
        ),
        DisruptionEvent(
            id="D3",
            type=DisruptionType.MATERIAL_SHORTAGE,
            affected_task_id="T5",
            delay_days=5,
            fire_on_day=28,
            description="Electrical panel shortage — systems installation blocked.",
        ),
        DisruptionEvent(
            id="D4",
            type=DisruptionType.LABOR_SHORTAGE,
            affected_task_id="T6",
            delay_days=3,
            fire_on_day=30,
            description="HVAC crew strike delays plumbing & HVAC installation.",
        ),
        DisruptionEvent(
            id="D5",
            type=DisruptionType.WEATHER,
            affected_task_id="T9",
            delay_days=2,
            fire_on_day=42,
            description="Late-season rain delays exterior landscaping.",
        ),
    ]


DISRUPTION_FACTORIES = {
    "easy":   easy_disruptions,
    "medium": medium_disruptions,
    "hard":   hard_disruptions,
}
