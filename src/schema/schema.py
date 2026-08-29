"""Extraction schema for AV crash-narrative information extraction.

This module defines the structured representation that the LLM pipeline extracts
from each free-text collision narrative. Field design is grounded in NHTSA's
pre-crash typology (DOT HS 811 366) and the SGO 2021-01 data dictionary so the
labels are interpretable and map onto existing structured fields where possible.

IMPORTANT (research-integrity note):
    `contributory_party` is an EXTRACTION target only. It is validated against
    human coding in the extraction task. It is NEVER used as a target in the
    downstream lift test, because its only source is the narrative itself, which
    would make any text->fault prediction circular. See src/models/lift_test.py.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CollisionType(str, Enum):
    rear_end = "rear_end"
    cross_path = "cross_path"          # turning / crossing paths at junction
    sideswipe = "sideswipe"
    head_on = "head_on"
    backing = "backing"
    vru = "vru"                        # vulnerable road user (ped/cyclist/etc.)
    single_vehicle = "single_vehicle"  # object/road departure, no second party
    other = "other"
    unknown = "unknown"


class PreCrashManeuver(str, Enum):
    stopped = "stopped"
    proceeding_straight = "proceeding_straight"
    decelerating = "decelerating"
    accelerating = "accelerating"
    turning_left = "turning_left"
    turning_right = "turning_right"
    lane_change = "lane_change"
    backing = "backing"
    parked = "parked"
    other = "other"
    unknown = "unknown"


class ContributoryParty(str, Enum):
    """Narrative-described contributory party. NOT adjudicated legal fault."""
    av = "av"                  # narrative attributes the precipitating action to the AV
    other_party = "other_party"
    shared = "shared"
    ambiguous = "ambiguous"    # narrative does not permit a determination
    not_applicable = "not_applicable"


class EngagementState(str, Enum):
    ads_engaged = "ads_engaged"            # SAE L3-5 automated driving system active
    adas_engaged = "adas_engaged"          # SAE L2 driver assistance active
    manual = "manual"                      # human in control at time of incident
    disengaged_prior = "disengaged_prior"  # disengaged within seconds before crash
    unknown = "unknown"


class Lighting(str, Enum):
    daylight = "daylight"
    dark_lighted = "dark_lighted"
    dark_unlighted = "dark_unlighted"
    dawn_dusk = "dawn_dusk"
    unknown = "unknown"


class Weather(str, Enum):
    clear = "clear"
    rain = "rain"
    fog = "fog"
    snow = "snow"
    other = "other"
    unknown = "unknown"


class RoadClass(str, Enum):
    surface_street = "surface_street"
    arterial = "arterial"
    highway_freeway = "highway_freeway"
    parking_lot_private = "parking_lot_private"
    other = "other"
    unknown = "unknown"


class Locality(str, Enum):
    intersection = "intersection"
    intersection_related = "intersection_related"
    segment = "segment"          # mid-block / road segment, not at a junction
    driveway = "driveway"
    other = "other"
    unknown = "unknown"


class NarrativeInjurySeverity(str, Enum):
    """Severity AS DESCRIBED IN THE NARRATIVE.

    Kept deliberately separate from the structured SGO 'Highest Injury Severity
    Alleged' field, which is the lift-test target. Do not merge the two.
    """
    none = "none"
    minor = "minor"
    moderate = "moderate"
    serious = "serious"
    fatal = "fatal"
    unknown = "unknown"


class CrashExtraction(BaseModel):
    """Structured extraction for a single collision narrative."""

    collision_type: CollisionType
    subject_pre_crash_maneuver: PreCrashManeuver
    other_party_present: bool = Field(
        description="True if a second road user (vehicle, VRU, etc.) is involved."
    )
    contributory_party: ContributoryParty
    engagement_state: EngagementState

    lighting: Lighting
    weather: Weather
    road_class: RoadClass
    locality: Locality

    narrative_injury_severity: NarrativeInjurySeverity
    av_moving: bool = Field(
        description="True if the subject AV was in motion (speed > 0) at impact."
    )

    # Free-text evidence span supporting contributory_party, for auditability.
    contributory_evidence: Optional[str] = Field(
        default=None,
        description="Short verbatim phrase from the narrative justifying the "
        "contributory_party label. Used for error analysis, not as a feature.",
    )

    # Model self-reported confidence in {0,1} for calibration analysis.
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model's calibrated confidence that this extraction is correct.",
    )


# Fields scored for inter-annotator and model-vs-human agreement.
CATEGORICAL_FIELDS = [
    "collision_type",
    "subject_pre_crash_maneuver",
    "contributory_party",
    "engagement_state",
    "lighting",
    "weather",
    "road_class",
    "locality",
    "narrative_injury_severity",
]
BOOLEAN_FIELDS = ["other_party_present", "av_moving"]
SCORED_FIELDS = CATEGORICAL_FIELDS + BOOLEAN_FIELDS


def json_schema() -> dict:
    """Return the JSON schema used for constrained decoding / tool calling."""
    return CrashExtraction.model_json_schema()


if __name__ == "__main__":
    import json
    print(json.dumps(json_schema(), indent=2))
