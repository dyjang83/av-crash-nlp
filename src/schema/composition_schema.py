"""The CRSS-compatible extraction layer: SGO narratives coded the way CRSS codes.

This is the schema IIHS asked NHTSA for. Their conclusion was that manual
narrative coding does not scale and that SGO should adopt FARS/CRSS-style
fields -- `ACC_TYPE`, KABCO severity, standardized damage, and mode at impact.
This module is the automated version of that request: every field below maps
onto a named CRSS variable, so an SGO filing and a CRSS vehicle record can be
placed in one distribution without anyone hand-coding either.

-----------------------------------------------------------------------------
WHY THIS IS A SECOND SCHEMA AND NOT AN EXTENSION OF `schema.CrashExtraction`
-----------------------------------------------------------------------------
`CrashExtraction` is frozen. The lift test, the two-signal calibration study
and the distant-supervision evaluation are all computed against extractions
produced under it, and adding fields changes the constrained-decoding target,
which changes the decode-time probabilities that the calibration study
measures. A schema change would therefore invalidate published numbers for
reasons that have nothing to do with the composition question.

Running a second pass costs a second extraction, but only over the 2,353 ADS
INCIDENTS rather than the full 4,367-narrative corpus, because the composition
study is ADS-only by construction.

-----------------------------------------------------------------------------
TWO DESIGN CHOICES THAT ARE NOT NEGOTIABLE
-----------------------------------------------------------------------------
(1) ROLE COMES FROM CONTACT GEOMETRY, NOT FROM FAULT.
    `schema.CrashExtraction.contributory_party` asks the model who caused the
    crash. That field inherits the filer's framing -- the reporting entity
    wrote the narrative, and it is a party to the incident -- and this
    repository already flags it as unusable for anything but extraction
    evaluation. `striking_role` replaces it here. Front-to-rear is a fact about
    where the vehicles touched, it is structurally keyed by the SGO contact-area
    columns (`schema.acc_type.sgo_distant_key`), and it is what quasi-induced
    exposure actually needs. Nothing downstream of this module asks who was at
    fault.

(2) SENSOR-ONLY DAMAGE IS ITS OWN CATEGORY.
    An ADS vehicle carries exposed lidar, radar and camera housings that a
    human-driven car does not. A contact that scuffs a sensor pod can require a
    tow and produce a four-figure repair on a vehicle with no structural damage
    at all. Pooling that with "disabling damage" would make the ADS fleet look
    systematically more severely damaged than human vehicles in the same crash,
    and would corrupt the reportability coding, which is supposed to ignore
    repair cost. `sensor_only_damage` exists so the analysis can hold it out.

-----------------------------------------------------------------------------
THE FIELDS THE AMENDMENT TOOK AWAY
-----------------------------------------------------------------------------
`posted_speed_limit_bin` and `lighting` are extracted here even though SGO
filed both as structured columns until June 2025. The third amendment dropped
them, and they are precisely the covariates the human-side reweighting
conditions on (`human.balance`). Extracting them turns a dead end into a
validation: on the 1,384 pre-amendment incidents the structured value still
exists, so the extractor can be scored against it, and the measured accuracy is
what licenses using the extracted value on the 969 post-amendment incidents
where no structured value survives.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Crash type and role -- the CRSS ACC_TYPE collapse
# ---------------------------------------------------------------------------
class AccTypeCategory(str, Enum):
    """Mirrors `schema.acc_type.CATEGORIES` exactly. Do not diverge."""
    rear_end = "rear_end"
    sideswipe_same_direction = "sideswipe_same_direction"
    head_on = "head_on"
    sideswipe_opposite_direction = "sideswipe_opposite_direction"
    turn_across_path = "turn_across_path"
    turn_into_path = "turn_into_path"
    straight_paths_intersecting = "straight_paths_intersecting"
    backing = "backing"
    single_vehicle = "single_vehicle"
    pedestrian_animal = "pedestrian_animal"
    other_unknown = "other_unknown"


class StrikingRole(str, Enum):
    """Role of the SUBJECT (ADS) vehicle, from contact geometry.

    `undetermined` is a real answer, not a failure: CRSS's own 'Specifics
    Other/Unknown' codes occupy the same position, and forcing a choice would
    make the ADS side artificially more decided than the human side it is
    compared against.
    """
    striking = "striking"
    struck = "struck"
    undetermined = "undetermined"
    not_applicable = "not_applicable"   # no second motor vehicle


# ---------------------------------------------------------------------------
# Pre-crash maneuver -- CRSS P_CRASH1
# ---------------------------------------------------------------------------
class PreCrashMovement(str, Enum):
    """Aligned to CRSS `P_CRASH1` values, not to the SGO movement vocabulary.

    The SGO list and the CRSS list differ in ways that matter: SGO has a single
    `Stopped`, while CRSS separates `Stopped in Roadway` from `Disabled or
    "Parked" in Travel lane` and from `Starting in Road`. Extracting into the
    CRSS space and mapping SGO INTO it -- rather than the reverse -- keeps the
    finer distinctions the human side can express, and loses them only where
    the narrative genuinely does not say.
    """
    going_straight = "going_straight"
    stopped_in_roadway = "stopped_in_roadway"
    decelerating = "decelerating"
    accelerating = "accelerating"
    starting_in_road = "starting_in_road"
    turning_left = "turning_left"
    turning_right = "turning_right"
    making_u_turn = "making_u_turn"
    changing_lanes = "changing_lanes"
    merging = "merging"
    passing = "passing"
    backing = "backing"
    negotiating_curve = "negotiating_curve"
    parked = "parked"
    entering_parking = "entering_parking"
    leaving_parking = "leaving_parking"
    other = "other"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Junction -- CRSS RELJCT2 and TYP_INT
# ---------------------------------------------------------------------------
class RelationToJunction(str, Enum):
    intersection = "intersection"
    intersection_related = "intersection_related"
    driveway_access = "driveway_access"
    non_junction = "non_junction"
    unknown = "unknown"


class IntersectionType(str, Enum):
    four_way = "four_way"
    t_intersection = "t_intersection"
    y_intersection = "y_intersection"
    roundabout_traffic_circle = "roundabout_traffic_circle"
    five_or_more = "five_or_more"
    not_an_intersection = "not_an_intersection"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Severity -- KABCO, as CRSS codes it
# ---------------------------------------------------------------------------
class KABCO(str, Enum):
    """KABCO as a police officer would assign it, from the narrative.

    THE MAPPING FROM SGO IS LOSSY IN A DIRECTION WORTH NAMING. SGO's 'Highest
    Injury Severity Alleged' is ALLEGED by the reporting entity, and its
    `Moderate` grade has no KABCO counterpart. `schema.crss_map` handles the
    structured version by merging Moderate down into minor and KABCO C+B
    together. Extracting KABCO directly from the narrative avoids that merge
    where the narrative supports it, and `unknown` is used where it does not --
    which is most of the time, and is reported as such rather than defaulted to
    `O`.
    """
    K = "K"   # fatal
    A = "A"   # suspected serious injury
    B = "B"   # suspected minor injury
    C = "C"   # possible injury
    O = "O"   # no apparent injury
    unknown = "unknown"


class YesNoUnknown(str, Enum):
    yes = "yes"
    no = "no"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Damage -- CRSS DEFORMED, plus the ADS-specific category
# ---------------------------------------------------------------------------
class DamageDescriptor(str, Enum):
    """Standardized damage extent. Maps coarsely onto CRSS `DEFORMED`.

    `sensor_only` has NO CRSS counterpart and must not be mapped onto one. It
    is the category that makes the ADS/human damage comparison honest rather
    than the one that breaks it: see module docstring, design choice (2).
    """
    none = "none"
    scratch_scuff = "scratch_scuff"        # CRSS: No Damage / Minor
    panel_dent = "panel_dent"              # CRSS: Minor / Functional
    deformation = "deformation"            # CRSS: Functional
    intrusion = "intrusion"                # CRSS: Disabling
    sensor_only = "sensor_only"            # no CRSS counterpart
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Operating domain -- the covariates the amendment dropped
# ---------------------------------------------------------------------------
class SpeedLimitBin(str, Enum):
    """Binned rather than continuous, because narratives state the limit rarely.

    MEASURED RESULT: "RARELY" IS ACTUALLY "NEVER". On a 300-narrative
    validation run the extractor answered `unknown` 298 times out of 298. SGO
    narratives describe what the vehicles did, not what the sign said, and the
    posted limit essentially never appears in the text.

    This kills the plan that motivated extracting it. The third amendment
    dropped SGO's structured `Posted Speed Limit (MPH)` column, and the
    intention was to recover it from the narrative so the human-side
    reweighting could condition on speed limit across the whole period. It
    cannot be recovered. The consequence, which belongs in the limitations
    rather than in a footnote: **speed-limit conditioning is available for
    pre-amendment incidents only** (1,384 of 2,353), and any post-amendment
    analysis must either drop the covariate or restrict to the pre-amendment
    window.

    The field is retained rather than deleted because the 100%-abstention rate
    is itself the evidence for that limitation, and because a future SGO
    amendment could reinstate the column.
    """
    le_25 = "le_25"
    mph_30_35 = "mph_30_35"
    mph_40_45 = "mph_40_45"
    gt_45 = "gt_45"
    unknown = "unknown"


class LightingCondition(str, Enum):
    daylight = "daylight"
    dark_lighted = "dark_lighted"
    dark_unlighted = "dark_unlighted"
    dawn_dusk = "dawn_dusk"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Reportability -- Tier B
# ---------------------------------------------------------------------------
class Reportability(str, Enum):
    """Would a reasonable person have reported this crash to the police?

    NOT "was the filer legally obliged to report it", and NOT "was it in fact
    reported". The standard is deliberately the one IIHS used, so the marginals
    are comparable to theirs, and it deliberately codes UP to 'would report'
    rather than to 'must report' -- because the human comparator (CRSS)
    contains only crashes that actually generated a police report, and over
    half of all crashes do not.

    `maybe` is a real level and is reported both pooled with `yes` (IIHS's
    convention, which they called conservative) and separately.
    """
    yes = "yes"
    maybe = "maybe"
    no = "no"


class CompositionExtraction(BaseModel):
    """CRSS-compatible structured extraction for one ADS crash narrative."""

    # --- crash type and role ---------------------------------------------
    acc_type_category: AccTypeCategory
    striking_role: StrikingRole
    # REQUIRED, not optional. Made required after an open-weight trial run
    # returned `None` for every evidence span while still committing to a
    # label: an optional audit field is one the model is free to skip, and a
    # label whose span is absent cannot be adjudicated against the human
    # double-coders. Forcing the span also forces the model to locate the
    # narrative text it is relying on before it answers.
    role_evidence: str = Field(
        description="Short verbatim phrase from the narrative establishing "
                    "which vehicle struck which. Required.")

    # --- mode at impact ---------------------------------------------------
    sv_pre_crash_movement: PreCrashMovement
    cp_pre_crash_movement: PreCrashMovement

    # --- junction ---------------------------------------------------------
    relation_to_junction: RelationToJunction
    intersection_type: IntersectionType

    # --- outcome ----------------------------------------------------------
    kabco_severity: KABCO
    tow_away_due_to_damage: YesNoUnknown = Field(
        description="Towed BECAUSE of damage. A convenience tow is 'no' -- CRSS "
                    "separates these and the severity floor depends on it.")
    airbag_deployed: YesNoUnknown
    damage_descriptor: DamageDescriptor
    sensor_only_damage: bool = Field(
        description="True if the only damage described is to exposed sensing "
                    "hardware (lidar pod, radar, camera housing, mirror-mounted "
                    "unit) with no damage to structure, panels or glazing.")

    # --- operating domain (structured equivalents dropped in June 2025) ---
    posted_speed_limit_bin: SpeedLimitBin
    lighting: LightingCondition

    # --- reportability (Tier B) -------------------------------------------
    police_reportable: Reportability
    reportable_evidence: str = Field(
        description="Verbatim span from the narrative that drives the "
                    "reportability judgement. Required: this is what the human "
                    "double-coders adjudicate against.")
    reportable_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the reportability label specifically, "
                    "separate from overall extraction confidence.")

    # --- exclusion flags ---------------------------------------------------
    # These decide corpus membership and therefore every denominator. They are
    # extracted rather than inferred because SGO has no structured value for
    # the first two, and the structured value for the third is the filer's.
    #
    # TRI-STATE, NOT BOOLEAN. These were `bool` in the first version, and on a
    # 300-narrative validation run the model twice tried to answer "unknown"
    # for `on_public_road` and failed schema validation instead -- losing the
    # whole extraction over one field it was right to be unsure about. A
    # narrative that never says where the crash happened supports no answer,
    # and this repository's position is that abstention is not error. Forcing a
    # binary here would have made the model either fail or guess, and a guessed
    # exclusion flag silently moves every denominator downstream.
    is_true_crash: YesNoUnknown = Field(
        description="'no' if the narrative describes something that was not a "
                    "collision -- a reported contact that proved to be road "
                    "debris, a filing made out of caution with no contact, a "
                    "door opened against a stationary vehicle by a third party. "
                    "'unknown' if the narrative does not say.")
    on_public_road: YesNoUnknown = Field(
        description="'no' for private property, parking structures and depots. "
                    "'unknown' if the narrative does not locate the crash.")
    ads_engaged_at_impact: YesNoUnknown

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence that this extraction is correct.")


# Fields scored against human double-coding and, where one exists, the distant
# key. `police_reportable` heads the list because its agreement with human
# coding is the validity number the whole Tier B analysis rests on.
CATEGORICAL_FIELDS = [
    "police_reportable",
    "acc_type_category",
    "striking_role",
    "sv_pre_crash_movement",
    "cp_pre_crash_movement",
    "relation_to_junction",
    "intersection_type",
    "kabco_severity",
    "tow_away_due_to_damage",
    "airbag_deployed",
    "damage_descriptor",
    "posted_speed_limit_bin",
    "lighting",
]
CATEGORICAL_FIELDS += ["is_true_crash", "on_public_road", "ads_engaged_at_impact"]
BOOLEAN_FIELDS = ["sensor_only_damage"]
SCORED_FIELDS = CATEGORICAL_FIELDS + BOOLEAN_FIELDS

# Fields with an independent SGO structured counterpart, so they can be scored
# at corpus scale without annotation. The value is the column in the ADS
# incident frame; None means the key is derived, not a single column.
DISTANT_KEYED = {
    "acc_type_category": None,          # schema.acc_type.sgo_distant_key
    "striking_role": None,              # ditto
    "sv_pre_crash_movement": "SV Pre-Crash Movement",
    "cp_pre_crash_movement": "CP Pre-Crash Movement",
    "kabco_severity": "Highest Injury Severity Alleged",
    "tow_away_due_to_damage": "sv_towed",
    "airbag_deployed": "sv_airbag",
    "posted_speed_limit_bin": "posted_speed_limit",   # pre-amendment only
    "lighting": "lighting_struct",                    # pre-amendment only
    "ads_engaged_at_impact": "ads_engaged",
    "on_public_road": "public_road",
}

# Fields with NO structured counterpart anywhere. These are the ones that
# require human double-coding, and the sample in `annotate.make_composition_sample`
# is sized on them.
HUMAN_CODED_ONLY = ["police_reportable", "damage_descriptor",
                    "sensor_only_damage", "is_true_crash",
                    "relation_to_junction", "intersection_type"]


def json_schema() -> dict:
    return CompositionExtraction.model_json_schema()


if __name__ == "__main__":
    import json
    print(json.dumps(json_schema(), indent=2))
