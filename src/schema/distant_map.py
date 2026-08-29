"""Distant-supervision label mapping: regulator-filed structured fields -> schema enums.

MOTIVATION
    Both source corpora already carry reporter-filed structured codes for
    several of the fields the extraction schema targets. SGO's incident CSVs
    code lighting, weather, roadway type, pre-crash movement, crash partner and
    pre-crash speed; OL 316's fillable PDF carries the equivalent checkboxes.
    These are labels produced by the reporting entity at filing time -- not by
    us, not by a model -- so they can serve as a large, independent answer key
    for extraction accuracy at corpus scale rather than at hand-annotation
    scale.

GRANULARITY IS THE WHOLE PROBLEM
    The structured codes and the schema enums do not agree on granularity, and
    they disagree in BOTH directions:

      - The structured code is coarser. SGO records "Dark - Unknown Lighting",
        which cannot distinguish the schema's `dark_lighted` from
        `dark_unlighted`. Scoring the model's `dark_lighted` as wrong here would
        penalize it for a distinction the answer key cannot make.

      - The schema is coarser. SGO records "Merging", which has no schema
        counterpart; forcing it to `other` would penalize a model that
        reasonably answered `lane_change`.

    Both directions are handled the same way: every field defines a *reducer*
    that receives the structured value AND the model value and returns the pair
    projected into the coarsest space both can express, or None to drop the row
    from that field's evaluation entirely. Dropping is the default whenever a
    mapping would require inventing a convention. Coverage is reported per
    field so the reader can see exactly how much of the corpus each number
    rests on -- a field evaluated on 3198 records and a field evaluated on 214
    are not the same evidence, and the tables say so.

WHAT THIS DELIBERATELY DOES NOT COVER
    `contributory_party` has no structured counterpart in either corpus. It is
    the field with the lowest model agreement and the most human disagreement,
    and it stays a human-annotated task. Distant supervision replaces the
    convention-driven fields, not the judgment-driven one.

RESEARCH-INTEGRITY NOTE
    `struct_severity` is the lift-test target. It appears here ONLY as a
    diagnostic answer key for the narrative-vs-structured severity concordance
    (Section: narrative severity), and is reported separately from extraction
    accuracy because `narrative_injury_severity` is defined as severity AS
    DESCRIBED IN THE NARRATIVE, which is a different quantity from the
    structured allegation. It must never enter the lift-test feature matrix.
    See src/models/lift_test.py. The lift test itself is restricted to SGO
    rows (features/build_features.py filters on source=="sgo") because it
    needs the full none/minor/moderate/serious/fatal ordinal scale; OL 316
    also populates `struct_severity` now (see parse_ol316.py's
    _ol316_injury_flag), but only at the coarser none/some_injury/fatal
    resolution its form supports, so it is used here in the diagnostic only,
    never in the lift test's target vector.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------

# SGO CSVs are read with keep_default_na=False, so empty cells arrive as "" or
# " ". Rows sourced from a CSV that lacks a given column arrive as float NaN
# after the concat in _load_sgo_frame. Both are missing; neither is a label.
# NOTE: float NaN is truthy in Python, so `value or None` does NOT filter it.
_MISSING_TOKENS = {"", "unknown", "unknown, see narrative", "not reported",
                   "n/a", "na", "none reported", "other, see narrative"}


def is_missing(v) -> bool:
    """Missingness on the STRUCTURED side: no label was filed.

    "Unknown" and "Other, see Narrative" count as missing here because the
    reporting entity declined to code the field, so there is nothing to score
    against.
    """
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str):
        return v.strip().lower() in _MISSING_TOKENS
    return False


def is_missing_pred(v) -> bool:
    """Missingness on the PREDICTION side: the model produced no value at all.

    Deliberately narrower than is_missing. On this side `unknown` is a real
    schema value meaning "the narrative does not state it" -- it is the model's
    answer, not the absence of one, and it is the single most common answer on
    the text-silent fields. Treating it as missing would drop exactly the rows
    where the model declined to guess, silently removing them from the
    denominator and inflating accuracy against the distant key.
    """
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _norm(v) -> Optional[str]:
    """Normalize a structured code for table lookup, or None if missing."""
    if is_missing(v):
        return None
    s = str(v).strip().lower()
    s = re.sub(r"\s*[/\-]\s*", " ", s)   # "Dark - Lighted" -> "dark lighted"
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Per-field reducers
#
# Each reducer: (struct_row: dict, model_value) -> (gold, pred) | None
#   None  => this row provides no usable supervision for this field
# ---------------------------------------------------------------------------

# --- lighting --------------------------------------------------------------
_LIGHTING = {
    "daylight": "daylight",
    "dark lighted": "dark_lighted",
    "dark not lighted": "dark_unlighted",
    "dawn dusk": "dawn_dusk",
    "dark unknown lighting": "dark_any",   # coarse: cannot resolve lit vs unlit
}
_MODEL_DARK = {"dark_lighted", "dark_unlighted"}


def reduce_lighting(row: dict, pred) -> Optional[tuple]:
    g = _LIGHTING.get(_norm(row.get("struct_lighting")))
    if g is None or is_missing_pred(pred):
        return None
    p = str(pred)
    if p == "unknown":
        # The model declined to commit. That is an extraction failure against a
        # key that did commit, so it is scored, not dropped.
        p = "unknown"
    if g == "dark_any":
        # Project both sides onto {dark, non-dark}: the key cannot say more.
        if p not in _MODEL_DARK:
            return ("dark", "non_dark" if p != "unknown" else "unknown")
        return ("dark", "dark")
    return (g, p)


# --- weather ---------------------------------------------------------------
# SGO encodes weather as a row of independent Y/blank indicator columns, not a
# single categorical. The original column resolver matched "weather - clear"
# and therefore read the CLEAR INDICATOR as if it were the weather value,
# yielding a field whose only observed values were 'Y' and ' '. All indicator
# columns must be collected and reduced jointly.
#
# Precipitation dominates when several indicators are set (a "Rain + Cloudy"
# filing is a rain filing). "Cloudy" alone is dropped rather than mapped: the
# schema has no cloudy category, annotators plausibly coded it `clear`, and
# inventing that convention here would manufacture agreement.
# Ordered by precedence. Values observed across both SGO schema generations:
# clear, cloudy, partly cloudy, rain, snow, fog/smoke, fog/smoke/haze,
# severe wind, dust storm, severe hurricane, structure-indoor, unknown,
# unk - see narrative.
_WEATHER_PRIORITY = [
    ("snow", "snow"), ("sleet", "snow"), ("hail", "snow"),
    ("rain", "rain"),
    ("fog", "fog"), ("smoke", "fog"), ("haze", "fog"),
    ("dust", "other"), ("hurricane", "other"),
    ("severe wind", "other"), ("wind", "other"),
    ("clear", "clear"),
]

# Indicator values that carry no usable weather label. "Structure-Indoor" is a
# location, not a weather condition; the cloudy variants have no schema
# counterpart (see reduce_weather).
_WEATHER_DROP = {"unknown", "unk - see narrative", "unk", "structure-indoor",
                 "cloudy", "partly cloudy", "other"}


def reduce_weather(row: dict, pred) -> Optional[tuple]:
    flags = row.get("struct_weather_flags") or {}
    on = {k for k, v in flags.items() if str(v).strip().lower() in {"y", "yes", "1", "true"}}
    if not on or is_missing_pred(pred):
        return None
    if on <= _WEATHER_DROP:
        # Only uninformative or non-schema indicators are set. "Cloudy" and
        # "Partly Cloudy" are dropped rather than mapped: the schema has no
        # cloudy category, annotators plausibly coded them `clear`, and
        # inventing that convention here would manufacture agreement.
        return None
    for token, label in _WEATHER_PRIORITY:
        if any(token in f for f in on):
            return (label, str(pred))
    return None


# --- road_class ------------------------------------------------------------
# SGO's "Roadway Type" is a single-select that mixes a road-class concept
# (Street / Highway / Parking Lot) with a locality concept (Intersection). It
# therefore supervises two different schema fields, partially, and must be
# split rather than mapped wholesale.
#
# The schema splits surface roads into `surface_street` and `arterial`; SGO
# does not. Both are projected onto `surface` so the model is not charged for a
# distinction the key cannot express.
_ROAD_SURFACE = {"street", "rural road"}
_ROAD_MAP = {
    "highway freeway": "highway_freeway",
    "parking lot": "parking_lot_private",
}
_MODEL_SURFACE = {"surface_street", "arterial"}


def reduce_road_class(row: dict, pred) -> Optional[tuple]:
    raw = _norm(row.get("struct_roadway"))
    if raw is None or is_missing_pred(pred):
        return None
    p = str(pred)
    if raw in _ROAD_SURFACE:
        return ("surface", "surface" if p in _MODEL_SURFACE else p)
    if raw in _ROAD_MAP:
        return (_ROAD_MAP[raw], "surface" if p in _MODEL_SURFACE else p)
    # "Intersection" carries no road-class information; "Traffic Circle" (n=6)
    # has no clean counterpart. Both drop.
    return None


# --- locality --------------------------------------------------------------
# Only the positive class and the unambiguous negatives are usable. A filing
# coded "Street" is NOT evidence of a mid-block location: the reporter chose a
# single code and "Street" does not exclude an intersection, so those rows drop.
# "Highway / Freeway" and "Parking Lot" do exclude an at-grade intersection.
_MODEL_INTERSECTION = {"intersection", "intersection_related"}


def reduce_locality(row: dict, pred) -> Optional[tuple]:
    raw = _norm(row.get("struct_roadway"))
    if raw is None or is_missing_pred(pred):
        return None
    p = "intersection" if str(pred) in _MODEL_INTERSECTION else "not_intersection"
    if raw == "intersection":
        return ("intersection", p)
    if raw in {"highway freeway", "parking lot"}:
        return ("not_intersection", p)
    return None


# --- subject_pre_crash_maneuver -------------------------------------------
# Only codes with a clean schema counterpart are kept. SGO values such as
# "Merging", "Passing", "Entering Traffic", "Lane / Road Departure" and
# "Crossing into Opposing Lane" are dropped rather than collapsed to `other`:
# collapsing would score a model that answered `lane_change` on a merge as
# wrong, which measures the mapping rather than the model.
# Keys are the normalized forms of BOTH corpora's movement codes: SGO's
# "Proceeding Straight" and OL 316's letter-decoded "proceeding straight"
# normalize identically, so one table serves both.
_MANEUVER = {
    "stopped": "stopped",
    "proceeding straight": "in_motion_straight",
    "parked": "parked",
    "making left turn": "turning_left",
    "making right turn": "turning_right",
    "changing lanes": "lane_change",
    "backing": "backing",
}
# The schema resolves longitudinal control (`decelerating`, `accelerating`)
# that SGO folds into "Proceeding Straight".
_MODEL_STRAIGHT = {"proceeding_straight", "decelerating", "accelerating"}


def reduce_maneuver(row: dict, pred) -> Optional[tuple]:
    g = _MANEUVER.get(_norm(row.get("struct_movement")))
    if g is None or is_missing_pred(pred):
        return None
    p = str(pred)
    if p in _MODEL_STRAIGHT:
        p = "in_motion_straight"
    return (g, p)


# --- other_party_present ---------------------------------------------------
_CRASH_WITH_PARTY = {
    "passenger car", "suv", "pickup truck", "heavy truck", "van", "bus",
    "motorcycle", "first responder vehicle", "motorized scooter",
    "non motorist cyclist", "non motorist pedestrian", "non motorist other",
    "non motorist scooter rider", "school bus", "truck", "trailer",
}
_CRASH_WITH_NO_PARTY = {
    "other fixed object", "pole tree", "curb", "barrier", "guardrail",
    "building", "parked vehicle no occupant",
}


def reduce_other_party(row: dict, pred) -> Optional[tuple]:
    raw = _norm(row.get("struct_crashwith"))
    if raw is None or is_missing_pred(pred):
        return None
    if raw in _CRASH_WITH_PARTY:
        g = True
    elif raw in _CRASH_WITH_NO_PARTY:
        g = False
    else:
        # "Animal" is deliberately excluded: whether a struck animal counts as
        # a second road user is a convention the schema does not settle.
        return None
    return (str(g), str(bool(pred)))


# --- collision_type (partial) ---------------------------------------------
# The crash-partner code identifies two collision types unambiguously and says
# nothing about the rest. Supervision is restricted to what it can actually
# support, evaluated as a one-vs-rest decision per identifiable class.
_VRU = {"non motorist cyclist", "non motorist pedestrian", "non motorist other",
        "non motorist scooter rider", "motorized scooter"}
_FIXED_OBJECT = {"other fixed object", "pole tree", "curb", "barrier",
                 "guardrail", "building"}


def _one_vs_rest(row: dict, pred, positive_codes: set, positive_label: str,
                 negative_codes: set) -> Optional[tuple]:
    """One-vs-rest reduction for a single identifiable collision class.

    The two identifiable classes MUST be scored as separate binary problems.
    Folding them into one label vector -- gold in {vru, single_vehicle}, pred in
    {vru, not_vru, single_vehicle, not_single_vehicle} -- makes the classes
    almost perfectly separable by which branch produced the row rather than by
    the model's answer, which inflates kappa toward 1.0 while measuring nothing.
    """
    raw = _norm(row.get("struct_crashwith"))
    if raw is None or is_missing_pred(pred):
        return None
    if raw in positive_codes:
        g = positive_label
    elif raw in negative_codes:
        g = f"not_{positive_label}"
    else:
        return None
    p = positive_label if str(pred) == positive_label else f"not_{positive_label}"
    return (g, p)


# Negatives for the VRU decision are the unambiguous motor-vehicle partners;
# fixed objects are excluded because a fixed-object crash is not evidence about
# whether a VRU was involved under this schema's `vru` definition.
_MOTOR_VEHICLE = {"passenger car", "suv", "pickup truck", "heavy truck", "van",
                  "bus", "motorcycle", "first responder vehicle", "school bus",
                  "truck", "trailer"}


def reduce_collision_vru(row: dict, pred) -> Optional[tuple]:
    direct = _norm(row.get("struct_collision_type"))
    if direct is not None and not is_missing_pred(pred):
        # OL 316 codes collision type directly, so no one-vs-rest projection is
        # needed and the full label is used.
        p = "vru" if str(pred) == "vru" else "not_vru"
        return ("vru" if direct == "vru" else "not_vru", p)
    return _one_vs_rest(row, pred, _VRU, "vru", _MOTOR_VEHICLE)


def reduce_collision_single_vehicle(row: dict, pred) -> Optional[tuple]:
    direct = _norm(row.get("struct_collision_type"))
    if direct is not None and not is_missing_pred(pred):
        p = ("single_vehicle" if str(pred) == "single_vehicle"
             else "not_single_vehicle")
        return ("single_vehicle" if direct == "single_vehicle"
                else "not_single_vehicle", p)
    return _one_vs_rest(row, pred, _FIXED_OBJECT, "single_vehicle",
                        _MOTOR_VEHICLE | _VRU)


# --- av_moving -------------------------------------------------------------
# SGO records pre-crash speed, not speed at impact. The schema defines
# `av_moving` at impact. The two coincide except where the AV decelerated to a
# stop during the pre-crash window, which is a genuine and reported source of
# disagreement (see threats.tex). Speed is used only for the zero / non-zero
# distinction, never as a magnitude.
def reduce_av_moving(row: dict, pred) -> Optional[tuple]:
    raw = row.get("struct_speed")
    if is_missing(raw) or is_missing_pred(pred):
        return None
    try:
        speed = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(speed) or speed < 0:
        return None
    return (str(speed > 0), str(bool(pred)))


# --- engagement_state ------------------------------------------------------
# "Automation Engaged?" is Yes/No; whether an engaged system is ADS or ADAS is
# determined by WHICH SGO file the row came from, which is why parse_ol316.py
# now records `struct_report_type` from the source filename. Without it the
# Yes case cannot be resolved to a schema value.
#
# The No case is genuinely coarse: SGO cannot distinguish the schema's `manual`
# from `disengaged_prior`, so both sides project onto `not_engaged`.
_MODEL_ENGAGED = {"ads_engaged", "adas_engaged"}


def reduce_engagement(row: dict, pred) -> Optional[tuple]:
    raw = _norm(row.get("struct_engaged"))
    rtype = _norm(row.get("struct_report_type"))
    if raw is None or is_missing_pred(pred):
        return None
    p = str(pred)
    # The third amendment replaced the Yes/No "ADS Equipped?" column with a
    # graded "Engagement Status". "Alleged Engaged" is the reporting entity's
    # unverified assertion and is dropped rather than treated as engaged.
    if raw in {"alleged engaged"}:
        return None
    if raw in {"verified not engaged"}:
        return ("not_engaged",
                "engaged" if p in _MODEL_ENGAGED else "not_engaged")
    if raw in {"yes", "y", "verified engaged"}:
        if rtype not in {"ads", "adas"}:
            return None                    # cannot resolve which system
        g = "ads_engaged" if rtype == "ads" else "adas_engaged"
        return (g, p)
    if raw in {"no", "n"}:
        return ("not_engaged",
                "engaged" if p in _MODEL_ENGAGED else "not_engaged")
    return None


# --- narrative severity concordance (DIAGNOSTIC ONLY) ----------------------
# Reported separately from extraction accuracy. `narrative_injury_severity` is
# severity as described in the narrative; `struct_severity` is the reporting
# entity's alleged highest severity. Divergence is a finding about the corpus,
# not necessarily an extraction error.
#
# Two sources feed struct_severity at different granularity. SGO's "Highest
# Injury Severity Alleged" resolves the full none/minor/moderate/serious/
# fatal scale below. OL 316 has no such grade -- parse_ol316.py's
# _ol316_injury_flag can only resolve "none"/"fatal" (unambiguous) or "some
# injury" (an Injured tick with no severity grade). "some injury" is
# therefore evaluated as a genuinely coarser bucket, the same way
# reduce_lighting projects "Dark - Unknown Lighting" onto dark_any: both
# sides of the comparison collapse onto {none, some_injury, fatal} rather
# than crediting/penalizing the model against a grade OL 316 cannot express.
_SEVERITY = {
    "no injuries reported": "none",
    "no injured reported": "none",
    "property damage no injured reported": "none",
    "minor": "minor",
    "minor wo hospitalization": "minor",
    "minor w hospitalization": "minor",
    "moderate": "moderate",
    "moderate wo hospitalization": "moderate",
    "moderate w hospitalization": "moderate",
    "serious": "serious",
    "serious w hospitalization": "serious",
    "serious wo hospitalization": "serious",
    "fatality": "fatal",
    # OL 316 unambiguous buckets (see _ol316_injury_flag).
    "none": "none",
    "fatal": "fatal",
}
_OL316_SOME_INJURY_KEY = "some injury"
_MODEL_SOME_INJURY = {"minor", "moderate", "serious"}


def reduce_narrative_severity(row: dict, pred) -> Optional[tuple]:
    raw = _norm(row.get("struct_severity"))
    if raw is None or is_missing_pred(pred):
        return None
    p = str(pred)
    if raw == _OL316_SOME_INJURY_KEY:
        return ("some_injury", "some_injury" if p in _MODEL_SOME_INJURY else p)
    g = _SEVERITY.get(raw)
    if g is None:
        return None
    return (g, p)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Registry maps an EVALUATION ROW NAME to the (schema field, reducer) pair it
# scores. The two are not one-to-one: `collision_type` supports two independent
# binary decisions and therefore contributes two rows, each scored separately.
REDUCERS: dict[str, tuple[str, Callable]] = {
    "lighting": ("lighting", reduce_lighting),
    "weather": ("weather", reduce_weather),
    "road_class": ("road_class", reduce_road_class),
    "locality": ("locality", reduce_locality),
    "subject_pre_crash_maneuver": ("subject_pre_crash_maneuver", reduce_maneuver),
    "other_party_present": ("other_party_present", reduce_other_party),
    "collision_type (VRU)": ("collision_type", reduce_collision_vru),
    "collision_type (single-veh.)": ("collision_type",
                                     reduce_collision_single_vehicle),
    "av_moving": ("av_moving", reduce_av_moving),
    "engagement_state": ("engagement_state", reduce_engagement),
}

# Scored as extraction accuracy against the distant key.
DISTANT_FIELDS = list(REDUCERS)

# No structured counterpart in either corpus -- remains human-annotated.
HUMAN_ONLY_FIELDS = ["contributory_party"]

# Fields whose evaluation space is coarser than the schema, and the reason.
# Surfaced in the paper so no reader mistakes a coarsened score for a
# full-granularity one.
COARSENED = {
    "lighting": "'Dark - Unknown Lighting' collapses dark_lighted/dark_unlighted",
    "road_class": "surface_street and arterial are not distinguished by SGO",
    "locality": "evaluated as intersection vs. not-intersection",
    "subject_pre_crash_maneuver": "decelerating/accelerating fold into proceeding_straight",
    "collision_type (VRU)": "one-vs-rest: VRU vs. motor-vehicle partner",
    "collision_type (single-veh.)": "one-vs-rest: fixed object vs. road-user partner",
    "engagement_state": "'No' collapses manual/disengaged_prior",
}
