"""Distant-supervision key for the CRSS-compatible composition layer.

Same contract as `schema.distant_map`, applied to `schema.composition_schema`:
each field defines a REDUCER taking the SGO-filed structured value and the
model's prediction, returning the pair projected into the coarsest space both
can express, or None to drop the row from that field's evaluation.

The point of the contract is that a mapping is never invented to make a number
computable. A reducer that would have to guess returns None, and coverage is
reported per field, because a field scored on 1,384 records and one scored on
2,353 are not the same evidence.

-----------------------------------------------------------------------------
WHAT IS KEYED, AND WHAT THE AMENDMENT MADE CONDITIONAL
-----------------------------------------------------------------------------
Two fields -- `posted_speed_limit_bin` and `lighting` -- have a structured
counterpart ONLY in the pre-amendment (archive) generation, because the SGO
third amendment dropped both columns. That makes their evaluation a
PRE-AMENDMENT evaluation, and the accuracy measured there is what licenses
using the extracted value on the 969 post-amendment incidents where no
structured value exists. This is stated in `PRE_AMENDMENT_ONLY` and surfaced in
the coverage table rather than left for a reader to infer from an n.

Three more -- `acc_type_category`, `striking_role` and the junction fields --
are keyed only where the contact-area geometry determines them
(`acc_type.sgo_distant_key`), which is 64% and 54% of the corpus respectively.
The residual is exactly the junction crashes the structured fields cannot
resolve, and it is where the extraction is doing work no key can check. Said
plainly: the validity claim covers the keyed portion, and the unkeyed portion
rests on the human double-coding sample instead.
"""
from __future__ import annotations

from typing import Callable, Optional

from schema.acc_type import sgo_distant_key


def _s(v) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    return "" if s in {"nan", "none", "n/a", "-", "<na>"} else s


def _bool(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = _s(v)
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return None


# ---------------------------------------------------------------------------
# Crash type and role
# ---------------------------------------------------------------------------
def reduce_acc_type(row: dict, pred) -> Optional[tuple]:
    """Key from contact geometry; drop where geometry does not determine it.

    The key covers six of the eleven categories. The five it cannot reach --
    turn_across_path, turn_into_path, straight_paths_intersecting,
    sideswipe_opposite_direction and other_unknown -- all require knowing which
    way the vehicles were heading, which the contact areas do not say.

    A prediction in an UNREACHABLE category is not scored as wrong: the key has
    no opinion there, so the row is dropped. Scoring it wrong would penalise
    the extractor for resolving exactly the cases the key was built to concede.
    """
    key = sgo_distant_key(row)
    if key is None:
        return None
    return (key[0], _s(pred))


def reduce_striking_role(row: dict, pred) -> Optional[tuple]:
    """Role from contact geometry.

    Dropped when the key returns `undetermined` as well as when it returns
    nothing: an undetermined key cannot adjudicate a committed prediction. It
    is `not_applicable` that IS scored, because "no second vehicle" is a
    determinate fact the contact areas and partner class establish.
    """
    key = sgo_distant_key(row)
    if key is None or key[1] == "undetermined":
        return None
    return (key[1], _s(pred))


# ---------------------------------------------------------------------------
# Pre-crash movement -- SGO vocabulary into the CRSS P_CRASH1 space
# ---------------------------------------------------------------------------
# SGO and CRSS disagree on granularity in both directions, so the shared space
# is coarser than either. Notably SGO's single `Stopped` cannot separate CRSS's
# `Stopped in Roadway` from `Disabled or Parked in Travel lane`, so those
# collapse; and SGO's `Lane / Road Departure` has no CRSS P_CRASH1 counterpart
# at all (CRSS puts departures in ACC_TYPE, not the maneuver field), so it is
# dropped rather than forced into `other`.
_SGO_MOVE = {
    "stopped": "stopped",
    "parked": "stopped",
    "proceeding straight": "going_straight",
    "making left turn": "turning_left",
    "making right turn": "turning_right",
    "making u-turn": "making_u_turn",
    "changing lanes": "lane_change",
    "merging": "lane_change",
    "passing": "lane_change",
    "backing": "backing",
    "parking maneuver": "parking",
    "entering traffic": "starting_in_road",
    "decelerating": "decelerating",
    "accelerating": "accelerating",
}
_PRED_MOVE = {
    "stopped_in_roadway": "stopped",
    "parked": "stopped",
    "going_straight": "going_straight",
    "turning_left": "turning_left",
    "turning_right": "turning_right",
    "making_u_turn": "making_u_turn",
    "changing_lanes": "lane_change",
    "merging": "lane_change",
    "passing": "lane_change",
    "backing": "backing",
    "entering_parking": "parking",
    "leaving_parking": "parking",
    "starting_in_road": "starting_in_road",
    "decelerating": "decelerating",
    "accelerating": "accelerating",
}
# SGO values with no defensible placement: dropped, never mapped to `other`.
_SGO_MOVE_DROP = {"other, see narrative", "unknown", "", "lane / road departure",
                  "crossing into opposing lane", "traveling wrong way",
                  "nm crossing roadway", "negotiating a curve"}


def _reduce_move(col: str) -> Callable:
    def fn(row: dict, pred) -> Optional[tuple]:
        raw = _s(row.get(col))
        if raw in _SGO_MOVE_DROP:
            return None
        key = _SGO_MOVE.get(raw)
        if key is None:
            return None
        p = _PRED_MOVE.get(_s(pred))
        if p is None:
            # The model answered something the shared space cannot express
            # (`other`, `unknown`, `negotiating_curve`). That is an abstention
            # or an out-of-space answer, not a wrong answer against this key.
            return None
        return (key, p)
    return fn


reduce_sv_movement = _reduce_move("SV Pre-Crash Movement")
reduce_cp_movement = _reduce_move("CP Pre-Crash Movement")


# ---------------------------------------------------------------------------
# Severity -- SGO allegation into KABCO, at the granularity both support
# ---------------------------------------------------------------------------
# SGO's ladder is none / minor / moderate / serious / fatal, with
# hospitalization qualifiers. KABCO is O / C / B / A / K. The two cannot be
# aligned at full resolution: SGO's `Moderate` has no KABCO grade, and KABCO's
# C-vs-B distinction has no SGO counterpart. The shared space is four levels,
# matching `crss_map.SEVERITY_COMMON` exactly so the two crosswalks cannot
# drift apart.
_KABCO_COMMON = {"k": "fatal", "a": "serious", "b": "minor", "c": "minor",
                 "o": "none"}


def _sgo_severity_common(s) -> Optional[str]:
    t = _s(s)
    if not t:
        return None
    if "fatal" in t:
        return "fatal"
    if "serious" in t:
        return "serious"
    if "moder" in t or "minor" in t:
        return "minor"
    if "no inj" in t or "property damage" in t or t in {"none", "no"}:
        return "none"
    return None            # 'Unknown'


def reduce_kabco(row: dict, pred) -> Optional[tuple]:
    key = _sgo_severity_common(row.get("Highest Injury Severity Alleged"))
    if key is None:
        return None
    p = _KABCO_COMMON.get(_s(pred))
    if p is None:
        return None        # model answered `unknown`
    return (key, p)


# ---------------------------------------------------------------------------
# Tow-away and airbag
# ---------------------------------------------------------------------------
# Keyed against the SUBJECT VEHICLE columns specifically. The extraction asks
# about tow "due to damage"; SGO's field does not carry the reason, so the key
# is coarser than the prediction and a convenience tow filed as a tow reads as
# a disagreement. That is a known, directional limitation of the key rather
# than of the extraction, and it is named here so the measured accuracy is not
# read as a pure error rate.
def _reduce_yn(col: str) -> Callable:
    def fn(row: dict, pred) -> Optional[tuple]:
        k = _bool(row.get(col))
        if k is None:
            return None
        p = _s(pred)
        if p not in {"yes", "no"}:
            return None
        return ("yes" if k else "no", p)
    return fn


reduce_tow = _reduce_yn("sv_towed")
reduce_airbag = _reduce_yn("sv_airbag")


# ---------------------------------------------------------------------------
# Operating domain -- PRE-AMENDMENT ONLY
# ---------------------------------------------------------------------------
def reduce_speed_bin(row: dict, pred) -> Optional[tuple]:
    """Posted speed limit, binned. Structured value exists pre-amendment only."""
    v = row.get("posted_speed_limit")
    try:
        mph = float(v)
    except (TypeError, ValueError):
        return None
    if mph <= 0:
        return None
    key = ("le_25" if mph <= 25 else "mph_30_35" if mph <= 35
           else "mph_40_45" if mph <= 45 else "gt_45")
    p = _s(pred)
    if p not in {"le_25", "mph_30_35", "mph_40_45", "gt_45"}:
        return None
    return (key, p)


# SGO 'Dark - Unknown Lighting' cannot separate lit from unlit dark, exactly as
# CRSS cannot (see crss_map). The shared space pools them, so the key never
# penalises a distinction it cannot make.
_SGO_LIGHT = {"daylight": "daylight", "dark - lighted": "dark",
              "dark - not lighted": "dark", "dark - unknown lighting": "dark",
              "dawn / dusk": "dawn_dusk"}
_PRED_LIGHT = {"daylight": "daylight", "dark_lighted": "dark",
               "dark_unlighted": "dark", "dawn_dusk": "dawn_dusk"}


def reduce_lighting(row: dict, pred) -> Optional[tuple]:
    key = _SGO_LIGHT.get(_s(row.get("lighting_struct")))
    if key is None:
        return None
    p = _PRED_LIGHT.get(_s(pred))
    if p is None:
        return None
    return (key, p)


# ---------------------------------------------------------------------------
# Exclusion flags
# ---------------------------------------------------------------------------
def _reduce_flag(col: str) -> Callable:
    def fn(row: dict, pred) -> Optional[tuple]:
        k = _bool(row.get(col))
        p = _bool(pred)
        if k is None or p is None:
            return None
        return (str(k), str(p))
    return fn


reduce_ads_engaged = _reduce_flag("ads_engaged")
reduce_public_road = _reduce_flag("public_road")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REDUCERS: dict[str, Callable] = {
    "acc_type_category": reduce_acc_type,
    "striking_role": reduce_striking_role,
    "sv_pre_crash_movement": reduce_sv_movement,
    "cp_pre_crash_movement": reduce_cp_movement,
    "kabco_severity": reduce_kabco,
    "tow_away_due_to_damage": reduce_tow,
    "airbag_deployed": reduce_airbag,
    "posted_speed_limit_bin": reduce_speed_bin,
    "lighting": reduce_lighting,
    "ads_engaged_at_impact": reduce_ads_engaged,
    "on_public_road": reduce_public_road,
}

# Fields whose structured counterpart the third amendment removed. Their
# evaluation is a pre-amendment evaluation, and the reported n reflects that.
PRE_AMENDMENT_ONLY = {"posted_speed_limit_bin", "lighting"}

COARSENED = {
    "acc_type_category": "keyed only where contact geometry determines it "
                         "(~64% of incidents); junction categories unreachable",
    "striking_role": "keyed only where geometry determines it (~54%); "
                     "'undetermined' keys are dropped, not scored",
    "sv_pre_crash_movement": "stopped/parked pooled; changing lanes, merging "
                             "and passing pooled as lane_change",
    "cp_pre_crash_movement": "as sv_pre_crash_movement",
    "kabco_severity": "four levels: SGO Minor+Moderate merged, KABCO C+B merged",
    "tow_away_due_to_damage": "SGO does not record the REASON for the tow, so a "
                              "convenience tow reads as a disagreement",
    "lighting": "dark_lighted/dark_unlighted pooled: SGO 'Dark - Unknown Lighting'",
    "posted_speed_limit_bin": "pre-amendment filings only",
}


def score(rows, predictions, fields=None) -> dict:
    """Accuracy and coverage per field against the structured key.

    `rows` and `predictions` are parallel: row i's structured fields and the
    extraction for the same incident. Coverage is the share of rows the reducer
    kept; accuracy is over those rows only. Both are reported because either
    alone is misleading.
    """
    fields = fields or list(REDUCERS)
    out = {}
    for f in fields:
        fn = REDUCERS.get(f)
        if fn is None:
            continue
        n_kept = n_ok = 0
        for row, pred in zip(rows, predictions):
            if pred is None:
                continue
            r = fn(row, pred.get(f))
            if r is None:
                continue
            n_kept += 1
            n_ok += int(r[0] == r[1])
        out[f] = {
            "n_scored": n_kept,
            "coverage": n_kept / max(len(rows), 1),
            "accuracy": (n_ok / n_kept) if n_kept else None,
            "coarsened": COARSENED.get(f, ""),
            "pre_amendment_only": f in PRE_AMENDMENT_ONLY,
        }
    return out
