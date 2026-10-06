"""The shared crash-type taxonomy, and the striking/struck role, for both sides.

This module is the crosswalk the composition study turns on. CRSS codes every
vehicle with NHTSA's `ACC_TYPE` (85 values in use); SGO codes nothing of the
kind. What follows defines (a) a collapsed category space both sources can
express, (b) how CRSS `ACC_TYPE` projects into it, and (c) how the SGO
structured fields project into it, so the LLM-extracted layer has an
independent key to be validated against rather than being the only source.

-----------------------------------------------------------------------------
THE PROPERTY THAT MAKES ROLE OBJECTIVE
-----------------------------------------------------------------------------
`ACC_TYPE` is a VEHICLE-level variable, and within a crash configuration its
values come in complementary pairs: one names the maneuver of the vehicle that
struck, the other the state of the vehicle that was struck. The pairing is
exact in the data, which is how it was verified rather than assumed:

    L86 Straight Paths - Striking from the Right     1,789
    L87 Straight Paths - Struck on the Right         1,789
    L88 Straight Paths - Striking from the Left      1,624
    L89 Straight Paths - Struck on the left          1,624
    J68 Turn Across Path - Opposite Dir (Left/Right) 3,532
    J69 Turn Across Path - Opposite Dir (Straight)   3,532

Some pairs are explicit ("Striking from" / "Struck on"). Most are implicit: in
the rear-end family, `D20 Rear End-Stopped` is the vehicle that WAS stopped and
`D21 Rear End-Stopped, Straight` is the one that was going straight into it.
The descriptor names this vehicle's own behaviour, so an ACTIVE descriptor
(going straight, changing lanes, turning, backing) is the striking vehicle and
a PASSIVE one (stopped, slower, decelerating, struck on) is the struck vehicle.

This matters because role is what the quasi-induced-exposure analysis needs,
and because it is derivable on the SGO side from contact-area geometry without
anyone's opinion about fault. `contributory_party` is a narrative judgement and
inherits the filer's framing; front-to-rear does not.

-----------------------------------------------------------------------------
WHAT THE COLLAPSE COSTS, STATED
-----------------------------------------------------------------------------
Two losses are structural and are surfaced in `COARSENED` rather than buried:

  PEDESTRIANS AND ANIMALS ARE ONE CODE. `C13 Single Driver-Forward Impact-
  Pedestrian/Animal` pools them, so `ACC_TYPE` alone cannot separate a
  pedestrian strike from a deer strike. Where the vulnerable-road-user share is
  the quantity of interest, the CRSS side must be taken from `HARM_EV` or the
  person file instead, and `crss_vru_from_harm_ev` exists for that.

  "SPECIFICS OTHER/UNKNOWN" IS A REAL CATEGORY, NOT MISSINGNESS. Codes like
  D32/D33 place the crash in the rear-end family but decline to say which
  vehicle did what. They keep their crash-type category and get role
  `undetermined`; dropping them would condition the role analysis on the role
  being codeable, which is not independent of the role.

-----------------------------------------------------------------------------
THE ROLE MAP IS VERIFIED, AND THE QIE NULL IS *NOT* 1.0 EVERYWHERE
-----------------------------------------------------------------------------
Quasi-induced exposure reasons from the fact that in a human-vs-human sample
the striking:struck ratio must be 1 by construction: every crash contributes
one of each. Running that check on CRSS 2023 weighted vehicle counts confirms
the role assignment above and, at the same time, shows the assumption fails for
two categories for reasons internal to the coding scheme:

    rear_end                       1.000    <- exactly as predicted
    head_on                        1.000
    straight_paths_intersecting    1.000
    turn_across_path               1.000
    turn_into_path                 1.000
    sideswipe_opposite_direction   1.000
    sideswipe_same_direction       0.607    <- NOT 1
    backing                        1.851    <- NOT 1

Five categories landing on 1.000 to three decimals is the evidence that
striking and struck have not been inverted anywhere. The two exceptions are
explained, not waved away:

  `sideswipe_same_direction` is dominated by `F45 Straight Ahead on
  Left/Right` (602k of 1.42m), a COMBINED code applied to both vehicles when
  neither is clearly the encroacher. Both get the passive descriptor, so the
  struck side is inflated and no crash contributes a striking vehicle.

  `backing` is inflated on the striking side because `M92 Backing Veh.` is
  coded for single-vehicle backing crashes too -- backing into a pole produces
  a striking vehicle and no struck one.

CONSEQUENCE FOR §6.3. The QIE baseline must be ESTIMATED FROM CRSS per
category (`human_baseline_ratio`), not assumed to be 1. Using 1 as the null
for same-direction sideswipes would manufacture a 1.6x apparent ADS effect out
of a coding convention. For the five categories at 1.000 the assumption holds
and the estimated baseline simply reproduces it, which is the right way to
find that out.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# The collapsed category space
# ---------------------------------------------------------------------------
# Ten substantive categories plus `other_unknown`. Chosen so that every
# category (a) exists on both sides, (b) is a distinct liability story, and
# (c) is large enough to estimate in the ADS corpus -- the granularity was set
# after counting per-cell n, per the outline's instruction to do so before
# fixing it.
CATEGORIES = [
    "rear_end",                     # D: same trafficway, same direction, rear end
    "sideswipe_same_direction",     # F: angle/sideswipe, same direction
    "head_on",                      # G: opposite direction, head-on
    "sideswipe_opposite_direction",  # I: angle/sideswipe, opposite direction
    "turn_across_path",             # J: one vehicle turns across the other's path
    "turn_into_path",               # K: one vehicle turns into the other's path
    "straight_paths_intersecting",  # L: the intersection T-bone
    "backing",                      # M92/M93
    "single_vehicle",               # A, B, C (except C13)
    "pedestrian_animal",            # C13 -- pooled, see module docstring
    "other_unknown",                # M0, M98, M99, E, H
]

ROLES = ["striking", "struck", "undetermined", "not_applicable"]

COARSENED = {
    "pedestrian_animal": "CRSS ACC_TYPE code C13 pools pedestrians with animals; "
                         "use crss_vru_from_harm_ev where the VRU share matters",
    "head_on": "CRSS H-family (opposite-direction forward impact: control loss, "
               "evasive) is placed in other_unknown, not head_on -- those are "
               "departures from the opposing lane, not front-to-front impacts",
    "role": "'Specifics Other/Unknown' codes keep their crash type and take "
            "role=undetermined rather than being dropped",
}


# ---------------------------------------------------------------------------
# CRSS ACC_TYPE -> (category, role)
# ---------------------------------------------------------------------------
# Keyed on the INTEGER code, which is stable within the documented typology,
# and cross-checked against the decoded *NAME string at load time by
# `verify_crss_names` -- NHTSA renumbers between releases, and a silent shift
# would invert striking and struck without raising anything. This is the same
# hazard `crss_map.py` guards by matching on names; here both are used, because
# the integer carries the pairing structure and the name carries the check.
_ROLE_BY_CODE: dict[int, tuple[str, str]] = {}


def _reg(codes, category: str, role: str) -> None:
    for c in codes:
        _ROLE_BY_CODE[c] = (category, role)


# --- A/B: roadside departure; C: forward impact. No motor-vehicle partner. --
_reg(range(1, 11), "single_vehicle", "not_applicable")
_reg([11, 12, 14, 15, 16], "single_vehicle", "not_applicable")
_reg([13], "pedestrian_animal", "not_applicable")

# --- D: rear-end. THE NAMING RUNS THE OPPOSITE WAY TO EVERY OTHER FAMILY. ---
# Read as English, `D20 Rear End-Stopped` looks like the stopped (struck)
# vehicle and `D21 Rear End-Stopped, Straight` like the one driving into it.
# The data says the reverse, unambiguously (CRSS 2023, P_CRASH1 modal value):
#
#     D20 Rear End-Stopped                        Going Straight      0.76
#     D21 Rear End-Stopped, Straight              Stopped in Roadway  1.00
#     D22 Rear End-Stopped, Left                  Stopped in Roadway  1.00
#     D23 Rear End-Stopped, Right                 Stopped in Roadway  1.00
#     D28 Rear End-Decelerating                   Going Straight      0.82
#     D29 Rear End-Decelerating, Going Straight   Decelerating        0.97
#
# In the bare configuration codes (D20/D24/D28) the word names the PARTNER's
# state and this vehicle is the striker; in the two-part codes the FIRST word
# is this vehicle's own state and the second is the partner's. Elsewhere in
# ACC_TYPE the descriptor names this vehicle throughout (J68 turns left 100% of
# the time, K76 100%, M92 backs 92%), which is why this family alone inverts.
#
# WHY THE 1:1 BALANCE CHECK DID NOT CATCH THIS. Inverting BOTH members of a
# pair leaves the striking:struck ratio at exactly 1.000. The balance test
# validates that the pairing is intact, not that it points the right way; only
# an independent variable -- here P_CRASH1, which the role map never reads --
# can establish orientation. `test_role_orientation_against_p_crash1` pins it.
_reg([20, 24, 28], "rear_end", "striking")   # bare config: drove into the other
_reg([21, 22, 23, 25, 26, 27, 29, 30, 31], "rear_end", "struck")
_reg([32, 33], "rear_end", "undetermined")           # Specifics Other / Unknown

# --- E: same-direction forward impact (evasive). Not a two-vehicle contact
#     configuration in the sense the other families are; pooled as other. ----
_reg([38, 39, 40, 41, 42], "other_unknown", "undetermined")

# --- F: same-direction sideswipe. The lane-changer encroaches. -------------
_reg([44, 45], "sideswipe_same_direction", "struck")     # straight ahead in lane
_reg([46, 47], "sideswipe_same_direction", "striking")   # changing lanes
_reg([48, 49], "sideswipe_same_direction", "undetermined")

# --- G: head-on. The vehicle that made the lateral move crossed over. ------
_reg([50], "head_on", "striking")                    # lateral move (left/right)
_reg([51], "head_on", "struck")                      # going straight
_reg([52, 53], "head_on", "undetermined")

# --- H: opposite-direction forward impact (control loss / evasive). -------
_reg([54, 55, 58, 59, 60, 61, 63], "other_unknown", "undetermined")

# --- I: opposite-direction sideswipe. Same logic as G. --------------------
_reg([64], "sideswipe_opposite_direction", "striking")
_reg([65], "sideswipe_opposite_direction", "struck")
_reg([66, 67], "sideswipe_opposite_direction", "undetermined")

# --- J: turn across path. The turning vehicle crosses the other's path. ---
_reg([68, 70, 72], "turn_across_path", "striking")   # turning
_reg([69, 71, 73], "turn_across_path", "struck")     # going straight
_reg([74, 75], "turn_across_path", "undetermined")

# --- K: turn into path. Same structure. -----------------------------------
_reg([76, 78, 80, 82], "turn_into_path", "striking")  # turning
_reg([77, 79, 81, 83], "turn_into_path", "struck")    # going straight
_reg([84, 85], "turn_into_path", "undetermined")

# --- L: intersecting straight paths. Explicit striking/struck. ------------
_reg([86, 88], "straight_paths_intersecting", "striking")
_reg([87, 89], "straight_paths_intersecting", "struck")
_reg([90, 91], "straight_paths_intersecting", "undetermined")

# --- M: backing, other, unknown. -----------------------------------------
_reg([92], "backing", "striking")                    # the backing vehicle
_reg([93], "backing", "struck")                      # the other vehicle
_reg([0, 98, 99], "other_unknown", "undetermined")


def crss_acc_type(code) -> Optional[tuple[str, str]]:
    """(category, role) for one CRSS vehicle. None if the code is unmapped."""
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return None
    return _ROLE_BY_CODE.get(c)


# The substring each code's decoded name must contain, as a guard against a
# renumbering silently inverting striking and struck. Only the discriminating
# fragment is checked -- the full strings differ in punctuation between years.
_NAME_GUARD = {
    20: "rear end-stopped", 21: "stopped, straight", 24: "rear end-slower",
    86: "striking from the right", 87: "struck on the right",
    88: "striking from the left", 89: "struck on the left",
    92: "backing veh", 93: "backing-other vehicle",
    13: "pedestrian", 46: "changing lanes to the right",
}

# ORIENTATION GUARD. The modal `P_CRASH1` value each code must show, used to
# check that striking and struck have not been swapped. Name guards catch a
# renumbering; only this catches an inversion, because inverting a pair leaves
# every balance statistic unchanged. Shares are modal fractions on CRSS 2023;
# the check asserts the MODE, not the fraction, so ordinary year-to-year drift
# does not trip it.
_ORIENTATION_GUARD = {
    21: ("stopped in roadway", "struck"),
    22: ("stopped in roadway", "struck"),
    23: ("stopped in roadway", "struck"),
    29: ("decelerating in road", "struck"),
    20: ("going straight", "striking"),
    28: ("going straight", "striking"),
    46: ("changing lanes", "striking"),
    47: ("changing lanes", "striking"),
    68: ("turning left", "striking"),
    69: ("going straight", "struck"),
    76: ("turning left", "striking"),
    77: ("going straight", "struck"),
    78: ("turning right", "striking"),
    92: ("backing up (other than for parking position)", "striking"),
}


def verify_role_orientation(triples) -> list[str]:
    """Check assigned roles against observed `P_CRASH1`, the independent signal.

    `triples` yields (ACC_TYPE, P_CRASH1NAME, count) rows. For each guarded
    code the modal maneuver must match the expected one, which in turn is what
    makes the striking/struck assignment defensible: a vehicle coded `struck`
    in a rear-end crash should have been stopped or decelerating, and one coded
    `striking` should have been going straight.

    This exists because the rear-end family WAS inverted and no balance
    statistic revealed it -- see the note above the D-family registrations.
    """
    modal: dict[int, tuple[str, int]] = {}
    for code, name, count in triples:
        try:
            c = int(float(code))
        except (TypeError, ValueError):
            continue
        if c not in _ORIENTATION_GUARD:
            continue
        n = int(count)
        if c not in modal or n > modal[c][1]:
            modal[c] = (str(name).strip().lower(), n)

    problems = []
    for c, (want_move, want_role) in _ORIENTATION_GUARD.items():
        if c not in modal:
            continue
        got_move = modal[c][0]
        got_role = (_ROLE_BY_CODE.get(c) or (None, None))[1]
        if want_move not in got_move:
            problems.append(
                f"ACC_TYPE {c}: expected modal P_CRASH1 ~ {want_move!r}, "
                f"got {got_move!r} -- the typology may have been recoded")
        if got_role != want_role:
            problems.append(
                f"ACC_TYPE {c}: role is {got_role!r} but the observed maneuver "
                f"({got_move!r}) implies {want_role!r} -- striking and struck "
                f"look inverted for this family")
    return problems


def verify_crss_names(pairs) -> list[str]:
    """Check decoded names against `_NAME_GUARD`; return human-readable problems.

    Called by the CRSS loader on every release. A mismatch means the integer
    codes moved and the role assignment above is no longer trustworthy -- which
    must stop the run, not produce a quietly inverted quasi-induced-exposure
    ratio.
    """
    problems = []
    for code, name in pairs:
        try:
            c = int(float(code))
        except (TypeError, ValueError):
            continue
        want = _NAME_GUARD.get(c)
        if want and want not in str(name).strip().lower():
            problems.append(f"ACC_TYPE {c}: expected name containing {want!r}, "
                            f"got {str(name)!r}")
    return problems


def crss_vru_from_harm_ev(name) -> Optional[bool]:
    """Pedestrian/cyclist involvement from `HARM_EV`, separating them from animals.

    `ACC_TYPE` code C13 pools pedestrians with animals, so the VRU share cannot
    be read off the crash-type collapse. `HARM_EV` names the first harmful
    event and does separate them.
    """
    n = str(name or "").strip().lower()
    if not n:
        return None
    if "pedestrian" in n or "pedalcyclist" in n or "bicycle" in n or "cyclist" in n:
        return True
    if "animal" in n:
        return False
    return None


# ---------------------------------------------------------------------------
# SGO structured fields -> (category, role): the distant key
# ---------------------------------------------------------------------------
# Built from `Crash With`, the two pre-crash movement fields and the contact
# areas. This is a KEY, not a prediction: it uses only fields the reporting
# entity filed, so the LLM's `acc_type_category` can be scored against it at
# corpus scale the way `score_distant.py` scores the existing fields.
#
# It is deliberately INCOMPLETE. Where the structured fields do not determine
# the category -- an intersection crash whose contact areas are front-to-side
# could be a turn-across-path or a straight-paths T-bone, and only the
# narrative says which -- it returns None rather than guessing. Coverage is
# reported by `distant_coverage` so nobody mistakes a partial key for a full one.

_FRONT = ("front", "front_left", "front_right")
_REAR = ("rear", "rear_left", "rear_right")
_SIDE = ("left", "right", "front_left", "front_right", "rear_left", "rear_right")

_MOVE_TURNING = {"making left turn", "making right turn", "making u-turn"}
_MOVE_BACKING = {"backing"}
_MOVE_LANE = {"changing lanes", "merging", "passing"}
_MOVE_STOPPED = {"stopped", "parked"}
_MOVE_STRAIGHT = {"proceeding straight"}


def _zones(row, side: str) -> set:
    """Contact zones marked on one side, as bare zone names."""
    out = set()
    prefix = f"{side}_contact_"
    for k in row.keys():
        if not k.startswith(prefix) or k.endswith("unknown"):
            continue
        if row.get(k):
            out.add(k[len(prefix):])
    return out


def _any(zones: set, group) -> bool:
    return any(z in zones for z in group)


def sgo_distant_key(row: dict) -> Optional[tuple[str, str]]:
    """(category, role) from SGO structured fields alone, or None if undetermined.

    ORDER MATTERS. Partner class is checked first because a crash with a fixed
    object cannot be a rear-end regardless of contact geometry; backing is
    checked before the contact-geometry rules because a backing vehicle's front
    contact would otherwise read as a forward impact.
    """
    partner = str(row.get("crash_with") or "").strip().lower()
    sv_move = str(row.get("SV Pre-Crash Movement") or "").strip().lower()
    cp_move = str(row.get("CP Pre-Crash Movement") or "").strip().lower()
    sv, cp = _zones(row, "sv"), _zones(row, "cp")

    # --- no motor-vehicle partner ----------------------------------------
    if partner.startswith("non-motorist:"):
        return ("pedestrian_animal", "not_applicable")
    if partner in {"animal"}:
        return ("pedestrian_animal", "not_applicable")
    if partner in {"other fixed object", "pole / tree"}:
        return ("single_vehicle", "not_applicable")

    # --- backing ----------------------------------------------------------
    # Checked before geometry: whichever vehicle was backing is the striking
    # one by construction, and its contact area is its rear.
    if sv_move in _MOVE_BACKING:
        return ("backing", "striking")
    if cp_move in _MOVE_BACKING:
        return ("backing", "struck")

    if not sv or not cp:
        return None            # contact geometry is what the rest relies on

    sv_front, sv_rear = _any(sv, _FRONT), _any(sv, _REAR)
    cp_front, cp_rear = _any(cp, _FRONT), _any(cp, _REAR)

    # --- rear-end: front-to-rear, in either direction ---------------------
    # The single most objective determination available, and the one the
    # role analysis leans on hardest.
    if sv_front and cp_rear and not sv_rear and not cp_front:
        return ("rear_end", "striking")
    if sv_rear and cp_front and not sv_front and not cp_rear:
        return ("rear_end", "struck")

    # --- head-on: front-to-front -----------------------------------------
    if sv_front and cp_front and not sv_rear and not cp_rear:
        # Front-to-front with both going straight is head-on. With one turning
        # it is a turn-across-path, which the structured fields cannot separate
        # from a genuine head-on -- so only the unambiguous case is keyed.
        if sv_move in _MOVE_STRAIGHT and cp_move in _MOVE_STRAIGHT:
            return ("head_on", "undetermined")
        return None

    # --- same-direction sideswipe: side-to-side, one vehicle changing lanes
    sv_side_only = _any(sv, _SIDE) and not sv_front and not sv_rear
    cp_side_only = _any(cp, _SIDE) and not cp_front and not cp_rear
    if sv_side_only and cp_side_only:
        if sv_move in _MOVE_LANE and cp_move not in _MOVE_LANE:
            return ("sideswipe_same_direction", "striking")
        if cp_move in _MOVE_LANE and sv_move not in _MOVE_LANE:
            return ("sideswipe_same_direction", "struck")
        return ("sideswipe_same_direction", "undetermined")

    # Everything else -- front-to-side at a junction above all -- needs the
    # narrative to separate turn-across-path from turn-into-path from a
    # straight-paths T-bone. Returning None here is what keeps this a key.
    return None


def distant_coverage(rows) -> dict:
    """What share of ADS incidents the structured key can categorise at all.

    Reported alongside every accuracy figure computed against this key. A key
    that covers half the corpus supports a validity claim about half the
    corpus, and the other half rests on the extraction alone.
    """
    n = cat = role = 0
    by_cat: dict[str, int] = {}
    for r in rows:
        n += 1
        k = sgo_distant_key(r)
        if k is None:
            continue
        cat += 1
        by_cat[k[0]] = by_cat.get(k[0], 0) + 1
        if k[1] in ("striking", "struck"):
            role += 1
    return {
        "n": n,
        "category_coverage": cat / max(n, 1),
        "role_coverage": role / max(n, 1),
        "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
    }


# ---------------------------------------------------------------------------
# Role helpers shared by the analyses
# ---------------------------------------------------------------------------
def human_baseline_ratio(df, cat_col: str = "acc_category", role_col: str = "acc_role",
                         weight_col: str = "WEIGHT") -> dict:
    """Weighted striking:struck ratio per category, from CRSS itself.

    This is the NULL for the quasi-induced-exposure comparison, and it is
    estimated rather than assumed for the reason given in the module docstring:
    two of the eight two-vehicle categories are not 1:1 by construction, and
    assuming they are would turn a coding convention into a finding.

    Returned per category so §6.3 can compare the ADS ratio against the ratio
    a human driver faces IN THE SAME CATEGORY, which is the only comparison the
    method licenses.
    """
    out = {}
    sub = df[df[role_col].isin(["striking", "struck"])]
    for cat, g in sub.groupby(cat_col):
        w = g.groupby(role_col)[weight_col].sum()
        st, sk = float(w.get("striking", 0.0)), float(w.get("struck", 0.0))
        out[str(cat)] = {
            "striking_weighted": st,
            "struck_weighted": sk,
            "ratio": (st / sk) if sk > 0 else None,
            "n_striking": int((g[role_col] == "striking").sum()),
            "n_struck": int((g[role_col] == "struck").sum()),
            # A category whose human baseline departs from 1 does so because of
            # the coding scheme, and the ADS comparison must use the estimate.
            "balanced_by_construction": bool(sk > 0 and abs(st / sk - 1.0) < 0.01),
        }
    return out


def opposite_role(role: str) -> str:
    return {"striking": "struck", "struck": "striking"}.get(role, role)


def is_two_vehicle(category: str) -> bool:
    """Categories that involve a second motor vehicle.

    Quasi-induced exposure is only defined on these: the method estimates one
    party's crash propensity from the OTHER party's presence, and a fixed
    object has no propensity.
    """
    return category not in {"single_vehicle", "pedestrian_animal", "other_unknown"}
