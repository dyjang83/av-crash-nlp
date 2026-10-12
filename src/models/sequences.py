"""Conditional pre-crash sequences: what each party was doing, given crash type.

WHAT THIS ADDS OVER THE SHARE TABLE. "ADS vehicles have more rear-end crashes"
is a composition fact. "In intersection crashes the ADS is going straight while
the human turns across it" is a mechanism, and it is what makes "structurally
different" concrete rather than a table of proportions. This module compares
the SV/CP pre-crash maneuver PAIR, conditional on crash type and role.

-----------------------------------------------------------------------------
HOW THE TWO SIDES ARE MADE COMPARABLE
-----------------------------------------------------------------------------
The ADS side has a designated subject vehicle; the human side does not. A
human-vs-human crash has two vehicles and no reason to call either one the
subject, so there is no such thing as "the human SV maneuver distribution"
without first saying which vehicle plays the subject's part.

ROLE is what supplies that. Within a (crash type, role) cell -- say
`rear_end` / `struck` -- the ADS vehicle and a human struck vehicle occupy the
same structural position, and their maneuver distributions are comparable. The
partner's maneuver is then read off the other vehicle in the same crash.

This is why the human side must be built from the CRASH-LEVEL frame
(`crss_vehicles_qie.parquet`): pairing a vehicle with its partner requires both
to be present, and the vehicle-level filter can admit one and drop the other.
The same hazard documented in `models.qie` applies here for the same reason.

-----------------------------------------------------------------------------
WHAT THE CELL COUNTS WILL DO TO THIS
-----------------------------------------------------------------------------
Conditioning on crash type AND role AND a maneuver pair divides a few hundred
ADS crashes into a lot of cells. Most will be empty or near-empty, and a pair
seen three times is not a finding. Every cell therefore reports its n, cells
below `--min-cell` are suppressed rather than displayed with a noisy
percentage, and the suppression count is printed so the reader knows how much
of the table was withheld rather than absent.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.acc_type import is_two_vehicle  # noqa: E402
from models.composition import attach_structured_key  # noqa: E402

RESULTS = os.path.join("data", "processed")

# CRSS P_CRASH1 -> the schema's PreCrashMovement vocabulary. Coarser than
# either, because the two disagree in both directions -- the same discipline
# `schema.composition_distant` applies, and deliberately the same buckets, so a
# maneuver means one thing across the whole study.
_CRSS_MOVE = {
    "going straight": "going_straight",
    "decelerating in road": "decelerating",
    "accelerating in road": "accelerating",
    "stopped in roadway": "stopped",
    "disabled or \"parked\" in travel lane": "stopped",
    "starting in road": "starting",
    "turning left": "turning_left",
    "turning right": "turning_right",
    "making a u-turn": "turning_left",
    "changing lanes": "lane_change",
    "merging": "lane_change",
    "passing or overtaking another vehicle": "lane_change",
    "backing up (other than for parking position)": "backing",
    "negotiating a curve": "going_straight",
    "entering a parking position": "parking",
    "leaving a parking position": "parking",
}
_AV_MOVE = {
    "going_straight": "going_straight", "decelerating": "decelerating",
    "accelerating": "accelerating", "stopped_in_roadway": "stopped",
    "parked": "stopped", "starting_in_road": "starting",
    "turning_left": "turning_left", "turning_right": "turning_right",
    "making_u_turn": "turning_left", "changing_lanes": "lane_change",
    "merging": "lane_change", "passing": "lane_change", "backing": "backing",
    "negotiating_curve": "going_straight", "entering_parking": "parking",
    "leaving_parking": "parking",
}
MOVES = ["going_straight", "decelerating", "stopped", "starting", "accelerating",
         "turning_left", "turning_right", "lane_change", "backing", "parking"]


def _norm_crss_move(s) -> Optional[str]:
    return _CRSS_MOVE.get(str(s or "").strip().lower())


def _norm_av_move(s) -> Optional[str]:
    return _AV_MOVE.get(str(s or "").strip().lower())


# ---------------------------------------------------------------------------
# Human side: pair each vehicle with its partner
# ---------------------------------------------------------------------------
def human_pairs(h: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle in a two-vehicle crash, carrying the partner's maneuver.

    Restricted to crashes with exactly two vehicle records. Three-vehicle
    crashes have no single partner and pairing them would require choosing one,
    which is a convention rather than a measurement; they are dropped and
    counted.
    """
    d = h.copy()
    d["move"] = d["pre_crash_maneuver"].map(_norm_crss_move)
    sizes = d.groupby("CASENUM")["VEH_NO"].transform("size")
    two = d[sizes == 2].copy()
    if two.empty:
        return two.assign(partner_move=pd.Series(dtype=object))

    # The partner is the other row in the same crash. Summing the crash's
    # maneuvers and subtracting this row's would not work for strings, so the
    # partner is taken by reversing the within-crash order -- exact for n == 2.
    two = two.sort_values(["CASENUM", "VEH_NO"])
    two["partner_move"] = two.groupby("CASENUM")["move"].transform(
        lambda s: s.iloc[::-1].to_numpy())
    two["partner_role"] = two.groupby("CASENUM")["acc_role"].transform(
        lambda s: s.iloc[::-1].to_numpy())
    return two


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def _dist(values, weights, levels) -> np.ndarray:
    w = np.asarray(weights, float)
    v = np.asarray(values, dtype=object)
    tot = w.sum()
    if tot <= 0:
        return np.full(len(levels), np.nan)
    return np.array([w[v == lv].sum() / tot for lv in levels])


def subject_maneuver_by_cell(ads: pd.DataFrame, human: pd.DataFrame,
                             cat_col: str, role_col: str, move_col: str,
                             min_cell: int = 10) -> list[dict]:
    """Subject's pre-crash maneuver, by (crash type, role), both sides."""
    out = []
    a = ads.copy()
    a["_move"] = a[move_col].map(_norm_av_move)
    h = human.copy()
    h["_move"] = h["pre_crash_maneuver"].map(_norm_crss_move)

    for cat in sorted(set(a[cat_col].dropna().astype(str))):
        if not is_two_vehicle(cat):
            continue
        for role in ("striking", "struck"):
            ac = a[(a[cat_col].astype(str) == cat) & (a[role_col] == role)
                   & a["_move"].notna()]
            hc = h[(h["acc_category"] == cat) & (h["acc_role"] == role)
                   & h["_move"].notna()]
            if len(ac) < min_cell or hc.empty:
                out.append({"category": cat, "role": role, "n_ads": int(len(ac)),
                            "n_human": int(len(hc)), "suppressed": True})
                continue
            av = _dist(ac["_move"], np.ones(len(ac)), MOVES)
            hu = _dist(hc["_move"], hc["_w"], MOVES)
            out.append({
                "category": cat, "role": role,
                "n_ads": int(len(ac)), "n_human": int(len(hc)),
                "suppressed": False, "levels": MOVES,
                "ads": av.tolist(), "human": hu.tolist(),
                "ratio": (av / np.where(hu > 0, hu, np.nan)).tolist(),
            })
    return out


def maneuver_pairs_by_cell(ads: pd.DataFrame, human_pairs_df: pd.DataFrame,
                           cat_col: str, role_col: str, sv_col: str,
                           cp_col: str, min_cell: int = 10,
                           top_k: int = 6) -> list[dict]:
    """The (subject maneuver, partner maneuver) joint, by crash type and role.

    This is the sequence itself -- not "the ADS was going straight" and
    separately "the partner turned", but the two together, which is the claim
    worth making.
    """
    out = []
    a = ads.copy()
    a["_sv"] = a[sv_col].map(_norm_av_move)
    a["_cp"] = a[cp_col].map(_norm_av_move)
    h = human_pairs_df

    for cat in sorted(set(a[cat_col].dropna().astype(str))):
        if not is_two_vehicle(cat):
            continue
        for role in ("striking", "struck"):
            ac = a[(a[cat_col].astype(str) == cat) & (a[role_col] == role)
                   & a["_sv"].notna() & a["_cp"].notna()]
            hc = h[(h["acc_category"] == cat) & (h["acc_role"] == role)
                   & h["move"].notna() & h["partner_move"].notna()]
            if len(ac) < min_cell or hc.empty:
                continue

            ads_pairs = (ac["_sv"] + " -> " + ac["_cp"]).value_counts()
            ads_share = ads_pairs / ads_pairs.sum()
            hk = (hc["move"] + " -> " + hc["partner_move"])
            hw = hc["_w"].groupby(hk).sum()
            human_share = hw / hw.sum()

            rows = []
            for pair in ads_share.head(top_k).index:
                hs = float(human_share.get(pair, 0.0))
                rows.append({
                    "pair": pair, "n_ads": int(ads_pairs[pair]),
                    "ads_share": float(ads_share[pair]), "human_share": hs,
                    "ratio": (float(ads_share[pair]) / hs) if hs > 0 else None,
                })
            out.append({"category": cat, "role": role, "n_ads": int(len(ac)),
                        "n_human": int(len(hc)), "top_pairs": rows})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Conditional pre-crash maneuver sequences, ADS vs human.")
    ap.add_argument("--ads", default=os.path.join("data", "interim",
                                                  "ads_reportability.parquet"))
    ap.add_argument("--human-qie",
                    default=os.path.join("data", "interim", "crss_vehicles_qie.parquet"),
                    help="The CRASH-LEVEL frame. Pairing a vehicle with its "
                         "partner needs both present.")
    ap.add_argument("--tier", default="tier_a")
    ap.add_argument("--source", default="extracted",
                    choices=["extracted", "structured"])
    ap.add_argument("--driverless-only", action="store_true")
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(RESULTS, "sequences.json"))
    a = ap.parse_args()

    ads = attach_structured_key(pd.read_parquet(a.ads))
    if a.driverless_only:
        ads = ads[ads["driverless"].eq(True)]
    ads = ads[ads["incident_year"].isin(a.years)]
    if a.tier in ads.columns:
        ads = ads[ads[a.tier]]
    ads = ads.copy()

    if a.source == "extracted":
        cat_col, role_col = "acc_type_category", "striking_role"
        sv_col, cp_col = "sv_pre_crash_movement", "cp_pre_crash_movement"
        if cat_col not in ads.columns or ads[cat_col].isna().all():
            raise SystemExit(
                "[sequences] no extracted crash-type labels. Run "
                "extract.composition_extract, then report.reportability, or "
                "pass --source structured.")
    else:
        cat_col, role_col = "acc_category_struct", "acc_role_struct"
        sv_col, cp_col = "SV Pre-Crash Movement", "CP Pre-Crash Movement"
        # The structured movement columns use SGO's vocabulary, not the
        # schema's, so they are normalised into the schema space first.
        sgo_to_schema = {
            "stopped": "stopped_in_roadway", "parked": "parked",
            "proceeding straight": "going_straight",
            "making left turn": "turning_left", "making right turn": "turning_right",
            "making u-turn": "making_u_turn", "changing lanes": "changing_lanes",
            "merging": "merging", "passing": "passing", "backing": "backing",
            "parking maneuver": "entering_parking",
            "entering traffic": "starting_in_road",
        }
        for c, nc in ((sv_col, "_sv_norm"), (cp_col, "_cp_norm")):
            ads[nc] = ads[c].astype(str).str.strip().str.lower().map(sgo_to_schema)
        sv_col, cp_col = "_sv_norm", "_cp_norm"

    human = pd.read_parquet(a.human_qie)
    if a.tier == "tier_a" and "tier_a_crash" in human.columns:
        human = human[human["tier_a_crash"]]
    hp = human_pairs(human)

    subj = subject_maneuver_by_cell(ads, human, cat_col, role_col, sv_col,
                                    min_cell=a.min_cell)
    pairs = maneuver_pairs_by_cell(ads, hp, cat_col, role_col, sv_col, cp_col,
                                   min_cell=a.min_cell)

    rep = {"tier": a.tier, "source": a.source, "years": a.years,
           "driverless_only": a.driverless_only, "min_cell": a.min_cell,
           "n_ads": int(len(ads)), "n_human_vehicles": int(len(human)),
           "n_human_paired": int(len(hp)),
           "subject_maneuver": subj, "maneuver_pairs": pairs}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    n_sup = sum(1 for s in subj if s.get("suppressed"))
    print(f"[sequences] ADS n={len(ads):,} (tier={a.tier}, {a.source} labels); "
          f"human {len(human):,} vehicles, {len(hp):,} in two-vehicle crashes")
    print(f"[sequences] {n_sup} of {len(subj)} (type, role) cells suppressed "
          f"below n={a.min_cell}")
    for s in subj:
        if s.get("suppressed"):
            continue
        print(f"\n=== {s['category']} / SV is {s['role']}  "
              f"(ADS n={s['n_ads']}, human n={s['n_human']:,})")
        print(f"{'subject maneuver':20s} {'ADS':>8s} {'human':>8s} {'ratio':>7s}")
        for i, lv in enumerate(s["levels"]):
            if s["ads"][i] < 0.02 and s["human"][i] < 0.02:
                continue
            r = s["ratio"][i]
            rs = f"{r:.2f}x" if r and np.isfinite(r) else "   -  "
            print(f"{lv:20s} {s['ads'][i]:8.3f} {s['human'][i]:8.3f} {rs:>7s}")
    for p in pairs:
        print(f"\n--- sequences: {p['category']} / SV {p['role']} "
              f"(ADS n={p['n_ads']})")
        for r in p["top_pairs"]:
            rs = f"{r['ratio']:.2f}x" if r["ratio"] else "  n/a "
            print(f"    {r['pair']:42s} ADS {r['ads_share']:.3f} "
                  f"human {r['human_share']:.3f}  {rs}  (n={r['n_ads']})")
    print(f"\n[sequences] wrote {a.out}")


if __name__ == "__main__":
    main()
