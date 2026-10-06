"""San Francisco (SWITRS) as a geographically matched human comparator.

WHY THIS EXISTS. CRSS is nationally representative and cannot isolate a city --
its PSUs are confidential. The ADS fleet is concentrated in a handful of metros
whose crash mix differs from the urban US average, so "ADS crashes differ from
human crashes" is confounded with "San Francisco differs from urban America".
That is the single largest remaining threat to the composition result, and the
only way to address it is a human comparator drawn from the same streets.

San Francisco is the right city: 452 of the 829 ADS incidents in the 2021-24
window are there (99 of them clearing Tier A), against 44 in Austin and 83 in
Phoenix. DataSF publishes the SWITRS extract as `Traffic Crashes Resulting in
Injury` plus a party-level table, which is what makes the reconstruction below
possible.

-----------------------------------------------------------------------------
WHAT SWITRS CANNOT SAY, AND HOW THE MOVEMENT PAIR RECOVERS IT
-----------------------------------------------------------------------------
SWITRS `type_of_collision` is COARSER than the CRSS typology in exactly the
place that matters. Its `Broadside` value pools three CRSS categories --
turn-across-path, turn-into-path and intersecting-straight-paths -- and the
strongest finding in this study lives in the first of them. Taken at face value
the SF comparator could not test that finding at all.

The party table supplies the missing dimension: `move_pre_acc` records each
party's pre-crash movement, so a Broadside in which one party was `Making Left
Turn` and the other `Proceeding Straight` is a turn-across-path, and one in
which both were `Proceeding Straight` is an intersecting-straight-paths crash.
That is the same information CRSS folds into `ACC_TYPE`, arriving as two fields
instead of one.

THE RECONSTRUCTION IS A CONVENTION, SO IT IS VALIDATED RATHER THAN ASSERTED.
`validate_on_crss` applies the identical rule to CRSS -- which carries both the
movement fields AND the true `ACC_TYPE` -- and reports how often the
reconstruction recovers the true category. A rule that cannot reproduce CRSS's
own categories from CRSS's own movements has no business being applied to SWITRS.

-----------------------------------------------------------------------------
WHAT THIS COMPARATOR IS AND IS NOT
-----------------------------------------------------------------------------
INJURY CRASHES ONLY. DataSF publishes injury crashes; property-damage-only
crashes are absent. That is a severity floor built into the source, and it is
close to the `tier_a_no_tow` definition on the ADS side (injury or airbag, no
tow leg) -- which is the tow-de-confounded floor, so the pairing is a
reasonable one. It is NOT comparable to the permissive Tier A, and the ADS side
must be restricted accordingly.

NO SURVEY WEIGHTS. SWITRS is a census of reported injury crashes in the city,
not a sample, so every crash has weight 1 and uncertainty is an ordinary
bootstrap. This differs from the CRSS side and is not a defect of either.
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
from schema.acc_type import CATEGORIES, crss_acc_type  # noqa: E402

RAW = os.path.join("data", "raw", "state")
CRASHES = os.path.join(RAW, "sf_crashes.json")
PARTIES = os.path.join(RAW, "sf_parties.json")

SF_CRASH_DATASET = "ubvf-ztfx"
SF_PARTY_DATASET = "8gtc-pjc6"


# ---------------------------------------------------------------------------
# Movement vocabulary, shared by the reconstruction on both sources
# ---------------------------------------------------------------------------
# SWITRS `move_pre_acc` and CRSS `P_CRASH1` use different words for the same
# small set of movements. Both are normalised into one vocabulary so the
# reconstruction rule is literally the same function on both sides -- which is
# what makes validating it on CRSS meaningful.
_MOVE = {
    # SWITRS
    "proceeding straight": "straight", "making left turn": "left",
    "making right turn": "right", "making u turn": "left",
    "stopped": "stopped", "parked": "stopped", "slowing/stopping": "straight",
    "backing": "backing", "changing lanes": "lane_change",
    "passing other vehicle": "lane_change", "entering traffic": "entering",
    "parking maneuver": "parking", "merging": "lane_change",
    # CRSS P_CRASH1
    "going straight": "straight", "turning left": "left",
    "turning right": "right", "making a u-turn": "left",
    "stopped in roadway": "stopped", "decelerating in road": "straight",
    "negotiating a curve": "straight", "changing lanes": "lane_change",
    "passing or overtaking another vehicle": "lane_change",
    "backing up (other than for parking position)": "backing",
    "starting in road": "entering", "merging": "lane_change",
    "entering a parking position": "parking",
    "leaving a parking position": "parking",
    "disabled or \"parked\" in travel lane": "stopped",
    "accelerating in road": "straight",
}


def norm_move(v) -> Optional[str]:
    return _MOVE.get(str(v or "").strip().lower())


# ---------------------------------------------------------------------------
# The reconstruction rule
# ---------------------------------------------------------------------------
def reconstruct(manner: str, moves: list, has_nonmotorist: bool = False
                ) -> Optional[str]:
    """Crash category from a coarse manner code plus the pair of movements.

    `manner` is a normalised family: angle / rear_end / head_on / sideswipe /
    object / pedestrian / other. `moves` is the normalised movement of each
    party.

    Returns None where the inputs do not determine a category. Guessing here
    would be the same invented convention the SGO distant key refuses, and the
    coverage is reported so a reader can see how much of the comparator rests
    on it.
    """
    if has_nonmotorist or manner == "pedestrian":
        return "pedestrian_animal"
    if manner == "object":
        return "single_vehicle"
    if manner == "rear_end":
        return "rear_end"
    if manner == "head_on":
        return "head_on"
    if manner == "sideswipe":
        # SWITRS does not separate same- from opposite-direction sideswipes.
        # Mapped to the same-direction category, which is ~92% of sideswipes in
        # CRSS, and flagged in COARSENED.
        return "sideswipe_same_direction"
    if "backing" in moves:
        return "backing"
    if manner == "angle":
        turns = [m for m in moves if m in ("left", "right")]
        straights = [m for m in moves if m == "straight"]
        if len(moves) < 2:
            return None
        if not turns and len(straights) >= 2:
            return "straight_paths_intersecting"
        if len(turns) == 1 and straights:
            # A left turn crosses the other party's path; a right turn merges
            # into it. This is the distinction CRSS draws between its J and K
            # families, and it is the one the whole SF comparator exists to
            # recover.
            return "turn_across_path" if turns[0] == "left" else "turn_into_path"
        return None
    return None


_SWITRS_MANNER = {
    "broadside": "angle", "rear end": "rear_end", "head-on": "head_on",
    "sideswipe": "sideswipe", "hit object": "object", "overturned": "object",
    "vehicle/pedestrian": "pedestrian", "other": "other",
    "not stated": "other",
}


def _crss_manner(name) -> Optional[str]:
    """CRSS `MAN_COLL` into the same coarse family SWITRS `type_of_collision` gives.

    CRSS spells these with HYPHENS -- `Front-to-Rear`, `Front-to-Front`,
    `Sideswipe - Same Direction` -- which a space-separated substring test
    silently misses. It did: the first validation run scored rear-end recall at
    0.000 and head-on at 0.000, which reads as a catastrophic rule failure and
    was a string-matching bug. Hyphens are normalised to spaces before any
    comparison, and the validation is what surfaced it.
    """
    n = str(name or "").strip().lower().replace("-", " ")
    n = " ".join(n.split())
    if not n or n == "not reported":
        return None
    if n.startswith("angle"):
        return "angle"
    if "front to rear" in n:
        return "rear_end"
    if "front to front" in n:
        return "head_on"
    if "sideswipe" in n:
        return "sideswipe"
    if "not a collision with a motor vehicle" in n:
        return "object"
    return "other"


# ---------------------------------------------------------------------------
# Validation on CRSS, where the true category is known
# ---------------------------------------------------------------------------
def validate_on_crss(years=(2023,), crss_dir=os.path.join("data", "raw", "crss")
                     ) -> dict:
    """Apply the reconstruction to CRSS and score it against true ACC_TYPE.

    CRSS carries `MAN_COLL` (the coarse manner, equivalent to SWITRS
    `type_of_collision`) and `PCRASH1_IM` (the movement, equivalent to
    `move_pre_acc`) AND the full `ACC_TYPE`. So the rule can be run on exactly
    the inputs SWITRS provides and scored against the answer SWITRS lacks.

    Restricted to two-vehicle crashes, which is what the rule is defined on.
    """
    frames = []
    for y in years:
        v = pd.read_csv(os.path.join(crss_dir, str(y), "vehicle.csv"),
                        low_memory=False,
                        usecols=["CASENUM", "VEH_NO", "ACC_TYPE", "MAN_COLLNAME",
                                 "PCRASH1_IMNAME", "VE_FORMS"])
        frames.append(v)
    v = pd.concat(frames, ignore_index=True)
    v = v[v["VE_FORMS"] == 2]

    truth = v["ACC_TYPE"].map(crss_acc_type)
    v["_true"] = truth.map(lambda x: x[0] if x else None)
    v["_move"] = v["PCRASH1_IMNAME"].map(norm_move)
    v["_manner"] = v["MAN_COLLNAME"].map(_crss_manner)

    rows = []
    for case, g in v.groupby("CASENUM"):
        moves = [m for m in g["_move"].tolist() if m]
        manner = g["_manner"].iloc[0]
        pred = reconstruct(manner, moves)
        for t in g["_true"].tolist():
            rows.append({"true": t, "pred": pred})
    df = pd.DataFrame(rows).dropna(subset=["true"])

    covered = df[df["pred"].notna()]
    per = {}
    for cat in CATEGORIES:
        sub = df[df["true"] == cat]
        if not len(sub):
            continue
        hit = (sub["pred"] == cat).sum()
        per[cat] = {"n_true": int(len(sub)),
                    "recall": float(hit / len(sub)),
                    "n_predicted": int((df["pred"] == cat).sum()),
                    "precision": (float(hit / (df["pred"] == cat).sum())
                                  if (df["pred"] == cat).sum() else None)}
    # THE TWO TURNING CATEGORIES ARE CONFUSED WITH EACH OTHER, NOT WITH
    # ANYTHING ELSE. `turn_across_path` recalls at 0.571 and `turn_into_path`
    # at 0.155, which looks like the rule failing on both. It is one failure:
    # the rule keys on the TURN DIRECTION (left crosses, right merges) while
    # CRSS keys on the two vehicles' INITIAL DIRECTIONS, so a left turn into
    # the path of same-direction traffic is misfiled as across-path. SWITRS
    # carries `dir_of_travel` and could in principle resolve it; CRSS's vehicle
    # file carries no comparable heading, so a corrected rule could not be
    # validated here and is not attempted.
    #
    # What can be salvaged is the union. If the confusion is internal, the
    # merged "turning conflict" category should recover well, and it still
    # answers a real question -- whether ADS vehicles are under-represented in
    # crashes where one party turns into or across another's path.
    turning = {"turn_across_path", "turn_into_path"}
    t_true = df["true"].isin(turning)
    t_pred = df["pred"].isin(turning)
    merged = {
        "n_true": int(t_true.sum()),
        "recall": float((t_pred & t_true).sum() / max(t_true.sum(), 1)),
        "precision": float((t_pred & t_true).sum() / max(t_pred.sum(), 1))
        if t_pred.sum() else None,
    }
    return {
        "years": list(years),
        "n_vehicles": int(len(df)),
        "coverage": float(len(covered) / max(len(df), 1)),
        "accuracy_on_covered": float((covered["pred"] == covered["true"]).mean())
        if len(covered) else None,
        "per_category": per,
        "turning_conflict_merged": merged,
    }


# ---------------------------------------------------------------------------
# SF
# ---------------------------------------------------------------------------
def load_sf(years=(2021, 2022, 2023, 2024)) -> pd.DataFrame:
    """One row per SF injury crash with a reconstructed category."""
    if not (os.path.exists(CRASHES) and os.path.exists(PARTIES)):
        raise SystemExit(
            f"{CRASHES} / {PARTIES} missing. Fetch them from DataSF datasets "
            f"{SF_CRASH_DATASET} and {SF_PARTY_DATASET}.")
    c = pd.DataFrame(json.load(open(CRASHES)))
    p = pd.DataFrame(json.load(open(PARTIES)))

    c["_year"] = pd.to_numeric(c["accident_year"], errors="coerce")
    c = c[c["_year"].isin(list(years))].copy()
    p["_year"] = pd.to_numeric(p["accident_year"], errors="coerce")
    p = p[p["_year"].isin(list(years))].copy()

    p["_move"] = p["move_pre_acc"].map(norm_move)
    p["_nonmotorist"] = p["party_type"].astype(str).str.lower().isin(
        {"pedestrian", "bicyclist"})

    agg = p.groupby("case_id_pkey").agg(
        moves=("_move", lambda s: [x for x in s if isinstance(x, str)]),
        nonmotorist=("_nonmotorist", "any"),
        n_parties=("case_id_pkey", "size"),
    )
    c = c.merge(agg, left_on="case_id_pkey", right_index=True, how="left")
    c["moves"] = c["moves"].apply(lambda x: x if isinstance(x, list) else [])
    c["nonmotorist"] = c["nonmotorist"].fillna(False)

    c["_manner"] = c["type_of_collision"].astype(str).str.strip().str.lower().map(
        _SWITRS_MANNER).fillna("other")
    c["category"] = [reconstruct(m, mv, nm) for m, mv, nm in
                     zip(c["_manner"], c["moves"], c["nonmotorist"])]
    c["_w"] = 1.0
    c["side"] = "sf_switrs"
    return c


COARSENED = {
    "sideswipe": "SWITRS does not separate same- from opposite-direction "
                 "sideswipes; both map to same_direction",
    "severity": "DataSF publishes INJURY crashes only, so the comparator has a "
                "built-in severity floor closest to tier_a_no_tow",
    "angle": "turn_across_path / turn_into_path / straight_paths_intersecting "
             "are reconstructed from the party movement pair, not read off a "
             "code; see validate_on_crss for the recovery rate",
}


def main():
    import argparse
    from models.composition import ads_shares, attach_structured_key, filter_entities

    ap = argparse.ArgumentParser(
        description="San Francisco (SWITRS) geographically matched comparator.")
    ap.add_argument("--ads", default=os.path.join("data", "interim",
                                                  "ads_reportability.parquet"))
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2021, 2022, 2023, 2024])
    ap.add_argument("--tier", default="tier_a_no_tow",
                    help="ADS-side floor. Default matches the injury-only "
                         "floor built into the SF source.")
    ap.add_argument("--driverless-only", action="store_true")
    ap.add_argument("--validate-years", nargs="+", type=int, default=[2023])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join("data", "processed",
                                                  "state_sf.json"))
    a = ap.parse_args()

    val = validate_on_crss(years=tuple(a.validate_years))
    print(f"[sf] reconstruction validated on CRSS {a.validate_years}: "
          f"covers {val['coverage']:.1%} of two-vehicle crash records, "
          f"{val['accuracy_on_covered']:.1%} accurate where it commits")
    print(f"[sf] {'category':32s} {'n_true':>8s} {'recall':>8s} {'prec':>8s}")
    for cat, m in sorted(val["per_category"].items(), key=lambda kv: -kv[1]["n_true"]):
        pr = f"{m['precision']:.3f}" if m["precision"] is not None else "   -  "
        print(f"[sf] {cat:32s} {m['n_true']:8d} {m['recall']:8.3f} {pr:>8s}")

    m = val["turning_conflict_merged"]
    print(f"[sf] merged turning-conflict (across+into): recall {m['recall']:.3f}, "
          f"precision {m['precision']:.3f} on n={m['n_true']:,} -- the two are "
          f"confused with EACH OTHER, so the union is usable where neither is")

    sf = load_sf(tuple(a.years))
    cov = sf["category"].notna().mean()
    print(f"\n[sf] SF injury crashes {a.years}: {len(sf):,}; "
          f"category reconstructed for {cov:.1%}")

    ads = attach_structured_key(pd.read_parquet(a.ads))
    ads = ads[ads["City"].astype(str).str.strip().str.lower().eq("san francisco")]
    ads = ads[ads["incident_year"].isin(a.years)]
    if a.driverless_only:
        ads = ads[ads["driverless"].eq(True)]
    ads = ads[ads[a.tier]].copy() if a.tier in ads.columns else ads.copy()

    levels = CATEGORIES
    sub = sf[sf["category"].notna()]
    human = np.array([(sub["category"] == lv).mean() for lv in levels])
    av = ads_shares(ads, "acc_type_category", levels, n_boot=a.n_boot)

    # Merged turning-conflict row, reported because the two components are
    # individually unreliable in the reconstruction but their union is not.
    turning = ["turn_across_path", "turn_into_path"]
    sf_turn = float(sub["category"].isin(turning).mean())
    ads_turn = float(ads["acc_type_category"].isin(turning).mean()) if len(ads) else float("nan")

    rows = []
    for i, lv in enumerate(levels):
        h, v = float(human[i]), av["share"][i]
        lo, hi = av.get("ci_lo", [None] * len(levels))[i], av.get("ci_hi", [None] * len(levels))[i]
        rows.append({"category": lv, "sf_human_share": h, "ads_share": v,
                     "ads_n": av.get("counts", [0] * len(levels))[i],
                     "ratio": (v / h) if h > 0 else None,
                     "ci_excludes_human": bool(lo is not None and (lo > h or hi < h))})

    rep = {"years": a.years, "tier": a.tier,
           "driverless_only": a.driverless_only,
           "n_sf_crashes": int(len(sf)), "n_sf_categorised": int(len(sub)),
           "sf_category_coverage": float(cov),
           "n_ads_sf": int(len(ads)),
           "reconstruction_validation": val, "coarsened": COARSENED,
           "turning_conflict": {"sf_human_share": sf_turn, "ads_share": ads_turn,
                                "ratio": (ads_turn / sf_turn) if sf_turn else None,
                                "n_ads": int(ads["acc_type_category"].isin(turning).sum()) if len(ads) else 0},
           "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    print(f"\n[sf] ADS in San Francisco, tier={a.tier}: n={len(ads)}")
    print(f"{'category':32s} {'ADS':>8s} {'SF human':>9s} {'ratio':>8s}  n")
    for r in rows:
        if not r["ads_n"] and r["sf_human_share"] < 0.005:
            continue
        rt = f"{r['ratio']:.2f}x" if r["ratio"] else "   -  "
        print(f"{r['category']:32s} {r['ads_share']:8.3f} "
              f"{r['sf_human_share']:9.3f} {rt:>8s}  {r['ads_n']}"
              f"{' *' if r['ci_excludes_human'] else ''}")
    tc = rep["turning_conflict"]
    print(f"{'TURNING CONFLICT (across+into)':32s} {tc['ads_share']:8.3f} "
          f"{tc['sf_human_share']:9.3f} "
          f"{(str(round(tc['ratio'],2))+'x') if tc['ratio'] else '   -  ':>8s}  {tc['n_ads']}")
    print("\n[sf] * = ADS 95% CI excludes the SF human share")
    print(f"[sf] wrote {a.out}")


if __name__ == "__main__":
    main()
