"""The human comparator: a vehicle-level CRSS extract, frozen before comparison.

WHY VEHICLE-LEVEL. The composition question is about a VEHICLE's crash -- what
type it was, what the vehicle was doing before it, whether it struck or was
struck. `ACC_TYPE`, `P_CRASH1`, `IMPACT1` and `TOWED` are all vehicle-level in
CRSS, and rolling them up to the crash would destroy the role structure the
quasi-induced-exposure analysis depends on. The existing `models/av_vs_human.py`
works crash-level because it compares severity distributions, which are crash
outcomes; this module exists beside it rather than replacing it.

The corresponding ADS unit is the SUBJECT VEHICLE of one SGO filing, which is
one vehicle in one crash. So the two sides pair correctly: one row per
vehicle-in-a-crash, and the ADS side is the ADS vehicle's row.

-----------------------------------------------------------------------------
SURVEY DESIGN IS NOT OPTIONAL HERE
-----------------------------------------------------------------------------
CRSS is a stratified cluster probability sample. Every estimate must be
`WEIGHT`-ed and every variance must come from resampling `PSU_VAR` within
`PSUSTRAT`. An unweighted CRSS count is not an estimate of anything, and an iid
bootstrap over rows understates the variance badly because PSUs are geographic
and internally homogeneous. The design columns travel with the frame so no
downstream step can lose them.

Pooling years namespaces both design identifiers by year: a PSU identifier
means a different unit in a different annual sample, and pooling 2021's PSU 12
with 2024's PSU 12 would treat two unrelated clusters as one.

-----------------------------------------------------------------------------
THE OPERATING-DOMAIN RESTRICTIONS, AND WHAT THEY CANNOT FIX
-----------------------------------------------------------------------------
ADS vehicles do not operate everywhere. Comparing their crash mix against all
US police-reported crashes would mostly measure that ADS fleets avoid rural
roads and interstates. The restrictions below (urban, non-interstate,
non-freeway, posted speed <= 45) are the HARD part of the alignment; the
reweighting in `human.balance` handles the residual within-domain mismatch.

What no restriction fixes, and what belongs in the limitations rather than in
this module:

  GEOGRAPHY. CRSS cannot isolate cities or even states -- its PSUs are
  confidential. The ADS fleet is concentrated in a handful of metros whose
  crash mix differs from the urban US average. This is the critique the
  state-database sensitivity run answers, not this extract.

  SPEED LIMIT IS 13% NOT-REPORTED. `VSPD_LIM` is missing on about one vehicle
  in eight. Dropping those conditions the comparison on the speed limit being
  recorded, which correlates with crash severity and with road type. They are
  therefore retained under a `speed_known` flag, and the restriction is applied
  in two versions -- drop-unknown and keep-unknown -- so the reader can see
  which results depend on the choice.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.acc_type import (crss_acc_type, crss_vru_from_harm_ev,  # noqa: E402
                             verify_crss_names, verify_role_orientation)
from schema.crss_map import (WEIGHT_COL, PSU_COL, STRATUM_COL,  # noqa: E402
                             crss_severity_common, is_missing, _norm)

CRSS_DIR = os.path.join("data", "raw", "crss")

# Columns pulled from each file. Named explicitly rather than read wholesale:
# the vehicle file is 169 columns and loading it four times over is the
# difference between a 20-second and a 3-minute run.
_VEH_COLS = [
    "CASENUM", "VEH_NO", "PSU_VAR", "PSUSTRAT", "WEIGHT", "URBANICITYNAME",
    "VE_FORMS", "BODY_TYPNAME", "TOWEDNAME", "DEFORMEDNAME", "IMPACT1NAME",
    "ACC_TYPE", "ACC_TYPENAME", "P_CRASH1NAME", "PCRASH1_IMNAME",
    "VSPD_LIM", "VSPD_LIMNAME", "VTRAFWAYNAME", "HARM_EVNAME",
    "MAX_VSEVNAME", "MXVSEV_IMNAME", "HIT_RUNNAME", "TRAV_SP",
]
_ACC_COLS = [
    "CASENUM", "URBANICITYNAME", "INT_HWYNAME", "RELJCT2NAME", "TYP_INTNAME",
    "LGT_CONDNAME", "WEATHERNAME", "MAN_COLLNAME", "MAXSEV_IMNAME", "MAX_SEVNAME",
    "HOUR_IM", "MONTH", "WEIGHT",
]
_PER_COLS = ["CASENUM", "VEH_NO", "AIR_BAGNAME", "INJSEV_IMNAME", "PER_TYPNAME"]


# ---------------------------------------------------------------------------
# Value projections
# ---------------------------------------------------------------------------
def _speed_limit(row) -> Optional[float]:
    """Posted speed limit in mph, or None.

    `VSPD_LIM` carries two non-numeric sentinels that mean different things:
    'No Statutory Limit/Non-Trafficway or Driveway Access' is a real road
    condition (treated as unknown for the <=45 restriction, because a driveway
    is not a 45 mph road and is not a 25 mph road either), and 'Not Reported' /
    'Reported as Unknown' are missingness.
    """
    n = _norm(row.get("VSPD_LIMNAME"))
    if n is None:
        return None
    m = pd.Series([n]).str.extract(r"^(\d+)\s*mph")[0].iloc[0]
    if pd.isna(m):
        return None
    v = float(m)
    return v if 0 < v <= 85 else None


_FREEWAY_VTRAFWAY = {"entrance exit ramp"}


def _towed(v) -> Optional[bool]:
    """Tow-away of THIS vehicle.

    `TOWED` is this vehicle's disposition; `TOW_VEH` is about trailers and is
    not it. Values distinguish towed-due-to-damage from towed-for-other-reason,
    and only the former is the severity signal SGO's tow field corresponds to.
    """
    n = _norm(v)
    if n is None or is_missing(n):
        return None
    if n.startswith("not towed"):
        return False
    if "towed" in n and "not towed" not in n:
        # 'Towed Due to Disabling Damage' and the bare 'Towed' both count;
        # 'Towed, But Not Due to Disabling Damage' does not -- it is a
        # convenience tow and carries no damage information.
        if "not due to disabling damage" in n:
            return False
        return True
    return None


def _airbag_deployed(v) -> Optional[bool]:
    n = _norm(v)
    if n is None or is_missing(n):
        return None
    if n.startswith("deployed") or n.startswith("deployment") and "unknown" not in n:
        return True
    if n.startswith("not deployed"):
        return False
    return None


def _deformed(v) -> Optional[str]:
    """CRSS extent of damage, coarsened to the three levels SGO could express.

    SGO files no damage-extent field at all before the third amendment, so this
    is the human side of a comparison whose AV side must be hand-coded on a
    sample. Kept coarse deliberately: `DEFORMED` has a 'Minor' / 'Functional' /
    'Disabling' ladder, and a finer split would not survive projection.
    """
    n = _norm(v)
    if n is None or is_missing(n):
        return None
    if "no damage" in n:
        return "none"
    if "minor" in n:
        return "minor"
    if "functional" in n:
        return "functional"
    if "disabling" in n:
        return "disabling"
    return None


def _intersection(row) -> Optional[str]:
    n = _norm(row.get("RELJCT2NAME"))
    if n is None or is_missing(n):
        return None
    return "intersection" if n in {"intersection", "intersection related"} \
        else "non_intersection"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_year(year: int, crss_dir: str = CRSS_DIR) -> pd.DataFrame:
    """One CRSS annual release as a vehicle-level frame with derived fields."""
    d = os.path.join(crss_dir, str(year))
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"{d} not found. Run `python3 -m fetch.fetch_crss --years {year}`.")

    def _read(name, cols):
        path = os.path.join(d, name)
        have = set(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in cols if c not in have]
        if missing:
            # A renamed column must stop the run. CRSS renames between
            # releases, and silently returning NaN for a restriction variable
            # would quietly widen the comparison population.
            raise KeyError(f"{name} ({year}) is missing {missing}. Check the "
                           f"Analytical User's Manual for this release.")
        return pd.read_csv(path, usecols=cols, low_memory=False)

    veh = _read("vehicle.csv", _VEH_COLS)
    acc = _read("accident.csv", _ACC_COLS)
    per = _read("person.csv", _PER_COLS)

    # Two guards, catching two different failures.
    #
    # Names catch a RENUMBERING: NHTSA shifts codes between releases and the
    # role map keys on integers.
    problems = verify_crss_names(
        veh[["ACC_TYPE", "ACC_TYPENAME"]].drop_duplicates().itertuples(
            index=False, name=None))
    # P_CRASH1 catches an INVERSION, which names cannot: a struck vehicle in a
    # rear-end crash was stopped or decelerating, and a striking one was going
    # straight. The rear-end family was in fact inverted on the first pass here
    # -- read as English the code names imply the opposite of what they mean --
    # and every balance statistic stayed at exactly 1.000 throughout, because
    # swapping both members of a pair is invisible to a ratio.
    counts = (veh.groupby(["ACC_TYPE", "P_CRASH1NAME"]).size()
                 .reset_index(name="n"))
    problems += verify_role_orientation(counts.itertuples(index=False, name=None))
    if problems:
        raise RuntimeError(
            f"CRSS {year} ACC_TYPE does not match schema/acc_type.py:\n  "
            + "\n  ".join(problems)
            + "\nThe role map assigns striking/struck by integer code; a "
              "renumbering or an inversion would corrupt every quasi-induced-"
              "exposure estimate without raising anything else.")

    # Airbag rolls up from the person file: deployed for THIS vehicle if any
    # occupant's airbag deployed. Non-occupants (pedestrians coded against a
    # vehicle) are excluded -- they have no airbag and would dilute the flag.
    occ = per[~per["PER_TYPNAME"].map(_norm).fillna("").str.contains("not a motor vehicle")]
    ab = (occ.assign(_d=occ["AIR_BAGNAME"].map(_airbag_deployed))
             .groupby(["CASENUM", "VEH_NO"])["_d"]
             .agg(lambda s: True if (s == True).any()  # noqa: E712
                  else (False if (s == False).any() else None)))  # noqa: E712

    acc = acc.rename(columns={"URBANICITYNAME": "_acc_urbanicity",
                              "WEIGHT": "_acc_weight"})
    v = veh.merge(acc, on="CASENUM", how="left", validate="many_to_one")
    v["airbag_deployed"] = pd.MultiIndex.from_frame(
        v[["CASENUM", "VEH_NO"]]).map(ab)

    # --- crash type and role ---------------------------------------------
    m = v["ACC_TYPE"].map(crss_acc_type)
    v["acc_category"] = m.map(lambda x: x[0] if x else None)
    v["acc_role"] = m.map(lambda x: x[1] if x else None)
    v["vru_involved"] = v["HARM_EVNAME"].map(crss_vru_from_harm_ev)

    # --- pre-crash maneuver ----------------------------------------------
    # The imputed variable is preferred for the same reason MAXSEV_IM is in
    # av_vs_human.py: the ~2% left unknown are not unknown at random.
    v["pre_crash_maneuver"] = v["PCRASH1_IMNAME"].fillna(v["P_CRASH1NAME"])

    # --- severity, tow, airbag, damage -----------------------------------
    v["severity"] = v["MAXSEV_IMNAME"].map(crss_severity_common)
    v["vehicle_severity"] = v["MXVSEV_IMNAME"].map(crss_severity_common)
    v["towed"] = v["TOWEDNAME"].map(_towed)
    v["deformed"] = v["DEFORMEDNAME"].map(_deformed)

    # --- operating domain -------------------------------------------------
    v["speed_limit"] = v.apply(_speed_limit, axis=1)
    v["speed_known"] = v["speed_limit"].notna()
    v["urban"] = v["_acc_urbanicity"].map(_norm).eq("urban area")
    v["interstate"] = v["INT_HWYNAME"].map(_norm).eq("yes")
    v["freeway_ramp"] = v["VTRAFWAYNAME"].map(_norm).isin(_FREEWAY_VTRAFWAY)
    v["intersection"] = v.apply(_intersection, axis=1)
    v["lighting"] = v["LGT_CONDNAME"]
    v["weather"] = v["WEATHERNAME"]
    v["hour"] = pd.to_numeric(v["HOUR_IM"], errors="coerce")

    # Passenger vehicles, for the sensitivity run. ADS fleets are passenger
    # cars and SUVs; heavy trucks and motorcycles have different crash physics.
    body = v["BODY_TYPNAME"].map(_norm).fillna("")
    v["passenger_vehicle"] = ~body.str.contains(
        "truck|motorcycle|moped|bus|atv|snowmobile|large limousine", regex=True)

    # --- design ------------------------------------------------------------
    v["_w"] = v[WEIGHT_COL].astype(float)
    v["_psu"] = v[PSU_COL]
    v["_stratum"] = v[STRATUM_COL]
    v["_year"] = year
    v["side"] = "human"
    return v


def load_years(years: Iterable[int], crss_dir: str = CRSS_DIR) -> pd.DataFrame:
    """Pool annual releases, namespacing the design identifiers by year."""
    frames = []
    for y in years:
        f = load_year(y, crss_dir)
        f["_psu"] = f"{y}_" + f["_psu"].astype(str)
        f["_stratum"] = f"{y}_" + f["_stratum"].astype(str)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    print(f"[crss] pooled {list(years)}: {len(out):,} vehicles, "
          f"weighted {out['_w'].sum():,.0f}")
    return out


# ---------------------------------------------------------------------------
# Operating-domain restriction
# ---------------------------------------------------------------------------
RESTRICTIONS = [
    ("urban", lambda d: d["urban"],
     "URBANICITY = Urban Area"),
    ("non_interstate", lambda d: ~d["interstate"],
     "INT_HWY != Yes"),
    ("non_freeway_ramp", lambda d: ~d["freeway_ramp"],
     "VTRAFWAY != Entrance/Exit Ramp"),
    ("speed_le_45", lambda d: d["speed_limit"].le(45),
     "posted speed limit <= 45 mph (IIHS found no Waymo crash above it)"),
]


def restrict(d: pd.DataFrame, steps: Optional[list] = None,
             keep_unknown_speed: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the hard operating-domain restrictions, returning the funnel.

    `keep_unknown_speed` governs the 13% of vehicles whose `VSPD_LIM` is not
    reported. Dropping them conditions the comparison on the speed limit having
    been recorded, which is not independent of road type or severity; keeping
    them admits vehicles that may have been on a 65 mph road. Neither is
    correct, so both are run and reported (see module docstring).
    """
    steps = steps or RESTRICTIONS
    funnel = [{"step": "all_vehicles", "n": int(len(d)),
               "weighted_n": float(d["_w"].sum()), "removed": 0, "why": ""}]
    out = d
    for name, pred, why in steps:
        mask = pred(out)
        if name == "speed_le_45" and keep_unknown_speed:
            mask = mask | ~out["speed_known"]
            why += " (vehicles with speed not reported RETAINED)"
        mask = mask.fillna(False) if hasattr(mask, "fillna") else mask
        removed = int((~mask).sum())
        out = out[mask].copy()
        funnel.append({"step": name, "n": int(len(out)),
                       "weighted_n": float(out["_w"].sum()),
                       "removed": removed, "why": why})
    return out, funnel


# ---------------------------------------------------------------------------
# Crash-level promotion: what quasi-induced exposure requires
# ---------------------------------------------------------------------------
# QUASI-INDUCED EXPOSURE BREAKS UNDER VEHICLE-LEVEL FILTERING, AND IT BREAKS
# SILENTLY. The method rests on the striking:struck ratio being 1 in a
# human-vs-human sample, because every two-vehicle crash contributes exactly
# one of each. Any filter that can admit ONE vehicle of a crash and reject the
# other destroys that guarantee, and the resulting ratio is then a property of
# the filter rather than of anyone's driving.
#
# Both filters in this module do exactly that:
#
#   `VSPD_LIM` IS VEHICLE-LEVEL. Two vehicles meeting at an intersection are on
#   different roads with different posted limits, and one may be not-reported
#   while the other is not. Measured on the pooled 2021-2024 extract, the
#   vehicle-level domain restriction moved `turn_into_path` from 1.000 to
#   0.682 and `straight_paths_intersecting` from 1.000 to 0.992 -- the
#   intersection categories, exactly where the two vehicles are most likely to
#   face different posted limits.
#
#   TIER A IS AN OUTCOME, AND OUTCOMES DIFFER BY ROLE. In a rear-end crash the
#   striking vehicle's front deforms and its airbag fires far more often than
#   the struck vehicle's rear. Applying the severity floor per vehicle moved
#   `rear_end` from 1.000 to 0.771 -- a 23% apparent asymmetry manufactured
#   entirely by the filter.
#
# The promotion below keeps a crash whole or drops it whole. `restrict` and
# `tier_a` remain available for the composition analyses, which are marginal
# distributions and do not depend on pairing.


def restrict_crash_level(d: pd.DataFrame, max_speed: int = 45
                         ) -> tuple[pd.DataFrame, list[dict]]:
    """Operating-domain restriction applied to whole crashes.

    `urban` and `interstate` come from the accident file and are already
    crash-level, so they need no promotion. `speed_limit` and `freeway_ramp`
    are vehicle-level and are promoted with an ALL rule: the crash is kept only
    if every vehicle in it that reports a speed limit reports one at or below
    the bar. The alternative (ANY) would admit crashes in which one party was
    on a 65 mph road, which is not the ADS operating domain.

    Crashes where NO vehicle reports a speed limit are kept or dropped by the
    same `speed_known` logic as `restrict`, and counted separately.
    """
    funnel = [{"step": "all_vehicles", "n": int(len(d)),
               "weighted_n": float(d["_w"].sum()), "removed": 0, "why": ""}]
    out = d

    for name, pred, why in [
        ("urban", lambda x: x["urban"], "URBANICITY = Urban Area (crash-level)"),
        ("non_interstate", lambda x: ~x["interstate"], "INT_HWY != Yes (crash-level)"),
    ]:
        mask = pred(out).fillna(False)
        funnel.append({"step": name, "n": int(mask.sum()),
                       "weighted_n": float(out.loc[mask, "_w"].sum()),
                       "removed": int((~mask).sum()), "why": why})
        out = out[mask].copy()

    # Promote the two vehicle-level conditions with an ALL rule over the crash.
    ok_speed = out["speed_limit"].isna() | out["speed_limit"].le(max_speed)
    ok_ramp = ~out["freeway_ramp"]
    keep = (out.assign(_ok=ok_speed & ok_ramp)
               .groupby("CASENUM")["_ok"].transform("all"))
    funnel.append({
        "step": f"crash_speed_le_{max_speed}_and_non_ramp",
        "n": int(keep.sum()), "weighted_n": float(out.loc[keep, "_w"].sum()),
        "removed": int((~keep).sum()),
        "why": f"every vehicle in the crash reporting a speed limit is <= "
               f"{max_speed} mph and none is on a ramp (promoted to crash level "
               f"so the striking/struck pairing survives)"})
    return out[keep].copy(), funnel


def tier_a_crash_level(d: pd.DataFrame) -> pd.Series:
    """Tier A promoted to the crash: true for every vehicle in a qualifying crash.

    A crash qualifies if ANY vehicle in it meets the objective floor, which is
    the same rule `av_vs_human.load_crss` already uses for tow-away and the
    same one SGO's own combined tow/airbag fields encode. Returned per vehicle
    so it can be used as a row mask without regrouping.
    """
    return d.assign(_a=tier_a(d)).groupby("CASENUM")["_a"].transform("any")


# ---------------------------------------------------------------------------
# Tier A on the human side
# ---------------------------------------------------------------------------
def tier_a(d: pd.DataFrame) -> pd.Series:
    """Objective severity floor: KABCO K/A/B, OR tow-away, OR airbag deployment.

    Defined identically on both sides -- these are the three SGO fields whose
    CRSS counterparts mean the same thing. It is the only bar under which the
    two reporting regimes are comparable without a judgement call, which is why
    every role-based result is reported under it.

    KABCO here is B or worse. `crss_severity_common` merges C and B into
    'minor', so 'minor' is C-or-B and cannot be split; the floor is therefore
    minor-or-worse, which is slightly LOOSER than K/A/B alone. Stated rather
    than silently absorbed: it makes the human side marginally more inclusive,
    which biases against finding an ADS-specific difference rather than for it.
    """
    sev = d["severity"].isin(["minor", "serious", "fatal"])
    return sev | d["towed"].eq(True) | d["airbag_deployed"].eq(True)


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Build the CRSS human comparator.")
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--keep-unknown-speed", action="store_true")
    ap.add_argument("--out", default=os.path.join("data", "interim", "crss_vehicles.parquet"))
    ap.add_argument("--report", default=os.path.join("data", "processed", "crss_extract.json"))
    a = ap.parse_args()

    d = load_years(a.years)
    r, funnel = restrict(d, keep_unknown_speed=a.keep_unknown_speed)
    r["tier_a"] = tier_a(r)

    # The QIE frame is a different object: whole crashes, not qualifying
    # vehicles. Built and saved alongside rather than derived downstream, so
    # no analysis can reach for the vehicle-level frame by accident.
    q, q_funnel = restrict_crash_level(d)
    q["tier_a"] = tier_a(q)
    q["tier_a_crash"] = tier_a_crash_level(q)

    from schema.acc_type import human_baseline_ratio
    rep = {
        "years": a.years,
        "keep_unknown_speed": a.keep_unknown_speed,
        "restriction_funnel": funnel,
        "n_restricted": int(len(r)),
        "weighted_n_restricted": float(r["_w"].sum()),
        "tier_a_share": float(r.loc[r["tier_a"], "_w"].sum() / r["_w"].sum()),
        "tier_a_n": int(r["tier_a"].sum()),
        "acc_category_coverage": float(r["acc_category"].notna().mean()),
        "role_coverage": float(r["acc_role"].isin(["striking", "struck"]).mean()),
        "n_psu": int(r["_psu"].nunique()), "n_stratum": int(r["_stratum"].nunique()),
        # Four baselines, because which one is right depends on the filter, and
        # the difference between them IS the methodological point.
        "qie_frame": {
            "restriction_funnel": q_funnel,
            "n": int(len(q)), "weighted_n": float(q["_w"].sum()),
            "n_tier_a_crash": int(q["tier_a_crash"].sum()),
        },
        "human_baseline_ratio": {
            "vehicle_filtered_no_floor": human_baseline_ratio(r),
            "vehicle_filtered_tier_a": human_baseline_ratio(r[r["tier_a"]]),
            "crash_filtered_no_floor": human_baseline_ratio(q),
            "crash_filtered_tier_a": human_baseline_ratio(q[q["tier_a_crash"]]),
        },
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    r.to_parquet(a.out, index=False)
    q.to_parquet(a.out.replace(".parquet", "_qie.parquet"), index=False)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    print("\n[crss] operating-domain restriction funnel:")
    for s in funnel:
        print(f"[crss]   {s['step']:20s} n={s['n']:7,d} "
              f"wtd={s['weighted_n']:12,.0f}  -{s['removed']:,}  {s['why']}")
    print(f"\n[crss] Tier A (injury B+ or tow-away or airbag): "
          f"{rep['tier_a_n']:,} vehicles, {rep['tier_a_share']:.1%} weighted")
    print(f"[crss] crash-type coverage {rep['acc_category_coverage']:.1%}, "
          f"role {rep['role_coverage']:.1%}")
    print(f"\n[crss] QIE frame (whole crashes): {rep['qie_frame']['n']:,} vehicles, "
          f"{rep['qie_frame']['n_tier_a_crash']:,} in Tier A crashes")
    print("\n[crss] human striking:struck baseline -- the QIE null, per category.")
    print("[crss] A vehicle-level filter breaks the 1:1 pairing; a crash-level "
          "one preserves it.")
    b = rep["human_baseline_ratio"]
    cats = sorted({c for v in b.values() for c in v})
    print(f"[crss]   {'category':30s} {'veh/none':>9s} {'veh/tierA':>10s} "
          f"{'crash/none':>11s} {'crash/tierA':>12s}")
    for c in cats:
        vals = []
        for k in ("vehicle_filtered_no_floor", "vehicle_filtered_tier_a",
                  "crash_filtered_no_floor", "crash_filtered_tier_a"):
            r_ = b[k].get(c, {}).get("ratio")
            vals.append(f"{r_:.3f}" if r_ else "  -  ")
        print(f"[crss]   {c:30s} {vals[0]:>9s} {vals[1]:>10s} "
              f"{vals[2]:>11s} {vals[3]:>12s}")
    print(f"\n[crss] wrote {a.out}, {a.out.replace('.parquet', '_qie.parquet')} "
          f"and {a.report}")


if __name__ == "__main__":
    main()
