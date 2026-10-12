"""Crash-type composition: ADS versus human, conditional on a crash occurring.

THE QUESTION. Not "how often do ADS vehicles crash" -- that needs an exposure
denominator this study deliberately does not have -- but "GIVEN a crash, is the
ADS crash-type mix different from the human one?". Composition comparisons need
no VMT, which is what allows this to cover every ADS operator rather than the
one whose mileage is published.

TWO SIDES, TWO ESTIMATORS, ON PURPOSE.

    ADS    a census of filings, not a sample. Every incident has weight 1 and
           uncertainty comes from an ordinary bootstrap over incidents. There
           is no design to respect, because nobody sampled anything.

    CRSS   a stratified cluster probability sample. Every estimate is
           WEIGHT-ed and uncertainty comes from resampling PSU_VAR within
           (year, PSUSTRAT). An iid bootstrap over CRSS rows would report
           intervals several times too narrow, because PSUs are geographic and
           internally homogeneous.

WHERE THE ADS CRASH TYPE COMES FROM. Two independent sources, reported
side by side rather than blended:

    structured   `schema.acc_type.sgo_distant_key`, derived from the filed
                 contact areas, pre-crash movements and crash-partner class.
                 Covers ~64% of incidents. Involves no model at all.
    extracted    the LLM composition layer. Covers every incident with a
                 narrative, including the junction crashes the structured key
                 cannot resolve.

Presenting both is the point. Where they agree, the composition result does not
depend on the extraction being right; where they diverge, the divergence is
itself reportable. Blending them would destroy exactly that information.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.acc_type import CATEGORIES, sgo_distant_key, is_two_vehicle  # noqa: E402

RESULTS = os.path.join("data", "processed")
TABLES = os.path.join("paper", "tables")

# ---------------------------------------------------------------------------
# THE CATEGORY SPACE THE STRUCTURED KEY CAN ACTUALLY EXPRESS
# ---------------------------------------------------------------------------
# `sgo_distant_key` derives the crash type from contact areas, pre-crash
# movements and crash-partner class. Five categories are UNREACHABLE by that
# construction, because separating them requires knowing which way the vehicles
# were heading, which the contact areas do not say:
#
#     turn_across_path, turn_into_path, straight_paths_intersecting,
#     sideswipe_opposite_direction, other_unknown
#
# WHY THIS HAD TO BE MADE EXPLICIT. Comparing the full eleven-category space
# against CRSS puts a structural ZERO on the ADS side for each of those, next
# to a real human share -- and then renormalises the remaining ADS categories
# over a denominator that excluded them. That does not merely omit five
# categories: it inflates every surviving ADS share. On the first run it made
# the ADS rear-end share 0.606 against a human 0.310, a 1.96x "finding" that
# was substantially an artifact of the key's blind spot.
#
# This is the same rule `schema.crss_map` states for the severity and weather
# crosswalks -- compare in the coarsest space BOTH sides express TOTALLY, so
# projection drops only genuinely missing values -- applied here. When the
# structured key is the ADS source, both sides are restricted to the reachable
# set and both are renormalised within it. The resulting quantity is a share
# CONDITIONAL on the crash being of a structurally-keyable type, and it is
# labelled that way in the output.
#
# The extracted layer has no such blind spot and is compared over all eleven.
REACHABLE_BY_STRUCTURED_KEY = [
    "rear_end", "sideswipe_same_direction", "head_on", "backing",
    "single_vehicle", "pedestrian_animal",
]
UNREACHABLE_BY_STRUCTURED_KEY = [c for c in CATEGORIES
                                 if c not in REACHABLE_BY_STRUCTURED_KEY]


# ---------------------------------------------------------------------------
# Operator restriction, shared by the composition analyses
# ---------------------------------------------------------------------------
# Waymo files 79-100% of the driverless corpus depending on the driverless
# definition (see corpus.sgo_ads), so any driverless result is at risk of being
# a Waymo result wearing a fleet-wide label. `--exclude-entities waymo` is the
# §7 robustness arm: it answers whether the finding reproduces on the operators
# that are not driving the estimate.
#
# Transdev is excluded alongside Waymo. It supplied personnel for Waymo
# operations and co-files the same crashes under a shared Same Incident ID;
# leaving it in would keep Waymo crashes in a Waymo-excluded run.
_ENTITY_GROUPS = {
    "waymo": ("waymo", "transdev"),
    "cruise": ("cruise", "general motors"),
    "zoox": ("zoox",),
}


def filter_entities(d, exclude=None, only=None):
    """Drop or keep reporting entities by group name. Returns (frame, note)."""
    if not exclude and not only:
        return d, ""
    ent = d["Reporting Entity"].astype(str).str.lower()

    def _mask(groups):
        pats = [p for g in groups for p in _ENTITY_GROUPS.get(g, (g,))]
        m = ent.str.contains("|".join(pats), regex=True)
        return m

    note = []
    if exclude:
        m = _mask(exclude)
        note.append(f"excluded {sorted(exclude)}: -{int(m.sum())} incidents")
        d = d[~m]
    if only:
        m = _mask(only)
        note.append(f"restricted to {sorted(only)}: {int(m.sum())} incidents")
        d = d[m]
    return d.copy(), "; ".join(note)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def weighted_share(df: pd.DataFrame, col: str, levels: list,
                   weight: str = "_w") -> np.ndarray:
    w = df[weight].to_numpy(float)
    v = df[col].to_numpy()
    tot = w.sum()
    if tot <= 0:
        return np.full(len(levels), np.nan)
    return np.array([w[v == lv].sum() / tot for lv in levels])


def psu_bootstrap(df: pd.DataFrame, stat: Callable, n_boot: int = 1000,
                  seed: int = 11) -> tuple:
    """Design-consistent bootstrap: resample PSUs with replacement in strata.

    Falls back to an ordinary bootstrap over rows when the frame carries no
    `_psu`, which is the ADS census case. The two are different estimators for
    different objects and the branch is deliberate, not a convenience.
    """
    rng = np.random.default_rng(seed)
    if "_psu" not in df.columns or df["_psu"].isna().all():
        idx = np.arange(len(df))
        vals = [stat(df.iloc[rng.choice(idx, len(idx), replace=True)])
                for _ in range(n_boot)]
    else:
        groups = {s: g["_psu"].unique() for s, g in df.groupby("_stratum")}
        by_psu = {k: g for k, g in df.groupby(["_stratum", "_psu"])}
        vals = []
        for _ in range(n_boot):
            parts = []
            for s, psus in groups.items():
                for p in rng.choice(psus, len(psus), replace=True):
                    parts.append(by_psu[(s, p)])
            vals.append(stat(pd.concat(parts, ignore_index=True)))
    a = np.vstack(vals)
    return np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0)


# ---------------------------------------------------------------------------
# ADS side
# ---------------------------------------------------------------------------
def attach_structured_key(d: pd.DataFrame) -> pd.DataFrame:
    """Add `acc_category_struct` / `acc_role_struct` from the filed fields."""
    keys = [sgo_distant_key(r) for r in d.to_dict("records")]
    return d.assign(
        acc_category_struct=[k[0] if k else None for k in keys],
        acc_role_struct=[k[1] if k else None for k in keys],
    )


def ads_shares(d: pd.DataFrame, col: str, levels: list, n_boot: int = 1000,
               seed: int = 11) -> dict:
    """Crash-type shares on the ADS census, with an ordinary bootstrap CI.

    Rows where `col` is null are DROPPED and the drop is reported. For the
    structured key that is ~36% of incidents, and those are not missing at
    random -- they are disproportionately the junction crashes the contact
    areas cannot resolve. A share computed on the remainder is a share among
    resolvable crashes, and `coverage` is what says so.
    """
    sub = d[d[col].notna()].copy()
    sub["_w"] = 1.0
    if sub.empty:
        return {"n": 0, "coverage": 0.0, "share": [np.nan] * len(levels)}
    est = weighted_share(sub, col, levels)
    lo, hi = psu_bootstrap(sub, lambda x: weighted_share(x, col, levels),
                           n_boot=n_boot, seed=seed)
    return {
        "n": int(len(sub)),
        "n_total": int(len(d)),
        "coverage": float(len(sub) / max(len(d), 1)),
        "share": est.tolist(),
        "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
        "counts": [int((sub[col] == lv).sum()) for lv in levels],
    }


def apply_balanced_weights(human: pd.DataFrame, path: str,
                           which: str = "weights") -> tuple:
    """Swap the CRSS design weight for the entropy-balanced one.

    WHY THIS IS A SEPARATE, EXPLICIT STEP. `human.balance` writes the balanced
    weights to disk and nothing read them, so §5's reweighting was computed and
    then never used -- the hard operating-domain restrictions were doing all the
    alignment and the reported results were the unweighted-within-domain ones.
    Loading them here makes the reweighted comparison an actual arm rather than
    an intention.

    The weights are positional: `balance.py` writes them in the row order of the
    frame it was given, which is `crss_vehicles.parquet` as loaded. A length
    mismatch therefore means the human frame was rebuilt since the weights were
    computed, and that must raise rather than silently align the wrong rows to
    the wrong weights -- which would look like a plausible result.
    """
    if not os.path.exists(path):
        return human, {"applied": False,
                       "why": f"{path} not found; run `python3 -m human.balance`"}
    z = np.load(path)
    w = z[which]
    # Balancing drops out-of-domain rows before solving, so the weights are
    # shorter than the frame. The saved mask says which rows they belong to.
    if "keep_mask" in z:
        mask = z["keep_mask"].astype(bool)
        if len(mask) != len(human):
            raise SystemExit(
                f"[composition] balanced-weight mask covers {len(mask):,} rows "
                f"but the human frame has {len(human):,}. Re-run "
                f"`python3 -m human.balance`.")
        human = human[mask].copy()
    if len(w) != len(human):
        raise SystemExit(
            f"[composition] balanced weights have {len(w):,} rows but the human "
            f"frame has {len(human):,}. The weights are positional, so this "
            f"means one was rebuilt without the other. Re-run "
            f"`python3 -m human.balance` against the current "
            f"crss_vehicles.parquet.")
    out = human.copy()
    # Rescale to the original total so weighted N stays interpretable; shares
    # are scale-invariant but the reported weighted_n should not silently
    # become 1.0.
    out["_w"] = w / w.sum() * human["_w"].sum()
    from human.balance import effective_n
    return out, {
        "applied": True, "which": which,
        "effective_n_before": effective_n(human["_w"].to_numpy(float)),
        "effective_n_after": effective_n(w),
    }


def human_shares(c: pd.DataFrame, col: str, levels: list, n_boot: int = 300,
                 seed: int = 11) -> dict:
    sub = c[c[col].notna()]
    if sub.empty:
        return {"n": 0, "share": [np.nan] * len(levels)}
    est = weighted_share(sub, col, levels)
    lo, hi = psu_bootstrap(sub, lambda x: weighted_share(x, col, levels),
                           n_boot=n_boot, seed=seed)
    return {
        "n": int(len(sub)),
        "weighted_n": float(sub["_w"].sum()),
        "share": est.tolist(),
        "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
    }


def compare(ads: pd.DataFrame, human: pd.DataFrame, ads_col: str,
            levels: Optional[list] = None, n_boot_ads: int = 1000,
            n_boot_human: int = 300, restrict_human: bool = True) -> dict:
    """Side-by-side composition, with a ratio and a non-overlap flag.

    `levels` is the category space BOTH sides are renormalised within. When the
    ADS source is the structured key this must be `REACHABLE_BY_STRUCTURED_KEY`
    and `restrict_human` must stay True -- see the note at the top of this
    module for what goes wrong otherwise.

    `ci_disjoint` is CI NON-OVERLAP, a conservative screen and NOT a hypothesis
    test: non-overlapping 95% intervals imply a difference at well below 5%,
    while overlapping intervals do not imply no difference. Labelled a screen so
    nobody reports it as a p-value. The multinomial model holds the conditional
    test.
    """
    levels = levels or CATEGORIES
    # Restricting the human side to the same space is what makes the two
    # denominators the same object. Without it the human shares are computed
    # over eleven categories and the ADS shares over six.
    h_src = human[human["acc_category"].isin(levels)] if restrict_human else human
    a = ads_shares(ads, ads_col, levels, n_boot=n_boot_ads)
    h = human_shares(h_src, "acc_category", levels, n_boot=n_boot_human)

    rows = []
    for i, lv in enumerate(levels):
        av, hv = a["share"][i], h["share"][i]
        rows.append({
            "category": lv,
            "ads_share": av, "ads_ci": [a.get("ci_lo", [None] * len(levels))[i],
                                        a.get("ci_hi", [None] * len(levels))[i]],
            "ads_n": a.get("counts", [None] * len(levels))[i],
            "human_share": hv, "human_ci": [h.get("ci_lo", [None] * len(levels))[i],
                                            h.get("ci_hi", [None] * len(levels))[i]],
            "ratio": (av / hv) if (hv and hv > 0) else None,
            "ci_disjoint": bool(
                a.get("ci_lo") and h.get("ci_lo")
                and (a["ci_lo"][i] > h["ci_hi"][i] or h["ci_lo"][i] > a["ci_hi"][i])),
        })
    return {"levels": levels, "ads": a, "human": h, "rows": rows,
            "human_restricted_to_levels": bool(restrict_human),
            "human_share_of_all_categories_retained": (
                float(h_src["_w"].sum() / human["_w"].sum())
                if restrict_human and human["_w"].sum() > 0 else 1.0),
            "screen_note": "ci_disjoint is a conservative non-overlap screen, "
                           "not a hypothesis test; see the multinomial model"}


# ---------------------------------------------------------------------------
# Multinomial model -- §6.2
# ---------------------------------------------------------------------------
def multinomial_model(ads: pd.DataFrame, human: pd.DataFrame, ads_col: str,
                      covariates=("intersection", "lighting_bin", "speed_bin"),
                      seed: int = 11, n_boot: int = 0) -> dict:
    """Crash type ~ ADS indicator + environment, on the pooled data.

    WHAT THIS BUYS OVER THE SHARE TABLE. A raw share difference confounds "ADS
    vehicles crash differently" with "ADS vehicles drive in places where
    everyone crashes differently". Conditioning on the environment separates
    them, and the ADS coefficient is then the difference that survives.

    WEIGHTING. CRSS rows enter at their survey weight, rescaled so the human
    side contributes its effective sample size rather than its population
    total -- an unrescaled weight would give the human side ~20 million
    observations against the ADS side's ~2,000 and drive every standard error
    to zero. ADS rows enter at weight 1. This is a pragmatic pooling, and the
    standard errors it produces are NOT design-consistent for the human side;
    the design-consistent uncertainty is in the bootstrap CIs above, and this
    model is reported for its coefficients rather than its p-values.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    cols = [c for c in covariates if c in ads.columns and c in human.columns]
    if not cols:
        return {"available": False,
                "why": f"no shared covariates among {list(covariates)}"}

    a = ads[ads[ads_col].notna()].copy()
    h = human[human["acc_category"].notna()].copy()
    if a.empty or h.empty:
        return {"available": False, "why": "one side is empty after dropping nulls"}

    a["_y"] = a[ads_col]
    h["_y"] = h["acc_category"]
    a["_ads"] = 1.0
    h["_ads"] = 0.0
    # Effective sample size (Kish) keeps the human side's influence
    # proportional to the information it actually carries.
    w = h["_w"].to_numpy(float)
    ess = (w.sum() ** 2) / max((w ** 2).sum(), 1e-9)
    h["_mw"] = w / w.sum() * ess
    a["_mw"] = 1.0

    pooled = pd.concat([a[["_y", "_ads", "_mw"] + cols],
                        h[["_y", "_ads", "_mw"] + cols]], ignore_index=True)
    pooled = pooled.dropna(subset=["_y"])
    for c in cols:
        pooled[c] = pooled[c].astype(str).fillna("unknown")

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                        drop="first")
    X = np.hstack([pooled[["_ads"]].to_numpy(float), enc.fit_transform(pooled[cols])])
    y = pooled["_y"].to_numpy()

    # `multi_class` is deprecated in scikit-learn >= 1.5 and multinomial is the
    # only remaining behaviour, so the argument is omitted rather than pinned.
    m = LogisticRegression(max_iter=2000, random_state=seed)
    m.fit(X, y, sample_weight=pooled["_mw"].to_numpy(float))

    # The ADS coefficient is column 0. Exponentiated it is the multiplicative
    # change in the odds of that crash type for an ADS vehicle, holding the
    # environment fixed.
    out = {}
    for k, cls in enumerate(m.classes_):
        out[str(cls)] = {
            "ads_log_odds": float(m.coef_[k, 0]),
            "ads_odds_ratio": float(np.exp(m.coef_[k, 0])),
        }
    res = {
        "available": True,
        "covariates": cols,
        "n_ads": int(len(a)), "n_human": int(len(h)),
        "human_effective_n": float(ess),
        "coefficients": out,
    }
    if n_boot:
        res["design_bootstrap"] = _multinomial_psu_bootstrap(
            a, h, cols, enc, m.classes_, n_boot=n_boot, seed=seed)
        res["caveat"] = ("CIs come from resampling PSUs within (year, stratum) "
                         "on the human side and incidents on the ADS side, so "
                         "they respect the CRSS design")
    else:
        res["caveat"] = ("point estimates only; pass n_boot for "
                         "design-consistent intervals")
    return res


def _multinomial_psu_bootstrap(a: pd.DataFrame, h: pd.DataFrame, cols: list,
                               enc, classes, n_boot: int = 200,
                               seed: int = 11) -> dict:
    """Design-consistent CIs for the ADS odds ratios.

    THE TWO SIDES RESAMPLE DIFFERENTLY, FOR THE SAME REASON THEY DO ELSEWHERE
    IN THIS MODULE. CRSS is a stratified cluster sample, so its replicates
    resample PSUs within (year, stratum); the ADS corpus is a census of
    filings, so its replicates resample incidents. Resampling CRSS rows
    independently would ignore the clustering and report intervals several
    times too narrow -- which matters more here than for a share, because the
    ADS indicator is estimated against a human side that is three orders of
    magnitude larger and will otherwise appear to be measured without error.

    The encoder is reused rather than refit so that a replicate missing a rare
    covariate level does not silently change the design matrix's columns
    between replicates.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    groups = {s: g["_psu"].unique() for s, g in h.groupby("_stratum")} \
        if "_psu" in h.columns else {}
    by_psu = {k: g for k, g in h.groupby(["_stratum", "_psu"])} if groups else {}
    a_idx = np.arange(len(a))

    draws: dict[str, list] = {str(c): [] for c in classes}
    n_ok = 0
    for _ in range(n_boot):
        if groups:
            parts = [by_psu[(s, p)] for s, psus in groups.items()
                     for p in rng.choice(psus, len(psus), replace=True)]
            hb = pd.concat(parts, ignore_index=True)
        else:
            hb = h
        ab = a.iloc[rng.choice(a_idx, len(a_idx), replace=True)]

        # Both frames already carry `_y` (the crash category) and `_ads`,
        # assigned by `multinomial_model` before it called this. The human
        # side's modelling weight is recomputed per replicate because the
        # effective sample size changes with which PSUs were drawn.
        w = hb["_w"].to_numpy(float)
        ess = (w.sum() ** 2) / max((w ** 2).sum(), 1e-9)
        keep = ["_y", "_ads"] + cols
        pooled = pd.concat([ab[keep].assign(_mw=1.0),
                            hb[keep].assign(_mw=w / w.sum() * ess)],
                           ignore_index=True).dropna(subset=["_y"])
        for c in cols:
            pooled[c] = pooled[c].astype(str)
        if pooled["_y"].nunique() < 2:
            continue

        X = np.hstack([pooled[["_ads"]].to_numpy(float),
                       enc.transform(pooled[cols])])
        try:
            mb = LogisticRegression(max_iter=2000, random_state=seed)
            mb.fit(X, pooled["_y"].to_numpy(),
                   sample_weight=pooled["_mw"].to_numpy(float))
        except Exception:
            continue
        n_ok += 1
        for k, cls in enumerate(mb.classes_):
            if str(cls) in draws:
                draws[str(cls)].append(float(np.exp(mb.coef_[k, 0])))

    out = {}
    for cls, vals in draws.items():
        if len(vals) < max(10, n_boot // 10):
            out[cls] = {"n_replicates": len(vals), "ci": None,
                        "note": "too few successful replicates to report"}
            continue
        out[cls] = {"n_replicates": len(vals),
                    "ci": [float(np.percentile(vals, 2.5)),
                           float(np.percentile(vals, 97.5))],
                    "median_odds_ratio": float(np.median(vals))}
    return {"n_boot_requested": n_boot, "n_boot_fitted": n_ok,
            "by_category": out}


# ---------------------------------------------------------------------------
# Environment binning, shared by both sides
# ---------------------------------------------------------------------------
def bin_environment(d: pd.DataFrame, side: str) -> pd.DataFrame:
    """Add `speed_bin`, `lighting_bin`, `intersection` in one shared vocabulary.

    Defined once, applied to both sides, so a bin edge cannot drift between
    them. The ADS side's speed and lighting come from the structured columns,
    which the third amendment dropped -- so post-amendment incidents land in
    `unknown` unless the extraction layer has filled them, and that is visible
    in the cell counts rather than hidden.
    """
    out = d.copy()
    if side == "human":
        spd = out["speed_limit"]
        lgt = out["lighting"].astype(str).str.strip().str.lower()
        out["lighting_bin"] = np.where(
            lgt.str.startswith("daylight"), "daylight",
            np.where(lgt.str.startswith("dark"), "dark",
                     np.where(lgt.str.startswith("dawn") | lgt.str.startswith("dusk"),
                              "dawn_dusk", "unknown")))
        out["intersection"] = out["intersection"].fillna("unknown")
    else:
        spd = out.get("posted_speed_limit")
        lgt = out.get("lighting_struct", pd.Series(index=out.index, dtype=object))
        lgt = lgt.astype(str).str.strip().str.lower()
        out["lighting_bin"] = np.where(
            lgt.str.startswith("daylight"), "daylight",
            np.where(lgt.str.startswith("dark"), "dark",
                     np.where(lgt.str.startswith("dawn"), "dawn_dusk", "unknown")))
        rt = out.get("roadway_type", pd.Series(index=out.index, dtype=object))
        out["intersection"] = np.where(
            rt.astype(str).str.lower().eq("intersection"), "intersection",
            "non_intersection")

    spd = pd.to_numeric(spd, errors="coerce")
    out["speed_bin"] = np.where(
        spd.isna(), "unknown",
        np.where(spd <= 25, "le_25",
                 np.where(spd <= 35, "mph_30_35",
                          np.where(spd <= 45, "mph_40_45", "gt_45"))))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ADS vs human crash-type composition.")
    ap.add_argument("--ads", default=os.path.join("data", "interim",
                                                  "ads_reportability.parquet"))
    ap.add_argument("--human", default=os.path.join("data", "interim",
                                                    "crss_vehicles.parquet"))
    ap.add_argument("--tiers", nargs="+",
                    default=["all", "tier_a", "tier_b_yes", "tier_b_yes_maybe"])
    ap.add_argument("--exclude-entities", nargs="+", default=None,
                    help="Operator groups to drop (waymo / cruise / zoox). "
                         "The §7 robustness arm: Waymo dominates n, so "
                         "`--exclude-entities waymo` asks whether a finding "
                         "reproduces without it.")
    ap.add_argument("--driverless-only", action="store_true",
                    help="Primary analysis. Omit for the supervised-inclusive run.")
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024],
                    help="Restrict the ADS side to the CRSS release years.")
    ap.add_argument("--balanced-weights", default=None,
                    help="Path to crss_balanced_weights.npz from human.balance. "
                         "Replaces the CRSS design weight with the "
                         "entropy-balanced one, making the §7 'reweighted vs "
                         "hard-restricted only' comparison an actual arm.")
    ap.add_argument("--trimmed", action="store_true",
                    help="Use the trimmed balanced weights. Trimming trades "
                         "exact moment-matching for stability; both are "
                         "reported because an estimate that moves a lot under "
                         "trimming rests on a few rows.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--multinomial-boot", type=int, default=0,
                    help="Design-consistent bootstrap replicates for the "
                         "multinomial ADS odds ratios (PSUs within stratum on "
                         "the human side, incidents on the ADS side). ~2.6s "
                         "each; 0 skips and reports point estimates only.")
    ap.add_argument("--n-boot-human", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(RESULTS, "composition.json"))
    a = ap.parse_args()

    ads = pd.read_parquet(a.ads)
    ads = attach_structured_key(ads)
    ads = bin_environment(ads, "ads")
    ads["all"] = True
    for t in ("tier_a", "tier_b_yes", "tier_b_yes_maybe"):
        if t not in ads.columns:
            ads[t] = False

    if a.driverless_only:
        ads = ads[ads["driverless"].eq(True)].copy()
    ads, ent_note = filter_entities(ads, exclude=a.exclude_entities)
    if ent_note:
        print(f"[composition] {ent_note}")
    # Same window on both sides: the ADS corpus runs to 2026 and CRSS ends at
    # its latest release, so an unrestricted ADS side compares roughly a third
    # of its crashes against human years that do not exist.
    ads = ads[ads["incident_year"].isin(a.years)].copy()

    human = bin_environment(pd.read_parquet(a.human), "human")
    bal = {"applied": False}
    if a.balanced_weights:
        human, bal = apply_balanced_weights(
            human, a.balanced_weights,
            which="weights_trimmed" if a.trimmed else "weights")
        if bal["applied"]:
            print(f"[composition] balanced weights applied ({bal['which']}): "
                  f"effective n {bal['effective_n_before']:,.0f} -> "
                  f"{bal['effective_n_after']:,.0f}")
        else:
            print(f"[composition] {bal['why']}")

    rep = {
        "years": a.years, "driverless_only": a.driverless_only,
        "n_ads": int(len(ads)), "n_human": int(len(human)),
        "human_weighted_n": float(human["_w"].sum()),
        "balancing": bal,
        "by_tier": {},
    }

    for t in a.tiers:
        sub = ads[ads[t]] if t in ads.columns else ads.iloc[0:0]
        if sub.empty:
            rep["by_tier"][t] = {"n_ads": 0,
                                 "skipped": "no ADS incidents in this tier "
                                            "(extraction layer not yet run?)"}
            continue
        # The human side is filtered to the matching floor only for Tier A,
        # which is the one tier defined identically on both sides. Tiers B and
        # C are properties of the SGO reporting regime and have no human
        # counterpart, so the human side stays at the full restricted domain
        # and that asymmetry is recorded rather than papered over.
        h = human[human["tier_a"]] if t == "tier_a" else human
        rep["by_tier"][t] = {
            "n_ads": int(len(sub)),
            "human_filter": "tier_a" if t == "tier_a" else "domain_restricted_only",
            # Structured key: both sides renormalised within the six categories
            # the key can express. Anything else compares two different
            # denominators (see REACHABLE_BY_STRUCTURED_KEY).
            "structured_key": compare(sub, h, "acc_category_struct",
                                      levels=REACHABLE_BY_STRUCTURED_KEY,
                                      n_boot_ads=a.n_boot,
                                      n_boot_human=a.n_boot_human),
            "structured_key_note": (
                "shares are CONDITIONAL on the crash being of a type the "
                "structured key can express; the five junction/other "
                f"categories {UNREACHABLE_BY_STRUCTURED_KEY} are excluded from "
                "BOTH sides, not zeroed on one"),
        }
        if "acc_type_category" in sub.columns and sub["acc_type_category"].notna().any():
            # The extracted layer reaches every category, so it is compared
            # over the full space with no renormalisation on either side.
            rep["by_tier"][t]["extracted"] = compare(
                sub, h, "acc_type_category", levels=CATEGORIES,
                n_boot_ads=a.n_boot, n_boot_human=a.n_boot_human,
                restrict_human=False)
            rep["by_tier"][t]["multinomial"] = multinomial_model(
                sub, h, "acc_type_category", n_boot=a.multinomial_boot)
        else:
            rep["by_tier"][t]["extracted"] = {
                "skipped": "no composition extractions; run "
                           "extract.composition_extract"}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    print(f"[composition] ADS n={rep['n_ads']:,} ({'driverless' if a.driverless_only else 'all supervision'}), "
          f"years {a.years}")
    print(f"[composition] human n={rep['n_human']:,} "
          f"(weighted {rep['human_weighted_n']:,.0f})")
    for t in a.tiers:
        b = rep["by_tier"].get(t, {})
        if b.get("skipped"):
            print(f"\n=== {t}: {b['skipped']}")
            continue
        sk = b["structured_key"]
        print(f"\n=== {t}  (ADS n={b['n_ads']:,}, keyed {sk['ads']['n']:,} = "
              f"{sk['ads']['coverage']:.0%}; human filter: {b['human_filter']})")
        print(f"{'category':30s} {'ADS':>8s} {'human':>8s} {'ratio':>7s}  n")
        for r in sk["rows"]:
            if r["ads_n"] in (0, None) and not r["human_share"]:
                continue
            ratio = f"{r['ratio']:.2f}x" if r["ratio"] else "   -  "
            star = " *" if r["ci_disjoint"] else ""
            print(f"{r['category']:30s} {r['ads_share']:8.3f} "
                  f"{r['human_share']:8.3f} {ratio:>7s}  {r['ads_n']}{star}")
    print("\n[composition] * = 95% CIs disjoint (conservative screen, not a test)")
    print(f"[composition] wrote {a.out}")


if __name__ == "__main__":
    main()
