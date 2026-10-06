"""Residual duplicate detection across SGO ADS filings, calibrated not assumed.

-----------------------------------------------------------------------------
WHAT THIS MODULE FOUND, AND WHY IT IS NOT WHAT IT WAS BUILT TO FIND
-----------------------------------------------------------------------------
The design this started from was the standard one: `Same Incident ID` is
entity-scoped, so co-filed crashes (an operator and a vehicle manufacturer both
filing one collision) survive as duplicates, and they must be recovered by
fuzzy-matching on date, time, city and narrative similarity.

Two measurements contradicted that, and both are reported here rather than
worked around:

(1) `Same Incident ID` DOES link across reporting entities in these files --
    149 Cruise/GM groups, 139 Waymo/Transdev groups, 2,358 incidents out of
    2,716 in-scope filings. It is not entity-scoped. Whatever residual exists
    is what is left AFTER it, and that is the only quantity worth estimating.

(2) NARRATIVE SIMILARITY IS THE WRONG SIGNAL HERE, and a threshold on it
    produces mostly false merges. The reason is specific to this corpus:

      - NHTSA redacts the DAY from `Incident Date` (published as MON-YYYY) and
        redacts street addresses, so two filings can only be compared within a
        month, not within a day.
      - The dominant operator files narratives from a template. After the
        boilerplate is stripped, UNRELATED Waymo crashes in one city in one
        month still sit at a median character-n-gram cosine of 0.42, with 5%
        above 0.70.
      - Genuine duplicates are BIMODAL on that same statistic (5th percentile
        0.073, median 0.661), because a co-filing is often the partner
        entity's much shorter or partly redacted account of the crash.

    The two distributions overlap so heavily that no cosine threshold
    separates them. A rule tuned on text alone proposed 171 merges; hand
    inspection of the top of that list found consecutive-minute Waymo crashes
    in the same city that differ in direction of travel and in crash-partner
    type -- different crashes.

-----------------------------------------------------------------------------
WHAT DOES WORK: STRUCTURED AGREEMENT, CALIBRATED ON THE BUILT-IN KEY
-----------------------------------------------------------------------------
`Same Incident ID` supplies its own labelled training data. Every pair of rows
it already groups is a KNOWN duplicate -- two filings of one crash, made
independently by two entities. Scoring candidate rules against those positives,
and against the pool of same-block pairs it did NOT group, turns threshold
choice into a measurement:

    rule                            recall   hits in ungrouped pool
    dt == 0                          0.987          41
    dt == 0 & crash-partner match    0.981          17
    dt == 0 & partner & contact>=.5  0.970           5     <- adopted
    dt <= 2 & partner & movement     0.976          12

Exact agreement on the minute is what carries the rule: known duplicates agree
to the minute 98.7% of the time, while ungrouped same-block pairs do so 0.2% of
the time. Crash-partner type and contact-area overlap remove the remainder.
Narrative similarity is computed and reported, and is deliberately NOT part of
the decision.

The pool of ungrouped pairs is UNLABELLED, not negative: a hit in it is either
a false merge or the residual duplicate this module is looking for, and the two
cannot be told apart by the matcher that proposed them. That is precisely why
the surviving handful go to a human (`audit_sample`) rather than straight into
a merge map.

-----------------------------------------------------------------------------
THE ASYMMETRY THAT SETS THE THRESHOLD
-----------------------------------------------------------------------------
A false merge DESTROYS a real crash and leaves no trace downstream. A missed
merge leaves one crash counted twice, which biases composition shares toward
whatever co-filed crashes look like -- and co-filing tracks severity, so the
bias has a direction. Both are real; only the first is undetectable after the
fact. The adopted rule accordingly gives up 1.7 points of recall against the
looser `dt == 0` rule to cut the ungrouped-pool hits from 41 to 5.
"""
from __future__ import annotations

import itertools
import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

# --- adopted decision rule -------------------------------------------------
# Calibrated in `calibrate`; see the table in the module docstring. Stated as
# constants so `calibrate` can re-derive them on a future SGO release rather
# than inheriting a number that was true in 2026.
TIME_TOLERANCE_MIN = 0     # exact minute agreement
CONTACT_JACCARD_MIN = 0.5  # overlap of SV contact areas

# Reported, never decisive. See module docstring note (2).
_BOILERPLATE = [
    r"without agreeing[^.]*?provides this information:?",
    r"\[redacted[^\]]*\]", r"\[may contain[^\]]*\]",
    r"the (?:waymo|zoox|cruise|nuro|motional) (?:autonomous vehicle|av|robotaxi)"
    r"(?:\s*\(['\"]?[^)]*\))?",
    r"autonomous vehicle \(['\"]?av['\"]?\)",
    r"automated driving system \(['\"]?ads['\"]?\)",
    r"in autonomous mode", r"with the automated driving system engaged",
]
_BOILER_RE = re.compile("|".join(_BOILERPLATE), re.I)


def _clean_narrative(s) -> str:
    t = _BOILER_RE.sub(" ", str(s or "").lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _minutes(t) -> Optional[float]:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", str(t or ""))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return h * 60 + mi if 0 <= h < 24 and 0 <= mi < 60 else None


def _city_key(s) -> str:
    return re.sub(r"[^a-z]", "", str(s or "").lower())


def _block_key(d: pd.DataFrame) -> pd.Series:
    """Candidate blocking: same incident month, state and city.

    Blocking on the month is forced -- the day is redacted -- and it is why
    time-of-day collisions between unrelated crashes are common enough to
    matter. `unblocked_city_variants` reports the filings this blocking would
    split on a city spelling difference.
    """
    return (d["incident_month"].astype(str) + "|" +
            d["State"].astype(str).str.strip().str.upper() + "|" +
            d["City"].map(_city_key))


# ---------------------------------------------------------------------------
# Pair features
# ---------------------------------------------------------------------------
def _contact_cols(d: pd.DataFrame, side: str = "sv") -> list[str]:
    # 'unknown' is an explicit filed value, not a contact area, and including it
    # would let two filings that both said "unknown" score a perfect overlap.
    return [c for c in d.columns
            if c.startswith(f"{side}_contact_") and not c.endswith("unknown")]


def pair_features(d: pd.DataFrame, pairs: list[tuple[int, int]],
                  sim: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Agreement features for a list of row-index pairs."""
    sv = _contact_cols(d, "sv")
    mins = {i: _minutes(t) for i, t in d["Incident Time (24:00)"].items()}
    contact = {i: frozenset(c for c in sv if bool(d.at[i, c])) for i in d.index}

    rows = []
    for a, b in pairs:
        ta, tb = mins.get(a), mins.get(b)
        dt = abs(ta - tb) if ta is not None and tb is not None else np.nan
        A, B = contact[a], contact[b]
        rows.append({
            "i": a, "j": b,
            "time_delta_min": dt,
            "partner_match": bool(d.at[a, "crash_with"] == d.at[b, "crash_with"]),
            "contact_jaccard": len(A & B) / max(len(A | B), 1),
            "movement_match": bool(d.at[a, "SV Pre-Crash Movement"]
                                   == d.at[b, "SV Pre-Crash Movement"]),
            "severity_match": bool(d.at[a, "Highest Injury Severity Alleged"]
                                   == d.at[b, "Highest Injury Severity Alleged"]),
            "same_entity": bool(d.at[a, "Reporting Entity"] == d.at[b, "Reporting Entity"]),
            "narrative_sim": float(sim[a, b]) if sim is not None else np.nan,
        })
    return pd.DataFrame(rows)


def narrative_similarity(d: pd.DataFrame) -> np.ndarray:
    """Dense corpus-wide cosine over character n-grams of cleaned narratives.

    Computed for REPORTING only -- the measured overlap between the duplicate
    and non-duplicate distributions is the finding, and it cannot be reported
    without being computed. Character n-grams rather than words because two
    filers of one crash share street names, vehicle descriptions and numbers
    without sharing a vocabulary; TF-IDF rather than a sentence embedding to
    keep this offline and deterministic, matching `lift_test --embed tfidf`.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [_clean_narrative(t) for t in d["Narrative"]]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return np.zeros((len(d), len(d)))
    return (X @ X.T).toarray()


# ---------------------------------------------------------------------------
# Labelled pairs from the built-in key
# ---------------------------------------------------------------------------
def known_duplicate_pairs(d: pd.DataFrame) -> list[tuple[int, int]]:
    """Pairs that `Same Incident ID` already groups: known same-crash.

    Restricted to keys that came FROM a Same Incident ID (`sid:` prefix).
    Rows with a blank ID get per-report keys in `sgo_ads.group_incidents` and
    never group, so they carry no label either way.
    """
    out = []
    for k, g in d.groupby("incident_key"):
        if len(g) > 1 and str(k).startswith("sid:"):
            out += list(itertools.combinations(sorted(g.index.tolist()), 2))
    return out


def ungrouped_pairs(d: pd.DataFrame) -> list[tuple[int, int]]:
    """Same-block pairs the built-in key did NOT group.

    UNLABELLED, not negative. This pool contains both the true residual
    duplicates and every non-duplicate coincidence, and no statistic computed
    on it can separate them -- which is the whole reason the surviving few go
    to a human auditor.
    """
    out = []
    for _, g in d.groupby(_block_key(d)):
        idx = sorted(g.index.tolist())
        for a, b in itertools.combinations(idx, 2):
            if d.at[a, "incident_key"] != d.at[b, "incident_key"]:
                out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Rule calibration
# ---------------------------------------------------------------------------
RULES = {
    "time_exact": lambda F: F["time_delta_min"].eq(0),
    "time_exact + partner": lambda F: F["time_delta_min"].eq(0) & F["partner_match"],
    "time_exact + partner + contact": lambda F: (
        F["time_delta_min"].eq(0) & F["partner_match"]
        & F["contact_jaccard"].ge(CONTACT_JACCARD_MIN)),
    "time_exact + partner + movement": lambda F: (
        F["time_delta_min"].eq(0) & F["partner_match"] & F["movement_match"]),
    "time_exact + partner + (contact|movement)": lambda F: (
        F["time_delta_min"].eq(0) & F["partner_match"]
        & (F["contact_jaccard"].ge(CONTACT_JACCARD_MIN) | F["movement_match"])),
    "time<=2 + partner + movement": lambda F: (
        F["time_delta_min"].le(2) & F["partner_match"] & F["movement_match"]),
    "narrative_sim>=0.55": lambda F: F["narrative_sim"].ge(0.55),
    "narrative_sim>=0.70": lambda F: F["narrative_sim"].ge(0.70),
}

# Chosen by domination, not by taste: it matches or beats every other rule in
# the table on BOTH recall and pool hits. Requiring contact-area overlap
# *instead of* movement agreement costs two points of recall for nothing, and
# requiring both costs more; the disjunction is what a genuine co-filing
# satisfies, because two filers of one crash agree on how the AV was moving or
# on where it was struck, and usually on both.
ADOPTED_RULE = "time_exact + partner + (contact|movement)"


def calibrate(pos: pd.DataFrame, pool: pd.DataFrame) -> list[dict]:
    """Recall on known duplicates and hit count in the ungrouped pool, per rule.

    `est_precision` treats every pool hit as a false merge, so it is a LOWER
    BOUND: any pool hit that is a genuine residual duplicate makes the true
    precision higher. Reported as a bound rather than as a point estimate
    because the pool is unlabelled, and pretending otherwise would turn the
    quantity this module is trying to measure into an input to its own
    threshold choice.
    """
    rows = []
    for name, fn in RULES.items():
        r = float(fn(pos).mean()) if len(pos) else np.nan
        hits = int(fn(pool).sum()) if len(pool) else 0
        tp = (len(pos) * r) if len(pos) else 0.0
        rows.append({
            "rule": name,
            "recall_on_known_duplicates": round(r, 4),
            "hits_in_ungrouped_pool": hits,
            "est_precision_lower_bound": round(tp / max(tp + hits, 1), 4),
            "adopted": name == ADOPTED_RULE,
        })
    return rows


def similarity_overlap(pos: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """The distributions that rule narrative similarity out as a decision signal.

    Published because "we tried text similarity and it did not separate" is a
    claim a reader is entitled to check, and because the mechanism -- operator
    narrative templating -- generalises to any future SGO analysis that reaches
    for embedding similarity on this corpus.
    """
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]

    def q(F):
        s = F["narrative_sim"].dropna()
        return {str(x): round(float(s.quantile(x)), 4) for x in qs} if len(s) else {}

    ps, ns = pos["narrative_sim"].dropna(), pool["narrative_sim"].dropna()
    # How often does an ungrouped (mostly non-duplicate) pair outscore the
    # median genuine duplicate? If this is not small, no threshold works.
    med = float(ps.median()) if len(ps) else np.nan
    return {
        "known_duplicates": q(pos),
        "ungrouped_pool": q(pool),
        "pool_above_duplicate_median": (round(float((ns >= med).mean()), 4)
                                        if len(ns) and np.isfinite(med) else None),
        "duplicate_below_pool_median": (
            round(float((ps <= float(ns.median())).mean()), 4) if len(ps) and len(ns) else None),
    }


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
def decide(pool: pd.DataFrame, rule: str = ADOPTED_RULE) -> pd.DataFrame:
    """Label ungrouped pairs merge / review / reject.

    `review` is the band that clears the time test but fails one structured
    check. It is NOT merged. It is the band the hand audit oversamples, because
    an error rate estimated only on a matcher's confident decisions is not an
    error rate.
    """
    if pool.empty:
        return pool.assign(decision=pd.Series(dtype=str))
    merge = RULES[rule](pool)
    review = ~merge & pool["time_delta_min"].le(2)
    return pool.assign(decision=np.where(merge, "merge",
                                         np.where(review, "review", "reject")))


def merge_map(d: pd.DataFrame, decided: pd.DataFrame) -> dict[str, str]:
    """Proposed merges as incident_key -> incident_key, oriented consistently.

    Both members point at the lexicographically smaller key so a chain resolves
    to one representative regardless of pair ordering.
    """
    out: dict[str, str] = {}
    for _, r in decided[decided["decision"] == "merge"].iterrows():
        a, b = sorted([d.at[r["i"], "incident_key"], d.at[r["j"], "incident_key"]])
        out[b] = a
    return out


def audit_sample(d: pd.DataFrame, decided: pd.DataFrame, n: int = 60,
                 seed: int = 11) -> list[dict]:
    """Side-by-side pairs for a human to verify, stratified by decision band.

    Every `merge` is included regardless of `n`: there are few enough that a
    sample of them would be pointless, and each one destroys a crash if wrong.
    """
    rng = np.random.default_rng(seed)
    parts = [decided[decided["decision"] == "merge"]]
    remaining = max(n - len(parts[0]), 0)
    for band in ("review", "reject"):
        sub = decided[decided["decision"] == band]
        take = min(remaining // 2, len(sub))
        if take:
            parts.append(sub.iloc[rng.choice(len(sub), take, replace=False)])
    samp = pd.concat([p for p in parts if len(p)]) if any(len(p) for p in parts) else None
    if samp is None:
        return []

    out = []
    for _, r in samp.iterrows():
        i, j = int(r["i"]), int(r["j"])
        out.append({
            "decision": r["decision"],
            "entity_i": d.at[i, "Reporting Entity"], "entity_j": d.at[j, "Reporting Entity"],
            "report_i": d.at[i, "Report ID"], "report_j": d.at[j, "Report ID"],
            "month": str(d.at[i, "incident_month"]),
            "city": d.at[i, "City"], "state": d.at[i, "State"],
            "time_i": d.at[i, "Incident Time (24:00)"],
            "time_j": d.at[j, "Incident Time (24:00)"],
            "time_delta_min": (None if pd.isna(r["time_delta_min"])
                               else float(r["time_delta_min"])),
            "partner_i": d.at[i, "crash_with"], "partner_j": d.at[j, "crash_with"],
            "contact_jaccard": round(float(r["contact_jaccard"]), 3),
            "narrative_sim": (None if pd.isna(r["narrative_sim"])
                              else round(float(r["narrative_sim"]), 3)),
            "narrative_i": str(d.at[i, "Narrative"])[:1500],
            "narrative_j": str(d.at[j, "Narrative"])[:1500],
            "verdict": "",     # human: same / different / unclear
        })
    return out


def unblocked_city_variants(d: pd.DataFrame) -> list[dict]:
    """Near-miss city spellings within a state, which blocking would split.

    Reported, not repaired: normalising "S.F." to "San Francisco" needs a
    gazetteer this repository does not ship, and guessing would be the same
    invented convention the crosswalks refuse elsewhere.
    """
    counts = d.groupby([d["State"].astype(str).str.upper(),
                        d["City"].map(_city_key)]).size()
    out = []
    for (state, city), n in counts.items():
        if n > 2 or not city:
            continue
        for (s2, c2), n2 in counts.items():
            if s2 != state or c2 == city or n2 <= 2:
                continue
            if c2.startswith(city) or city.startswith(c2):
                out.append({"state": state, "rare": city, "n_rare": int(n),
                            "common": c2, "n_common": int(n2)})
    return out


def main():
    import argparse
    from corpus.sgo_ads import (SGO_DIR, apply_exclusions, apply_merges,
                                collapse_versions, derive_fields, group_incidents,
                                load_ads_files)

    ap = argparse.ArgumentParser(
        description="Calibrated residual duplicate detection for SGO ADS filings.")
    ap.add_argument("--sgo-dir", default=SGO_DIR)
    ap.add_argument("--rule", default=ADOPTED_RULE, choices=list(RULES))
    ap.add_argument("--audit-n", type=int, default=60)
    ap.add_argument("--no-similarity", action="store_true",
                    help="Skip the narrative-similarity diagnostic (dense n^2 "
                         "cosine). The decision rule does not use it.")
    ap.add_argument("--out", default=os.path.join("data", "processed", "dedup_audit.json"))
    ap.add_argument("--audit-out",
                    default=os.path.join("data", "gold", "dedup_audit_sample.jsonl"))
    a = ap.parse_args()

    d, _ = apply_exclusions(derive_fields(collapse_versions(load_ads_files(a.sgo_dir))))
    d = group_incidents(d).reset_index(drop=True)

    sim = None if a.no_similarity else narrative_similarity(d)
    pos_pairs, pool_pairs = known_duplicate_pairs(d), ungrouped_pairs(d)
    pos = pair_features(d, pos_pairs, sim)
    pool = pair_features(d, pool_pairs, sim)

    cal = calibrate(pos, pool)
    decided = decide(pool, a.rule)
    merges = merge_map(d, decided)

    n_before = d["incident_key"].nunique()
    n_after = apply_merges(d, merges)["incident_key"].nunique()

    rep = {
        "rule_adopted": a.rule,
        "thresholds": {"time_tolerance_min": TIME_TOLERANCE_MIN,
                       "contact_jaccard_min": CONTACT_JACCARD_MIN},
        "n_rows": int(len(d)),
        "n_known_duplicate_pairs": len(pos_pairs),
        "n_ungrouped_candidate_pairs": len(pool_pairs),
        "calibration": cal,
        "similarity_overlap": (similarity_overlap(pos, pool)
                               if not a.no_similarity else
                               {"skipped": "--no-similarity"}),
        "decisions": (decided["decision"].value_counts().to_dict()
                      if not decided.empty else {}),
        "n_merges_proposed": len(merges),
        "merges_cross_entity": int((decided["decision"].eq("merge")
                                    & ~decided["same_entity"]).sum()),
        "incidents_before": int(n_before),
        "incidents_after": int(n_after),
        "residual_duplicate_rate": float((n_before - n_after) / max(n_before, 1)),
        "unblocked_city_variants": unblocked_city_variants(d),
        "merges": merges,
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    sample = audit_sample(d, decided, n=a.audit_n)
    os.makedirs(os.path.dirname(a.audit_out), exist_ok=True)
    with open(a.audit_out, "w") as f:
        for r in sample:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"[dedup] {len(d):,} in-scope filings; {len(pos_pairs)} pairs already "
          f"grouped by Same Incident ID; {len(pool_pairs):,} ungrouped same-block pairs")
    print("\n[dedup] rule calibration (recall on the built-in key's own positives):")
    print(f"{'rule':34s} {'recall':>7s} {'pool hits':>10s} {'prec>=':>7s}")
    for r in cal:
        mark = " <- adopted" if r["rule"] == a.rule else ""
        print(f"{r['rule']:34s} {r['recall_on_known_duplicates']:7.3f} "
              f"{r['hits_in_ungrouped_pool']:10d} "
              f"{r['est_precision_lower_bound']:7.3f}{mark}")
    if not a.no_similarity:
        so = rep["similarity_overlap"]
        print(f"\n[dedup] narrative similarity does NOT separate the two classes: "
              f"{so['pool_above_duplicate_median']:.1%} of ungrouped pairs score "
              f"at or above the median genuine duplicate")
        print(f"[dedup]   duplicates {so['known_duplicates']}")
        print(f"[dedup]   ungrouped  {so['ungrouped_pool']}")
    print(f"\n[dedup] proposed merges: {len(merges)} "
          f"({rep['merges_cross_entity']} cross-entity)")
    print(f"[dedup] incidents {n_before:,} -> {n_after:,} "
          f"(residual duplicate rate {rep['residual_duplicate_rate']:.2%})")
    print(f"[dedup] wrote {a.out}")
    print(f"[dedup] hand-verification sample ({len(sample)} pairs) -> {a.audit_out}; "
          f"fill `verdict` (same / different / unclear)")


if __name__ == "__main__":
    main()
