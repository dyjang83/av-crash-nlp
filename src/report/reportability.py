"""Reportability: two definitions, bracketed, never one.

THE PROBLEM. SGO compels a filing far below the bar at which a human crash
generates a police report, and CRSS samples police-reported crashes. Any
comparison between them is a comparison of reporting rules unless the ADS side
is first cut down to crashes that would have entered the human data. There is
no single correct cut. So two are constructed, and a structural difference is
claimed ONLY where it survives both (and, for Tier A, all three of its tow
variants).

    TIER A  objective severity floor.  KABCO B-or-worse, OR tow-away due to
            damage, OR airbag deployment. Built from structured fields on BOTH
            sides, with near-identical definitions and minimal underreporting
            at this severity. No judgement, no model. The cost is sample size.
            PRIMARY for anything role-based.

    TIER B  the reasonable-person rubric. An operationalization of the standard
            IIHS applied by hand, run by the extraction layer, yielding
            yes/maybe/no with an evidence span. Reported yes-only AND yes+maybe.
            PRIMARY for composition.

WHY THERE IS NO TIER C. The post-amendment filing regime was originally carried
as a third tier: filings made under the third-amended order, taken as filed, on
the reasoning that its built-in damage thresholds sit closer to a police
reporting bar. It cannot serve as a tier, for a reason that is arithmetic
rather than conceptual. Every post-amendment filing concerns a 2025 or 2026
incident, and CRSS ends at 2024, so the post-amendment population is EMPTY in
the comparison window -- 0 of 829 incidents. A tier that selects nothing from
the population being compared is not a bracket on the comparison; it is a
different corpus.

The regime split is still used, and is still informative, as a VALIDATION CHECK
rather than a tier (see `regime_check`). It answers whether a reportability
coder responds to a change in filing thresholds. That is a question about the
coder, answered within the ADS corpus, and it needs no human comparator -- which
is exactly why it does not belong in a list of ways to cut the ADS side down to
the human population.

-----------------------------------------------------------------------------
THE ASYMMETRY THAT MUST BE STATED, NOT CORRECTED
-----------------------------------------------------------------------------
CRSS contains only crashes that were ACTUALLY reported, and Blincoe et al.
estimate that over half of all crashes never are. Tier B deliberately codes the
ADS side up to "a reasonable person WOULD report this", not "this was reported"
and not "the law required a report". So Tier B's ADS population is, if
anything, more inclusive than the human population it is compared against --
which biases toward finding the ADS mix similar to the human mix, not toward
finding it different.

Tier A is the leg that is insensitive to this: at injury-B-or-worse, tow-away,
or airbag deployment, human underreporting is small. Where Tier A and Tier B
agree, the asymmetry is not driving the result.

-----------------------------------------------------------------------------
WHAT VALIDATES TIER B
-----------------------------------------------------------------------------
Four checks, in increasing order of how much they would cost to fake:

  1. HUMAN DOUBLE-CODING. kappa / AC1 / majority share against two independent
     human coders on a stratified sample. This is the headline validity number
     and nothing substitutes for it.
  2. POLICE-INVOLVEMENT GRADIENT. IIHS found SGO-noted police involvement in
     16% / 42% / 68% of their no / maybe / yes cases. The gradient must be
     monotone and of similar steepness, and it uses a structured field the
     rubric never sees.
  3. REGIME CHECK -- AND THE EXPECTATION BEHIND IT WAS WRONG. The check was
     specified as "the reportable share must rise sharply after June 2025, when
     the amendment raised the filing threshold; a coder that cannot see this is
     not measuring reportability". Measured, Tier A rises 0.313 -> 0.543 while
     Tier B FALLS 0.238 -> 0.152, and the reason is that the amendment raised a
     DAMAGE-AND-TOW bar, not a severity bar:

         SV towed          0.253 -> 0.498      injury (structured)  0.095 -> 0.110
         CP towed          0.051 -> 0.078      airbag               0.038 -> 0.049
         police involved   0.206 -> 0.233      narrative injury     0.250 -> 0.210

     Almost the whole Tier A jump is the subject-AV tow leg, which is fleet
     recovery rather than damage (see `tier_a_variants`). Under the
     de-confounded definitions the rise is +3.6 points (cp_tow) or +1.4 points
     (no_tow), not +23. The post-amendment filing population is more TOWED, not
     more injurious, so a reportability coder SHOULD NOT show a sharp rise --
     and Tier B declining to is evidence it is measuring severity rather than
     filing volume.

     The usable form of this check is therefore: Tier B must track the
     de-confounded severity legs (injury, airbag, police involvement), not the
     raw filing count. A coder that reproduced the +23-point jump would be
     tracking AV recovery tows.
  4. IIHS MARGINALS. Restricted to their frame, the overall reportable share
     and the by-crash-type shares should land near their published values.
     Case-level agreement would be far stronger, and is not available.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

TIERS = ["tier_a", "tier_a_cp_tow", "tier_a_no_tow",
         "tier_b_yes", "tier_b_yes_maybe"]

# IIHS published marginals, for the external calibration in `iihs_marginals`.
# Held here as data, clearly labelled as THEIR numbers, so the comparison code
# cannot drift from what is being compared against. Case-level codes were
# requested and are not available; this is a marginals-only check and is
# reported as such.
IIHS_PUBLISHED = {
    "frame": "Waymo, driverless, Jul 2021 - Dec 2024, public road, ADS engaged, "
             "true crash",
    "reportable_share_yes_or_maybe": 0.216,
    "by_crash_type_yes_or_maybe": {
        "head_on": 0.83,
        "straight_paths_intersecting": 0.72,   # "T-bone"
        "backing": 0.07,
        "sideswipe_same_direction": 0.11,
    },
    "police_involvement_by_label": {"no": 0.16, "maybe": 0.42, "yes": 0.68},
    "source": "Teoh, Kidd & Riexinger (IIHS). Marginals transcribed from the "
              "published tables; case-level codes were not available.",
}


# ---------------------------------------------------------------------------
# Tier A -- structured, both sides, no model
# ---------------------------------------------------------------------------
def _sgo_kabco_floor(severity) -> Optional[bool]:
    """True if SGO's alleged severity is minor-or-worse.

    SGO's ladder has no KABCO letters. `Minor` is the lowest injury grade it
    files and corresponds to KABCO B/C, so "minor or worse" is the closest
    available reading of the B-or-worse floor. It is marginally LOOSER than
    K/A/B alone on both sides -- `human.crss_extract.tier_a` makes the same
    concession for the same reason -- and the looseness is symmetric, which is
    what matters.
    """
    t = str(severity or "").strip().lower()
    if not t or t == "unknown":
        return None
    if "fatal" in t or "serious" in t or "moder" in t or "minor" in t:
        return True
    if "no inj" in t or "property damage" in t or t in {"none", "no"}:
        return False
    return None


def tier_a(d: pd.DataFrame) -> pd.Series:
    """Objective severity floor on the ADS side.

    Uses the harmonized `any_towed` / `any_airbag` built by `corpus.sgo_ads`,
    which parse both the pre-amendment two-column form and the post-amendment
    combined-value form. Reading the raw columns here instead would silently
    make Tier A a pre-amendment-only analysis.

    A row is Tier A if ANY leg is affirmatively true. A row with every leg
    unknown is NOT Tier A -- but it is counted separately by `tier_a_coverage`,
    because "we could not tell" and "it did not qualify" are different, and the
    share that is merely untellable bounds how much Tier A could move.
    """
    sev = d["Highest Injury Severity Alleged"].map(_sgo_kabco_floor)
    return (sev.eq(True) | d["any_towed"].eq(True) | d["any_airbag"].eq(True))


def tier_a_coverage(d: pd.DataFrame) -> dict:
    sev = d["Highest Injury Severity Alleged"].map(_sgo_kabco_floor)
    legs = pd.DataFrame({"sev": sev, "tow": d["any_towed"], "bag": d["any_airbag"]})
    known = legs.notna().any(axis=1)
    all_unknown = ~known
    return {
        "n": int(len(d)),
        "n_tier_a": int(tier_a(d).sum()),
        "n_all_legs_unknown": int(all_unknown.sum()),
        "share_all_legs_unknown": float(all_unknown.mean()),
        "by_leg": {
            "severity_minor_or_worse": int(sev.eq(True).sum()),
            "tow_away": int(d["any_towed"].eq(True).sum()),
            "airbag": int(d["any_airbag"].eq(True).sum()),
        },
    }


# ---------------------------------------------------------------------------
# Tier B -- the rubric, from the extraction layer
# ---------------------------------------------------------------------------
def attach_extractions(d: pd.DataFrame, path: str, model: Optional[str] = None
                       ) -> pd.DataFrame:
    """Join the composition extraction onto the incident frame.

    Filtered to ONE model for the same reason `build_features.load()` is: an
    unfiltered dedup resolves by JSONL append order, and the contested rows are
    exactly the ones where one model failed schema validation, so the other
    silently backfills the hardest records.
    """
    if not os.path.exists(path):
        return d.assign(police_reportable=pd.NA, reportable_confidence=np.nan,
                        _has_extraction=False)
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok") or r.get("extraction") is None:
                continue
            if model and r.get("model") != model:
                continue
            e = dict(r["extraction"])
            e["report_id"] = str(r["report_id"])
            e["_model"] = r.get("model")
            rows.append(e)
    if not rows:
        return d.assign(police_reportable=pd.NA, reportable_confidence=np.nan,
                        _has_extraction=False)
    ext = pd.DataFrame(rows).drop_duplicates("report_id", keep="last")
    out = d.merge(ext, left_on=d["Report ID"].astype(str), right_on="report_id",
                  how="left", suffixes=("", "_ext"))
    out["_has_extraction"] = out["police_reportable"].notna()
    return out


# Fields the human census codes and the gold file carries. With
# --label-source gold these REPLACE the extracted values, so every downstream
# module (composition, qie, sequences) runs on human labels unchanged, by being
# pointed at the gold parquet with --ads.
GOLD_FIELDS = ["acc_type_category", "striking_role", "police_reportable",
               "is_true_crash", "on_public_road", "ads_engaged_at_impact"]


def attach_gold(d: pd.DataFrame, gold_path: str, meta_path: str) -> pd.DataFrame:
    """Restrict to the census frame and overwrite the coded fields with gold.

    The census covers the comparison window only (batches 1 and 2 of
    `annotate.make_composition_census`), so the gold frame is that window, not
    the full corpus. Every frame incident must have a gold label: a partly
    coded frame would mix human and model labels in one distribution, which is
    exactly the comparison this exists to avoid, so it raises instead.
    """
    with open(meta_path) as f:
        meta = json.load(f)
    frame = set(meta["batch_order"]["1"]) | set(meta["batch_order"]["2"])
    gold = {}
    with open(gold_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if str(r["report_id"]) in frame:
                    gold[str(r["report_id"])] = r
    missing = sorted(frame - set(gold))
    if missing:
        raise SystemExit(
            f"[report] gold labels cover {len(gold)} of {len(frame)} census "
            f"incidents; finish coding and adjudication first (e.g. missing "
            f"{missing[:3]})")
    rid = d["Report ID"].astype(str)
    out = d[rid.isin(frame)].copy()
    rid = out["Report ID"].astype(str)
    for fld in GOLD_FIELDS:
        vals = rid.map(lambda i: str(gold[i].get(fld) or "").strip().lower() or None)
        if vals.isna().any():
            raise SystemExit(f"[report] gold field {fld!r} is blank for "
                             f"{int(vals.isna().sum())} census incidents")
        out[fld] = vals
    out["_has_extraction"] = True
    out["label_source"] = "gold"
    return out


def tier_b(d: pd.DataFrame, include_maybe: bool) -> pd.Series:
    """Reasonable-person reportability. `include_maybe` pools maybe with yes.

    Both are reported, always. IIHS pooled them and called that conservative;
    whether it is conservative depends on which direction the result moves, so
    the pooled and unpooled versions are carried side by side rather than one
    being chosen.
    """
    lab = d.get("police_reportable")
    if lab is None:
        return pd.Series(False, index=d.index)
    keep = {"yes", "maybe"} if include_maybe else {"yes"}
    return lab.astype(str).str.lower().isin(keep)


# ---------------------------------------------------------------------------
# The post-amendment regime: a validation check, NOT a tier
# ---------------------------------------------------------------------------
# There is deliberately no `tier_c` here. The split lives on `filing_regime`,
# which `corpus.sgo_ads` reads off the source file rather than the incident
# date -- the filer was subject to the threshold in force when the filing was
# made, and in any case the incident day is redacted so June 2025 cannot be cut
# at the 16th. `regime_check` below consumes it. See the module docstring for
# why it is not a reportability tier.


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
def police_gradient(d: pd.DataFrame) -> dict:
    """Police involvement by reportability label -- must be monotone.

    `police_named_agency` is a structured SGO field the rubric never sees, so
    agreement between it and the label is genuine external evidence rather than
    a restatement. IIHS reported 16% / 42% / 68% for no / maybe / yes.
    """
    if "police_reportable" not in d.columns or d["police_reportable"].isna().all():
        return {"available": False,
                "why": "no composition extractions joined; run extract.composition_extract"}
    sub = d[d["police_reportable"].notna()]
    out = {}
    for lab in ("no", "maybe", "yes"):
        g = sub[sub["police_reportable"].astype(str).str.lower() == lab]
        out[lab] = {"n": int(len(g)),
                    "police_share": (float(g["police_named_agency"].mean())
                                     if len(g) else None)}
    shares = [out[l]["police_share"] for l in ("no", "maybe", "yes")]
    known = [s for s in shares if s is not None]
    return {
        "available": True,
        "by_label": out,
        "monotone_increasing": bool(len(known) == 3 and known[0] <= known[1] <= known[2]),
        "iihs_published": IIHS_PUBLISHED["police_involvement_by_label"],
    }


def regime_check(d: pd.DataFrame) -> dict:
    """Does the reportable share rise after the amendment?

    Reported for BOTH the model rubric and Tier A. Tier A is the control: it is
    a fixed severity floor that the amendment did not change, so if Tier A
    moves as much as the rubric does, the rubric is tracking a change in the
    severity MIX of what gets filed rather than a change in thresholds -- which
    is a different (and still interesting) finding, but not the validation it
    would otherwise be.
    """
    out = {}
    for reg in ("pre", "post"):
        g = d[d["filing_regime"] == reg]
        if not len(g):
            continue
        row = {"n": int(len(g)), "tier_a_share": float(tier_a(g).mean())}
        if "police_reportable" in g.columns and g["police_reportable"].notna().any():
            lab = g["police_reportable"].astype(str).str.lower()
            row["tier_b_yes_share"] = float(lab.eq("yes").mean())
            row["tier_b_yes_maybe_share"] = float(lab.isin({"yes", "maybe"}).mean())
        row["police_named_share"] = float(g["police_named_agency"].mean())
        out[reg] = row
    if "pre" in out and "post" in out:
        out["delta_tier_a"] = out["post"]["tier_a_share"] - out["pre"]["tier_a_share"]
        if "tier_b_yes_share" in out["pre"]:
            out["delta_tier_b_yes"] = (out["post"]["tier_b_yes_share"]
                                       - out["pre"]["tier_b_yes_share"])
    return out


def iihs_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Restrict to IIHS's analysis frame for the external calibration.

    Waymo operations means Waymo LLC AND Transdev Alternative Services: Transdev
    supplied personnel for Waymo operations and co-filed 139 incidents under a
    shared Same Incident ID. Treating Transdev as a separate operator would
    drop or double-count exactly the co-filed subset.
    """
    waymo = {"Waymo LLC", "Transdev Alternative Services"}
    return d[
        d["Reporting Entity"].isin(waymo)
        & d["driverless"].eq(True)
        & d["incident_month"].between("2021-07-01", "2024-12-01")
    ].copy()


def iihs_marginals(d: pd.DataFrame) -> dict:
    """Compare this pipeline's marginals against IIHS's published ones.

    MARGINALS ONLY. Case-level agreement would be a far stronger check -- two
    pipelines can reach the same 21.6% by disagreeing on which crashes -- and
    the case-level codes were not available. That limitation is carried in the
    output rather than left to the write-up.
    """
    f = iihs_frame(d)
    res = {
        "frame": IIHS_PUBLISHED["frame"],
        "n_in_frame": int(len(f)),
        "case_level_agreement": None,
        "case_level_note": "IIHS case-level codes not available; marginals only. "
                           "Two pipelines can agree on a marginal while "
                           "disagreeing on which crashes produced it.",
        "iihs": IIHS_PUBLISHED,
    }
    if not len(f):
        return res
    res["tier_a_share"] = float(tier_a(f).mean())
    if "police_reportable" in f.columns and f["police_reportable"].notna().any():
        lab = f["police_reportable"].astype(str).str.lower()
        res["ours_yes_or_maybe"] = float(lab.isin({"yes", "maybe"}).mean())
        res["ours_yes"] = float(lab.eq("yes").mean())
        res["abs_diff_vs_iihs"] = abs(res["ours_yes_or_maybe"]
                                      - IIHS_PUBLISHED["reportable_share_yes_or_maybe"])
        if "acc_type_category" in f.columns:
            by = {}
            for cat, share in IIHS_PUBLISHED["by_crash_type_yes_or_maybe"].items():
                g = f[f["acc_type_category"] == cat]
                by[cat] = {
                    "n": int(len(g)),
                    "ours": (float(g["police_reportable"].astype(str).str.lower()
                                   .isin({"yes", "maybe"}).mean()) if len(g) else None),
                    "iihs": share,
                }
            res["by_crash_type"] = by
    else:
        res["note"] = ("no composition extractions joined, so only the "
                       "structured Tier A share is comparable")
    return res


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def narrative_exclusions(d: pd.DataFrame) -> tuple[pd.Series, dict]:
    """IIHS's remaining exclusions, which only the narrative can supply.

    `corpus.sgo_ads.apply_exclusions` handles the two with structured
    counterparts (ADS engaged, public road). The third -- "this filing does not
    describe a crash" -- has none: SGO has no not-a-crash value, and the
    determination is a reading of the text (a reported contact that turned out
    to be a pothole, a filing made out of caution with no contact, a third
    party opening a door against a stationary AV). That is why it is extracted.

    ONLY AN AFFIRMATIVE 'no' EXCLUDES. A record whose flag is `unknown`, or
    which has no extraction at all, is RETAINED. Dropping those would make
    corpus membership depend on the extractor having an opinion, and the
    records it has no opinion about are disproportionately the short and
    redacted ones -- which is not independent of anything. The counts are
    reported so the reader can see both what was excluded and how much the
    extractor declined to judge.
    """
    n = len(d)
    if "is_true_crash" not in d.columns:
        return pd.Series(True, index=d.index), {
            "available": False,
            "why": "no composition extractions; narrative exclusions not applied"}

    def _no(col):
        return d[col].astype(str).str.lower().eq("no") if col in d.columns \
            else pd.Series(False, index=d.index)

    def _unk(col):
        return d[col].astype(str).str.lower().eq("unknown") if col in d.columns \
            else pd.Series(False, index=d.index)

    not_crash = _no("is_true_crash")
    not_public = _no("on_public_road")
    not_engaged = _no("ads_engaged_at_impact")
    dropped = not_crash | not_public | not_engaged
    keep = ~dropped

    # THE THREE CRITERIA OVERLAP, SO THEIR COUNTS DO NOT ADD UP TO THE TOTAL.
    # A filing can be both "not a crash" and "ADS not engaged" -- 4 are -- and
    # one is both private property and not engaged. Reporting the three sums
    # alone invites a reader to add them (205) and find it does not reconcile
    # against the frame (200 actually removed). The union and the overlap are
    # therefore reported alongside, and `n_excluded` is the number that
    # reconciles the funnel.
    naive = int(not_crash.sum() + not_public.sum() + not_engaged.sum())
    return keep, {
        "available": True,
        "n_before": int(n), "n_after": int(keep.sum()),
        "n_excluded": int(dropped.sum()),
        "sum_of_criteria": naive,
        "overlap_double_counted": naive - int(dropped.sum()),
        "excluded_not_a_crash": int(not_crash.sum()),
        "excluded_not_public_road": int(not_public.sum()),
        "excluded_ads_not_engaged": int(not_engaged.sum()),
        "pairwise_overlap": {
            "not_crash_and_not_public": int((not_crash & not_public).sum()),
            "not_crash_and_not_engaged": int((not_crash & not_engaged).sum()),
            "not_public_and_not_engaged": int((not_public & not_engaged).sum()),
            "all_three": int((not_crash & not_public & not_engaged).sum()),
        },
        "unknown_retained": {
            "is_true_crash": int(_unk("is_true_crash").sum()),
            "on_public_road": int(_unk("on_public_road").sum()),
            "ads_engaged_at_impact": int(_unk("ads_engaged_at_impact").sum()),
        },
        "no_extraction_retained": int((~d["_has_extraction"]).sum()),
    }


def tier_a_variants(d: pd.DataFrame) -> dict:
    """Tier A under three tow definitions. The spread is a finding, not a knob.

    THE PROBLEM. Tier A's tow leg carries 806 of its 896 records, and SGO
    records 672 crashes in which ONLY the subject AV was towed against 134 in
    which the crash partner was. A five-to-one asymmetry in that direction is
    not what damage would produce: the AV is disproportionately the stationary
    struck party (see `models.sequences`), so if tows tracked damage the AV
    would be towed LESS, not more. It is fleet recovery -- an automated vehicle
    that cannot resume autonomously is collected regardless of damage.

    Nothing equivalent happens on the human side. CRSS's `TOWED` is dominated
    by a bare 'Towed' whose reason is unstated, so the human side is not
    restricted to disabling tows either; but no one recovers a member of the
    public's car as a matter of policy, so a human tow still implies the car
    was not driven away.

    Three definitions are therefore carried, and a Tier A result is only as
    strong as its stability across them:

        any_tow   sev | (SV or CP towed) | airbag   -- the permissive reading
        cp_tow    sev | CP towed          | airbag  -- drops fleet recovery
        no_tow    sev | airbag                      -- the floor, tow-free
    """
    sev = d["Highest Injury Severity Alleged"].map(_sgo_kabco_floor)
    bag = d["any_airbag"].eq(True)
    return {
        "any_tow": sev.eq(True) | d["any_towed"].eq(True) | bag,
        "cp_tow": sev.eq(True) | d["cp_towed"].eq(True) | bag,
        "no_tow": sev.eq(True) | bag,
    }


def build_tiers(d: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per tier."""
    out = d.copy()
    out["tier_a"] = tier_a(out)
    for name, mask in tier_a_variants(out).items():
        out[f"tier_a_{name}"] = mask
    out["tier_b_yes"] = tier_b(out, include_maybe=False)
    out["tier_b_yes_maybe"] = tier_b(out, include_maybe=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build and validate reportability tiers.")
    ap.add_argument("--incidents",
                    default=os.path.join("data", "interim", "ads_incidents.parquet"))
    ap.add_argument("--extractions",
                    default=os.path.join("data", "processed",
                                         "composition_extractions.jsonl"))
    ap.add_argument("--model", default=None,
                    help="Restrict to one extraction model. Omit to take "
                         "whatever is in the file (single-model runs only).")
    ap.add_argument("--keep-non-crashes", action="store_true",
                    help="Skip the narrative exclusions (not-a-crash, private "
                         "property, ADS not engaged). Sensitivity run only -- "
                         "these are IIHS's exclusions and the default applies them.")
    ap.add_argument("--label-source", choices=["extracted", "gold"],
                    default="extracted",
                    help="gold = replace the extracted crash type, role, "
                         "reportability and exclusion flags with adjudicated "
                         "human labels, on the census frame only. Writes to "
                         "*_gold outputs unless --out/--report are given.")
    ap.add_argument("--gold", default=os.path.join("data", "gold",
                                                   "composition_gold.jsonl"))
    ap.add_argument("--census-meta", default=os.path.join("data", "gold",
                                                          "census_meta.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    suffix = "_gold" if a.label_source == "gold" else ""
    a.out = a.out or os.path.join("data", "interim", f"ads_reportability{suffix}.parquet")
    a.report = a.report or os.path.join("data", "processed", f"reportability{suffix}.json")

    d = pd.read_parquet(a.incidents)
    d = attach_extractions(d, a.extractions, a.model)
    if a.label_source == "gold":
        d = attach_gold(d, a.gold, a.census_meta)
        print(f"[report] label source: GOLD, {len(d):,} census incidents")

    keep, excl = narrative_exclusions(d)
    if excl.get("available") and not a.keep_non_crashes:
        d = d[keep].copy()
    d = build_tiers(d)

    rep = {
        "label_source": a.label_source,
        "n_incidents": int(len(d)),
        "n_with_extraction": int(d["_has_extraction"].sum()),
        "narrative_exclusions": excl,
        "tier_a_coverage": tier_a_coverage(d),
        "tier_counts": {t: int(d[t].sum()) for t in TIERS},
        "tier_shares": {t: float(d[t].mean()) for t in TIERS},
        "regime_check": regime_check(d),
        "police_gradient": police_gradient(d),
        "iihs_calibration": iihs_marginals(d),
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    d.to_parquet(a.out, index=False)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    print(f"[report] {rep['n_incidents']:,} incidents, "
          f"{rep['n_with_extraction']:,} with a composition extraction")
    ex = rep["narrative_exclusions"]
    if ex.get("available"):
        print(f"[report] narrative exclusions: {ex['n_before']:,} -> "
              f"{ex['n_after']:,} (-{ex['n_excluded']} union)")
        print(f"[report]   criteria overlap: {ex['excluded_not_a_crash']} not a "
              f"crash + {ex['excluded_not_public_road']} private property + "
              f"{ex['excluded_ads_not_engaged']} ADS not engaged = "
              f"{ex['sum_of_criteria']}, but {ex['overlap_double_counted']} fail "
              f"more than one, so {ex['n_excluded']} records are removed")
        print(f"[report]   retained despite no verdict: "
              f"{ex['unknown_retained']}, {ex['no_extraction_retained']:,} "
              f"with no extraction at all")
    else:
        print(f"[report] narrative exclusions: {ex['why']}")
    cov = rep["tier_a_coverage"]
    print(f"[report] Tier A: {cov['n_tier_a']:,} "
          f"({cov['n_tier_a']/cov['n']:.1%}); legs {cov['by_leg']}")
    print(f"[report]   {cov['share_all_legs_unknown']:.1%} of incidents have every "
          f"Tier A leg unknown (upper bound on how much Tier A could move)")
    print("\n[report] tier counts:")
    for t in TIERS:
        print(f"[report]   {t:18s} {rep['tier_counts'][t]:6,d}  "
              f"{rep['tier_shares'][t]:6.1%}")
    print("\n[report] regime check (does the reportable share rise post-amendment?):")
    for k, v in rep["regime_check"].items():
        print(f"[report]   {k}: {v}")
    pg = rep["police_gradient"]
    if pg.get("available"):
        print(f"\n[report] police gradient: {pg['by_label']} "
              f"(monotone={pg['monotone_increasing']}, IIHS={pg['iihs_published']})")
    else:
        print(f"\n[report] police gradient unavailable: {pg['why']}")
    ic = rep["iihs_calibration"]
    print(f"\n[report] IIHS frame: n={ic['n_in_frame']:,}, "
          f"Tier A share {ic.get('tier_a_share', float('nan')):.1%}")
    if "ours_yes_or_maybe" in ic:
        print(f"[report]   ours yes+maybe {ic['ours_yes_or_maybe']:.1%} vs "
              f"IIHS {IIHS_PUBLISHED['reportable_share_yes_or_maybe']:.1%} "
              f"(|diff| {ic['abs_diff_vs_iihs']:.3f})")
    print(f"\n[report] wrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
