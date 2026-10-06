"""Build the human double-coding census for the composition comparison.

WHAT THIS IS FOR. The composition findings the paper rests on -- fewer
turn-across-path and single-vehicle crashes among ADS crashes -- are computed on
the comparison window: driverless incidents from 2021-2024. The stratified
300-row sample (`make_composition_sample`) cannot validate them. It holds only
~70 window driverless incidents, and a handful of turn-across-path crashes, so
the per-class accuracy the findings depend on stays unmeasured. The fix is a
CENSUS of the window: every incident the comparison uses is double-coded, so
the human-coded composition can be computed directly rather than inferred from
a model-vs-human error rate.

THE FRAME. Window driverless incidents under the `no_onboard` definition (the
`strict` ones are a subset), taken BEFORE the narrative exclusions -- so the
human exclusion flags decide membership too, instead of inheriting the model's.

BATCHES, IN CODING ORDER.
  0  calibration   20 driverless incidents from 2025-26, outside the window.
                   Coded together and discussed; excluded from every statistic.
  1  candidates    Window incidents that could be turn-across-path or
                   single-vehicle: a left or U-turn coded by either vehicle, a
                   movement coded vague ("Other, see Narrative" / unknown), a
                   model junction or other crash type, or a model single-vehicle
                   code.
  2  remainder     Every other window incident.
  3  corpus        The existing stratified 300 minus the rows already in the
                   window, reweighted to the current corpus. Fields as in
                   `make_composition_sample.CODED_FIELDS`.

Batches 1 and 2 together are the whole frame, so using the model's own output
to pick batch 1 affects only coding ORDER, never which incidents are coded.
Within each batch the order is random and shared by both coders, so if coding
stops early, the completed prefix of a batch is a random sample of that batch
and both coders' prefixes line up.

BLINDING. Coder sheets carry the narrative, the reporting entity and an
identifier -- what the model saw -- and nothing else: no model prediction, no
structured SGO field, no batch-selection reason. The dedup-audit sheet drops the
pipeline's decision and similarity scores for the same reason.

OUTPUT. CSV sheets for a spreadsheet (layout in `annotate.census_csv`):
census_coder_{A,B}.csv, dedup_coder_{A,B}.csv, coding_codes.csv (allowed
values) and census_meta.json.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.composition import attach_structured_key  # noqa: E402
from annotate.make_composition_sample import CODED_FIELDS, _strata  # noqa: E402
from annotate import census_csv  # noqa: E402

OUT_DIR = os.path.join("data", "gold")
INCIDENTS = os.path.join("data", "interim", "ads_incidents.parquet")
EXTRACTIONS = os.path.join("data", "processed", "composition_extractions.jsonl")
PRIOR_SAMPLE = os.path.join(OUT_DIR, "composition_coder_A.jsonl")
DEDUP_SAMPLE = os.path.join(OUT_DIR, "dedup_audit_sample_v2.jsonl")

# Fields coded for every window incident (batches 0-2). Free-text fields are
# recorded but never adjudicated or scored.
WINDOW_FIELDS = [
    "acc_type_category",
    "striking_role",
    "police_reportable",
    "reportable_evidence",
    "is_true_crash",
    "on_public_road",
    "ads_engaged_at_impact",
    "coder_notes",
]
FREE_TEXT_FIELDS = {"reportable_evidence", "coder_notes"}

# Allowed values per coded field. Taken from `schema.composition_schema` so the
# human form and the model are constrained to the same label space; the
# annotator page carries the same lists and a test checks they agree.
ALLOWED = {
    "acc_type_category": ["rear_end", "sideswipe_same_direction", "head_on",
                          "sideswipe_opposite_direction", "turn_across_path",
                          "turn_into_path", "straight_paths_intersecting",
                          "backing", "single_vehicle", "pedestrian_animal",
                          "other_unknown"],
    "striking_role": ["striking", "struck", "undetermined", "not_applicable"],
    "police_reportable": ["yes", "maybe", "no"],
    "is_true_crash": ["yes", "no", "unknown"],
    "on_public_road": ["yes", "no", "unknown"],
    "ads_engaged_at_impact": ["yes", "no", "unknown"],
    "damage_descriptor": ["none", "scratch_scuff", "panel_dent", "deformation",
                          "intrusion", "sensor_only", "unknown"],
    "sensor_only_damage": ["true", "false"],
    "relation_to_junction": ["intersection", "intersection_related",
                             "driveway_access", "non_junction", "unknown"],
    "intersection_type": ["four_way", "t_intersection", "y_intersection",
                          "roundabout_traffic_circle", "five_or_more",
                          "not_an_intersection", "unknown"],
    "verdict": ["same", "different", "unclear"],
}

TURN = {"making left turn", "making u-turn"}
VAGUE = {"other, see narrative", "unknown", ""}
JUNCTION_OR_OTHER = {"turn_across_path", "turn_into_path",
                     "straight_paths_intersecting", "head_on",
                     "sideswipe_opposite_direction", "other_unknown"}
NON_MV_PARTNER = {"other fixed object", "pole / tree", "animal"}


def _seed_order(ids: list[str], seed: int, batch: int) -> list[str]:
    """Deterministic random order, shared by every coder."""
    rng = np.random.default_rng([seed, batch])
    ids = sorted(ids)
    return [ids[i] for i in rng.permutation(len(ids))]


def _model_predictions(path: str) -> dict[str, str]:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("extraction"):
                out[str(r["report_id"])] = r["extraction"].get("acc_type_category")
    return out


def census_frame(d: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Window driverless (no_onboard) incidents with an extractable narrative."""
    words = d["Narrative"].astype(str).str.split().str.len()
    keep = (d["incident_year"].isin(years) & d["driverless_no_onboard"].eq(True)
            & (words >= 8))
    return d[keep].copy()


def candidate_mask(d: pd.DataFrame, pred: dict[str, str]) -> pd.Series:
    """Incidents that could be turn-across-path or single-vehicle (batch 1)."""
    mv = lambda c: d[c].astype(str).str.strip().str.lower()
    partner = d["crash_with"].astype(str).str.strip().str.lower()
    mv_partner = ~partner.str.startswith("non-motorist") & ~partner.isin(NON_MV_PARTNER)
    turn = mv_partner & (mv("SV Pre-Crash Movement").isin(TURN)
                         | mv("CP Pre-Crash Movement").isin(TURN))
    vague = mv_partner & (mv("SV Pre-Crash Movement").isin(VAGUE)
                          | mv("CP Pre-Crash Movement").isin(VAGUE))
    p = d["Report ID"].astype(str).map(pred)
    return turn | vague | p.isin(JUNCTION_OR_OTHER) | p.eq("single_vehicle")


def corpus_weights(d: pd.DataFrame, prior_ids: list[str]) -> dict[str, float]:
    """Reweight the existing stratified sample to the CURRENT corpus.

    The 300 were drawn from an earlier, smaller corpus. Their strata are
    recomputed on the current incidents, and each row's weight is the current
    stratum size over the number of the 300 in that stratum.
    """
    words = d["Narrative"].astype(str).str.split().str.len()
    elig = attach_structured_key(d[words >= 8].copy())
    elig["_stratum"] = _strata(elig)
    sizes = elig["_stratum"].value_counts()
    s = elig.set_index(elig["Report ID"].astype(str))["_stratum"]
    drawn = s.reindex(prior_ids).value_counts()
    return {rid: float(sizes[s[rid]] / drawn[s[rid]])
            for rid in prior_ids if rid in s.index}


def _row(r: pd.Series, batch: int, fields: list[str]) -> dict:
    return {
        "report_id": str(r["Report ID"]),
        "incident_key": str(r["incident_key"]),
        "entity": str(r["Reporting Entity"]),
        "batch": batch,
        "fields": fields,
        "narrative": str(r["Narrative"]),
        **{f: "" for f in fields},
    }


def _dedup_rows(path: str, seed: int) -> list[dict]:
    """Blinded side-by-side pairs: no pipeline decision, no similarity score."""
    if not os.path.exists(path):
        return []
    keep = ["entity_i", "entity_j", "report_i", "report_j", "month", "city",
            "state", "time_i", "time_j", "partner_i", "partner_j",
            "narrative_i", "narrative_j"]
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"report_id": f"{r['report_i']}|{r['report_j']}",
                         "batch": 4, "fields": ["verdict", "coder_notes"],
                         **{k: r.get(k) for k in keep},
                         "verdict": "", "coder_notes": ""})
    order = _seed_order([r["report_id"] for r in rows], seed, 4)
    by_id = {r["report_id"]: r for r in rows}
    return [by_id[i] for i in order]


def build(incidents: str = INCIDENTS, extractions: str = EXTRACTIONS,
          prior_sample: str = PRIOR_SAMPLE, dedup_sample: str = DEDUP_SAMPLE,
          years=(2021, 2022, 2023, 2024), n_calibration: int = 20,
          seed: int = 11) -> tuple[list[dict], list[dict], dict]:
    """Return (census rows, dedup rows, meta). Pure: writes nothing."""
    years = list(years)
    d = pd.read_parquet(incidents)
    pred = _model_predictions(extractions)

    frame = census_frame(d, years)
    cand = candidate_mask(frame, pred)
    b1_ids = _seed_order(frame.loc[cand, "Report ID"].astype(str).tolist(), seed, 1)
    b2_ids = _seed_order(frame.loc[~cand, "Report ID"].astype(str).tolist(), seed, 2)

    prior_ids = []
    if os.path.exists(prior_sample):
        with open(prior_sample) as f:
            prior_ids = [str(json.loads(l)["report_id"]) for l in f if l.strip()]

    # Calibration rows are discussed jointly, so they must never be a row that
    # is also coded independently -- the stratified sample reaches into
    # 2025-26 too, and a shared incident would leak the discussion into batch 3.
    words = d["Narrative"].astype(str).str.split().str.len()
    calib_pool = d[(d["incident_year"] >= max(years) + 1)
                   & d["driverless_no_onboard"].eq(True) & (words >= 8)
                   & ~d["Report ID"].astype(str).isin(prior_ids)]
    calib_ids = _seed_order(calib_pool["Report ID"].astype(str).tolist(),
                            seed, 0)[:n_calibration]
    weights = corpus_weights(d, prior_ids)
    in_frame = set(b1_ids) | set(b2_ids)
    b3_ids = _seed_order([i for i in prior_ids if i not in in_frame
                          and i in weights], seed, 3)

    by_id = d.set_index(d["Report ID"].astype(str))
    rows = []
    for batch, ids, fields in [(0, calib_ids, WINDOW_FIELDS),
                               (1, b1_ids, WINDOW_FIELDS),
                               (2, b2_ids, WINDOW_FIELDS),
                               (3, b3_ids, CODED_FIELDS)]:
        rows += [_row(by_id.loc[i], batch, list(fields)) for i in ids]
    dedup = _dedup_rows(dedup_sample, seed)
    ids = [r["report_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("[census] an incident landed in two batches")

    def _md5(p):
        return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None

    meta = {
        "frame": ("incident_year in %s, driverless_no_onboard, narrative >= 8 "
                  "words, BEFORE narrative exclusions" % years),
        "seed": seed,
        "order": "random within batch, shared by all coders",
        "n_by_batch": {"0_calibration": len(calib_ids), "1_candidates": len(b1_ids),
                       "2_remainder": len(b2_ids), "3_corpus_sample": len(b3_ids),
                       "4_dedup_pairs": len(dedup)},
        "n_frame": int(len(frame)),
        "n_frame_strict": int(frame["driverless_strict"].eq(True).sum()),
        "batch": {**{i: 0 for i in calib_ids}, **{i: 1 for i in b1_ids},
                  **{i: 2 for i in b2_ids}, **{i: 3 for i in b3_ids}},
        "batch_order": {"1": b1_ids, "2": b2_ids},
        "corpus_sample_weight": weights,
        "corpus_sample_ids": prior_ids,
        "window_fields": WINDOW_FIELDS,
        "corpus_fields": CODED_FIELDS,
        "free_text_fields": sorted(FREE_TEXT_FIELDS),
        "allowed": ALLOWED,
        "blinding": ("coder files carry narrative, entity and ids only; no model "
                     "prediction, structured field or batch-selection reason"),
        "inputs_md5": {"incidents": _md5(incidents), "extractions": _md5(extractions),
                       "prior_sample": _md5(prior_sample),
                       "dedup_sample": _md5(dedup_sample)},
    }
    return rows, dedup, meta


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build the window double-coding census.")
    ap.add_argument("--incidents", default=INCIDENTS)
    ap.add_argument("--extractions", default=EXTRACTIONS)
    ap.add_argument("--prior-sample", default=PRIOR_SAMPLE)
    ap.add_argument("--dedup-sample", default=DEDUP_SAMPLE)
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--n-calibration", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--coders", nargs="+", default=["A", "B"])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing coder sheets. Refused by default, "
                         "because they may already hold a coder's work.")
    a = ap.parse_args()

    rows, dedup, meta = build(a.incidents, a.extractions, a.prior_sample,
                              a.dedup_sample, a.years, a.n_calibration, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    targets = [os.path.join(a.out_dir, f"{kind}_coder_{c}.csv")
               for c in a.coders for kind in ("census", "dedup")]
    existing = [p for p in targets if os.path.exists(p)]
    if existing and not a.force:
        raise SystemExit(f"[census] refusing to overwrite {existing}; pass --force")
    for c in a.coders:
        path = os.path.join(a.out_dir, f"census_coder_{c}.csv")
        census_csv.write_census(rows, path)
        print(f"[census] wrote {path} ({len(rows)} rows)")
        path = os.path.join(a.out_dir, f"dedup_coder_{c}.csv")
        census_csv.write_dedup(dedup, path)
        print(f"[census] wrote {path} ({len(dedup)} pairs)")
    cpath = os.path.join(a.out_dir, "coding_codes.csv")
    census_csv.write_codes(ALLOWED, cpath)
    print(f"[census] wrote {cpath}")
    mpath = os.path.join(a.out_dir, "census_meta.json")
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[census] frame: {meta['n_frame']} window driverless incidents "
          f"({meta['n_frame_strict']} strict)")
    print(f"[census] batches: {meta['n_by_batch']}")
    print(f"[census] wrote {mpath}")


if __name__ == "__main__":
    main()
