"""Held-out validation of the token-probability recalibration.

The two-confidence-signals analysis reports ECE after isotonic recalibration
fit out-of-fold with StratifiedKFold over RECORDS: every test fold shares
manufacturers, reporting templates and the corpus with the data the
recalibrator was fit on, so the low recalibrated ECE (0.006-0.077) is an
in-distribution figure. This re-scores it two stricter ways:

  held-out entity   GroupKFold by vehicle make: the recalibrator never sees the
                    manufacturer it is applied to. Both models.
  held-out corpus   fit on CA DMV OL 316 decisions, apply to NHTSA SGO
                    decisions: a different regulator, form and writing style.
                    Both models: each run covers all 864 OL 316 records.

Signal frames, correctness keys and ECE binning are imported from the
two-confidence-signals code itself, so "correct" and "ECE" mean exactly what
they mean in Table 6.

Usage:
    python -m report.two_signals_holdout
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

STUDY = os.path.expanduser(os.environ.get(
    "TWO_CONF_CODE", "~/Desktop/two-confidence-signals/code/score_confidence.py"))
NARR = os.path.join("data", "interim", "narratives.jsonl")
GOLD = os.path.join("data", "gold", "gold.jsonl")
RUNS = {"qwen2.5:7b": os.path.join("data", "processed", "extractions_qwen_full.jsonl"),
        "llama3.1:8b": os.path.join("data", "processed", "extractions_llama_full.jsonl")}
OUT = os.path.join("data", "processed", "two_signals_holdout.json")
MIN_N = 150
MIN_SIDE = 50   # decisions required on each side of the cross-corpus split


def _study():
    spec = importlib.util.spec_from_file_location("score_confidence", STUDY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iso_apply(conf_tr, y_tr, conf_te):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(conf_tr, y_tr)
    return iso.predict(conf_te)


def grouped_iso_ece(sc, conf, y, groups, n_splits=5):
    k = min(n_splits, len(np.unique(groups)))
    if k < 2 or len(set(y.tolist())) < 2:
        return None, 0
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(conf, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        oof[te] = _iso_apply(conf[tr], y[tr], conf[te])
    m = ~np.isnan(oof)
    if m.sum() < 10:
        return None, int(m.sum())
    e, _ = sc.ece_from(oof[m], y[m])
    return float(e), int(m.sum())


def run() -> dict:
    sc = _study()
    narr = pd.read_json(NARR, lines=True, dtype={"report_id": str}).set_index("report_id")
    out: dict = {"study_code": STUDY, "min_n": MIN_N, "models": {}}
    for model, path in RUNS.items():
        df = sc.signal_frame(path, model, committed_only=True)
        keys = {**sc.correctness_distant(NARR, path, model),
                **sc.correctness_human(GOLD, path, model)}
        df["correct"] = [keys.get((r, f)) for r, f in zip(df["report_id"], df["field"])]
        df = df[df["correct"].notna()].dropna(subset=["p_renorm"])
        df["make"] = narr.loc[df["report_id"], "manufacturer"].fillna("unknown") \
            .str.upper().to_numpy()
        df["source"] = narr.loc[df["report_id"], "source"].to_numpy()
        res = {}
        for field, g in df.groupby("field"):
            if len(g) < MIN_N:
                continue
            conf = g["p_renorm"].to_numpy(float)
            y = g["correct"].to_numpy(int)
            rec = {"n": int(len(g))}
            # The original, record-level out-of-fold figure, recomputed here with
            # the study's own function so the comparison is like for like.
            rec["ece_iso_record_oof"] = sc.isotonic_ece(conf, y)
            rec["ece_raw"], _ = sc.ece_from(conf, y)
            rec["ece_iso_heldout_make"], rec["n_heldout_make"] = \
                grouped_iso_ece(sc, conf, y, g["make"].to_numpy())
            rec["n_makes"] = int(g["make"].nunique())
            tr = (g["source"] == "ol316").to_numpy()
            te = (g["source"] == "sgo").to_numpy()
            if tr.sum() >= MIN_SIDE and te.sum() >= MIN_SIDE and len(set(y[tr])) == 2:
                pred = _iso_apply(conf[tr], y[tr], conf[te])
                rec["ece_iso_heldout_corpus"], _ = sc.ece_from(pred, y[te])
                rec["ece_raw_sgo"], _ = sc.ece_from(conf[te], y[te])
                rec["n_fit_ol316"], rec["n_eval_sgo"] = int(tr.sum()), int(te.sum())
            else:
                rec["ece_iso_heldout_corpus"] = None
                rec["n_fit_ol316"], rec["n_eval_sgo"] = int(tr.sum()), int(te.sum())
            res[field] = rec
        out["models"][model] = res
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    for model, res in out["models"].items():
        print(f"\n{model}")
        for field, r in res.items():
            hc = r["ece_iso_heldout_corpus"]
            print(f"  {field:28s} n={r['n']:5d} raw={r['ece_raw']:.3f} "
                  f"iso(record)={r['ece_iso_record_oof']:.3f} "
                  f"iso(held-out make, {r['n_makes']} makes)="
                  f"{r['ece_iso_heldout_make'] if r['ece_iso_heldout_make'] is None else round(r['ece_iso_heldout_make'], 3)} "
                  f"iso(OL316->SGO)={'--' if hc is None else round(hc, 3)} "
                  f"[fit {r['n_fit_ol316']}, eval {r['n_eval_sgo']}]")
    return out


if __name__ == "__main__":
    run()
