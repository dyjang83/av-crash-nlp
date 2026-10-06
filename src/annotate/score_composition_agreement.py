"""Score the composition double-coding: coder agreement and model accuracy.

Three questions, kept apart because they answer different things:

  1. CODER AGREEMENT (pre-adjudication). How reproducible is each field when two
     people apply the same guide? Percent agreement, Cohen's kappa, Gwet's AC1
     and the majority share, per field -- kappa alone misleads on the skewed
     fields (see `score_agreement.gwet_ac1`). These are the numbers the paper
     reports; agreement AFTER adjudication is 100% by construction.
  2. MODEL VS GOLD. Against the adjudicated labels: accuracy per field, and for
     every class precision, recall and F1 with bootstrap intervals, plus the
     crash-type confusion matrix. Per-class numbers are the point -- the
     composition findings rest on two classes, and an overall accuracy can hide
     either.
  3. DEDUP AUDIT. Coder agreement on the pair verdicts and, against the gold
     verdicts, how often the pipeline's merge / review / reject bands were right.

POPULATIONS AND WEIGHTS. Two populations are scored separately:

  window   the census frame (batches 1 and 2). If a batch was stopped early,
           the coded rows of that batch are a random prefix and get weight
           batch size / rows coded, so batch 1 (the candidate pool, where the
           rare classes concentrate) is not over-represented.
  corpus   the original stratified 300, wherever they were coded (batch 3, or
           the census if they fall in the window), weighted by
           `corpus_sample_weight` to the current corpus.

Calibration rows (batch 0) never enter a statistic.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotate.score_agreement import gwet_ac1  # noqa: E402
from annotate.adjudicate import COMPOSITION_FIELDS  # noqa: E402
from annotate.census_csv import read_rows  # noqa: E402

GOLD_DIR = os.path.join("data", "gold")
META = os.path.join(GOLD_DIR, "census_meta.json")
EXTRACTIONS = os.path.join("data", "processed", "composition_extractions.jsonl")
DEDUP_SAMPLE = os.path.join(GOLD_DIR, "dedup_audit_sample_v2.jsonl")
OUT = os.path.join("data", "processed", "composition_agreement.json")
CALIBRATION_BATCH = 0


def _norm(v) -> Optional[str]:
    """One comparable string per label, or None when uncoded."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip().lower()
    return s or None


def load_rows(path: str) -> dict[str, dict]:
    """A coder sheet (.csv) or gold file (.jsonl), calibration rows dropped."""
    return {k: r for k, r in read_rows(path).items()
            if r.get("batch") != CALIBRATION_BATCH}


def load_model(path: str, model: Optional[str] = None) -> dict[str, dict]:
    """report_id -> extraction, one model only (see reportability.attach_extractions)."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok") or r.get("extraction") is None:
                continue
            if model and r.get("model") != model:
                continue
            out[str(r["report_id"])] = r["extraction"]
    return out


def is_coded(row: dict, fields: list[str]) -> bool:
    own = [f for f in fields if f in (row.get("fields") or fields)]
    return all(_norm(row.get(f)) is not None for f in own)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def window_weights(meta: dict, coded_ids: set[str]) -> dict[str, float]:
    """Batch size / rows coded, per census batch (1 and 2)."""
    w = {}
    for b in ("1", "2"):
        order = meta["batch_order"][b]
        done = [i for i in order if i in coded_ids]
        if done:
            w.update({i: len(order) / len(done) for i in done})
    return w


def corpus_weights(meta: dict, coded_ids: set[str]) -> dict[str, float]:
    cw = meta.get("corpus_sample_weight", {})
    return {i: float(cw[i]) for i in meta.get("corpus_sample_ids", [])
            if i in coded_ids and i in cw}


# ---------------------------------------------------------------------------
# 1. Coder agreement
# ---------------------------------------------------------------------------
def interannotator(a: dict, b: dict, fields: list[str],
                   weights: Optional[dict] = None) -> list[dict]:
    """Per-field agreement on rows both coders coded.

    Percent agreement is weight-adjusted when `weights` is given; kappa and AC1
    are computed on the rows as coded, because neither has a standard weighted
    form -- the weighted percent agreement is the population-level figure.
    """
    ids = sorted(set(a) & set(b) & (set(weights) if weights is not None else set(a)))
    out = []
    for f in fields:
        pairs = [(i, _norm(a[i].get(f)), _norm(b[i].get(f))) for i in ids
                 if f in (a[i].get("fields") or [f])]
        pairs = [(i, x, y) for i, x, y in pairs if x is not None and y is not None]
        n = len(pairs)
        if not n:
            out.append({"field": f, "n": 0})
            continue
        ya, yb = [p[1] for p in pairs], [p[2] for p in pairs]
        agree = np.array([x == y for x, y in zip(ya, yb)], float)
        w = np.array([weights[p[0]] if weights else 1.0 for p in pairs])
        n_classes = len(set(ya) | set(yb))
        out.append({
            "field": f, "n": n,
            "pct_agree": float(agree.mean()),
            "pct_agree_weighted": float((agree * w).sum() / w.sum()),
            "kappa": (float(cohen_kappa_score(ya, yb))
                      if n > 1 and n_classes > 1 else None),
            "ac1": float(gwet_ac1(ya, yb)) if n > 1 else None,
            "majority_share": float(pd.Series(ya + yb).value_counts(normalize=True).iloc[0]),
            "n_classes": n_classes,
        })
    return out


# ---------------------------------------------------------------------------
# 2. Model vs gold
# ---------------------------------------------------------------------------
def _prf(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray, c: str) -> tuple:
    tp = w[(y_true == c) & (y_pred == c)].sum()
    npred = w[y_pred == c].sum()
    ntrue = w[y_true == c].sum()
    p = tp / npred if npred > 0 else np.nan
    r = tp / ntrue if ntrue > 0 else np.nan
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else np.nan
    return p, r, f1


def per_class(y_true: list, y_pred: list, w: list, n_boot: int = 1000,
              seed: int = 11) -> dict:
    """Weighted precision / recall / F1 per class, percentile bootstrap CIs.

    The bootstrap resamples rows (with their weights), which is the right unit:
    every row is a distinct incident.
    """
    yt, yp, ww = np.array(y_true, object), np.array(y_pred, object), np.array(w, float)
    classes = sorted(set(yt) | set(yp))
    rng = np.random.default_rng(seed)
    boots = {c: [] for c in classes}
    for _ in range(n_boot):
        ix = rng.integers(0, len(yt), len(yt))
        for c in classes:
            boots[c].append(_prf(yt[ix], yp[ix], ww[ix], c))
    out = {}
    for c in classes:
        p, r, f1 = _prf(yt, yp, ww, c)
        b = np.array(boots[c], float)
        lo = np.nanpercentile(b, 2.5, axis=0) if len(b) else [np.nan] * 3
        hi = np.nanpercentile(b, 97.5, axis=0) if len(b) else [np.nan] * 3
        out[c] = {"n_gold": int((yt == c).sum()), "n_pred": int((yp == c).sum()),
                  "precision": _f(p), "precision_ci": [_f(lo[0]), _f(hi[0])],
                  "recall": _f(r), "recall_ci": [_f(lo[1]), _f(hi[1])],
                  "f1": _f(f1)}
    return out


def _f(x) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def model_vs_gold(gold: dict, model: dict, fields: list[str], weights: dict,
                  n_boot: int = 1000) -> dict:
    out = {}
    for f in fields:
        rows = [(i, _norm(gold[i].get(f)), _norm((model.get(i) or {}).get(f)))
                for i in weights if i in gold]
        rows = [(i, t, p) for i, t, p in rows if t is not None]
        if not rows:
            continue
        # A missing model label (no extraction) is scored as wrong, not dropped:
        # dropping it would flatter the model on exactly the hard records.
        yt = [t for _, t, _ in rows]
        yp = [p if p is not None else "__none__" for _, _, p in rows]
        w = [weights[i] for i, _, _ in rows]
        ww = np.array(w)
        acc = float((ww * (np.array(yt) == np.array(yp))).sum() / ww.sum())
        res = {"n": len(rows), "accuracy_weighted": acc,
               "per_class": per_class(yt, yp, w, n_boot=n_boot)}
        if f == "acc_type_category":
            res["confusion"] = (pd.crosstab(pd.Series(yt, name="gold"),
                                            pd.Series(yp, name="model"))
                                .to_dict(orient="index"))
        out[f] = res
    return out


# ---------------------------------------------------------------------------
# 3. Dedup audit
# ---------------------------------------------------------------------------
def dedup_scores(a: dict, b: dict, gold: Optional[dict], sample_path: str) -> dict:
    res = {"coder_agreement": interannotator(a, b, ["verdict"])}
    if gold and os.path.exists(sample_path):
        with open(sample_path) as f:
            decided = {f"{r['report_i']}|{r['report_j']}": r["decision"]
                       for r in map(json.loads, f)}
        tab = {}
        for pid, g in gold.items():
            band = decided.get(pid)
            v = _norm(g.get("verdict"))
            if band and v:
                tab.setdefault(band, {}).setdefault(v, 0)
                tab[band][v] += 1
        res["pipeline_band_vs_gold"] = tab
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default=os.path.join(GOLD_DIR, "census_coder_A.csv"))
    ap.add_argument("--b", default=os.path.join(GOLD_DIR, "census_coder_B.csv"))
    ap.add_argument("--gold", default=os.path.join(GOLD_DIR, "composition_gold.jsonl"),
                    help="Adjudicated labels; model-vs-gold is skipped if absent.")
    ap.add_argument("--meta", default=META)
    ap.add_argument("--extractions", default=EXTRACTIONS)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dedup-a", default=os.path.join(GOLD_DIR, "dedup_coder_A.csv"))
    ap.add_argument("--dedup-b", default=os.path.join(GOLD_DIR, "dedup_coder_B.csv"))
    ap.add_argument("--dedup-gold", default=os.path.join(GOLD_DIR, "dedup_gold.jsonl"))
    ap.add_argument("--dedup-sample", default=DEDUP_SAMPLE)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=OUT)
    x = ap.parse_args()

    meta = json.load(open(x.meta))
    fields = COMPOSITION_FIELDS
    rep = {"fields": fields, "calibration_excluded": True}

    if os.path.exists(x.a) and os.path.exists(x.b):
        a, b = load_rows(x.a), load_rows(x.b)
        both = {i for i in set(a) & set(b) if is_coded(a[i], fields) and is_coded(b[i], fields)}
        ww, cw = window_weights(meta, both), corpus_weights(meta, both)
        rep["coverage"] = {"window_coded_by_both": len(ww),
                           "window_frame": meta["n_frame"],
                           "corpus_sample_coded_by_both": len(cw)}
        rep["coder_agreement"] = {"window": interannotator(a, b, fields, ww),
                                  "corpus": interannotator(a, b, fields, cw)}
        print(f"[agree] coded by both: window {len(ww)}/{meta['n_frame']}, "
              f"corpus sample {len(cw)}/{len(meta.get('corpus_sample_ids', []))}")
        for r in rep["coder_agreement"]["window"]:
            if r["n"]:
                k = "-" if r["kappa"] is None else f"{r['kappa']:.3f}"
                ac = "-" if r["ac1"] is None else f"{r['ac1']:.3f}"
                print(f"  {r['field']:24s} n={r['n']:4d}  agree {r['pct_agree']:.3f}  "
                      f"kappa {k}  AC1 {ac}")
    else:
        print(f"[agree] coder files not found ({x.a}, {x.b}); skipping agreement")

    if os.path.exists(x.gold):
        gold = load_rows(x.gold)
        model = load_model(x.extractions, x.model)
        coded = set(gold)
        rep["model_vs_gold"] = {
            "window": model_vs_gold(gold, model, fields, window_weights(meta, coded), x.n_boot),
            "corpus": model_vs_gold(gold, model, fields, corpus_weights(meta, coded), x.n_boot),
        }
        mv = rep["model_vs_gold"]["window"].get("acc_type_category")
        if mv:
            print(f"[gold] window crash type: weighted accuracy {mv['accuracy_weighted']:.3f} (n={mv['n']})")
            for c in ("turn_across_path", "single_vehicle"):
                pc = mv["per_class"].get(c)
                if pc:
                    print(f"  {c:18s} P={pc['precision']} R={pc['recall']} "
                          f"(gold n={pc['n_gold']}, model n={pc['n_pred']})")
    else:
        print(f"[gold] {x.gold} not found; skipping model-vs-gold")

    if os.path.exists(x.dedup_a) and os.path.exists(x.dedup_b):
        dg = load_rows(x.dedup_gold) if os.path.exists(x.dedup_gold) else None
        rep["dedup"] = dedup_scores(load_rows(x.dedup_a), load_rows(x.dedup_b),
                                    dg, x.dedup_sample)

    os.makedirs(os.path.dirname(x.out), exist_ok=True)
    with open(x.out, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"[agree] wrote {x.out}")


if __name__ == "__main__":
    main()
