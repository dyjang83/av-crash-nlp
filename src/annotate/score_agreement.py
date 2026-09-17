"""Agreement and extraction-accuracy scoring.

Computes:
  1. Inter-annotator agreement per field                   -> Table 2 (label ceiling)
     Reported as percent agreement, Cohen's kappa, Gwet's AC1 and the
     majority-class share together, because kappa alone is not interpretable on
     the prevalence-skewed fields in this schema (see gwet_ac1).
  2. Model-vs-gold agreement: per-field macro-F1 and kappa -> Table 3 (extraction)
  3. Confidence calibration (reliability curve + ECE)       -> Figure 2

All inputs are real annotation/extraction files; nothing is simulated here.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.schema import SCORED_FIELDS, CATEGORICAL_FIELDS  # noqa: E402

GOLD_DIR = os.path.join("data", "gold")
TABLES = os.path.join("paper", "tables")


def _load_jsonl(path: str) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame, indexed by report_id.

    Deliberately does NOT use pd.read_json: it silently coerces any column
    that mixes JSON null with booleans (other_party_present, av_moving) into
    float64, turning True/False into 1.0/0.0 and null into NaN rather than
    None. That corrupts both the agreement mask below (None-checks expect the
    string "None", which NaN's str() does not produce) and the boolean schema
    itself. Parsing lines manually and building the DataFrame from the
    resulting list of dicts preserves the original Python types exactly.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows).set_index("report_id")


def gwet_ac1(ya, yb) -> float:
    """Gwet's AC1: chance-corrected agreement that is stable under skew.

    Cohen's kappa estimates chance agreement from the observed marginals, which
    makes it behave badly exactly where these fields live. On a field that is
    97.6% one value, two coders who both apply the same "not stated => unknown"
    convention agree trivially, and kappa reports either ~1.0 (if their vectors
    are identical) or collapses toward 0 (if they differ on a handful of rare
    cases) -- neither of which describes how much genuine judgment was shared.
    AC1 estimates chance agreement from the propensity for random rating
    instead, and does not degrade this way.

    Reported alongside kappa and the majority-class share so a reader can see
    when a high agreement figure is carried by prevalence rather than by
    concordant coding.
    """
    ya, yb = list(ya), list(yb)
    n = len(ya)
    if n == 0:
        return np.nan
    cats = sorted(set(ya) | set(yb))
    if len(cats) < 2:
        return np.nan
    pa = np.mean([x == y for x, y in zip(ya, yb)])
    pi = [((ya.count(c) + yb.count(c)) / (2 * n)) for c in cats]
    pe = sum(p * (1 - p) for p in pi) / (len(cats) - 1)
    if np.isclose(pe, 1.0):
        return np.nan
    return float((pa - pe) / (1 - pe))


def interannotator(a_path: str, b_path: str) -> pd.DataFrame:
    a, b = _load_jsonl(a_path), _load_jsonl(b_path)
    idx = a.index.intersection(b.index)
    rows = []
    for field in SCORED_FIELDS:
        ya_raw, yb_raw = a.loc[idx, field], b.loc[idx, field]
        # Check missingness on the raw values via pd.isna(), which correctly
        # recognizes None/NaN/pd.NA regardless of which dtype backend pandas
        # chose for this column. Stringifying first and comparing against the
        # literal "None" is fragile: pandas' newer string-backed dtype (the
        # default since pandas 2.x/3.x) represents missing values with its own
        # NA sentinel that stringifies as "nan", not "None", which silently
        # left every missing value IN the comparison instead of excluding it.
        mask = ~ya_raw.isna() & ~yb_raw.isna()
        ya, yb = ya_raw[mask].astype(str), yb_raw[mask].astype(str)
        n = int(mask.sum())
        # A field on which both coders used a single value throughout carries no
        # agreement information. kappa is undefined there (0/0); reporting it as
        # 1.000 would put a field that was never really adjudicated at the top of
        # the table. Percent agreement and the majority share still describe it.
        n_classes = len(set(ya) | set(yb))
        kappa = (cohen_kappa_score(ya, yb)
                 if n > 1 and n_classes > 1 else np.nan)
        rows.append({
            "field": field, "n": n,
            "pct_agree": float(np.mean(ya.values == yb.values)) if n else np.nan,
            "kappa": kappa,
            "ac1": gwet_ac1(ya, yb) if n > 1 else np.nan,
            "majority_share": (float(pd.concat([ya, yb]).value_counts(
                normalize=True).iloc[0]) if n else np.nan),
            "n_classes": n_classes,
        })
    return pd.DataFrame(rows)


def model_vs_gold(gold_path: str, extractions_path: str,
                  model: Optional[str] = None) -> pd.DataFrame:
    gold = _load_jsonl(gold_path)
    ext = pd.read_json(extractions_path, lines=True)
    ext = ext[ext["ok"]].copy()
    if model:
        ext = ext[ext["model"] == model]
    flat = pd.json_normalize(ext["extraction"])
    flat["report_id"] = ext["report_id"].values
    flat = flat.set_index("report_id")
    idx = gold.index.intersection(flat.index)

    rows = []
    for field in SCORED_FIELDS:
        yg_raw = gold.loc[idx, field]
        yp_raw = flat.loc[idx, field] if field in flat.columns else pd.Series(
            index=idx, dtype=object)
        mask = ~yg_raw.isna() & ~yp_raw.isna()
        yg, yp = yg_raw[mask].astype(str), yp_raw[mask].astype(str)
        n = int(mask.sum())
        if n <= 1:
            rows.append({"field": field, "n": n, "macro_f1": np.nan,
                         "kappa": np.nan})
            continue
        macro = f1_score(yg, yp, average="macro",
                         labels=sorted(set(yg) | set(yp)),
                         zero_division=0)
        kappa = cohen_kappa_score(yg, yp)
        rows.append({"field": field, "n": n, "macro_f1": macro, "kappa": kappa})
    out = pd.DataFrame(rows)
    out["model"] = model or "all"
    return out


def ece_from(conf, correct, n_bins: int = 10) -> tuple[float, list]:
    """Expected calibration error and reliability curve for one signal.

    Equal-width binning over [0,1]; each bin contributes |accuracy - mean
    confidence| weighted by its share of the sample. Empty bins are skipped, so
    the weights sum to 1 over occupied bins.

    Factored out of calibration() so the token-probability signal
    (annotate/score_confidence.py) is scored by the identical arithmetic that
    produced the self-reported-confidence figures already in the paper -- a
    second implementation would make the two numbers incomparable.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=int)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, curve = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum() == 0:
            continue
        acc, avg_conf, w = correct[m].mean(), conf[m].mean(), m.mean()
        ece += w * abs(acc - avg_conf)
        curve.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                      "acc": float(acc), "conf": float(avg_conf)})
    return float(ece), curve


def calibration(extractions_path: str, gold_path: str, model: str,
                n_bins: int = 10) -> dict:
    """Expected calibration error of the model's self-reported confidence.

    'Correct' = exact match on the contributory_party field against gold
    (the hardest field), a conservative proxy for extraction correctness.
    """
    gold = _load_jsonl(gold_path)
    ext = pd.read_json(extractions_path, lines=True)
    ext = ext[(ext["ok"]) & (ext["model"] == model)].copy()
    if len(ext) == 0:
        print(f"[calibration] no extractions found for model={model!r} in "
              f"{extractions_path} -- skipping (did you run llm_extract.py "
              f"with --model {model}?)")
        return {"model": model, "ece": None, "curve": [], "n": 0}

    flat = pd.json_normalize(ext["extraction"])
    flat["report_id"] = ext["report_id"].values
    flat = flat.set_index("report_id")
    idx = gold.index.intersection(flat.index)

    if "contributory_party" not in flat.columns or "confidence" not in flat.columns:
        print(f"[calibration] extractions for model={model!r} are missing "
              f"expected fields -- skipping")
        return {"model": model, "ece": None, "curve": [], "n": 0}

    gold_cp = gold.loc[idx, "contributory_party"]
    flat_cp = flat.loc[idx, "contributory_party"]
    valid = (~gold_cp.isna() & ~flat_cp.isna()).to_numpy()
    idx = idx[valid]
    if len(idx) == 0:
        print(f"[calibration] no overlapping gold/extraction rows for "
              f"model={model!r} -- skipping")
        return {"model": model, "ece": None, "curve": [], "n": 0}

    conf = flat.loc[idx, "confidence"].astype(float).values
    correct = (flat.loc[idx, "contributory_party"].astype(str).values ==
               gold.loc[idx, "contributory_party"].astype(str).values).astype(int)

    ece, curve = ece_from(conf, correct, n_bins=n_bins)
    return {"model": model, "ece": ece, "curve": curve, "n": int(len(idx))}


def _to_latex(df: pd.DataFrame, path: str, caption: str, label: str,
              float_fmt: str = "%.3f"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: float_fmt % x,
                            na_rep="--", caption=caption, label=label,
                            longtable=False, escape=True))
    print(f"[score] wrote {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join(GOLD_DIR, "gold.jsonl"))
    ap.add_argument("--ann-a", default=os.path.join(GOLD_DIR, "annotator_A.jsonl"))
    ap.add_argument("--ann-b", default=os.path.join(GOLD_DIR, "annotator_B.jsonl"))
    ap.add_argument("--extractions",
                    default=os.path.join("data", "processed", "extractions.jsonl"))
    ap.add_argument("--models", nargs="+", default=None,
                    help="Models to score. Defaults to every model actually "
                    "present in --extractions, so it never fails on a model "
                    "you haven't run yet.")
    args = ap.parse_args()

    if os.path.exists(args.ann_a) and os.path.exists(args.ann_b):
        iaa = interannotator(args.ann_a, args.ann_b)
        iaa_table = iaa.drop(columns=["n", "n_classes"])
        _to_latex(iaa_table, os.path.join(TABLES, "tab_iaa.tex"),
                  "Inter-annotator agreement per field.",
                  "tab:iaa")

    models = args.models
    if models is None and os.path.exists(args.extractions):
        ext_all = pd.read_json(args.extractions, lines=True)
        models = sorted(ext_all.loc[ext_all["ok"], "model"].unique().tolist())
        print(f"[score] --models not given; found {len(models)} model(s) in "
              f"{args.extractions}: {models}")
    models = models or []

    all_rows = []
    cal_results = {}
    for m in models:
        if os.path.exists(args.extractions) and os.path.exists(args.gold):
            all_rows.append(model_vs_gold(args.gold, args.extractions, model=m))
            cal = calibration(args.extractions, args.gold, m)
            cal_results[m] = cal
            print(json.dumps(cal, indent=2))

    if cal_results:
        cal_path = os.path.join("data", "processed", "calibration.json")
        with open(cal_path, "w") as f:
            json.dump(cal_results, f, indent=2)
        print(f"[score] wrote {cal_path}")

    if all_rows:
        res = pd.concat(all_rows, ignore_index=True)
        res_table = res.drop(columns=["n"])
        _to_latex(res_table, os.path.join(TABLES, "tab_extraction.tex"),
                  "Model-vs-gold extraction accuracy (macro-F1 and $\\kappa$).",
                  "tab:extraction")