"""Cross-source transfer evaluation.

External-validity check: the extractor is developed and gold-validated primarily
on the richer CA DMV OL 316 narratives. This script measures how extraction
accuracy degrades when the same extractor is applied to the shorter, partially
redacted NHTSA SGO narratives, by computing per-source macro-F1 / kappa against
a source-stratified gold set.

Reports Table 4: extraction macro-F1 by field x source, and the average
degradation (DMV -> SGO).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotate.score_agreement import model_vs_gold  # noqa: E402

GOLD_DIR = os.path.join("data", "gold")
TABLES = os.path.join("paper", "tables")


def _gold_by_source(gold_path: str):
    g = pd.read_json(gold_path, lines=True)
    for src in ["ol316", "sgo"]:
        sub = g[g["source"] == src]
        if len(sub):
            p = os.path.join(GOLD_DIR, f"gold_{src}.jsonl")
            sub.to_json(p, orient="records", lines=True)
            yield src, p


def run(gold_path: str, extractions_path: str, model: str) -> dict:
    per_source = {}
    for src, gp in _gold_by_source(gold_path):
        res = model_vs_gold(gp, extractions_path, model=model)
        per_source[src] = res.set_index("field")["macro_f1"]

    if "ol316" in per_source and "sgo" in per_source:
        comp = pd.DataFrame(per_source)
        comp["degradation"] = comp["ol316"] - comp["sgo"]
        out = comp.reset_index()
        os.makedirs(TABLES, exist_ok=True)
        with open(os.path.join(TABLES, "tab_transfer.tex"), "w") as f:
            f.write(out.to_latex(index=False, na_rep="--",
                    float_format=lambda x: "%.3f" % x,
                    caption=("Extraction macro-F1 by source and degradation "
                             "(DMV OL\\,316 $\\rightarrow$ NHTSA SGO)."),
                    label="tab:transfer", escape=True))
        summary = {"model": model,
                   "mean_degradation": float(out["degradation"].mean()),
                   "by_field": out.to_dict(orient="records")}
        stats_path = os.path.join("data", "processed", "transfer_eval.json")
        with open(stats_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[transfer] wrote {stats_path}")
        print(json.dumps({"mean_degradation": summary["mean_degradation"]}, indent=2))
        return summary
    print("[transfer] need gold rows from BOTH ol316 and sgo to report transfer.")
    return {}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join(GOLD_DIR, "gold.jsonl"))
    ap.add_argument("--extractions",
                    default=os.path.join("data", "processed", "extractions.jsonl"))
    ap.add_argument("--model", default="claude-sonnet-4-6")
    a = ap.parse_args()
    run(a.gold, a.extractions, a.model)
