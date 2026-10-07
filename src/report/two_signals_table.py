"""Paper table and summary for the two-confidence-signals comparison.

Reads the scored results of the separate two-confidence-signals analysis
(confidence_eval_<model>-committed.json, produced by its score_confidence.py)
and writes paper/tables/tab_two_signals.tex plus a small JSON of the figures the
text quotes, so no number in the section is typed by hand.

The primary token signal is p_renorm for EVERY field -- the aggregation the
analysis declares as its headline. The analysis's own findings table instead
picked p_renorm or p_first per field after scoring against the answer key; that
selection is not reproduced here. p_first is reported beside it as a disclosed
sensitivity column, because on single-token booleans p_renorm saturates near 1.0
and its ranking is dominated by numerical noise.

Usage:
    python -m report.two_signals_table [--src ~/Desktop/two-confidence-signals/data]
"""
from __future__ import annotations

import argparse
import json
import os

SRC = os.path.expanduser(os.environ.get(
    "TWO_CONF_DATA", "~/Desktop/two-confidence-signals/data"))
TABLES = os.path.join("paper", "tables")
OUT_JSON = os.path.join("data", "processed", "two_signals_summary.json")
MODELS = {"qwen2-5-7b": "Qwen2.5-7B", "llama3-1-8b": "Llama-3.1-8B"}
MODEL_IDS = {"qwen2-5-7b": "qwen2.5:7b", "llama3-1-8b": "llama3.1:8b"}
# Held-out recalibration, from report.two_signals_holdout. Optional: without it
# the table falls back to the record-level out-of-fold figure only.
HOLDOUT = os.path.join("data", "processed", "two_signals_holdout.json")
MIN_N = 150   # the analysis's reporting threshold for committed decisions
BOOLEAN = {"av_moving", "other_party_present"}


def _load(src: str, slug: str) -> dict:
    with open(os.path.join(src, f"confidence_eval_{slug}-committed.json")) as f:
        return json.load(f)


def build(src: str = SRC) -> dict:
    summary: dict = {"source": src, "min_n": MIN_N, "models": {}}
    rows = []
    hold = {}
    if os.path.exists(HOLDOUT):
        with open(HOLDOUT) as fh:
            hold = json.load(fh)["models"]
    for slug, label in MODELS.items():
        d = _load(src, slug)
        by = {}
        for r in d["scored"]:
            by.setdefault(r["field"], {})[r["signal"]] = r
        fields = sorted((f for f, s in by.items() if s["self_conf"]["n"] >= MIN_N),
                        key=lambda f: -by[f]["self_conf"]["n"])
        wins = [f for f in fields
                if by[f]["p_renorm"]["auroc"] > by[f]["self_conf"]["auroc"]]
        wins_first = [f for f in fields
                      if by[f]["p_first"]["auroc"] > by[f]["self_conf"]["auroc"]]
        iso = [by[f]["p_renorm"]["ece_isotonic"] for f in fields]
        gap = {g["field"]: g for g in d["gap"]}
        summary["models"][label] = {
            "fields": fields,
            "n_fields": len(fields),
            "p_renorm_beats_self": wins,
            "p_renorm_loses_to_self": [f for f in fields if f not in wins],
            "p_first_beats_self": wins_first,
            # Absent from runs scored before the gap diagnostics were extended.
            "self_conf_distinct_values": sorted({gap[f]["self_conf_nunique"]
                                                 for f in fields
                                                 if gap.get(f, {}).get("self_conf_nunique")}),
            "spearman_range": [min(gap[f]["spearman"] for f in fields if f in gap),
                               max(gap[f]["spearman"] for f in fields if f in gap)],
            "p_renorm_ece_isotonic_range": [min(iso), max(iso)],
        }
        for f in fields:
            s, p, q = by[f]["self_conf"], by[f]["p_renorm"], by[f]["p_first"]
            h = hold.get(MODEL_IDS[slug], {}).get(f, {}).get("ece_iso_heldout_make")
            rows.append((label, f, s["n"], s["base_rate"], s["auroc"], p["auroc"],
                         q["auroc"], s["ece"], p["ece"], p["ece_isotonic"], h))

    total = sum(m["n_fields"] for m in summary["models"].values())
    summary["p_renorm_wins_total"] = sum(len(m["p_renorm_beats_self"])
                                         for m in summary["models"].values())
    summary["p_first_wins_total"] = sum(len(m["p_first_beats_self"])
                                        for m in summary["models"].values())
    summary["fields_total"] = total
    hm = [r[-1] for r in rows if r[-1] is not None]
    summary["ece_iso_heldout_make_range"] = [min(hm), max(hm)] if hm else None
    hc = [v["ece_iso_heldout_corpus"] for m in hold.values() for v in m.values()
          if v.get("ece_iso_heldout_corpus") is not None]
    summary["ece_iso_heldout_corpus_range"] = [min(hc), max(hc)] if hc else None

    esc = lambda s: s.replace("_", r"\_")
    L = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
         r" & & & & \multicolumn{3}{c}{AUROC} & \multicolumn{4}{c}{ECE} \\",
         r"\cmidrule(lr){5-7}\cmidrule(lr){8-11}",
         r" & & & & & & & & & \multicolumn{2}{c}{$p_{\mathrm{renorm}}$ recalibrated$^{\ddagger}$} \\",
         r"\cmidrule(lr){10-11}",
         r"Model & Field & $n$ & Acc. & Self & $p_{\mathrm{renorm}}$ & "
         r"$p_{\mathrm{first}}^{\dagger}$ & Self & $p_{\mathrm{renorm}}$ & "
         r"records & makes held out \\", r"\midrule"]
    prev = None
    for (label, f, n, base, a_s, a_p, a_q, e_s, e_p, e_i, e_h) in rows:
        if prev and label != prev:
            L.append(r"\midrule")
        prev = label
        a_p_str = f"\\textbf{{{a_p:.3f}}}" if a_p > a_s else f"{a_p:.3f}"
        fname = esc(f) + ("$^{b}$" if f in BOOLEAN else "")
        L.append(f"{label} & {fname} & {n:,} & {base:.3f} & {a_s:.3f} & {a_p_str} & "
                 f"{a_q:.3f} & {e_s:.3f} & {e_p:.3f} & {e_i:.3f} & "
                 f"{'--' if e_h is None else f'{e_h:.3f}'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "tab_two_signals.tex"), "w") as fh:
        fh.write("\n".join(L))
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    build(ap.parse_args().src)
