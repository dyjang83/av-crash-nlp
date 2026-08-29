"""Distant-supervision extraction evaluation.

Scores model extractions against the reporter-filed structured codes carried in
data/interim/narratives.jsonl, using the per-field reducers in
schema/distant_map.py to project both sides into the coarsest space the
structured key can express.

Outputs:
  data/processed/distant_eval.json    per-field metrics + coverage, all models
  paper/tables/tab_distant.tex        Table: extraction accuracy at corpus scale
  paper/tables/tab_distant_cov.tex    Table: coverage and coarsening per field

Usage:
    python -m annotate.score_distant
    python -m annotate.score_distant --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.distant_map import (  # noqa: E402
    COARSENED, HUMAN_ONLY_FIELDS, REDUCERS,
)

NARRATIVES = os.path.join("data", "interim", "narratives.jsonl")
EXTRACTIONS = os.path.join("data", "processed", "extractions.jsonl")
GOLD = os.path.join("data", "gold", "gold.jsonl")
OUT_JSON = os.path.join("data", "processed", "distant_eval.json")
TABLES = os.path.join("paper", "tables")


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_extractions(path: str) -> pd.DataFrame:
    """Flatten extractions.jsonl to one row per (report_id, model).

    Parsed line-by-line rather than with pd.read_json for the same reason as
    elsewhere in this codebase: read_json coerces boolean columns that contain
    any null into float64, turning True/False into 1.0/0.0, which then compares
    unequal against the string labels the reducers emit.
    """
    rows = []
    for r in _load_jsonl(path):
        if not r.get("ok"):
            continue
        ext = r.get("extraction") or {}
        rows.append({"report_id": r["report_id"], "model": r.get("model"), **ext})
    return pd.DataFrame(rows)


def _bootstrap_ci(y_true: list, y_pred: list, stat, n_boot: int = 1000,
                  seed: int = 11) -> tuple:
    """Percentile bootstrap CI for an agreement statistic.

    At corpus scale the sampling error on these estimates is small, but it is
    not zero and it varies by two orders of magnitude across fields (n=214 for
    weather vs n=3198 for maneuver). Reporting the interval keeps a
    high-coverage field and a thin one from being read as equally firm.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    yt, yp = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(stat(yt[b], yp[b]))
        except Exception:  # noqa: BLE001 - degenerate resample (single class)
            continue
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def evaluate(reducers: dict, narratives: list[dict], ext: pd.DataFrame,
             model: str, n_boot: int = 1000) -> list[dict]:
    by_id = {r["report_id"]: r for r in narratives}
    sub = ext[ext["model"] == model]
    rows = []
    for name, (field, reduce_fn) in reducers.items():
        gold_vals, pred_vals = [], []
        n_candidate = 0
        for _, e in sub.iterrows():
            row = by_id.get(e["report_id"])
            if row is None:
                continue
            n_candidate += 1
            out = reduce_fn(row, e.get(field))
            if out is None:
                continue
            gold_vals.append(out[0])
            pred_vals.append(out[1])

        n = len(gold_vals)
        if n <= 1 or len(set(gold_vals)) < 1:
            rows.append({"field": name, "model": model, "n": n,
                         "coverage": n / n_candidate if n_candidate else np.nan,
                         "macro_f1": np.nan, "kappa": np.nan,
                         "accuracy": np.nan, "kappa_lo": np.nan,
                         "kappa_hi": np.nan, "abstention_rate": np.nan,
                         "n_committed": 0, "accuracy_committed": np.nan,
                         "kappa_committed": np.nan,
                         "n_classes": len(set(gold_vals))})
            continue

        labels = sorted(set(gold_vals) | set(pred_vals))
        macro = f1_score(gold_vals, pred_vals, average="macro", labels=labels,
                         zero_division=0)
        acc = float(np.mean([g == p for g, p in zip(gold_vals, pred_vals)]))

        # Disagreement with the distant key has two very different causes, and
        # collapsing them understates the extractor while overstating what is
        # fixable. The reporting entity coded the CONDITION; the narrative often
        # does not state it. Where the model answers `unknown` it has not made
        # an error of judgment -- it has correctly reported that the text is
        # silent. Where it commits to a wrong value it has.
        #
        # `abstention_rate` is the share of keyed records the model declined.
        # `accuracy_committed` is accuracy on the remainder: the quantity that
        # actually measures extraction quality, and the one a downstream user
        # who filters on non-null extractions would experience.
        committed = [(g, p) for g, p in zip(gold_vals, pred_vals)
                     if p not in {"unknown", "non_dark"}]
        n_abstain = n - len(committed)
        if committed:
            cg, cp = zip(*committed)
            acc_c = float(np.mean([g == p for g, p in committed]))
            kappa_c = (cohen_kappa_score(list(cg), list(cp))
                       if len(set(cg)) > 1 else np.nan)
        else:
            acc_c, kappa_c = np.nan, np.nan
        # Cohen's kappa is undefined when the key is single-class; accuracy
        # still is. Report NaN rather than a spurious 1.0 or 0.0.
        if len(set(gold_vals)) < 2:
            kappa, lo, hi = np.nan, np.nan, np.nan
        else:
            kappa = cohen_kappa_score(gold_vals, pred_vals)
            lo, hi = _bootstrap_ci(gold_vals, pred_vals, cohen_kappa_score,
                                   n_boot=n_boot)
        rows.append({
            "field": name, "model": model, "n": n,
            "coverage": n / n_candidate if n_candidate else np.nan,
            "macro_f1": float(macro), "kappa": kappa, "accuracy": acc,
            "kappa_lo": lo, "kappa_hi": hi,
            "abstention_rate": n_abstain / n,
            "n_committed": len(committed),
            "accuracy_committed": acc_c, "kappa_committed": kappa_c,
            "n_classes": len(set(gold_vals)),
            "majority_share": float(pd.Series(gold_vals).value_counts(
                normalize=True).iloc[0]),
        })
    return rows


def human_vs_distant(gold_path: str, narratives: list[dict]) -> list[dict]:
    """Agreement between the human gold set and the distant key on the overlap.

    This is the validity check for the whole approach. If the reporter-filed
    codes and the human annotators agree on the 245 hand-coded records, the
    distant key is measuring the same construct as the human key and can stand
    in for it at scale. If they do not, the distant key is measuring something
    else and the paper must say so rather than quietly substituting one for the
    other.
    """
    by_id = {r["report_id"]: r for r in narratives}
    gold = _load_jsonl(gold_path)
    rows = []
    for name, (field, reduce_fn) in REDUCERS.items():
        gv, hv = [], []
        for g in gold:
            row = by_id.get(g["report_id"])
            if row is None:
                continue
            out = reduce_fn(row, g.get(field))
            if out is None:
                continue
            gv.append(out[0])
            hv.append(out[1])
        n = len(gv)
        # Cohen's kappa is only meaningful if BOTH sides vary. If one side is
        # constant (e.g. the human gold set contains no positive VRU cases in
        # this overlap), po reduces algebraically to pe and kappa is pinned to
        # exactly 0.0 regardless of the other side's accuracy -- a degenerate
        # value, not a real disagreement signal. Report it as undefined, same
        # as the fully-degenerate (both sides constant) case.
        if n <= 1 or len(set(gv)) < 2 or len(set(hv)) < 2:
            rows.append({"field": name, "n": n, "kappa": np.nan,
                         "accuracy": float(np.mean([a == b for a, b in zip(gv, hv)]))
                         if n else np.nan})
            continue
        rows.append({"field": name, "n": n,
                     "kappa": cohen_kappa_score(gv, hv),
                     "accuracy": float(np.mean([a == b for a, b in zip(gv, hv)]))})
    return rows


# ---------------------------------------------------------------------------
# Table emission
# ---------------------------------------------------------------------------

def _tex(s: str) -> str:
    """Escape LaTeX-special characters in generated table cells.

    Field names, model names and the coarsening notes all contain underscores;
    an unescaped one is a hard compile error ("Missing $ inserted"), and because
    these strings are generated rather than typed, the error surfaces only at
    build time.
    """
    out = str(s)
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    return out.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def _fmt_ci(lo, hi, nd=3):
    if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)) \
            or (isinstance(hi, float) and np.isnan(hi)):
        return "--"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def write_distant_table(df: pd.DataFrame, path: str) -> None:
    lines = [r"\begin{table}", r"\centering", r"\small",
             r"\caption{Extraction accuracy against the distant-supervision key "
             r"(reporter-filed structured codes), with 95\% bootstrap "
             r"confidence intervals on $\kappa$. Fields marked "
             r"* are evaluated in a coarsened label space "
             r"(Table~5). \emph{abst.} is the share of "
             r"keyed records on which the extractor answered "
             r"\texttt{unknown}; \emph{acc.\textsubscript{com}} is accuracy "
             r"on the remaining, committed records (Section~5).}",
             r"\label{tab:distant}",
             r"\begin{tabular}{llrlrr}", r"\toprule",
             r"field & model & $\kappa$ & 95\% CI & abst. & "
             r"acc.$_{\text{com}}$ \\",
             r"\midrule"]
    for _, r in df.iterrows():
        dag = "*" if r["field"] in COARSENED else ""
        field = _tex(r["field"]) + dag
        lines.append(f"{field} & {_tex(r['model'])} & "
                     f"{_fmt(r['kappa'])} & "
                     f"{_fmt_ci(r.get('kappa_lo'), r.get('kappa_hi'))} & "
                     f"{_fmt(r.get('abstention_rate'),2)} & "
                     f"{_fmt(r.get('accuracy_committed'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[distant] wrote {path}")


def write_coverage_table(df: pd.DataFrame, hvd: pd.DataFrame, path: str) -> None:
    hvd_map = {r["field"]: r for _, r in hvd.iterrows()}
    lines = [r"\begin{table}", r"\centering",
             r"\caption{Distant-supervision coverage, label-space coarsening, and "
             r"validity. \emph{Coverage} is the share of extracted records whose "
             r"structured code yields usable supervision for that field. "
             r"\emph{Human $\kappa$} is agreement between the adjudicated human "
             r"gold set and the distant key on the 245 overlapping records, in "
             r"the same coarsened space.}",
             r"\label{tab:distant_cov}",
             r"\begin{tabular}{lrrrp{4.6cm}}", r"\toprule",
             r"field & $n$ & cov. & human $\kappa$ & coarsening \\",
             r"\midrule"]
    for _, r in df.iterrows():
        h = hvd_map.get(r["field"], {})
        note = COARSENED.get(r["field"], "none (full schema granularity)")
        lines.append(f"{_tex(r['field'])} & {int(r['n'])} & "
                     f"{_fmt(r['coverage'],2)} & {_fmt(h.get('kappa'))} & "
                     f"{_tex(note)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[distant] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="score a single model (default: all models present)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    narratives = _load_jsonl(NARRATIVES)
    ext = _load_extractions(EXTRACTIONS)
    models = [args.model] if args.model else sorted(ext["model"].dropna().unique())

    all_rows = []
    for m in models:
        all_rows += evaluate(REDUCERS, narratives, ext, m, n_boot=args.n_boot)

    df = pd.DataFrame(all_rows)
    hvd = pd.DataFrame(human_vs_distant(GOLD, narratives))

    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    write_distant_table(df, os.path.join(TABLES, "tab_distant.tex"))
    primary = models[0] if len(models) == 1 else "claude-sonnet-4-6"
    cov = df[df["model"] == primary] if primary in set(df["model"]) else df
    write_coverage_table(cov, hvd, os.path.join(TABLES, "tab_distant_cov.tex"))

    with open(OUT_JSON, "w") as f:
        json.dump({"extraction": all_rows,
                   "human_vs_distant": hvd.to_dict("records"),
                   "human_only_fields": HUMAN_ONLY_FIELDS}, f, indent=2,
                  default=float)
    print(f"[distant] wrote {OUT_JSON}")

    print("\n--- distant-supervision extraction accuracy ---")
    print(df[["field", "model", "n", "accuracy", "kappa", "abstention_rate",
              "n_committed", "accuracy_committed"]].to_string(index=False))
    print("\n--- human gold vs. distant key (validity check, n<=245) ---")
    print(hvd.to_string(index=False))


if __name__ == "__main__":
    main()
