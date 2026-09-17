"""Two confidence signals, compared.

The extraction schema asks the model to write a `confidence` float into its own
output. That is a SELF-REPORT: one scalar for the whole record, produced by the
same forward pass that produced the answer. An open-weight model served locally
exposes a second, independent quantity -- the decode-time probability of each
field's value (extract/token_probs.py). The closed tool-calling backend does not
expose it, so this comparison is only possible on the open models.

The two are not expected to agree, and the question this module answers is not
"which is better" but "how much credit does the token signal earn". Those come
apart, so four things are measured rather than one:

  ECE          calibration -- is a stated 0.8 right 80% of the time? Comparable
               to the self-report figures already in the paper (identical
               binning, via score_agreement.ece_from).
  AUROC        DISCRIMINATION -- does the signal RANK correct extractions above
               wrong ones? A signal can be badly calibrated and still rank
               perfectly; recalibration fixes scale, nothing fixes ranking. This
               is the metric that decides whether the signal carries information.
  ECE_iso      ECE after isotonic recalibration, fit out-of-fold. Separates "the
               signal is uninformative" from "the signal is informative but on
               the wrong scale". Only the first is a reason to discount it.
  AURC         area under the risk-coverage curve: abstain on the least
               confident x%, measure accuracy on the rest. This is the
               operational form of the question, and it connects directly to the
               abstention-vs-committed-error decomposition in the evaluation
               protocol.

One asymmetry is structural and is reported, not smoothed over: the self-report
is a single number per RECORD, while the token signal is per FIELD. Pairing them
means comparing one scalar against each of ten field-level probabilities. That
the self-report has no field resolution at all is itself a finding about its
usefulness, not a nuisance in the comparison.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.distant_map import REDUCERS  # noqa: E402
from annotate.score_agreement import ece_from, _to_latex  # noqa: E402
from annotate.score_distant import _load_jsonl  # noqa: E402

RESULTS = os.path.join("data", "processed")
TABLES = os.path.join("paper", "tables")
FIGURES = os.path.join("paper", "figures")

# Aggregations available as the token-probability signal. p_renorm is the
# headline (a posterior over categories); the others are reported so the
# tokenization-dilution argument is visible rather than asserted.
SIGNALS = ["p_renorm", "p_first", "p_seq", "p_min"]


# ---------------------------------------------------------------------------
# Assemble the long signal frame
# ---------------------------------------------------------------------------
def signal_frame(extractions_path: str, model: str,
                 drop_truncated: bool = True,
                 committed_only: bool = False) -> pd.DataFrame:
    """One row per (report_id, field) with both signals attached.

    Rows whose renormalization collapsed for lack of any legal alternative in
    the top-k are dropped by default: their p_renorm is 1.0 by artifact, and
    leaving them in would manufacture a block of spuriously perfect confidence.

    committed_only drops rows where the extractor answered `unknown`, and it is
    the setting under which the two signals are actually comparable.

    WHY. `unknown` is the model's abstention, and the token distribution is
    SHARPLY PEAKED on it -- it is the safe continuation when the narrative is
    silent. But the distant key always committed to a value, so every abstention
    scores as an error. Confidence and correctness then move in opposite
    directions and AUROC inverts: measured on the full sample, p_renorm scores
    0.145 on weather (96.7% abstention), 0.157 on road_class (70.6%) and 0.225
    on lighting (85.3%), while reaching 0.790 on engagement_state (3.7%) and
    0.832 on contributory_party (0%). The ranking is abstention rate, not signal
    quality -- the confidence signal is measuring "how sure am I that this text
    says nothing", which is a different question from "how sure am I that this
    answer is right".

    This is the same decomposition the evaluation protocol already applies to
    extraction accuracy (abstention rate vs. accuracy on the committed subset);
    it applies here for the same reason and must not be pooled away.
    """
    rows = []
    n_trunc = 0
    n_abstain = 0
    for r in _load_jsonl(extractions_path):
        if not r.get("ok") or r.get("model") != model:
            continue
        tp = r.get("token_probs") or {}
        fields = tp.get("fields") or {}
        if not fields:
            continue
        self_conf = (r.get("extraction") or {}).get("confidence")
        for field, d in fields.items():
            if d.get("truncated"):
                n_trunc += 1
                if drop_truncated:
                    continue
            if str(d.get("value")) == "unknown":
                n_abstain += 1
                if committed_only:
                    continue
            rows.append({
                "report_id": r["report_id"], "field": field,
                "value": d.get("value"), "self_conf": self_conf,
                "n_tokens": d.get("n_tokens"), "truncated": d.get("truncated"),
                **{s: d.get(s) for s in SIGNALS},
            })
    df = pd.DataFrame(rows)
    print(f"[confidence] {model}: {len(df)} field-decisions "
          f"({n_trunc} truncated{' dropped' if drop_truncated else ' kept'}, "
          f"{n_abstain} abstentions{' dropped' if committed_only else ' kept'})")
    return df


# ---------------------------------------------------------------------------
# Correctness keys
# ---------------------------------------------------------------------------
def correctness_human(gold_path: str, extractions_path: str,
                      model: str) -> dict[tuple[str, str], int]:
    """Exact match on contributory_party against the adjudicated human key."""
    gold = {r["report_id"]: r for r in _load_jsonl(gold_path)}
    out = {}
    for r in _load_jsonl(extractions_path):
        if not r.get("ok") or r.get("model") != model:
            continue
        g = gold.get(r["report_id"])
        if g is None:
            continue
        gv, pv = g.get("contributory_party"), (r.get("extraction") or {}).get("contributory_party")
        if gv is None or pv is None:
            continue
        out[(r["report_id"], "contributory_party")] = int(str(gv) == str(pv))
    return out


def correctness_distant(narratives_path: str, extractions_path: str,
                        model: str) -> dict[tuple[str, str], int]:
    """Agreement with the reporting entity's own structured code, per field.

    Uses the same REDUCERS -- and therefore the same coarsening -- as the
    corpus-scale extraction-accuracy table, so "correct" means here exactly what
    it means there.
    """
    by_id = {r["report_id"]: r for r in _load_jsonl(narratives_path)}
    out = {}
    for r in _load_jsonl(extractions_path):
        if not r.get("ok") or r.get("model") != model:
            continue
        row = by_id.get(r["report_id"])
        if row is None:
            continue
        ext = r.get("extraction") or {}
        for name, (field, reduce_fn) in REDUCERS.items():
            res = reduce_fn(row, ext.get(field))
            if res is None:
                continue
            # Registry names can be finer than schema fields (collision_type
            # contributes two one-vs-rest rows); key on the SCHEMA field, which
            # is what the token probability was measured for.
            key = (r["report_id"], field)
            out[key] = int(res[0] == res[1]) if key not in out else \
                out[key] & int(res[0] == res[1])
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auroc(conf, correct) -> float:
    from sklearn.metrics import roc_auc_score
    correct = np.asarray(correct)
    if len(set(correct.tolist())) < 2:
        return float("nan")   # undefined with one class; do not report 0.5
    return float(roc_auc_score(correct, np.asarray(conf, dtype=float)))


def brier(conf, correct) -> float:
    conf = np.asarray(conf, dtype=float)
    return float(np.mean((conf - np.asarray(correct, dtype=float)) ** 2))


def isotonic_ece(conf, correct, n_splits: int = 5, seed: int = 11,
                 n_bins: int = 10) -> float:
    """ECE after out-of-fold isotonic recalibration.

    Fit in-fold and applied out-of-fold: a recalibration scored on the rows that
    fit it would report the training error, which is near zero by construction
    and says nothing.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=int)
    if len(set(correct.tolist())) < 2 or len(conf) < n_splits * 2:
        return float("nan")
    oof = np.full(len(conf), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(conf.reshape(-1, 1), correct):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(conf[tr], correct[tr])
        oof[te] = iso.predict(conf[te])
    e, _ = ece_from(oof, correct, n_bins=n_bins)
    return e


def risk_coverage(conf, correct) -> tuple[list, float]:
    """Accuracy on the retained set as the least-confident rows are abstained on.

    Returns (curve, AURC) where curve is [{coverage, accuracy, n}] and AURC is
    the mean accuracy across coverage levels -- higher is better, so a signal
    that ranks well scores well even if its absolute scale is wrong.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=int)
    order = np.argsort(-conf)           # most confident first
    c = correct[order]
    cum_acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    curve = []
    for frac in np.arange(0.1, 1.0001, 0.1):
        k = max(1, int(round(frac * len(c))))
        curve.append({"coverage": float(k / len(c)),
                      "accuracy": float(cum_acc[k - 1]), "n": int(k)})
    return curve, float(np.mean([p["accuracy"] for p in curve]))


def score_signal(conf, correct, n_bins: int = 10) -> dict:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=int)
    e, curve = ece_from(conf, correct, n_bins=n_bins)
    rc, aurc = risk_coverage(conf, correct)
    return {"n": int(len(conf)), "base_rate": float(correct.mean()),
            "mean_conf": float(conf.mean()), "ece": e,
            "ece_isotonic": isotonic_ece(conf, correct, n_bins=n_bins),
            "auroc": auroc(conf, correct), "brier": brier(conf, correct),
            "aurc": aurc, "reliability": curve, "risk_coverage": rc}


# ---------------------------------------------------------------------------
# Two-signal comparison
# ---------------------------------------------------------------------------
def signal_gap(df: pd.DataFrame, signal: str = "p_renorm") -> list[dict]:
    """How far apart are the self-report and the token signal, per field?

    Rank correlation rather than a difference of means: the two are on different
    scales by construction (the self-report is a stated belief about the whole
    record, the token probability is a decode-time quantity for one field), so
    the meaningful question is whether they ORDER the same records the same way.
    """
    from scipy.stats import spearmanr, kendalltau
    rows = []
    for field, g in df.groupby("field"):
        g = g.dropna(subset=[signal, "self_conf"])
        if len(g) < 10:
            continue
        a, b = g["self_conf"].to_numpy(float), g[signal].to_numpy(float)
        # A constant self-report has no ordering to correlate against. That is
        # itself the finding for models that emit the same number every time, so
        # it is recorded rather than skipped.
        if np.std(a) == 0 or np.std(b) == 0:
            rows.append({"field": field, "n": len(g), "spearman": np.nan,
                         "kendall": np.nan, "self_conf_sd": float(np.std(a)),
                         f"{signal}_sd": float(np.std(b)),
                         "mean_self_conf": float(a.mean()),
                         f"mean_{signal}": float(b.mean()),
                         "note": "constant signal -- no ordering to correlate"})
            continue
        rows.append({"field": field, "n": len(g),
                     "spearman": float(spearmanr(a, b).statistic),
                     "kendall": float(kendalltau(a, b).statistic),
                     "self_conf_sd": float(np.std(a)), f"{signal}_sd": float(np.std(b)),
                     "mean_self_conf": float(a.mean()), f"mean_{signal}": float(b.mean()),
                     "note": ""})
    return rows


def evaluate(df: pd.DataFrame, correct_map: dict, key_name: str,
             signals: Optional[list] = None) -> list[dict]:
    """Score every (field, signal) pair against one correctness key."""
    signals = signals or ["self_conf", "p_renorm", "p_first"]
    rows = []
    df = df.copy()
    df["correct"] = [correct_map.get((r, f)) for r, f in
                     zip(df["report_id"], df["field"])]
    df = df[df["correct"].notna()]
    for field, g in df.groupby("field"):
        for sig in signals:
            gg = g.dropna(subset=[sig])
            if len(gg) < 10:
                continue
            res = score_signal(gg[sig].to_numpy(float), gg["correct"].to_numpy(int))
            rows.append({"key": key_name, "field": field, "signal": sig, **res})
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    r"""Filesystem- and LaTeX-safe tag. \input chokes on spaces and parens."""
    out = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return re.sub(r"-+", "-", out)


def write_tables(scored: pd.DataFrame, gap: pd.DataFrame, model: str) -> None:
    os.makedirs(TABLES, exist_ok=True)
    tag = _slug(model)

    t = scored[scored["signal"].isin(["self_conf", "p_renorm"])].copy()
    t = t[["field", "signal", "n", "base_rate", "ece", "ece_isotonic",
           "auroc", "aurc"]]
    t.columns = ["Field", "Signal", "n", "base", "ECE", r"ECE$_{iso}$",
                 "AUROC", "AURC"]
    _to_latex(t, os.path.join(TABLES, f"tab_confidence_{tag}.tex"),
              f"Self-reported confidence vs.\\ decode-time token probability "
              f"({model}). ECE measures calibration, AUROC discrimination; "
              f"ECE$_{{iso}}$ is ECE after out-of-fold isotonic recalibration.",
              f"tab:confidence-{tag}")

    if len(gap):
        g = gap[["field", "n", "spearman", "kendall", "mean_self_conf",
                 "mean_p_renorm"]].copy()
        g.columns = ["Field", "n", r"Spearman $\rho$", r"Kendall $\tau$",
                     "mean self-conf.", "mean $p_{renorm}$"]
        _to_latex(g, os.path.join(TABLES, f"tab_conf_gap_{tag}.tex"),
                  f"Agreement between the two confidence signals ({model}).",
                  f"tab:conf-gap-{tag}")


def write_figures(scored: list[dict], model: str) -> None:
    """Reliability diagram and risk-coverage curve.

    score_agreement.py has named a reliability diagram 'Figure 2' since the
    project began and never drawn one; the curve data has been sitting unused in
    calibration.json the whole time. This draws it, for both signals.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES, exist_ok=True)
    tag = _slug(model)
    by_sig = {}
    for r in scored:
        if r["field"] == "contributory_party" or r["signal"] not in ("self_conf", "p_renorm"):
            continue
        by_sig.setdefault(r["signal"], []).append(r)
    if not by_sig:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax1.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, label="perfect calibration")
    for sig, rows in by_sig.items():
        # Pool bins across fields, weighting by bin count.
        agg: dict = {}
        for r in rows:
            for b in r["reliability"]:
                k = round(b["bin_lo"], 3)
                a = agg.setdefault(k, {"n": 0, "acc": 0.0, "conf": 0.0})
                a["n"] += b["n"]; a["acc"] += b["acc"] * b["n"]; a["conf"] += b["conf"] * b["n"]
        xs = sorted(agg)
        ax1.plot([agg[k]["conf"] / agg[k]["n"] for k in xs],
                 [agg[k]["acc"] / agg[k]["n"] for k in xs],
                 marker="o", ms=4, label=sig)
        cov = np.arange(0.1, 1.0001, 0.1)
        acc = []
        for i in range(len(cov)):
            num = sum(r["risk_coverage"][i]["accuracy"] * r["n"] for r in rows)
            acc.append(num / sum(r["n"] for r in rows))
        ax2.plot(cov, acc, marker="o", ms=4, label=sig)

    ax1.set_xlabel("confidence"); ax1.set_ylabel("empirical accuracy")
    ax1.set_title("Reliability"); ax1.legend(fontsize=8); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax2.set_xlabel("coverage (fraction retained)"); ax2.set_ylabel("accuracy on retained")
    ax2.set_title("Risk-coverage"); ax2.legend(fontsize=8)
    fig.suptitle(f"Confidence signals, {model}", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGURES, f"fig_confidence_{tag}.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[confidence] wrote {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare the two confidence signals.")
    ap.add_argument("--extractions", required=True,
                    help="JSONL containing token_probs (an open-weight run).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--narratives",
                    default=os.path.join("data", "interim", "narratives.jsonl"))
    ap.add_argument("--gold", default=os.path.join("data", "gold", "gold.jsonl"))
    ap.add_argument("--keep-truncated", action="store_true")
    ap.add_argument("--committed-only", action="store_true",
                    help="Score only decisions where the extractor committed to "
                    "a value. Without this, abstentions dominate the "
                    "high-abstention fields and AUROC inverts -- see "
                    "signal_frame's docstring.")
    a = ap.parse_args()

    df = signal_frame(a.extractions, a.model, drop_truncated=not a.keep_truncated,
                      committed_only=a.committed_only)
    if df.empty:
        raise SystemExit(f"[confidence] no token_probs for model={a.model!r} in "
                         f"{a.extractions}. Was it extracted with logprobs on?")

    scored = []
    scored += evaluate(df, correctness_distant(a.narratives, a.extractions, a.model),
                       "distant")
    scored += evaluate(df, correctness_human(a.gold, a.extractions, a.model), "human")
    gap = signal_gap(df)

    os.makedirs(RESULTS, exist_ok=True)
    tag = _slug(a.model) + ("-committed" if a.committed_only else "")
    out = os.path.join(RESULTS, f"confidence_eval_{tag}.json")
    with open(out, "w") as f:
        json.dump({"model": a.model, "scored": scored, "gap": gap}, f, indent=2)
    print(f"[confidence] wrote {out}")

    sdf, gdf = pd.DataFrame(scored), pd.DataFrame(gap)
    label = a.model + (" (committed)" if a.committed_only else "")
    if len(sdf):
        write_tables(sdf, gdf, label)
        write_figures(scored, label)
        print("\n" + sdf[["key", "field", "signal", "n", "ece", "auroc", "aurc"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
