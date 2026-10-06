"""Score the outcome masker against the two human coders.

Inputs (see outcome_masking_guide.md):
  data/gold/outcome_sent_{A,B}.csv       Task S, one row per sentence
  data/gold/outcome_sent_gold.csv        adjudicated Task S (written as a
                                         template by `adjudicate`, then resolved
                                         by hand)
  data/gold/outcome_recover_{A,B}.csv    Task R, masked narratives
  data/gold/outcome_sample_meta.json     severities and sampling weights (never
                                         shown to coders)

Reports:
  - inter-coder agreement on "is this sentence an outcome statement"
    (percent agreement, Cohen's kappa, Gwet's AC1);
  - sentence-level recall and precision of the lexicon, the LLM tagger and
    their union, for the outcome and strict tiers, against the adjudicated
    labels -- both on the sample and reweighted to the corpus's injured/not
    mix, since the sample oversamples injured records;
  - every adjudicated outcome sentence the union MISSED, verbatim;
  - Task R: how well a human reading the masked text recovers injury status.

Usage:
    python -m annotate.score_outcome_mask adjudicate   # disagreement template
    python -m annotate.score_outcome_mask score
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotate.score_agreement import gwet_ac1  # noqa: E402
from leakage.build_masked_corpus import SPANS, load_spans  # noqa: E402
from leakage.outcome_lexicon import TIERS, mask  # noqa: E402

GOLD = os.path.join("data", "gold")
NARR = os.path.join("data", "interim", "narratives.jsonl")
META = os.path.join(GOLD, "outcome_sample_meta.json")
OUT = os.path.join("data", "processed", "outcome_mask_validation.json")
TABLES = os.path.join("paper", "tables")


def _read_sent(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"report_id": str})
    d["outcome"] = pd.to_numeric(d["outcome"], errors="coerce")
    if d["outcome"].isna().any():
        raise SystemExit(f"{path}: {int(d['outcome'].isna().sum())} sentences not coded")
    d["outcome"] = d["outcome"].astype(int)
    d["category"] = d["category"].fillna("").astype(str).str.strip().str.upper()
    return d.set_index(["report_id", "sent_idx"]).sort_index()


SENT_A = os.path.join(GOLD, "outcome_sent_A.csv")
SENT_B = os.path.join(GOLD, "outcome_sent_B.csv")


def adjudicate(a_path: str = SENT_A, b_path: str = SENT_B) -> None:
    a = _read_sent(a_path)
    b = _read_sent(b_path)
    g = a[["sentence"]].copy()
    agree = (a["outcome"] == b["outcome"]) & \
            ((a["outcome"] == 0) | (a["category"] == b["category"]))
    g["outcome"] = np.where(agree, a["outcome"], "")
    g["category"] = np.where(agree, a["category"], "")
    g["A"] = a["outcome"].astype(str) + a["category"].radd(":")
    g["B"] = b["outcome"].astype(str) + b["category"].radd(":")
    g["needs_adjudication"] = ~agree
    p = os.path.join(GOLD, "outcome_sent_gold.csv")
    if os.path.exists(p):
        raise SystemExit(f"{p} exists; refusing to overwrite adjudicated labels")
    g.reset_index().to_csv(p, index=False)
    print(f"[outcome-score] {int((~agree).sum())} of {len(g)} sentences need "
          f"adjudication -> {p}")


def _detector_flags(ids: list[str]) -> dict[str, dict[tuple[str, int], bool]]:
    narr = pd.read_json(NARR, lines=True, dtype={"report_id": str}).set_index("report_id")
    spans = load_spans(SPANS)
    flags: dict[str, dict] = {}
    for tier in ["outcome", "strict"]:
        for det in ["lexicon", "llm", "union"]:
            flags[f"{tier}:{det}"] = {}
        for rid in ids:
            sp = spans.get(rid)
            r = mask(narr.loc[rid, "narrative"], tier,
                     llm_spans=[s["text"] for s in sp] if sp is not None else None,
                     llm_categories=[s["category"] for s in sp] if sp is not None else None)
            for i in range(r.n_sentences):
                flags[f"{tier}:lexicon"][(rid, i)] = i in r.removed_lexicon
                flags[f"{tier}:llm"][(rid, i)] = i in r.removed_llm
                flags[f"{tier}:union"][(rid, i)] = i in r.removed_lexicon or i in r.removed_llm
    return flags


def _cp(k: int, n: int) -> list[float]:
    """Exact (Clopper-Pearson) 95% interval for k successes in n trials."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def _pr(truth: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    tp = (truth & pred)
    return {"recall": float(tp.sum() / max(truth.sum(), 1)),
            "recall_ci95": _cp(int(tp.sum()), int(truth.sum())),
            "precision_ci95": _cp(int(tp.sum()), int(pred.sum())),
            "precision": float(tp.sum() / max(pred.sum(), 1)),
            "recall_weighted": float((w * tp).sum() / max((w * truth).sum(), 1e-12)),
            "precision_weighted": float((w * tp).sum() / max((w * pred).sum(), 1e-12)),
            "n_true": int(truth.sum()), "n_flagged": int(pred.sum())}


def score(a_path: str = SENT_A, b_path: str = SENT_B) -> dict:
    with open(META) as f:
        meta = json.load(f)
    rep: dict = {"coder_files": {"A": a_path, "B": b_path}}
    a = _read_sent(a_path)
    b = _read_sent(b_path)
    rep["iaa"] = {"n_sentences": int(len(a)),
                  "pct_agree": float((a["outcome"] == b["outcome"]).mean()),
                  "kappa": float(cohen_kappa_score(a["outcome"], b["outcome"])),
                  "ac1": gwet_ac1(a["outcome"].tolist(), b["outcome"].tolist())}
    both = (a["outcome"] == 1) & (b["outcome"] == 1)
    rep["iaa"]["category_kappa_on_joint_positives"] = float(
        cohen_kappa_score(a.loc[both, "category"], b.loc[both, "category"])) \
        if both.sum() > 1 else None

    gp = os.path.join(GOLD, "outcome_sent_gold.csv")
    if os.path.exists(gp):
        g = _read_sent(gp)
        recs = meta["sentences"]["records"]
        wts = meta["sentences"]["sampling_weights"]
        w = np.array([wts["injured" if recs[r]["injured"] else "not_injured"]
                      for r, _ in g.index])
        ids = sorted({r for r, _ in g.index})
        flags = _detector_flags(ids)
        rep["masker"] = {}
        missed = []
        for tier in ["outcome", "strict"]:
            truth = (g["outcome"].to_numpy() == 1) & g["category"].isin(TIERS[tier]).to_numpy()
            for det in ["lexicon", "llm", "union"]:
                f = flags[f"{tier}:{det}"]
                pred = np.array([f.get(k, False) for k in g.index])
                rep["masker"][f"{tier}:{det}"] = _pr(truth, pred, w)
                if det == "union":
                    for k, t, p in zip(g.index, truth, pred):
                        if t and not p:
                            missed.append({"tier": tier, "report_id": k[0],
                                           "sent_idx": int(k[1]),
                                           "category": g.loc[k, "category"],
                                           "sentence": g.loc[k, "sentence"]})
        rep["missed_by_union"] = missed
    else:
        print(f"[outcome-score] {gp} missing; run `adjudicate` and resolve it first")

    ra, rb = (os.path.join(GOLD, f"outcome_recover_{c}.csv") for c in "AB")
    if os.path.exists(ra) and os.path.exists(rb) and "recoverability" in meta:
        truth = meta["recoverability"]["records"]
        rep["recoverability"] = {}
        answers = {}
        for c, p in [("A", ra), ("B", rb)]:
            d = pd.read_csv(p, dtype={"report_id": str})
            d["injured"] = d["injured"].fillna("").str.strip().str.lower()
            bad = ~d["injured"].isin(["yes", "no", "cannot_tell"])
            if bad.any():
                raise SystemExit(f"{p}: {int(bad.sum())} rows without a valid answer")
            y = np.array([truth[r]["injured"] for r in d["report_id"]])
            ans = d["injured"].to_numpy()
            answers[c] = ans
            committed = ans != "cannot_tell"
            correct = (ans == "yes") == y
            rep["recoverability"][c] = {
                "n": int(len(d)), "share_cannot_tell": float(1 - committed.mean()),
                "accuracy_committed": float(correct[committed].mean()) if committed.any() else None,
                # "cannot tell" scored as a coin flip: 0.5 = no recoverable signal.
                "accuracy_cannot_tell_as_half": float(
                    np.where(committed, correct, 0.5).mean()),
                "tpr_yes_on_injured": float((ans[y] == "yes").mean()),
                "fpr_yes_on_uninjured": float((ans[~y] == "yes").mean()),
                "leftover_notes": d.loc[d["note"].fillna("").str.strip() != "",
                                        ["report_id", "note"]].to_dict("records"),
            }
        rep["recoverability"]["kappa_AB"] = float(cohen_kappa_score(answers["A"], answers["B"]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    _table(rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "missed_by_union"}, indent=2))
    return rep


def _table(rep: dict) -> None:
    if "masker" not in rep:
        return
    os.makedirs(TABLES, exist_ok=True)
    L = [r"\begin{tabular}{llrlrlrr}", r"\toprule",
         r"Tier & Detector & Recall & [95\% CI] & Precision & [95\% CI] & "
         r"Recall (wtd.) & Precision (wtd.) \\",
         r"\midrule"]
    for key, m in rep["masker"].items():
        tier, det = key.split(":")
        rc, pc = m["recall_ci95"], m["precision_ci95"]
        L.append(f"{tier} & {det} & {m['recall']:.3f} & [{rc[0]:.3f}, {rc[1]:.3f}] & "
                 f"{m['precision']:.3f} & [{pc[0]:.3f}, {pc[1]:.3f}] & "
                 f"{m['recall_weighted']:.3f} & {m['precision_weighted']:.3f} \\\\")
    iaa = rep["iaa"]
    L.append(r"\midrule")
    L.append(r"\multicolumn{8}{l}{Inter-coder agreement, " + f"{iaa['n_sentences']:,}" +
             r" sentences: " + f"{iaa['pct_agree']:.1%}".replace("%", r"\%") +
             r", Cohen's $\kappa$ = " + f"{iaa['kappa']:.3f}" +
             r", Gwet's AC1 = " + f"{iaa['ac1']:.3f}" + r"} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, "tab_mask_validation.tex"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["adjudicate", "score"])
    ap.add_argument("--a", default=SENT_A, help="Coder A's Task S file.")
    ap.add_argument("--b", default=SENT_B, help="Coder B's Task S file.")
    a = ap.parse_args()
    adjudicate(a.a, a.b) if a.task == "adjudicate" else score(a.a, a.b)
