"""Severity lift test (the paper's core empirical claim).

Question: does narrative text (T) add predictive value for structured injury
severity beyond the structured fields insurers already collect (S)?

Design (held fixed across models to isolate representation, not capacity):
  - Targets: ordinal severity (primary). Secondary binary proxies (tow-away,
    airbag) can be run by passing a binarized y upstream.
  - Models: penalized multinomial logistic regression AND gradient boosting,
    each fit on S, T, and S+T.
  - Validation: GroupKFold by reporting entity (manufacturer) to prevent
    entity-level leakage. A geography-grouped robustness run is supported by
    passing different `groups`.
  - Metrics: quadratic-weighted kappa (QWK) and MAE for the ordinal target.
  - Inference: paired bootstrap over the held-out predictions for S+T vs. S and
    T vs. S, reporting the mean lift, a 95% CI, and a bootstrap p-value. Effect
    size is reported alongside p; we do not report p alone.


"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

# GridSearchCV warns that our QWK scorer ignores sample_weight during inner
# scoring; fit() still applies it. Scoring stays unweighted deliberately, to
# match the unweighted QWK used for the final outer evaluation.
warnings.filterwarnings("ignore", message=".*does not support sample_weight.*")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import load, Dataset  # noqa: E402

RESULTS = os.path.join("data", "processed")
TABLES = os.path.join("paper", "tables")
RNG = np.random.default_rng(11)


# ----------------------------- metrics --------------------------------------
def quadratic_weighted_kappa(y_true, y_pred, n_classes) -> float:
    O = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    w = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    act = O.sum(axis=1)
    pred = O.sum(axis=0)
    E = np.outer(act, pred) / max(O.sum(), 1)
    denom = (w * E).sum()
    return 1.0 - (w * O).sum() / denom if denom > 0 else 0.0


def mae_ordinal(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ----------------------------- models ---------------------------------------
def make_model(kind: str):
    if kind == "logreg":
        # multi_class="multinomial" was deprecated in sklearn 1.5+ and removed
        # entirely in later versions -- the default lbfgs solver now always
        # fits a multinomial model for >2 classes, so no argument is needed.
        # C is tuned per fold via inner CV (see PARAM_GRID / cross_val_predict);
        # 1.0 here is just the GridSearchCV base-estimator placeholder.
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    if kind == "gbm":
        # max_iter=600 is a headroom cap, not a tuned value: early_stopping
        # (validation_fraction=0.1, n_iter_no_change=10 defaults) already
        # halts training on its own, so raising the cap just avoids
        # under-fitting the higher-dimensional S+T representation without
        # biasing model selection. depth/l2/learning_rate ARE tuned per fold
        # via inner CV -- see PARAM_GRID / cross_val_predict.
        return HistGradientBoostingClassifier(
            learning_rate=0.06, max_depth=3, max_iter=600,
            l2_regularization=1.0, early_stopping=True, random_state=11)
    raise ValueError(kind)


# Inner-CV search grids, keyed by model kind. Selection happens on
# train-fold-only data (see cross_val_predict), so the held-out predictions
# used for the reported lift/CI never inform the hyperparameters that
# produced them.
PARAM_GRID = {
    "logreg": {"C": [0.3, 1.0, 3.0, 10.0]},
    "gbm": {"max_depth": [3, 4], "learning_rate": [0.06, 0.1],
            "l2_regularization": [0.1, 1.0]},
}


@dataclass
class FoldPreds:
    rep: str          # feature set name
    model: str
    y_true: np.ndarray
    y_pred: np.ndarray


def cross_val_predict(ds: Dataset, rep: str, kind: str,
                      n_splits: int = 5, inner_splits: int = 3) -> FoldPreds:
    X, y, groups = ds.X[rep], ds.y, ds.groups
    n_classes = int(y.max()) + 1
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), -1)
    dense = kind == "gbm"
    qwk_scorer = make_scorer(quadratic_weighted_kappa, n_classes=n_classes)
    for tr, te in gkf.split(X, y, groups):
        Xtr = X[tr].toarray() if dense else X[tr]
        Xte = X[te].toarray() if dense else X[te]
        ytr, gtr = y[tr], groups[tr]

        # Inner GroupKFold on the training fold only -- hyperparameters are
        # selected without ever touching the held-out (te) rows, so the
        # outer OOF predictions used for the reported lift/CI stay unbiased.
        inner_n = min(inner_splits, len(np.unique(gtr)))
        inner_cv = list(GroupKFold(n_splits=inner_n).split(Xtr, ytr, gtr))
        search = GridSearchCV(make_model(kind), PARAM_GRID[kind], cv=inner_cv,
                              scoring=qwk_scorer, n_jobs=-1, refit=True)
        if kind == "gbm":
            # HistGradientBoostingClassifier has no class_weight constructor
            # argument (unlike LogisticRegression's class_weight="balanced"
            # above) -- without this, severe class imbalance
            # ([2825, 328, 82, 24, 14] here) makes it collapse to predicting
            # the majority class on low-signal representations like S,
            # confounding representation effects with an imbalance-handling
            # asymmetry between the two model families rather than a real
            # capacity difference.
            sw = compute_sample_weight("balanced", ytr)
            search.fit(Xtr, ytr, sample_weight=sw)
        else:
            search.fit(Xtr, ytr)
        oof[te] = search.best_estimator_.predict(Xte)
    assert (oof >= 0).all(), "some rows never received an out-of-fold prediction"
    return FoldPreds(rep, kind, y, oof)


# --------------------------- paired bootstrap -------------------------------
def paired_bootstrap(base: FoldPreds, cand: FoldPreds, n_classes: int,
                     metric: Callable, higher_better: bool,
                     n_boot: int = 5000) -> dict:
    """Bootstrap the per-example paired difference cand - base on one metric."""
    y = base.y_true
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        mb = metric(y[idx], base.y_pred[idx], n_classes) if metric is qwk_wrap \
            else metric(y[idx], base.y_pred[idx])
        mc = metric(y[idx], cand.y_pred[idx], n_classes) if metric is qwk_wrap \
            else metric(y[idx], cand.y_pred[idx])
        diffs[b] = (mc - mb) if higher_better else (mb - mc)
    point = (metric(y, cand.y_pred, n_classes) - metric(y, base.y_pred, n_classes)) \
        if metric is qwk_wrap else \
        (metric(y, cand.y_pred) - metric(y, base.y_pred))
    if not higher_better:
        point = -point
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_one_sided = float(np.mean(diffs <= 0))  # H0: lift <= 0
    return {"lift": float(point), "ci95": [float(lo), float(hi)],
            "p_one_sided": p_one_sided, "n_boot": n_boot}


def qwk_wrap(yt, yp, n_classes):
    return quadratic_weighted_kappa(yt, yp, n_classes)


def run(narratives: str, extractions: str, embed: str = "tfidf") -> dict:
    ds = load(narratives, extractions, embed=embed)
    n_classes = int(ds.y.max()) + 1
    reps = ["S", "T", "S+T"]
    kinds = ["logreg", "gbm"]

    preds = {(r, k): cross_val_predict(ds, r, k) for r in reps for k in kinds}

    report = {"n": int(len(ds.y)), "n_classes": n_classes,
              "class_counts": np.bincount(ds.y).tolist(),
              "embed": embed, "headline": {}, "by_model": {}}

    for k in kinds:
        base = preds[("S", k)]
        block = {}
        for r in reps:
            fp = preds[(r, k)]
            block[r] = {
                "qwk": float(quadratic_weighted_kappa(fp.y_true, fp.y_pred,
                                                      n_classes)),
                "mae": mae_ordinal(fp.y_true, fp.y_pred),
            }
        block["lift_S+T_vs_S_qwk"] = paired_bootstrap(
            base, preds[("S+T", k)], n_classes, qwk_wrap, higher_better=True)
        block["lift_S+T_vs_S_mae"] = paired_bootstrap(
            base, preds[("S+T", k)], n_classes, mae_ordinal, higher_better=False)
        block["lift_T_vs_S_qwk"] = paired_bootstrap(
            base, preds[("T", k)], n_classes, qwk_wrap, higher_better=True)
        report["by_model"][k] = block

    # Headline = the more conservative of the two models on QWK lift.
    lifts = {k: report["by_model"][k]["lift_S+T_vs_S_qwk"]["lift"] for k in kinds}
    hk = min(lifts, key=lifts.get)
    report["headline"] = {"model": hk,
                          **report["by_model"][hk]["lift_S+T_vs_S_qwk"]}

    # tfidf is the paper's canonical, reproducible representation and keeps the
    # plain filenames the paper's \input targets; any other embedding writes to
    # a suffixed filename so a robustness run never clobbers the canonical one.
    suffix = "" if embed == "tfidf" else f"_{embed}"

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"lift_test{suffix}.json"), "w") as f:
        json.dump(report, f, indent=2)
    _emit_table(report, suffix)
    print(json.dumps(report["headline"], indent=2))
    return report


def _emit_table(report: dict, suffix: str = ""):
    os.makedirs(TABLES, exist_ok=True)
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Model & Repr. & QWK & MAE & Lift vs. S (QWK, 95\% CI) \\",
             r"\midrule"]
    for k, block in report["by_model"].items():
        for r in ["S", "T", "S+T"]:
            q, m = block[r]["qwk"], block[r]["mae"]
            if r == "S+T":
                L = block["lift_S+T_vs_S_qwk"]
                lift = f"{L['lift']:+.3f} [{L['ci95'][0]:+.3f}, {L['ci95'][1]:+.3f}]"
            elif r == "T":
                L = block["lift_T_vs_S_qwk"]
                lift = f"{L['lift']:+.3f} [{L['ci95'][0]:+.3f}, {L['ci95'][1]:+.3f}]"
            else:
                lift = "--"
            lines.append(f"{k} & {r} & {q:.3f} & {m:.3f} & {lift} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    out_path = os.path.join(TABLES, f"tab_lift{suffix}.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[lift] wrote {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--narratives",
                    default=os.path.join("data", "interim", "narratives.jsonl"))
    ap.add_argument("--extractions",
                    default=os.path.join("data", "processed", "extractions.jsonl"))
    ap.add_argument("--embed", choices=["tfidf", "sbert"], default="tfidf")
    a = ap.parse_args()
    run(a.narratives, a.extractions, embed=a.embed)
