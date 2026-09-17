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
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

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
from utils.config import canonical_model  # noqa: E402

RESULTS = os.path.join("data", "processed")
TABLES = os.path.join("paper", "tables")
RNG = np.random.default_rng(11)

# GridSearchCV parallelism. -1 uses every core, which on a small-memory machine
# makes as many dense copies of the S+T matrix as there are cores and thrashes.
# Override with AV_CRASH_N_JOBS when running alongside anything else.
N_JOBS = int(os.environ.get("AV_CRASH_N_JOBS", "-1"))

# Estimator seeds for the stability sweep. HistGradientBoosting draws its
# early-stopping validation split from the training fold; with 14 fatalities in
# 3,272 rows that split lands differently per seed and moves QWK by up to ~0.10.
# GroupKFold does not shuffle, so folds are identical across seeds and the seed
# isolates estimator noise from fold composition. A single-seed headline is a
# draw from that distribution, not a property of the representation, so the
# reported lift is averaged over seeds and its spread is reported alongside the
# bootstrap CI -- the two intervals answer different questions and neither
# substitutes for the other.
DEFAULT_SEEDS = [11, 12, 13, 14, 15]


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
def make_model(kind: str, seed: int = 11):
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
            l2_regularization=1.0, early_stopping=True, random_state=seed)
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
    # Out-of-fold class-probability matrix, (n_samples, n_classes). Optional so
    # existing positional construction keeps working. Used for the severity
    # DISTRIBUTION (models/severity_dist.py): summing predicted probabilities
    # gives an expected-count distribution, whereas a histogram of argmax labels
    # badly understates rare classes -- with 14 fatalities in 3,272 records,
    # argmax can assign that class zero mass while the model still puts real
    # probability on it.
    y_proba: Optional[np.ndarray] = None


def cross_val_predict(ds: Dataset, rep: str, kind: str,
                      n_splits: int = 5, inner_splits: int = 3,
                      seed: int = 11) -> FoldPreds:
    X, y, groups = ds.X[rep], ds.y, ds.groups
    n_classes = int(y.max()) + 1
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), -1)
    oof_proba = np.zeros((len(y), n_classes), dtype=float)
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
        search = GridSearchCV(make_model(kind, seed), PARAM_GRID[kind], cv=inner_cv,
                              scoring=qwk_scorer, n_jobs=N_JOBS, refit=True)
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
        best = search.best_estimator_
        oof[te] = best.predict(Xte)

        # A training fold need not contain every class -- with 14 fatalities
        # across the corpus, some folds have none -- so predict_proba returns
        # only the classes it saw, in ITS order. Scatter through classes_ into
        # the fixed-width matrix; absent classes correctly keep probability 0.
        proba = best.predict_proba(Xte)
        for j, cls in enumerate(best.classes_):
            oof_proba[te, int(cls)] = proba[:, j]

    assert (oof >= 0).all(), "some rows never received an out-of-fold prediction"
    row_sums = oof_proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), \
        f"out-of-fold probabilities do not sum to 1 (min={row_sums.min():.6f}, " \
        f"max={row_sums.max():.6f})"
    # The distribution must be consistent with the point predictions it
    # accompanies, or the two would tell different stories from one fit.
    assert (oof_proba.argmax(axis=1) == oof).all(), \
        "argmax of out-of-fold probabilities disagrees with the predicted labels"
    return FoldPreds(rep, kind, y, oof, y_proba=oof_proba)


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


def run(narratives: str, extractions: str, embed: str = "tfidf",
        model: str | None = None, seeds: list[int] | None = None,
        confidence_gate: float | None = None,
        confidence_features: bool = False) -> dict:
    model = model or canonical_model()
    seeds = list(seeds or DEFAULT_SEEDS)
    ds = load(narratives, extractions, embed=embed, model=model,
              confidence_gate=confidence_gate,
              confidence_features=confidence_features)
    n_classes = int(ds.y.max()) + 1
    reps = ["S", "T", "S+T"]
    kinds = ["logreg", "gbm"]

    # Fit every (representation, family) once per seed. logreg is deterministic,
    # so its per-seed results are identical and the sweep costs nothing there;
    # the variation being measured is gradient boosting's.
    # LogisticRegression under lbfgs consumes no randomness (random_state applies
    # only to the sag/saga/liblinear solvers), and GroupKFold does not shuffle, so
    # its fit is identical for every seed. Fitting it once and reusing the result
    # avoids ~40% of the sweep's work and, more importantly, avoids presenting a
    # row of five identical numbers as though variance had been measured there.
    DETERMINISTIC = {"logreg"}
    per_seed = {}
    for sd in seeds:
        for r in reps:
            for k in kinds:
                if k in DETERMINISTIC and sd != seeds[0]:
                    per_seed[(r, k, sd)] = per_seed[(r, k, seeds[0])]
                    continue
                per_seed[(r, k, sd)] = cross_val_predict(ds, r, k, seed=sd)
        print(f"[lift] seed {sd} done", flush=True)

    # The bootstrap/CI reporting uses the FIRST seed, so the published interval
    # keeps its original meaning (sampling error over examples, one fitted
    # model). Seed spread is reported separately -- it is estimator variance,
    # a different source of uncertainty that a bootstrap CI cannot see.
    preds = {(r, k): per_seed[(r, k, seeds[0])] for r in reps for k in kinds}

    report = {"n": int(len(ds.y)), "n_classes": n_classes,
              "class_counts": np.bincount(ds.y).tolist(),
              "embed": embed, "extraction_model": model,
              "confidence_gate": confidence_gate,
              "confidence_features": confidence_features,
              "seeds": seeds, "headline": {}, "by_model": {}, "seed_stability": {}}

    # --- estimator-variance sweep --------------------------------------------
    for k in kinds:
        block = {}
        for r in reps:
            qwks = [quadratic_weighted_kappa(per_seed[(r, k, sd)].y_true,
                                             per_seed[(r, k, sd)].y_pred, n_classes)
                    for sd in seeds]
            block[r] = {"qwk_by_seed": [float(q) for q in qwks],
                        "qwk_mean": float(np.mean(qwks)),
                        "qwk_sd": float(np.std(qwks, ddof=1)) if len(qwks) > 1 else 0.0,
                        "qwk_min": float(np.min(qwks)), "qwk_max": float(np.max(qwks))}
        lifts = [block["S+T"]["qwk_by_seed"][i] - block["S"]["qwk_by_seed"][i]
                 for i in range(len(seeds))]
        block["lift_S+T_vs_S_by_seed"] = [float(x) for x in lifts]
        block["lift_mean"] = float(np.mean(lifts))
        block["lift_sd"] = float(np.std(lifts, ddof=1)) if len(lifts) > 1 else 0.0
        block["lift_min"], block["lift_max"] = float(np.min(lifts)), float(np.max(lifts))
        block["deterministic"] = k in DETERMINISTIC
        report["seed_stability"][k] = block

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

    # Headline = the more conservative of the two families on QWK lift, chosen on
    # the SEED-AVERAGED lift rather than a single seed. Choosing on one seed made
    # the choice itself unstable: gradient boosting's single-seed lift varies by
    # ~0.10, enough to flip which family looks conservative from run to run.
    mean_lifts = {k: report["seed_stability"][k]["lift_mean"] for k in kinds}
    hk = min(mean_lifts, key=mean_lifts.get)
    report["headline"] = {
        "model": hk,
        **report["by_model"][hk]["lift_S+T_vs_S_qwk"],
        "lift_seed_mean": report["seed_stability"][hk]["lift_mean"],
        "lift_seed_sd": report["seed_stability"][hk]["lift_sd"],
        "lift_seed_range": [report["seed_stability"][hk]["lift_min"],
                            report["seed_stability"][hk]["lift_max"]],
        "n_seeds": len(seeds),
    }

    # tfidf is the paper's canonical, reproducible representation and keeps the
    # plain filenames the paper's \input targets; any other embedding writes to
    # a suffixed filename so a robustness run never clobbers the canonical one.
    # The extraction model participates in the suffix for the same reason: a run
    # built from an open-weight model's extractions must not overwrite the
    # closed-model table the paper cites.
    suffix = "" if embed == "tfidf" else f"_{embed}"
    if model != canonical_model():
        suffix += "_" + re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
    if confidence_gate is not None:
        suffix += f"_gate{confidence_gate:g}".replace(".", "")
    if confidence_features:
        suffix += "_conffeat"

    # Predicted severity DISTRIBUTION per (representation, family): the column
    # mean of the out-of-fold probability matrix, i.e. expected share of the
    # corpus at each severity level. This is what the AV-vs-human distribution
    # comparison consumes; the observed marginal is stored beside it so the two
    # are never confused.
    report["predicted_distribution"] = {
        "levels_note": "index = ordinal severity, 0 = least severe",
        "observed": (np.bincount(ds.y, minlength=n_classes) / len(ds.y)).tolist(),
        "by_model": {
            k: {r: preds[(r, k)].y_proba.mean(axis=0).tolist() for r in reps}
            for k in kinds
        },
    }
    np.savez_compressed(
        os.path.join(RESULTS, f"lift_oof_proba{suffix}.npz"),
        y_true=ds.y, groups=ds.groups,
        **{f"{r}__{k}": preds[(r, k)].y_proba for r in reps for k in kinds})

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"lift_test{suffix}.json"), "w") as f:
        json.dump(report, f, indent=2)
    _emit_table(report, suffix)
    print(json.dumps(report["headline"], indent=2))
    return report


def _emit_table(report: dict, suffix: str = ""):
    """Main lift table, plus a seed-stability table when a sweep was run.

    The two uncertainty columns are NOT interchangeable and are printed side by
    side for that reason: the bootstrap CI is sampling error over examples for a
    single fitted model, while the seed range is estimator variance for a fixed
    sample. A bootstrap CI cannot see the second, which is how a headline whose
    QWK moves by ~0.10 across seeds could carry a tight-looking interval.
    """
    os.makedirs(TABLES, exist_ok=True)
    stab = report.get("seed_stability") or {}
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

    if stab:
        n_seeds = len(report.get("seeds", []))
        sl = [r"\begin{tabular}{llrrr}", r"\toprule",
              r"Model & Repr. & QWK (mean) & QWK (sd) & QWK range \\",
              r"\midrule"]
        for k, block in stab.items():
            for r in ["S", "T", "S+T"]:
                b = block[r]
                sl.append(f"{k} & {r} & {b['qwk_mean']:.3f} & {b['qwk_sd']:.3f} & "
                          f"[{b['qwk_min']:.3f}, {b['qwk_max']:.3f}] \\\\")
            sl.append(f"{k} & $S{{+}}T$ lift & {block['lift_mean']:+.3f} & "
                      f"{block['lift_sd']:.3f} & [{block['lift_min']:+.3f}, "
                      f"{block['lift_max']:+.3f}] \\\\")
            sl.append(r"\midrule")
        sl[-1] = r"\bottomrule"
        sl.append(r"\end{tabular}")
        sp = os.path.join(TABLES, f"tab_lift_seeds{suffix}.tex")
        with open(sp, "w") as f:
            f.write("\n".join(sl))
        print(f"[lift] wrote {sp}  ({n_seeds} seeds)")
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
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Estimator seeds for the stability sweep "
                    f"(default {DEFAULT_SEEDS}). Pass a single seed to reproduce "
                    "the original single-fit behaviour.")
    ap.add_argument("--confidence-gate", type=float, default=None,
                    help="Replace extracted field values whose decode-time "
                    "probability falls below TAU with `unknown`. Requires an "
                    "extraction run that captured token probabilities.")
    ap.add_argument("--confidence-features", action="store_true",
                    help="Append per-field token probabilities to T as numeric "
                    "features instead of gating on them.")
    ap.add_argument("--extraction-model", default=None,
                    help="Which model's extractions build T. Defaults to "
                    "extraction.canonical_model from config.yaml. A "
                    "non-canonical choice writes to suffixed output files.")
    a = ap.parse_args()
    run(a.narratives, a.extractions, embed=a.embed, model=a.extraction_model,
        seeds=a.seeds, confidence_gate=a.confidence_gate,
        confidence_features=a.confidence_features)
