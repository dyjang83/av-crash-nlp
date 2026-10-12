"""The outcome-masked lift ladder and residual-leakage audits.

Answers the reviewer's question -- how much of the text lift is the model
re-reading an outcome the narrative already states? -- by running the unchanged
lift test (models.lift_test.run) on a ladder of representations, all scored on
exactly the same records as the original analysis:

  orig         S+T from the original narrative and extractions (the old result)
  outcome      S+T from the outcome-masked narrative and its re-extraction  <- headline
  outcome_nf   S + extracted fields only (no embedding), masked extraction
  outcome_len  outcome + log-length control
  strict       strict mask (adds police/claims/allegation proxies)
  preimpact    text through the first contact sentence only (lower bound)
  outcome_sfull / orig_sfull   same, with the stronger structured baseline S_full
  outcome_sbert                outcome with SBERT in place of TF-IDF

Then, from the saved out-of-fold predictions:
  - decomposition: paired bootstrap of QWK(S+T | orig) - QWK(S+T | outcome),
    i.e. the part of the old lift that was outcome recovery;
  - natural control: S vs S+T(outcome) on records the mask did not touch;
and, independently of the ladder, the leakage audits:
  - extractor probe: narrative_injury_severity re-extracted from masked text;
  - mask artifact: does the amount removed predict severity by itself?
  - top TF-IDF terms for injured vs not, original vs masked.

Usage:
    python -m models.leakage_ladder                 # all rungs, then tables
    python -m models.leakage_ladder --rungs outcome orig --seeds 11
    python -m models.leakage_ladder --tables-only   # rebuild tables from JSON
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import (S_FULL_COLS, SEVERITY_ORDER,  # noqa: E402
                                     _norm_severity, load, sample_ids)
from models.lift_test import (RESULTS, TABLES, FoldPreds, mae_ordinal,  # noqa: E402
                              paired_bootstrap, quadratic_weighted_kappa, qwk_wrap,
                              run)

NARR = os.path.join("data", "interim", "narratives.jsonl")
MASKED = os.path.join("data", "interim", "narratives_masked_{tier}.jsonl")
EXT = os.path.join("data", "processed", "extractions.jsonl")
EXT_MASKED = os.path.join("data", "processed", "extractions_masked.jsonl")
OUT = os.path.join(RESULTS, "leakage_ladder.json")

# Every masked rung takes its extracted fields from the OUTCOME-masked
# re-extraction; only the free text differs by tier. For strict/preimpact that
# is conservative in the right direction: the fields never saw outcome text.
RUNGS: dict[str, dict] = {
    "orig": dict(narr=NARR, ext=EXT),
    "outcome": dict(narr=MASKED.format(tier="outcome"), ext=EXT_MASKED),
    "outcome_nf": dict(narr=MASKED.format(tier="outcome"), ext=EXT_MASKED,
                       load=dict(text_features=False)),
    "outcome_len": dict(narr=MASKED.format(tier="outcome"), ext=EXT_MASKED,
                        load=dict(length_feature=True)),
    "strict": dict(narr=MASKED.format(tier="strict"), ext=EXT_MASKED),
    "preimpact": dict(narr=MASKED.format(tier="preimpact"), ext=EXT_MASKED),
    "orig_sfull": dict(narr=NARR, ext=EXT, load=dict(struct_cols=S_FULL_COLS)),
    "outcome_sfull": dict(narr=MASKED.format(tier="outcome"), ext=EXT_MASKED,
                          load=dict(struct_cols=S_FULL_COLS)),
    "outcome_sbert": dict(narr=MASKED.format(tier="outcome"), ext=EXT_MASKED,
                          embed="sbert"),
}
LABELS = {
    "orig": "Original narrative (as submitted)",
    "outcome": "Outcome-masked (O1--O3, O5)",
    "outcome_nf": "Outcome-masked, extracted fields only",
    "outcome_len": "Outcome-masked + length control",
    "strict": "Strict mask (+ O4 response proxies)",
    "preimpact": "Pre-impact text only",
    "orig_sfull": "Original, $S_{\\mathrm{full}}$ baseline",
    "outcome_sfull": "Outcome-masked, $S_{\\mathrm{full}}$ baseline",
    "outcome_sbert": "Outcome-masked, SBERT",
}
KINDS = ["logreg", "gbm"]


def _tag(rung: str) -> str:
    return f"ladder-{rung}"


def _suffix(rung: str) -> str:
    cfg = RUNGS[rung]
    # Must mirror the suffix construction in models.lift_test.run.
    return ("" if cfg.get("embed", "tfidf") == "tfidf" else "_" + cfg["embed"]) \
        + "_" + re.sub(r"[^A-Za-z0-9]+", "-", _tag(rung)).strip("-")


def run_rungs(rungs: list[str], seeds: list[int] | None) -> None:
    for r in rungs:
        cfg = RUNGS[r]
        print(f"\n===== ladder rung: {r} =====", flush=True)
        run(cfg["narr"], cfg["ext"], embed=cfg.get("embed", "tfidf"),
            seeds=seeds, tag=_tag(r),
            load_kwargs={"align_to": NARR, **cfg.get("load", {})})


def _load_rung(r: str):
    sfx = _suffix(r)
    jp = os.path.join(RESULTS, f"lift_test{sfx}.json")
    npz = os.path.join(RESULTS, f"lift_oof_proba{sfx}.npz")
    if not (os.path.exists(jp) and os.path.exists(npz)):
        return None, None
    with open(jp) as f:
        rep = json.load(f)
    return rep, np.load(npz, allow_pickle=True)


def _fp(z, rep: str, kind: str) -> FoldPreds:
    p = z[f"{rep}__{kind}"]
    return FoldPreds(rep, kind, z["y_true"], p.argmax(axis=1), y_proba=p)


def _boot_diff(y, pa, pb, n_classes, idx=None, n_boot=5000):
    """Paired bootstrap of QWK(pb) - QWK(pa), optionally on a subset idx.

    Uses lift_test's own seeded RNG, so the intervals are reproducible run to run.
    """
    if idx is not None:
        y, pa, pb = y[idx], pa[idx], pb[idx]
    res = paired_bootstrap(FoldPreds("a", "", y, pa), FoldPreds("b", "", y, pb),
                           n_classes, qwk_wrap, True, n_boot=n_boot)
    res["qwk_a"] = float(quadratic_weighted_kappa(y, pa, n_classes))
    res["qwk_b"] = float(quadratic_weighted_kappa(y, pb, n_classes))
    res["n"] = int(len(y))
    return res


# ------------------------------- audits -------------------------------------
_NIS = {"none": 0, "minor": 1, "moderate": 2, "serious": 3, "fatal": 4}


def extractor_probe(ids: list[str]) -> dict:
    """narrative_injury_severity from original vs masked text, on the sample."""
    y = _target(ids)
    out = {}
    for name, path in [("orig", EXT), ("masked", EXT_MASKED)]:
        if not os.path.exists(path):
            continue
        e = pd.read_json(path, lines=True, dtype={"report_id": str})
        e = e[e["ok"] & (e["model"] == "claude-sonnet-4-6")].drop_duplicates("report_id")
        nis = pd.json_normalize(e["extraction"])["narrative_injury_severity"]
        m = dict(zip(e["report_id"].astype(str), nis))
        vals = [m.get(r, "unknown") for r in ids]
        known = np.array([v in _NIS for v in vals])
        pred = np.array([_NIS.get(v, 0) for v in vals])   # unknown -> none
        out[name] = {
            "share_unknown": float(1 - known.mean()),
            "share_nonnone": float(np.mean([v in _NIS and v != "none" for v in vals])),
            "qwk_vs_target_unknown_as_none": float(
                quadratic_weighted_kappa(y, pred, len(SEVERITY_ORDER))),
            "n_missing_extraction": int(sum(r not in m for r in ids)),
        }
    return out


def _target(ids: list[str]) -> np.ndarray:
    n = pd.read_json(NARR, lines=True, dtype={"report_id": str}).set_index("report_id")
    return np.array([SEVERITY_ORDER.index(_norm_severity(n.loc[r, "struct_severity"]))
                     for r in ids])


def mask_artifact(ids: list[str]) -> dict:
    """Does the AMOUNT the mask removed predict severity on its own?

    If it does, a masked representation could still leak through the shape of
    what is left (shorter texts = injury). Scored with the same grouped CV as
    the lift test, logreg only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    p = MASKED.format(tier="outcome")
    if not os.path.exists(p):
        return {}
    m = pd.read_json(p, lines=True, dtype={"report_id": str}).set_index("report_id").loc[ids]
    y = _target(ids)
    g = m["manufacturer"].fillna("unknown").astype(str).to_numpy()
    def _cv_qwk(X):
        oof = np.zeros(len(y), int)
        for tr, te in GroupKFold(5).split(X, y, g):
            oof[te] = LogisticRegression(max_iter=2000, class_weight="balanced") \
                .fit(X[tr], y[tr]).predict(X[te])
        return float(quadratic_weighted_kappa(y, oof, 5))

    removed = np.column_stack([np.log1p(m["mask_n_removed"]),
                               np.log1p(m["mask_chars_removed"])])
    words = lambda t: np.log1p(t.fillna("").str.split().str.len().to_numpy(float))
    o = pd.read_json(NARR, lines=True, dtype={"report_id": str}).set_index("report_id").loc[ids]
    # The relevant artifact test is the MASKED text's shape: a model sees only
    # what is left, so what matters is whether the remaining length alone
    # predicts severity -- and whether it does so more than the original
    # length already did. What was removed is informative by construction.
    return {"qwk_removed_amount_only": _cv_qwk(removed),
            "qwk_masked_length_only": _cv_qwk(words(m["narrative"]).reshape(-1, 1)),
            "qwk_original_length_only": _cv_qwk(words(o["narrative"]).reshape(-1, 1)),
            "share_records_touched": float((m["mask_n_removed"] > 0).mean()),
            "share_touched_by_class": {
                SEVERITY_ORDER[k]: float((m["mask_n_removed"].to_numpy()[y == k] > 0).mean())
                for k in range(len(SEVERITY_ORDER))}}


def top_terms(ids: list[str], k: int = 12) -> dict:
    """Top TF-IDF terms for injured vs not (binary logreg), original vs masked."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    y = (_target(ids) > 0).astype(int)
    out = {}
    for name, p in [("orig", NARR), ("masked", MASKED.format(tier="outcome"))]:
        if not os.path.exists(p):
            continue
        t = pd.read_json(p, lines=True, dtype={"report_id": str}) \
              .set_index("report_id").loc[ids, "narrative"].fillna("")
        vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=3,
                              stop_words="english", sublinear_tf=True)
        X = vec.fit_transform(t)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X, y)
        names = vec.get_feature_names_out()
        order = np.argsort(clf.coef_[0])
        out[name] = {"injured": [names[i] for i in order[::-1][:k]],
                     "not_injured": [names[i] for i in order[:k]]}
    return out


def corpus_facts(ids: list[str]) -> dict:
    """How much outcome text the ORIGINAL narratives carry, by severity class.

    Lexicon only, so these figures are reproducible without the LLM tagger.
    """
    from leakage.outcome_lexicon import sentence_categories, split_sentences
    n = pd.read_json(NARR, lines=True, dtype={"report_id": str}).set_index("report_id").loc[ids]
    y = _target(ids)
    cats = []
    for t in n["narrative"].fillna(""):
        c = set()
        for s, e in split_sentences(t):
            c |= sentence_categories(t[s:e])
        cats.append(c)
    has_inj = np.array([bool(c & {"O1", "O2", "O3"}) for c in cats])
    has_o5 = np.array(["O5" in c for c in cats])
    return {
        "share_O1_O3_injured": float(has_inj[y > 0].mean()),
        "share_O1_O3_not_injured": float(has_inj[y == 0].mean()),
        "share_O1_O3_by_class": {SEVERITY_ORDER[k]: float(has_inj[y == k].mean())
                                 for k in range(len(SEVERITY_ORDER)) if (y == k).any()},
        "share_O5_records": float(has_o5.mean()),
        "share_O5_injured": float(has_o5[y > 0].mean()),
        "share_O5_not_injured": float(has_o5[y == 0].mean()),
    }


# ------------------------------- report -------------------------------------
def summarize(n_boot: int = 5000) -> dict:
    ids = sample_ids(NARR)
    rep: dict = {"n": len(ids), "rungs": {}, "decomposition": {}, "untouched": {}}
    loaded = {}
    for r in RUNGS:
        j, z = _load_rung(r)
        if j is None:
            continue
        if list(z["report_ids"]) != ids:
            raise SystemExit(f"rung {r}: records differ from the original sample")
        loaded[r] = z
        rep["rungs"][r] = {
            k: {"S": j["by_model"][k]["S"], "T": j["by_model"][k]["T"],
                "S+T": j["by_model"][k]["S+T"],
                "lift": j["by_model"][k]["lift_S+T_vs_S_qwk"],
                "lift_T": j["by_model"][k]["lift_T_vs_S_qwk"],
                "seed_mean": j["seed_stability"][k]["lift_mean"],
                "seed_sd": j["seed_stability"][k]["lift_sd"]}
            for k in KINDS}
        rep["rungs"][r]["headline_model"] = j["headline"]["model"]

    y = np.asarray(_target(ids))
    nc = len(SEVERITY_ORDER)
    if "orig" in loaded and "outcome" in loaded:
        for k in KINDS:
            a = loaded["outcome"][f"S+T__{k}"].argmax(1)
            b = loaded["orig"][f"S+T__{k}"].argmax(1)
            rep["decomposition"][k] = _boot_diff(y, a, b, nc, n_boot=n_boot)
    mp = MASKED.format(tier="outcome")
    if "outcome" in loaded and os.path.exists(mp):
        m = pd.read_json(mp, lines=True, dtype={"report_id": str}).set_index("report_id").loc[ids]
        idx = np.where(m["mask_n_removed"].to_numpy() == 0)[0]
        for k in KINDS:
            z = loaded["outcome"]
            rep["untouched"][k] = _boot_diff(y, z[f"S__{k}"].argmax(1),
                                             z[f"S+T__{k}"].argmax(1), nc, idx=idx,
                                             n_boot=n_boot)
            rep["untouched"][k]["class_counts"] = np.bincount(y[idx], minlength=nc).tolist()

    # Best-vs-best. A within-family lift is measured against that family's own
    # S baseline, and gradient boosting's S is markedly weaker than logistic
    # regression's -- so a gbm "lift" can partly be gbm catching up on S. The
    # question an insurer faces is whether the best model that uses the text
    # beats the best model that does not. Families are selected on seed-
    # averaged QWK (not the bootstrap sample), then the pair is bootstrapped.
    rep["best_vs_best"] = {}
    for base_r, cand_r in [("outcome", "outcome"), ("outcome_sfull", "outcome_sfull"),
                           ("preimpact", "preimpact"), ("orig", "orig")]:
        if base_r not in loaded:
            continue
        j = {k: json.load(open(os.path.join(RESULTS, f"lift_test{_suffix(base_r)}.json")))
             for k in [base_r]}[base_r]
        qS = {k: j["seed_stability"][k]["S"]["qwk_mean"] for k in KINDS}
        qST = {k: j["seed_stability"][k]["S+T"]["qwk_mean"] for k in KINDS}
        kS, kST = max(qS, key=qS.get), max(qST, key=qST.get)
        z = loaded[base_r]
        d = _boot_diff(y, z[f"S__{kS}"].argmax(1), z[f"S+T__{kST}"].argmax(1), nc,
                       n_boot=n_boot)
        d.update(best_S=kS, best_ST=kST)
        rep["best_vs_best"][base_r] = d

    rep["corpus_facts"] = corpus_facts(ids)
    rep["audit"] = {"extractor_probe": extractor_probe(ids),
                    "mask_artifact": mask_artifact(ids),
                    "top_terms": top_terms(ids)}
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    _tables(rep)
    return rep


def _ci(L: dict) -> str:
    return f"{L['lift']:+.3f} [{L['ci95'][0]:+.3f}, {L['ci95'][1]:+.3f}]"


def _tables(rep: dict) -> None:
    os.makedirs(TABLES, exist_ok=True)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Representation & \multicolumn{2}{c}{logreg} & \multicolumn{2}{c}{gbm} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & QWK $S{+}T$ & lift vs.\ $S$ [95\% CI] & QWK $S{+}T$ & lift vs.\ $S$ [95\% CI] \\",
             r"\midrule"]
    for r in RUNGS:
        if r not in rep["rungs"]:
            continue
        b = rep["rungs"][r]
        cells = []
        for k in KINDS:
            cells += [f"{b[k]['S+T']['qwk']:.3f}", _ci(b[k]["lift"])]
        lines.append(f"{LABELS[r]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, "tab_lift_ladder.tex"), "w") as f:
        f.write("\n".join(lines))

    a = rep["audit"]
    al = [r"\begin{tabular}{lrr}", r"\toprule", r"Audit & Original & Masked \\", r"\midrule"]
    ep = a.get("extractor_probe", {})
    if "orig" in ep and "masked" in ep:
        o, m = ep["orig"], ep["masked"]
        al.append(f"Extractor: share coded injured (minor or worse) & "
                  f"{o['share_nonnone']:.3f} & {m['share_nonnone']:.4f} \\\\")
        al.append(f"Extractor: narrative injury severity QWK vs.\\ target & "
                  f"{o['qwk_vs_target_unknown_as_none']:.3f} & "
                  f"{m['qwk_vs_target_unknown_as_none']:.3f} \\\\")
    ma = a.get("mask_artifact", {})
    if ma:
        al.append(f"Narrative length alone, QWK vs.\\ target & "
                  f"{ma['qwk_original_length_only']:.3f} & "
                  f"{ma['qwk_masked_length_only']:.3f} \\\\")
        al.append(f"Amount removed alone, QWK vs.\\ target & -- & "
                  f"{ma['qwk_removed_amount_only']:.3f} \\\\")
    for k, d in rep.get("decomposition", {}).items():
        al.append(f"$S{{+}}T$ QWK attributable to outcome text ({k}) & "
                  f"\\multicolumn{{2}}{{r}}{{{_ci(d)}}} \\\\")
    for r, d in rep.get("best_vs_best", {}).items():
        al.append(f"Best $S{{+}}T$ ({d['best_ST']}) vs.\\ best $S$ ({d['best_S']}), "
                  f"{LABELS[r]} & \\multicolumn{{2}}{{r}}{{{_ci(d)}}} \\\\")
    for k, d in rep.get("untouched", {}).items():
        al.append(f"Lift on records the mask left untouched ({k}, $n={d['n']}$) & "
                  f"\\multicolumn{{2}}{{r}}{{{_ci(d)}}} \\\\")
    al += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, "tab_leak_audit.tex"), "w") as f:
        f.write("\n".join(al))

    tt = a.get("top_terms", {})
    if tt:
        tl = [r"\begin{tabular}{p{0.2\linewidth}p{0.36\linewidth}p{0.36\linewidth}}",
              r"\toprule", r"Text & Most injury-indicative terms & Most no-injury-indicative terms \\",
              r"\midrule"]
        for name in ["orig", "masked"]:
            if name in tt:
                tl.append(f"{'Original' if name == 'orig' else 'Outcome-masked'} & "
                          f"{', '.join(tt[name]['injured'])} & "
                          f"{', '.join(tt[name]['not_injured'])} \\\\")
        tl += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(TABLES, "tab_top_terms.tex"), "w") as f:
            f.write("\n".join(tl).replace("_", r"\_"))
    print(f"[ladder] wrote {OUT} and tab_lift_ladder / tab_leak_audit / tab_top_terms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS), choices=list(RUNGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    if not a.tables_only:
        run_rungs(a.rungs, a.seeds)
    summarize()
