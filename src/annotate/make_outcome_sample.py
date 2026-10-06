"""Draw the human-validation samples for the outcome masker.

Two disjoint samples from the lift-test pool (SGO records with a usable
narrative and a structured severity on the ordinal scale):

  sentences      ~300 narratives split into sentences with the SAME splitter the
                 masker uses, one CSV row per sentence, for Task S of
                 outcome_masking_guide.md. Half are drawn from injured records
                 (Minor or worse) and half from no-injury records, so recall is
                 measured where outcome statements actually occur; the sampling
                 weights needed to reweight back to the corpus are stored in the
                 meta file.
  recoverability ~100 MASKED narratives (Task R), 50 injured / 50 not. Needs
                 the masked corpus, so it is drawn after build_masked_corpus.

Coders never see the severity. It is written only to the meta file.

Usage:
    python -m annotate.make_outcome_sample sentences --n 300 --seed 23
    python -m annotate.make_outcome_sample recoverability --n 100 --seed 29
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import _norm_severity, SEVERITY_ORDER  # noqa: E402
from leakage.outcome_lexicon import split_sentences  # noqa: E402

NARR = os.path.join("data", "interim", "narratives.jsonl")
MASKED = os.path.join("data", "interim", "narratives_masked_outcome.jsonl")
GOLD = os.path.join("data", "gold")
META = os.path.join(GOLD, "outcome_sample_meta.json")


def _pool(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype={"report_id": str})
    df = df[df["source"] == "sgo"].copy()
    df["sev"] = df["struct_severity"].map(_norm_severity)
    df = df[df["sev"].isin(SEVERITY_ORDER)]
    df["injured"] = df["sev"] != SEVERITY_ORDER[0]
    return df.reset_index(drop=True)


def _stratified(df: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """n rows, allocated proportionally across manufacturers (>=1 where possible)."""
    if n >= len(df):
        return df
    frac = n / len(df)
    picks = []
    for _, g in df.groupby(df["manufacturer"].fillna("unknown")):
        k = min(len(g), max(1, int(round(len(g) * frac))))
        picks.append(g.sample(n=k, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks)
    if len(out) > n:
        out = out.sample(n=n, random_state=int(rng.integers(1 << 31)))
    return out


def _balanced(df: pd.DataFrame, n: int, rng) -> tuple[pd.DataFrame, dict]:
    half = n // 2
    inj = _stratified(df[df["injured"]], half, rng)
    non = _stratified(df[~df["injured"]], n - len(inj), rng)
    weights = {"injured": float(df["injured"].sum() / len(inj)),
               "not_injured": float((~df["injured"]).sum() / len(non))}
    return pd.concat([inj, non]).sample(frac=1, random_state=int(rng.integers(1 << 31))), weights


def _load_meta() -> dict:
    if os.path.exists(META):
        with open(META) as f:
            return json.load(f)
    return {}


def _save_meta(meta: dict) -> None:
    with open(META, "w") as f:
        json.dump(meta, f, indent=2)


def sentences(n: int, seed: int, coders: list[str]) -> None:
    rng = np.random.default_rng(seed)
    pool = _pool(NARR)
    pool = pool[pool["narrative"].fillna("").str.split().str.len() >= 8]
    meta = _load_meta()
    exclude = set(meta.get("recoverability", {}).get("report_ids", []))
    pool = pool[~pool["report_id"].isin(exclude)]
    sample, weights = _balanced(pool, n, rng)

    rows = []
    for _, r in sample.iterrows():
        t = r["narrative"]
        for i, (s, e) in enumerate(split_sentences(t)):
            rows.append({"report_id": r["report_id"], "sent_idx": i,
                         "sentence": t[s:e], "outcome": "", "category": "",
                         "note": ""})
    out = pd.DataFrame(rows)
    os.makedirs(GOLD, exist_ok=True)
    for c in coders:
        p = os.path.join(GOLD, f"outcome_sent_{c}.csv")
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite a coding file")
        out.to_csv(p, index=False)
        print(f"[outcome-sample] wrote {p}: {sample.shape[0]} narratives, "
              f"{len(out)} sentences")
    meta["sentences"] = {
        "seed": seed, "n": int(len(sample)), "n_sentences": int(len(out)),
        "sampling_weights": weights,
        "records": {r["report_id"]: {"sev": r["sev"], "injured": bool(r["injured"]),
                                     "manufacturer": r["manufacturer"]}
                    for _, r in sample.iterrows()},
    }
    _save_meta(meta)


def recoverability(n: int, seed: int, coders: list[str]) -> None:
    if not os.path.exists(MASKED):
        raise SystemExit(f"{MASKED} not found; run leakage.build_masked_corpus first")
    rng = np.random.default_rng(seed)
    pool = _pool(MASKED)
    meta = _load_meta()
    exclude = set(meta.get("sentences", {}).get("records", {}).keys())
    pool = pool[~pool["report_id"].isin(exclude)]
    # Rows the mask emptied carry nothing to judge; they are counted separately.
    pool = pool[pool["narrative"].fillna("").str.strip() != ""]
    sample, weights = _balanced(pool, n, rng)
    out = pd.DataFrame({"report_id": sample["report_id"].values,
                        "masked_narrative": sample["narrative"].values,
                        "injured": "", "note": ""})
    for c in coders:
        p = os.path.join(GOLD, f"outcome_recover_{c}.csv")
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite a coding file")
        out.to_csv(p, index=False)
        print(f"[outcome-sample] wrote {p}: {len(out)} masked narratives")
    meta["recoverability"] = {
        "seed": seed, "n": int(len(sample)), "masked_corpus": MASKED,
        "sampling_weights": weights,
        "report_ids": sample["report_id"].tolist(),
        "records": {r["report_id"]: {"sev": r["sev"], "injured": bool(r["injured"])}
                    for _, r in sample.iterrows()},
    }
    _save_meta(meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["sentences", "recoverability"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--coders", nargs="+", default=["A", "B"])
    a = ap.parse_args()
    if a.task == "sentences":
        sentences(a.n or 300, a.seed or 23, a.coders)
    else:
        recoverability(a.n or 100, a.seed or 29, a.coders)
