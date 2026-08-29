"""Build the three feature representations for the severity lift test.

  S   = structured SGO fields only (the signal insurers already collect)
  T   = text-derived: LLM-extracted schema fields (one-hot) + narrative embedding
  S+T = concatenation of S and T

Target: SGO 'Highest Injury Severity Alleged' (struct_severity), an ORDINAL field
that is independent of the narrative text. The lift test runs on the SGO corpus,
because each SGO record carries both a structured outcome and a narrative.

Leakage discipline:
  - contributory_party and narrative_injury_severity are DROPPED from T. The first
    is circular w.r.t. fault; the second is a text-derived restatement of the
    target. Including either would leak the outcome. See DROP_FROM_T.

Narrative embedding backend:
  - 'tfidf' (default): fully offline, deterministic, reproducible.
  - 'sbert': sentence-transformers if installed and weights available.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.schema import CATEGORICAL_FIELDS, BOOLEAN_FIELDS  # noqa: E402

# Severity ordinal map (adapt strings to the exact SGO categories at load time).
SEVERITY_ORDER = ["No Injuries Reported", "Minor", "Moderate", "Serious", "Fatality"]

# Extracted fields excluded from T to prevent target/fault leakage.
DROP_FROM_T = {"contributory_party", "narrative_injury_severity"}

STRUCT_COLS = ["manufacturer", "struct_engaged", "struct_towed", "struct_airbag"]


@dataclass
class Dataset:
    X: dict              # {"S": csr, "T": csr, "S+T": csr}
    y: np.ndarray        # ordinal severity (int)
    groups: np.ndarray   # grouping key (manufacturer) for grouped CV
    feature_names: dict


def _norm_severity(s: str) -> str:
    if not isinstance(s, str):
        return "unknown"
    t = s.strip().lower()
    if "fatal" in t:
        return "Fatality"
    if "serious" in t:
        return "Serious"
    if "moder" in t:
        return "Moderate"
    if "minor" in t:
        return "Minor"
    if "no inj" in t or t in {"none", "no"}:
        return "No Injuries Reported"
    return "unknown"


def load(narratives_jsonl: str, extractions_jsonl: str,
         embed: str = "tfidf", max_features: int = 4000) -> Dataset:
    narr = pd.read_json(narratives_jsonl, lines=True)
    narr = narr[narr["source"] == "sgo"].copy()  # lift test runs on SGO

    ext = pd.read_json(extractions_jsonl, lines=True)
    ext = ext[ext["ok"]].copy()
    # Use the closed model's extraction as the canonical T (configurable upstream).
    flat = pd.json_normalize(ext["extraction"])
    flat["report_id"] = ext["report_id"].values
    flat = flat.drop_duplicates("report_id")

    df = narr.merge(flat, on="report_id", how="inner", suffixes=("", "_ext"))
    df["sev"] = df["struct_severity"].map(_norm_severity)
    df = df[df["sev"].isin(SEVERITY_ORDER)].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("[features] no SGO rows with usable severity + extraction.")

    y = np.array([SEVERITY_ORDER.index(s) for s in df["sev"]])
    groups = df["manufacturer"].fillna("unknown").astype(str).values

    # ---- S: structured fields ----
    s_cols = [c for c in STRUCT_COLS if c in df.columns]
    ohe_s = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    S = ohe_s.fit_transform(df[s_cols].fillna("missing").astype(str))

    # ---- T: extracted schema fields (minus leakage) + narrative embedding ----
    t_cat = [c for c in (CATEGORICAL_FIELDS + BOOLEAN_FIELDS)
             if c in df.columns and c not in DROP_FROM_T]
    ohe_t = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    T_struct = ohe_t.fit_transform(df[t_cat].fillna("missing").astype(str))

    if embed == "sbert":
        emb = _sbert(df["narrative"].tolist())
        T_text = sparse.csr_matrix(emb)
        text_names = [f"sbert_{i}" for i in range(emb.shape[1])]
    else:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                              min_df=3, stop_words="english", sublinear_tf=True)
        T_text = vec.fit_transform(df["narrative"])
        text_names = [f"tfidf::{t}" for t in vec.get_feature_names_out()]

    T = sparse.hstack([T_struct, T_text]).tocsr()
    ST = sparse.hstack([S, T]).tocsr()

    names = {
        "S": list(ohe_s.get_feature_names_out(s_cols)),
        "T": list(ohe_t.get_feature_names_out(t_cat)) + text_names,
    }
    names["S+T"] = names["S"] + names["T"]

    print(f"[features] n={len(df)}  S={S.shape[1]}  T={T.shape[1]}  "
          f"S+T={ST.shape[1]}  classes={np.bincount(y).tolist()}")
    return Dataset(X={"S": S.tocsr(), "T": T, "S+T": ST}, y=y, groups=groups,
                   feature_names=names)


def _sbert(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(model.encode(texts, batch_size=64, show_progress_bar=True))


if __name__ == "__main__":
    ds = load(os.path.join("data", "interim", "narratives.jsonl"),
              os.path.join("data", "processed", "extractions.jsonl"))
    print("built feature sets:", {k: v.shape for k, v in ds.X.items()})
