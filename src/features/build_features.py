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
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.schema import CATEGORICAL_FIELDS, BOOLEAN_FIELDS  # noqa: E402
from utils.config import canonical_model  # noqa: E402

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


def _token_prob_frame(ext: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Per-field decode-time probabilities, one column per field.

    Absent for backends that cannot expose logprobs (the closed tool-calling
    API) and for rows extracted before token probabilities were captured. Those
    become NaN, which the callers below must treat as "not measured" -- never as
    zero confidence, which would gate away every field the closed model produced.
    """
    rows = []
    for tp in ext["token_probs"]:
        d = {}
        if isinstance(tp, dict):
            for f, v in (tp.get("fields") or {}).items():
                if f in fields and not v.get("truncated"):
                    # p_renorm is the posterior over CATEGORIES; p_first stands in
                    # where renormalization saturates (single-token booleans).
                    p = v.get("p_renorm")
                    if p is None or p >= 1.0:
                        p = v.get("p_first", p)
                    d[f] = p
        rows.append(d)
    return pd.DataFrame(rows, columns=fields)


def load(narratives_jsonl: str, extractions_jsonl: str,
         embed: str = "tfidf", max_features: int = 4000,
         model: Optional[str] = None,
         confidence_gate: Optional[float] = None,
         confidence_features: bool = False) -> Dataset:
    narr = pd.read_json(narratives_jsonl, lines=True)
    narr = narr[narr["source"] == "sgo"].copy()  # lift test runs on SGO

    ext = pd.read_json(extractions_jsonl, lines=True)
    ext = ext[ext["ok"]].copy()

    # Select the canonical model BEFORE deduplicating. Without this filter,
    # drop_duplicates("report_id") keeps whichever row was appended first, so a
    # report_id extracted by several models resolves by file order rather than
    # by choice -- two SGO rows in the lift-test pool were silently taking
    # qwen2.5:7b's extraction instead of the closed model's. That is negligible
    # while the open models cover only the 250-record gold sample, but it
    # becomes the whole corpus the moment an open model is run corpus-wide, and
    # the reported lift would then depend on JSONL append order.
    model = model or canonical_model()
    available = sorted(ext["model"].unique().tolist())
    ext = ext[ext["model"] == model].copy()
    if len(ext) == 0:
        raise SystemExit(
            f"[features] no successful extractions for model={model!r} in "
            f"{extractions_jsonl}. Models present: {available}. "
            f"Pass model=... (or --extraction-model) to select one."
        )

    flat = pd.json_normalize(ext["extraction"])
    flat["report_id"] = ext["report_id"].values

    # Carry per-field token probabilities alongside the extracted values so the
    # gate can be applied per (record, field) rather than per record.
    tp_fields = [c for c in (CATEGORICAL_FIELDS + BOOLEAN_FIELDS)
                 if c not in DROP_FROM_T]
    if "token_probs" in ext.columns:
        tpf = _token_prob_frame(ext, tp_fields)
        for c in tp_fields:
            flat[f"__p__{c}"] = tpf[c].values

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

    # Cast to object BEFORE gating. other_party_present and av_moving are
    # boolean columns, and writing the string "unknown" into one currently
    # upcasts silently but is deprecated and will raise in a future pandas.
    # Everything downstream goes through .astype(str) anyway, so this only makes
    # the existing conversion explicit and early -- the same boolean-coercion
    # hazard _load_jsonl already guards against on the scoring side.
    t_vals = df[t_cat].astype(object).copy()
    n_gated = 0

    # Requesting a confidence gate or confidence features against extractions
    # that carry no token probabilities is a silent no-op: the run completes,
    # writes plausible numbers, and measures nothing. That already happened once
    # by pointing --extractions at the default file (which predates token-probability
    # capture), producing a "gating sweep" whose entire spread was seed noise.
    # Fail loudly instead.
    if confidence_gate is not None or confidence_features:
        p_cols = [f"__p__{c}" for c in t_cat if f"__p__{c}" in df.columns]
        n_meas = int(sum(pd.to_numeric(df[c], errors="coerce").notna().sum()
                         for c in p_cols)) if p_cols else 0
        if n_meas == 0:
            raise SystemExit(
                "[features] confidence gating/features requested, but no token "
                "probabilities are present in these extractions. Point "
                "--extractions at a run captured with logprobs (an open-weight "
                "backend); the closed tool-calling API cannot provide them."
            )

    if confidence_gate is not None:
        # Where the model's own decode-time probability for a field falls below
        # tau, replace the value with `unknown` -- the extractor abstaining on
        # its own low-confidence cells rather than a downstream filter guessing
        # which ones to trust. A NaN probability means the field was never
        # measured (no logprobs for this backend), and is LEFT ALONE: gating it
        # would silently blank every field of every closed-model extraction.
        for c in t_cat:
            col = f"__p__{c}"
            if col not in df.columns:
                continue
            p = pd.to_numeric(df[col], errors="coerce")
            mask = p.notna() & (p < confidence_gate)
            n_gated += int(mask.sum())
            t_vals.loc[mask, c] = "unknown"
        print(f"[features] confidence gate tau={confidence_gate}: "
              f"{n_gated} of {len(df) * len(t_cat)} field-cells set to unknown "
              f"({n_gated / max(len(df) * len(t_cat), 1):.1%})")

    ohe_t = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    # .where rather than .fillna: on the object-dtype frame above, fillna
    # triggers pandas' deprecated silent downcasting. .where fills the same
    # cells without changing dtype.
    T_struct = ohe_t.fit_transform(
        t_vals.where(t_vals.notna(), "missing").astype(str))

    if embed == "sbert":
        emb = _sbert(df["narrative"].tolist())
        T_text = sparse.csr_matrix(emb)
        text_names = [f"sbert_{i}" for i in range(emb.shape[1])]
    else:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                              min_df=3, stop_words="english", sublinear_tf=True)
        T_text = vec.fit_transform(df["narrative"])
        text_names = [f"tfidf::{t}" for t in vec.get_feature_names_out()]

    t_blocks = [T_struct, T_text]
    conf_names: list[str] = []
    if confidence_features:
        # The alternative to gating: hand the model the probabilities themselves
        # and let it decide what to do with them. Unmeasured fields get 0.5 --
        # maximally uninformative -- rather than 0, which would read as certainty
        # that the extraction is wrong.
        cols = [f"__p__{c}" for c in t_cat if f"__p__{c}" in df.columns]
        if cols:
            C = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.5).to_numpy()
            t_blocks.append(sparse.csr_matrix(C))
            conf_names = [f"conf::{c}" for c in t_cat if f"__p__{c}" in df.columns]
            print(f"[features] appended {len(conf_names)} confidence features")

    T = sparse.hstack(t_blocks).tocsr()
    ST = sparse.hstack([S, T]).tocsr()

    names = {
        "S": list(ohe_s.get_feature_names_out(s_cols)),
        "T": list(ohe_t.get_feature_names_out(t_cat)) + text_names + conf_names,
    }
    names["S+T"] = names["S"] + names["T"]

    print(f"[features] model={model}  n={len(df)}  S={S.shape[1]}  T={T.shape[1]}  "
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
