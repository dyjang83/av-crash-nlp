"""Draw a stratified sample of narratives for human gold annotation.

Stratifies by (source, manufacturer, narrative-length tercile) so the gold set
spans short/long narratives and both data sources, then writes a blank
annotation template per record for each annotator.

Usage:
    python -m annotate.make_gold_sample --n 250 --seed 11
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

INTERIM = os.path.join("data", "interim", "narratives.jsonl")
GOLD_DIR = os.path.join("data", "gold")

BLANK = {
    "collision_type": None, "subject_pre_crash_maneuver": None,
    "other_party_present": None, "contributory_party": None,
    "engagement_state": None, "lighting": None, "weather": None,
    "road_class": None, "locality": None, "narrative_injury_severity": None,
    "av_moving": None, "contributory_evidence": None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--annotators", nargs="+", default=["A", "B"])
    args = ap.parse_args()

    df = pd.read_json(INTERIM, lines=True)
    df = df[df["narrative"].str.split().str.len() >= 8].reset_index(drop=True)
    df["len_tercile"] = pd.qcut(df["narrative"].str.len(), 3,
                                labels=["short", "mid", "long"], duplicates="drop")
    df["stratum"] = (df["source"].astype(str) + "|" +
                     df["manufacturer"].astype(str) + "|" +
                     df["len_tercile"].astype(str))

    rng = np.random.default_rng(args.seed)
    # Proportional allocation across strata, at least 1 per stratum where possible.
    frac = args.n / len(df)
    picks = []
    for _, g in df.groupby("stratum"):
        k = max(1, int(round(len(g) * frac)))
        k = min(k, len(g))
        picks.append(g.sample(n=k, random_state=int(rng.integers(1 << 31))))
    sample = pd.concat(picks).drop_duplicates("report_id")
    if len(sample) > args.n:
        sample = sample.sample(n=args.n, random_state=args.seed)

    os.makedirs(GOLD_DIR, exist_ok=True)
    sample[["report_id", "source", "manufacturer", "narrative"]].to_json(
        os.path.join(GOLD_DIR, "sample.jsonl"), orient="records", lines=True)

    for ann in args.annotators:
        path = os.path.join(GOLD_DIR, f"annotator_{ann}.jsonl")
        with open(path, "w") as f:
            for _, r in sample.iterrows():
                row = {"report_id": r["report_id"], "source": r["source"],
                       "narrative": r["narrative"], **BLANK}
                f.write(json.dumps(row) + "\n")
        print(f"[gold] wrote blank template {path} ({len(sample)} rows)")

    print(f"[gold] sampled {len(sample)} narratives across "
          f"{sample['stratum'].nunique()} strata")


if __name__ == "__main__":
    main()
