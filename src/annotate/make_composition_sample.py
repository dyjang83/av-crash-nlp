"""Draw the human double-coding sample for the composition layer.

WHAT THIS SAMPLE IS FOR. Six fields in `schema.composition_schema` have NO
structured counterpart anywhere in SGO -- `police_reportable`,
`damage_descriptor`, `sensor_only_damage`, `is_true_crash`,
`relation_to_junction`, `intersection_type`. Distant supervision cannot reach
them, so the only evidence about whether the extraction is right is two humans
coding the same narratives independently. `police_reportable` is the one that
matters most: every Tier B result in the paper rests on it, and the
LLM-versus-human agreement on it is the headline validity number.

WHY STRATIFIED, NOT RANDOM. A simple random sample of 2,353 ADS incidents is
dominated by minor rear-struck Waymo filings, because that is what the corpus
is. Coding 300 of those would establish agreement on the easy majority and say
nothing about the cells the analysis actually turns on. The sample is
therefore stratified on the three axes the outline names -- operator, crash
type, and filing regime -- with a floor per cell, so the rare cells are
represented well enough to estimate agreement in them.

THE COST OF STRATIFYING, AND HOW IT IS PAID BACK. A stratified sample's raw
agreement is not the corpus agreement: it over-weights rare strata. Every row
carries `stratum_weight`, the inverse of its sampling fraction, so
`score_composition_agreement` can report BOTH the unweighted agreement within
strata and a corpus-level estimate. Reporting only the raw number would
overstate or understate depending on which strata are hard, and there is no way
to know which without the weights.

BLINDING. The output carries the narrative and nothing else -- no model
prediction, no structured fields beyond what the coder needs to identify the
record. A coder who can see the model's answer is not independent of it, and
the agreement statistic would be meaningless. The model's predictions are
joined back in at scoring time, from a separate file.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.composition import attach_structured_key  # noqa: E402

DEFAULT_N = 300
OUT_DIR = os.path.join("data", "gold")


def _strata(d: pd.DataFrame) -> pd.Series:
    """operator x crash type x filing regime.

    Operator is collapsed to {waymo, zoox, other}: Waymo is 2,199 of 3,410 raw
    filings and would otherwise define every stratum on its own, while the long
    tail of one- and two-filing entities cannot support a stratum at all.
    Crash type uses the STRUCTURED key, not the extraction -- stratifying on
    the model's own output would correlate the sample with the thing being
    validated.
    """
    ent = d["Reporting Entity"].astype(str).str.lower()
    op = np.where(ent.str.contains("waymo|transdev"), "waymo",
                  np.where(ent.str.contains("zoox"), "zoox", "other"))
    cat = d["acc_category_struct"].fillna("unkeyed").astype(str)
    return pd.Series([f"{o}|{c}|{r}" for o, c, r in
                      zip(op, cat, d["filing_regime"].astype(str))], index=d.index)


def draw(d: pd.DataFrame, n: int = DEFAULT_N, seed: int = 11,
         min_per_cell: int = 3) -> pd.DataFrame:
    """Stratified draw with a per-cell floor, then proportional top-up.

    Two passes on purpose. The floor pass guarantees the rare cells are
    present; the proportional pass spends what is left in proportion to stratum
    size, so the sample is not so far from the corpus that the weights become
    extreme. Cells smaller than the floor contribute everything they have.
    """
    rng = np.random.default_rng(seed)
    d = d.copy()
    d["_stratum"] = _strata(d)
    sizes = d["_stratum"].value_counts()

    picked: list[int] = []
    # Pass 1: floor.
    for s, idx in d.groupby("_stratum").groups.items():
        idx = list(idx)
        take = min(min_per_cell, len(idx))
        picked += list(rng.choice(idx, take, replace=False))

    # Pass 2: proportional top-up on what remains.
    remaining = n - len(picked)
    if remaining > 0:
        pool = d.drop(index=picked)
        if len(pool):
            share = pool["_stratum"].map(sizes / sizes.sum()).fillna(0).to_numpy(float)
            share = share / share.sum() if share.sum() > 0 else None
            take = min(remaining, len(pool))
            picked += list(rng.choice(pool.index.to_numpy(), take,
                                      replace=False, p=share))

    samp = d.loc[picked].copy()
    # Sampling fraction per stratum, for the corpus-level estimate later.
    drawn = samp["_stratum"].value_counts()
    samp["stratum_size"] = samp["_stratum"].map(sizes).astype(int)
    samp["stratum_drawn"] = samp["_stratum"].map(drawn).astype(int)
    samp["stratum_weight"] = samp["stratum_size"] / samp["stratum_drawn"]
    return samp.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# The fields a human coder fills. Kept here rather than in the coding guide so
# the blank form and the scorer cannot drift apart.
CODED_FIELDS = [
    "police_reportable",        # yes / maybe / no
    "reportable_evidence",      # verbatim span
    "damage_descriptor",        # none/scratch_scuff/panel_dent/deformation/intrusion/sensor_only/unknown
    "sensor_only_damage",       # true / false
    "is_true_crash",            # true / false
    "relation_to_junction",     # intersection/intersection_related/driveway_access/non_junction/unknown
    "intersection_type",        # four_way/t_intersection/y_intersection/roundabout_traffic_circle/five_or_more/not_an_intersection/unknown
    "acc_type_category",        # the 11-category taxonomy
    "striking_role",            # striking/struck/undetermined/not_applicable
    "coder_notes",
]


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Draw the stratified human double-coding sample.")
    ap.add_argument("--incidents",
                    default=os.path.join("data", "interim", "ads_incidents.parquet"))
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--min-per-cell", type=int, default=3)
    ap.add_argument("--coders", nargs="+", default=["A", "B"])
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()

    d = attach_structured_key(pd.read_parquet(a.incidents))
    # Only rows with an extractable narrative: a human cannot code a redacted
    # filing any more than a model can, and including them would depress the
    # agreement statistic for a reason that has nothing to do with coding.
    d = d[d["Narrative"].astype(str).str.split().str.len() >= 8].copy()

    samp = draw(d, n=a.n, seed=a.seed, min_per_cell=a.min_per_cell)

    os.makedirs(a.out_dir, exist_ok=True)
    blank = {f: "" for f in CODED_FIELDS}
    for coder in a.coders:
        path = os.path.join(a.out_dir, f"composition_coder_{coder}.jsonl")
        with open(path, "w") as f:
            for _, r in samp.iterrows():
                f.write(json.dumps({
                    "report_id": str(r["Report ID"]),
                    "incident_key": str(r["incident_key"]),
                    "entity": str(r["Reporting Entity"]),
                    "filing_regime": str(r["filing_regime"]),
                    "stratum": str(r["_stratum"]),
                    "stratum_weight": float(r["stratum_weight"]),
                    "narrative": str(r["Narrative"]),
                    **blank,
                }) + "\n")
        print(f"[sample] wrote {path} ({len(samp)} rows, blank fields for coder {coder})")

    meta = {
        "n_drawn": int(len(samp)),
        "n_eligible": int(len(d)),
        "seed": a.seed, "min_per_cell": a.min_per_cell,
        "n_strata": int(samp["_stratum"].nunique()),
        "stratum_weight_range": [float(samp["stratum_weight"].min()),
                                 float(samp["stratum_weight"].max())],
        "by_regime": samp["filing_regime"].value_counts().to_dict(),
        "by_structured_category": samp["acc_category_struct"].fillna("unkeyed")
                                      .value_counts().to_dict(),
        "coded_fields": CODED_FIELDS,
        "blinding": "coder files carry the narrative only; model predictions "
                    "are joined at scoring time",
    }
    mpath = os.path.join(a.out_dir, "composition_sample_meta.json")
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[sample] {meta['n_drawn']} of {meta['n_eligible']:,} eligible, "
          f"across {meta['n_strata']} strata")
    print(f"[sample] stratum weights span "
          f"{meta['stratum_weight_range'][0]:.1f}-{meta['stratum_weight_range'][1]:.1f}x")
    print(f"[sample] by regime: {meta['by_regime']}")
    print(f"[sample] by structured crash type: {meta['by_structured_category']}")
    print(f"[sample] wrote {mpath}")


if __name__ == "__main__":
    main()
