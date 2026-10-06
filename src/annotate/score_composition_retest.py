"""Test-retest stability of the composition extraction across two runs.

WHAT THIS MEASURES, AND WHY IT IS ALMOST FREE. The reportability rubric was
revised after its first version produced a 99.6% reportable share against
IIHS's 21.6%, and the corpus was re-extracted. The revision changed the
REPORTABILITY section of the system prompt and nothing else -- so for every
field except `police_reportable` and its two companions, the two runs are the
same model, at temperature 0, on the same narrative, under a prompt that
differs only in a section about a different field.

Those fields should therefore agree almost perfectly. To the extent they do
not, the disagreement is a lower bound on the extraction's own instability, and
it is instability of a particularly awkward kind: caused by edits to an
unrelated part of the prompt. A field that moves when a distant paragraph
changes is a field whose value depends on prompt phrasing rather than on the
narrative, and no distant-supervision accuracy figure reveals that -- a coder
can be stably 85% accurate or unstably 85% accurate, and only a retest
separates them.

WHAT IT IS NOT. Not a validity measure. Two runs can agree perfectly and both
be wrong; `schema.composition_distant` is where correctness is measured. This
answers the narrower question of whether the number would come out the same
tomorrow.

THE EXPECTED EXCEPTION. `police_reportable`, `reportable_evidence` and
`reportable_confidence` SHOULD move -- that was the point of the revision -- and
they are reported separately as the intended change rather than pooled into a
stability figure they would drag down.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.composition_schema import SCORED_FIELDS  # noqa: E402

# Fields the revision deliberately targeted. Reported apart from the stability
# statistic, because a change here is the intended effect, not drift.
REVISED_FIELDS = ["police_reportable", "reportable_evidence",
                  "reportable_confidence"]


def load(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") and r.get("extraction") is not None:
                out[str(r["report_id"])] = r["extraction"]
    return out


def compare(a: dict, b: dict, fields: Optional[list] = None) -> dict:
    """Per-field agreement over the report_ids both runs extracted."""
    shared = sorted(set(a) & set(b))
    fields = fields or [f for f in SCORED_FIELDS if f not in REVISED_FIELDS]
    rows = []
    for f in fields:
        n = agree = 0
        moved: dict[str, int] = {}
        for rid in shared:
            va, vb = a[rid].get(f), b[rid].get(f)
            if va is None or vb is None:
                continue
            n += 1
            if va == vb:
                agree += 1
            else:
                moved[f"{va} -> {vb}"] = moved.get(f"{va} -> {vb}", 0) + 1
        rows.append({
            "field": f, "n": n,
            "agreement": (agree / n) if n else None,
            "top_transitions": dict(sorted(moved.items(), key=lambda kv: -kv[1])[:3]),
        })
    return {"n_shared": len(shared), "n_only_a": len(set(a) - set(b)),
            "n_only_b": len(set(b) - set(a)), "fields": rows}


def revised_shift(a: dict, b: dict) -> dict:
    """How the reportability label actually moved. The intended change."""
    shared = sorted(set(a) & set(b))
    pairs = [(a[r].get("police_reportable"), b[r].get("police_reportable"))
             for r in shared]
    pairs = [(x, y) for x, y in pairs if x and y]
    if not pairs:
        return {"n": 0}
    df = pd.DataFrame(pairs, columns=["before", "after"])
    ct = pd.crosstab(df["before"], df["after"])
    return {
        "n": len(df),
        "before": df["before"].value_counts().to_dict(),
        "after": df["after"].value_counts().to_dict(),
        "transition_matrix": ct.to_dict(),
        "unchanged": float((df["before"] == df["after"]).mean()),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Test-retest stability across two composition extraction runs.")
    ap.add_argument("--run-a",
                    default=os.path.join("data", "processed",
                                         "composition_extractions.rubric_v1.jsonl"))
    ap.add_argument("--run-b",
                    default=os.path.join("data", "processed",
                                         "composition_extractions.jsonl"))
    ap.add_argument("--out", default=os.path.join("data", "processed",
                                                  "composition_retest.json"))
    a = ap.parse_args()

    A, B = load(a.run_a), load(a.run_b)
    rep = compare(A, B)
    rep["reportability_shift"] = revised_shift(A, B)
    rep["run_a"], rep["run_b"] = a.run_a, a.run_b

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    print(f"[retest] {rep['n_shared']:,} report_ids in both runs "
          f"({rep['n_only_a']:,} only in A, {rep['n_only_b']:,} only in B)")
    print("\n[retest] fields the rubric revision did NOT touch -- these should "
          "be near-identical:")
    print(f"{'field':28s} {'n':>6s} {'agreement':>10s}   top transition")
    for r in sorted(rep["fields"], key=lambda x: x["agreement"] or 0):
        tt = next(iter(r["top_transitions"].items()), ("--", 0))
        print(f"{r['field']:28s} {r['n']:6d} "
              f"{(r['agreement'] or 0):10.3f}   {tt[0]} ({tt[1]})")

    s = rep["reportability_shift"]
    if s.get("n"):
        print(f"\n[retest] police_reportable -- the INTENDED change (n={s['n']:,}, "
              f"{s['unchanged']:.1%} unchanged)")
        print(f"[retest]   before: {s['before']}")
        print(f"[retest]   after:  {s['after']}")
    print(f"\n[retest] wrote {a.out}")


if __name__ == "__main__":
    main()
