"""Adjudicate annotator_A.jsonl vs annotator_B.jsonl into the final gold set.

Three-step workflow:

  1. `diff`     Compare A and B field-by-field. Fields where they agree are
                settled automatically. Fields where they disagree are written
                to data/gold/disagreements.jsonl, one row per (report_id,
                field) pair, with both values and the narrative for context, so
                a human (or a third adjudicator) can resolve each one without
                re-reading all 250 narratives from scratch.

  2. (human)    Open data/gold/disagreements.jsonl, fill in "resolved" for each
                row (the correct value -- typically A's, B's, or a third value
                if both coders missed something), save.

  3. `merge`    Combine the auto-agreed fields with the resolved disagreements
                into data/gold/gold.jsonl -- the final evaluation answer key
                used by score_agreement.py and transfer_eval.py.

Usage:
    python -m annotate.adjudicate diff
    # ... edit data/gold/disagreements.jsonl ...
    python -m annotate.adjudicate merge

Implementation note: annotation files are parsed with plain json.loads, not
pandas.read_json. pandas.read_json silently coerces any column mixing JSON
null with booleans (other_party_present, av_moving) into float64 -- True/False
become 1.0/0.0 and null becomes NaN rather than None -- which breaks both
equality comparisons here and, if written back out, the boolean schema itself.
Plain dicts preserve the original types exactly.
"""
from __future__ import annotations

import argparse
import json
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.schema import SCORED_FIELDS  # noqa: E402

GOLD_DIR = os.path.join("data", "gold")
ANN_A = os.path.join(GOLD_DIR, "annotator_A.jsonl")
ANN_B = os.path.join(GOLD_DIR, "annotator_B.jsonl")
DISAGREEMENTS = os.path.join(GOLD_DIR, "disagreements.jsonl")
GOLD_OUT = os.path.join(GOLD_DIR, "gold.jsonl")


def _load(path: str) -> dict:
    """Load a JSONL annotation file into {report_id: row_dict}, preserving
    exact JSON types (None stays None, true/false stay bool)."""
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["report_id"]] = r
    return rows


def diff(ann_a: str = ANN_A, ann_b: str = ANN_B,
         out_path: str = DISAGREEMENTS) -> dict:
    """Find every (report_id, field) where A and B disagree; write them for
    manual resolution. Returns a summary of agreement counts per field."""
    a, b = _load(ann_a), _load(ann_b)
    ids_a, ids_b = set(a), set(b)
    idx = sorted(ids_a & ids_b)
    if ids_a != ids_b:
        print(f"[adjudicate] WARNING: annotator files cover different "
              f"report_ids (A={len(ids_a)}, B={len(ids_b)}, "
              f"overlap={len(idx)}). Only the overlap will be adjudicated.")

    disagreements = []
    n_agree = {f: 0 for f in SCORED_FIELDS}
    n_total = {f: 0 for f in SCORED_FIELDS}
    for rid in idx:
        row_a, row_b = a[rid], b[rid]
        for field in SCORED_FIELDS:
            va, vb = row_a.get(field), row_b.get(field)
            if va is None and vb is None:
                continue  # both coders left it unknown -- not a disagreement
            n_total[field] += 1
            if va == vb:
                n_agree[field] += 1
                continue
            disagreements.append({
                "report_id": rid,
                "field": field,
                "narrative": row_a.get("narrative") or row_b.get("narrative") or "",
                "value_A": va,
                "value_B": vb,
                "resolved": None,  # <-- fill this in by hand
            })

    os.makedirs(GOLD_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        for d in disagreements:
            f.write(json.dumps(d) + "\n")

    print(f"[adjudicate] {len(disagreements)} disagreement(s) across "
          f"{len(idx)} narratives written to {out_path}")
    print("[adjudicate] agreement rate by field:")
    summary = {}
    for field in SCORED_FIELDS:
        tot = n_total[field]
        pct = round(100 * n_agree[field] / tot, 1) if tot else None
        summary[field] = {"agree": n_agree[field], "total": tot, "pct_agree": pct}
        label = f"{n_agree[field]:4d}/{tot:<4d} ({pct}%)" if tot else "n/a"
        print(f"    {field:28s} {label}")
    print(f"\n[adjudicate] Next: open {out_path}, fill in \"resolved\" for "
          f"each row, then run:\n    python -m annotate.adjudicate merge")
    return summary


def merge(ann_a: str = ANN_A, ann_b: str = ANN_B,
          disagreements_path: str = DISAGREEMENTS,
          out_path: str = GOLD_OUT) -> int:
    """Combine auto-agreed fields with resolved disagreements into gold.jsonl."""
    a, b = _load(ann_a), _load(ann_b)
    idx = sorted(set(a) & set(b))

    if not os.path.exists(disagreements_path):
        raise SystemExit(f"[adjudicate] {disagreements_path} not found. "
                         f"Run 'diff' first.")
    resolved_rows = []
    with open(disagreements_path) as f:
        for line in f:
            line = line.strip()
            if line:
                resolved_rows.append(json.loads(line))

    unresolved = [r for r in resolved_rows if r.get("resolved") is None]
    if unresolved:
        u = unresolved[0]
        raise SystemExit(
            f"[adjudicate] {len(unresolved)} disagreement(s) still have "
            f"\"resolved\": null in {disagreements_path}. Fill in every row "
            f"before merging -- e.g. the first unresolved one:\n"
            f"  report_id={u['report_id']!r} field={u['field']!r} "
            f"(A={u['value_A']!r}, B={u['value_B']!r})"
        )
    res_lookup = {(r["report_id"], r["field"]): r["resolved"] for r in resolved_rows}

    gold_rows = []
    for rid in idx:
        row_a = a[rid]
        gold = {"report_id": rid}
        for col in ("source", "narrative"):
            if col in row_a:
                gold[col] = row_a[col]
        for field in SCORED_FIELDS:
            key = (rid, field)
            gold[field] = res_lookup[key] if key in res_lookup else row_a.get(field)
        gold_rows.append(gold)

    os.makedirs(GOLD_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        for g in gold_rows:
            f.write(json.dumps(g) + "\n")
    print(f"[adjudicate] wrote {len(gold_rows)} adjudicated gold rows to "
          f"{out_path}")
    return len(gold_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["diff", "merge"])
    ap.add_argument("--ann-a", default=ANN_A)
    ap.add_argument("--ann-b", default=ANN_B)
    ap.add_argument("--disagreements", default=DISAGREEMENTS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.command == "diff":
        diff(args.ann_a, args.ann_b, args.disagreements)
    else:
        merge(args.ann_a, args.ann_b, args.disagreements,
              args.out or GOLD_OUT)
