"""CSV coder sheets for the composition census and the dedup audit.

Coders work in a spreadsheet: one row per crash (or per pair of filings), one
column per coded field. This module is the only place that knows the sheet
layout, so the census builder, the checker, the scorer and the adjudicator
cannot disagree about it.

LAYOUT. Census sheets hold every batch in one file, with the union of the
fields any batch codes. A cell a row does not code holds `n/a` and must be left
alone; a blank cell is one still to be coded. Allowed values are listed in
`coding_codes.csv`, written next to the sheets.

SPREADSHEET HAZARDS, HANDLED ON READ. Excel and Numbers rewrite true/false as
TRUE/FALSE and may save in a legacy encoding; values are compared
case-insensitively after stripping, and the reader tries UTF-8 (with or without
BOM), then cp1252, then Mac Roman.

Usage:
    python -m annotate.census_csv check data/gold/census_coder_A.csv
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NA = "n/a"
ID_COLS = ["report_id", "batch", "entity", "narrative"]
# Order of the coded columns: the window fields first, in the order a coder
# reads a narrative, then the corpus-sample-only fields, then notes.
CENSUS_FIELD_COLS = ["acc_type_category", "striking_role", "police_reportable",
                     "reportable_evidence", "is_true_crash", "on_public_road",
                     "ads_engaged_at_impact", "damage_descriptor",
                     "sensor_only_damage", "relation_to_junction",
                     "intersection_type", "coder_notes"]
DEDUP_INFO_COLS = ["report_id", "batch", "month", "city", "state",
                   "entity_i", "report_i", "time_i", "partner_i", "narrative_i",
                   "entity_j", "report_j", "time_j", "partner_j", "narrative_j"]
DEDUP_FIELD_COLS = ["verdict", "coder_notes"]
FREE_TEXT = {"reportable_evidence", "coder_notes"}
DISAGREEMENT_COLS = ["report_id", "field", "value_A", "value_B", "resolved",
                     "narrative"]
_ENCODINGS = ["utf-8-sig", "cp1252", "mac_roman"]


def _write(path: str, cols: list[str], rows: list[dict]) -> None:
    # utf-8-sig: the BOM makes Excel open the file as UTF-8 instead of guessing.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


def write_census(rows: list[dict], path: str) -> None:
    """Census rows (from make_composition_census.build) -> coder sheet."""
    out = []
    for r in rows:
        own = set(r["fields"])
        out.append({**{c: r.get(c) for c in ID_COLS},
                    **{c: ("" if c in own else NA) for c in CENSUS_FIELD_COLS}})
    _write(path, ID_COLS + CENSUS_FIELD_COLS, out)


def write_dedup(rows: list[dict], path: str) -> None:
    _write(path, DEDUP_INFO_COLS + DEDUP_FIELD_COLS,
           [{**r, "verdict": "", "coder_notes": ""} for r in rows])


def write_codes(allowed: dict[str, list[str]], path: str) -> None:
    """One row per coded field: what may go in the cell."""
    rows = [{"field": f, "allowed_values": "; ".join(v)} for f, v in allowed.items()]
    rows += [{"field": f, "allowed_values": "free text (optional)"} for f in sorted(FREE_TEXT)]
    _write(path, ["field", "allowed_values"], rows)


def _read_text(path: str) -> str:
    raw = open(path, "rb").read()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"[csv] cannot decode {path}; save it as 'CSV UTF-8'")


def read_csv(path: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(_read_text(path), newline="")))


def _norm_cell(v) -> str:
    return "" if v is None else str(v).strip()


def read_rows(path: str) -> dict[str, dict]:
    """report_id -> row, for a coder sheet (.csv) or a JSONL file.

    For sheets, `fields` is rebuilt from the cells: every coded column that is
    not `n/a`. Labels are lower-cased (TRUE -> true); free text is kept as
    written. `batch` becomes an int.
    """
    if not path.endswith(".csv"):
        rows = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[str(r["report_id"])] = r
        return rows
    rows = {}
    for r in read_csv(path):
        rid = _norm_cell(r.get("report_id"))
        if not rid:
            continue
        coded = [c for c in dict.fromkeys(CENSUS_FIELD_COLS + DEDUP_FIELD_COLS)
                 if c in r and _norm_cell(r[c]).lower() != NA]
        out = {k: _norm_cell(v) for k, v in r.items() if k is not None}
        for c in coded:
            if c not in FREE_TEXT:
                out[c] = out[c].lower()
        out["fields"] = coded
        b = out.get("batch", "")
        out["batch"] = int(float(b)) if b not in ("", None) else None
        rows[rid] = out
    return rows


def check(path: str, allowed: dict[str, list[str]]) -> dict:
    """Invalid cells and progress per batch. Invalid = not an allowed value."""
    rows = read_rows(path)
    bad, progress = [], {}
    for rid, r in rows.items():
        req = [c for c in r["fields"] if c not in FREE_TEXT]
        done = all(r.get(c) for c in req)
        p = progress.setdefault(r.get("batch"), [0, 0])
        p[1] += 1
        p[0] += int(done)
        for c in req:
            v = r.get(c)
            if v and v not in allowed.get(c, []):
                bad.append({"report_id": rid, "column": c, "value": v})
    return {"n_rows": len(rows), "invalid": bad,
            "progress": {str(k): {"complete": v[0], "rows": v[1]}
                         for k, v in sorted(progress.items(), key=lambda kv: str(kv[0]))}}


def write_disagreements(rows: list[dict], path: str) -> None:
    _write(path, DISAGREEMENT_COLS, rows)


def read_disagreements(path: str) -> list[dict]:
    """Resolved values keep their spelling but are compared lower-cased."""
    if not path.endswith(".csv"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    out = []
    for r in read_csv(path):
        res = _norm_cell(r.get("resolved"))
        out.append({"report_id": _norm_cell(r.get("report_id")),
                    "field": _norm_cell(r.get("field")),
                    "value_A": _norm_cell(r.get("value_A")),
                    "value_B": _norm_cell(r.get("value_B")),
                    "resolved": res.lower() if res else None})
    return out


def main():
    import argparse
    from annotate.make_composition_census import ALLOWED
    ap = argparse.ArgumentParser(description="Check a coder sheet.")
    ap.add_argument("command", choices=["check"])
    ap.add_argument("path")
    a = ap.parse_args()
    rep = check(a.path, ALLOWED)
    print(f"[csv] {a.path}: {rep['n_rows']} rows")
    for b, p in rep["progress"].items():
        print(f"  batch {b}: {p['complete']}/{p['rows']} complete")
    if rep["invalid"]:
        print(f"[csv] {len(rep['invalid'])} cell(s) hold a value that is not allowed "
              f"(see coding_codes.csv):")
        for x in rep["invalid"][:50]:
            print(f"  {x['report_id']:24s} {x['column']:22s} {x['value']!r}")
        raise SystemExit(1)
    print("[csv] every filled cell holds an allowed value")


if __name__ == "__main__":
    main()
