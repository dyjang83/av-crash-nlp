"""LLM detector for outcome statements: the second half of the masking union.

The lexicon (leakage.outcome_lexicon) is transparent but can only find words it
lists. This tagger asks the extraction model to copy out, verbatim, every span
that states or implies the crash's outcome, labelled with the same O1-O5
categories. build_masked_corpus deletes any sentence either detector flags, and
annotate.score_outcome_mask measures both against the human coders.

The tagger never sees the structured severity; it sees only the narrative.

Usage:
    python -m leakage.tag_outcome_spans                     # SGO, Batch API
    python -m leakage.tag_outcome_spans --limit 20          # small dry run
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract.batch import run_batch, tool_input  # noqa: E402
from leakage.outcome_lexicon import span_found  # noqa: E402

NARR = os.path.join("data", "interim", "narratives.jsonl")
OUT = os.path.join("data", "processed", "outcome_spans.jsonl")
STATE = os.path.join("data", "processed", "batches")
TOOL = "record_outcome_spans"

SYSTEM = """You audit autonomous-vehicle crash narratives for statements that \
reveal the OUTCOME of the crash, so they can be removed before the narrative is \
used to predict injury severity. Copy out every such span verbatim.

Outcome categories:
O1 injury status: any statement about whether anyone was injured, hurt, in pain, \
or complained of symptoms. INCLUDE absence statements ("no injuries were \
reported", "the passenger was uninjured", "no one was hurt").
O2 medical response: EMS/ambulance/paramedics, transport to or from care, \
hospital, treatment, medical evaluation, declining or refusing medical care.
O3 fatality: death, fatal, killed, pronounced dead.
O4 post-crash response proxies: police or fire attendance, first responders, \
claims, claimants, allegations, attorneys, citations.
O5 reportability statements: sentences explaining why or under which provision \
the crash is reported (e.g. "reporting this crash under Request No. 1 of \
Standing General Order 2021-01 because ..."), or that mention the injury \
severity field.

Do NOT tag crash context, even when it correlates with severity: speeds, \
maneuvers, what struck what, point of impact, vulnerable road users being \
involved or struck, vehicle damage, towing, airbag deployment, weather, \
lighting, road layout.

Rules:
- Each span must be an exact, contiguous copy of the narrative text (same words, \
same order). Prefer the whole clause that carries the outcome.
- If the narrative contains no outcome statement, return an empty list.
- Redaction markers such as [XXX] or [REDACTED ...] are not outcome statements."""

SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "Exact verbatim span from the narrative."},
                    "category": {"type": "string",
                                 "enum": ["O1", "O2", "O3", "O4", "O5"]},
                },
                "required": ["text", "category"],
            },
        }
    },
    "required": ["spans"],
}


_PAIR = re.compile(r'"text"\s*:\s*"(.*?)"\s*,\s*"category"\s*:\s*"(O[1-5])"', re.S)


def coerce_spans(raw) -> list[dict]:
    """Normalize the tool's `spans` to a list of {text, category} dicts.

    A small share of responses return `spans` as a JSON-encoded STRING rather
    than an array, sometimes with unescaped inner quotes, so json.loads fails
    on them. Those are recovered with a lenient text/category pair match rather
    than dropped: dropping them would silently remove the tagger's vote for
    exactly the records whose narratives are hardest to parse.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [{"text": t, "category": c} for t, c in _PAIR.findall(raw)]
    out = []
    for sp in raw or []:
        if isinstance(sp, str):
            out.append({"text": sp, "category": None})
        elif isinstance(sp, dict):
            out.append({"text": str(sp.get("text", "")), "category": sp.get("category")})
    return out


def build_params(narrative: str, model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 2048,
        "system": [{"type": "text", "text": SYSTEM,
                    "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": TOOL,
                   "description": "Record every verbatim outcome span.",
                   "input_schema": SCHEMA}],
        "tool_choice": {"type": "tool", "name": TOOL},
        "messages": [{"role": "user", "content":
                      f'Narrative:\n"""\n{narrative.strip()}\n"""'}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narratives", default=NARR)
    ap.add_argument("--output", default=OUT)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--source", default="sgo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default=None,
                    help="Batch state name (default derives from model/limit).")
    a = ap.parse_args()

    df = pd.read_json(a.narratives, lines=True)
    df = df[df["source"] == a.source]
    df = df[df["narrative"].fillna("").str.strip() != ""]
    if a.limit:
        df = df.head(a.limit)
    name = a.name or f"outcome_spans_{a.source}_{a.model}" + (
        f"_n{a.limit}" if a.limit else "")

    texts = dict(zip(df["report_id"].astype(str), df["narrative"]))
    res = run_batch(name, [(rid, build_params(t, a.model))
                           for rid, t in texts.items()], STATE)

    n_span = n_unaligned = n_fail = 0
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as f:
        for rid, o in res.items():
            row = {"report_id": rid, "source": a.source, "model": a.model,
                   "ok": False, "spans": None, "error": None}
            if o["ok"]:
                try:
                    spans = coerce_spans(tool_input(o["message"], TOOL).get("spans"))
                    for sp in spans:
                        sp["aligned"] = span_found(texts[rid], sp["text"])
                    row.update(ok=True, spans=spans)
                    n_span += len(spans)
                    n_unaligned += sum(not s["aligned"] for s in spans)
                except ValueError as e:
                    row["error"] = str(e)
            else:
                row["error"] = o["error"]
            n_fail += not row["ok"]
            f.write(json.dumps(row) + "\n")
    print(f"[tagger] {len(res)} records, {n_fail} failed, {n_span} spans "
          f"({n_unaligned} not found verbatim in the text) -> {a.output}")


if __name__ == "__main__":
    main()
