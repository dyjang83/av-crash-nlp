"""Write outcome-masked copies of the narrative corpus.

Output: data/interim/narratives_masked_{tier}.jsonl -- the same rows and columns
as narratives.jsonl (so features.build_features.load() consumes it unchanged),
with `narrative` replaced by the masked text and the mask bookkeeping added as
mask_* columns. Only the lift-test source (SGO) is masked; other rows are
dropped, since nothing downstream of this file reads them.

Every record is kept, including those the mask empties: dropping them would
make the analysis sample depend on the outcome.

Usage:
    python -m leakage.build_masked_corpus                  # lexicon + LLM spans
    python -m leakage.build_masked_corpus --no-llm         # lexicon only
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leakage.outcome_lexicon import (TIERS, mask, residual_hits,  # noqa: E402
                                     write_lexicon_table)

NARR = os.path.join("data", "interim", "narratives.jsonl")
SPANS = os.path.join("data", "processed", "outcome_spans.jsonl")
OUT_TMPL = os.path.join("data", "interim", "narratives_masked_{tier}.jsonl")
STATS = os.path.join("data", "processed", "mask_stats.json")


def load_spans(path: str) -> dict[str, list[dict]]:
    spans: dict[str, list[dict]] = {}
    if not os.path.exists(path):
        return spans
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok"):
                spans[str(r["report_id"])] = r["spans"] or []
    return spans


def build(narratives: str = NARR, spans_path: str = SPANS, use_llm: bool = True,
          tiers=tuple(TIERS), source: str = "sgo", suffix: str = "") -> dict:
    df = pd.read_json(narratives, lines=True, dtype={"report_id": str})
    df = df[df["source"] == source].reset_index(drop=True)
    spans = load_spans(spans_path) if use_llm else {}
    if use_llm:
        missing = int((~df["report_id"].isin(spans.keys())
                       & df["narrative"].fillna("").str.strip().ne("")).sum())
        if missing:
            # Not fatal -- the lexicon still applies to those rows -- but it must
            # be visible, because the union's recall claim does not cover them.
            print(f"[mask] WARNING: {missing} non-empty narratives have no LLM "
                  f"tagging result; they are masked by the lexicon only")

    stats = {"n": int(len(df)), "use_llm": use_llm, "tiers": {}}
    for tier in tiers:
        rows = []
        for _, r in df.iterrows():
            sp = spans.get(str(r["report_id"])) if use_llm else None
            res = mask(r["narrative"] or "", tier,
                       llm_spans=[s["text"] for s in sp] if sp is not None else None,
                       llm_categories=[s["category"] for s in sp] if sp is not None else None)
            assert residual_hits(res.text, tier) == 0, r["report_id"]
            d = r.to_dict()
            d.update(narrative=res.text,
                     mask_n_sentences=res.n_sentences,
                     mask_n_removed=len(res.removed),
                     mask_n_removed_lexicon=len(res.removed_lexicon),
                     mask_n_removed_llm=len(res.removed_llm),
                     mask_llm_only=len(set(res.removed_llm) - set(res.removed_lexicon)),
                     mask_truncated_after=res.truncated_after,
                     mask_categories=res.categories,
                     mask_chars_removed=res.chars_removed,
                     mask_orig_chars=len(r["narrative"] or ""))
            rows.append(d)
        out = pd.DataFrame(rows)
        path = OUT_TMPL.format(tier=tier + suffix)
        out.to_json(path, orient="records", lines=True)
        stats["tiers"][tier] = {
            "path": path,
            "share_records_masked": float((out["mask_n_removed"] > 0).mean()),
            "share_chars_removed": float(out["mask_chars_removed"].sum()
                                         / max(out["mask_orig_chars"].sum(), 1)),
            "sentences_removed": int(out["mask_n_removed"].sum()),
            "sentences_total": int(out["mask_n_sentences"].sum()),
            "sentences_llm_only": int(out["mask_llm_only"].sum()),
            "n_emptied": int((out["narrative"].str.strip() == "").sum()),
            "by_category": pd.Series([c for cs in out["mask_categories"] for c in cs])
                             .value_counts().to_dict(),
        }
        print(f"[mask] {tier}: {stats['tiers'][tier]}")
    with open(STATS.replace(".json", f"{suffix}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    write_lexicon_table(os.path.join("paper", "tables", "tab_lexicon.tex"))
    _stats_table(stats, os.path.join("paper", "tables", f"tab_mask_stats{suffix}.tex"))
    return stats


def _stats_table(stats: dict, path: str) -> None:
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Tier & Records touched & Sentences removed & of which LLM-only & "
         r"Characters removed & Emptied \\", r"\midrule"]
    for tier, t in stats["tiers"].items():
        L.append(f"{tier} & {t['share_records_masked']:.1%} & "
                 f"{t['sentences_removed']:,} / {t['sentences_total']:,} & "
                 f"{t['sentences_llm_only']:,} & {t['share_chars_removed']:.1%} & "
                 f"{t['n_emptied']} \\\\".replace("%", r"\%"))
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--narratives", default=NARR)
    ap.add_argument("--spans", default=SPANS)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--suffix", default="",
                    help="Appended to output names, e.g. _lexonly.")
    a = ap.parse_args()
    build(a.narratives, a.spans, use_llm=not a.no_llm, suffix=a.suffix)
