"""Validate the reportability rubric by controlled edits, with no human coding.

THE PROBLEM THIS SOLVES. `police_reportable` has no structured counterpart
anywhere in SGO, so there is no distant key for it, and the default answer is
human double-coding. Two cheaper checks already run without humans
(`report.reportability.police_gradient`, and the Tier-A-sufficiency
decomposition), but both are ONE-SIDED: they detect false negatives -- a
structured injury or airbag or named police agency where the rubric said "no" --
because a structured trigger being present is evidence the crash was
reportable. Nothing in SGO asserts that a crash was NOT reportable, so false
positives are invisible to them.

THE IDEA. Ground truth for an absolute label is expensive. Ground truth for the
DIRECTION a label must move under a controlled edit is free, because the edit
is constructed. Insert "a passenger complained of neck pain" into a narrative
the rubric called "no" and the label must become "yes" -- not because anyone
adjudicated the crash, but because the rubric says so in writing. Swap "Waymo"
for "Cruise" and the label must not move at all, because the rubric explicitly
forbids considering operator identity.

Each test therefore checks RUBRIC COMPLIANCE, which is the same thing human
double-coding measures, without needing a human to supply a label. And unlike
human coding it isolates the specific failure modes the first rubric version
actually exhibited: over-reading AV tows, over-reading undifferentiated damage,
and (untested until now) sensitivity to operator identity.

WHAT IT CANNOT DO. It cannot tell you whether the rubric's STANDARD is the
right standard -- whether "a reasonable person" really would report the crashes
it calls reportable. That is a question about the world, and the only external
anchors for it are IIHS's published marginals and human judgement. This
measures fidelity to a stated standard, not the merit of the standard.

-----------------------------------------------------------------------------
WHY THE EDITS ARE APPENDED, NOT SPLICED
-----------------------------------------------------------------------------
Every edit is a sentence appended to the narrative, or a whole-string
substitution. Nothing is spliced into the middle, and no existing sentence is
rewritten. Splicing would change the narrative's internal coherence and the
model could reasonably respond to the incoherence rather than to the inserted
fact, which would make a failed test uninterpretable. Appending keeps the
original text intact, so the only difference between the two conditions is the
added claim.

A consequence worth stating: an appended sentence is a slightly unnatural
narrative. The test measures whether the rubric responds to a stated fact, not
whether it would have spotted that fact in idiomatic prose.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

ORDER = {"no": 0, "maybe": 1, "yes": 2}


@dataclass
class Perturbation:
    """One controlled edit with a known-correct direction of effect.

    `direction` is what the label must do:
        "same" -- must not change. The rubric names this as a non-trigger.
        "up"   -- must move to a strictly higher level (and usually to "yes").
        "down" -- must move to a strictly lower level.
    """
    name: str
    direction: str
    apply: Callable[[str], Optional[str]]
    rationale: str
    # Which rubric clause this pins, for the write-up.
    clause: str


def _append(sentence: str) -> Callable[[str], Optional[str]]:
    def fn(narr: str) -> Optional[str]:
        return narr.rstrip() + " " + sentence
    return fn


def _swap_operator(narr: str) -> Optional[str]:
    """Rename the operator. Returns None when no known name is present."""
    subs = [(r"\bWaymo\b", "Cruise"), (r"\bZoox\b", "Waymo"),
            (r"\bCruise\b", "Zoox"), (r"\bMay Mobility\b", "Waymo")]
    for pat, rep in subs:
        if re.search(pat, narr):
            return re.sub(pat, rep, narr)
    return None


def _strip_damage(narr: str) -> Optional[str]:
    """Replace any damage statement with an explicit no-damage statement."""
    if not re.search(r"sustained damage|was damaged|damage to", narr, re.I):
        return None
    out = re.sub(r"[^.]*?(?:sustained damage|was damaged|damage to)[^.]*\.",
                 " Neither vehicle was damaged.", narr, flags=re.I)
    return out


def _remove_tow(narr: str) -> Optional[str]:
    """Delete any sentence mentioning a tow. Label must not move DOWN."""
    if not re.search(r"\btow", narr, re.I):
        return None
    return re.sub(r"[^.]*\btow[^.]*\.", " ", narr, flags=re.I)


# The five non-triggers the rubric names, plus the triggers it names. Each
# perturbation maps onto a clause, so a failure points at the text to fix.
PERTURBATIONS = [
    # --- invariance: the rubric's five explicit non-triggers ---------------
    Perturbation(
        "operator_identity", "same", _swap_operator,
        "Renaming the operating company cannot change whether a reasonable "
        "person would call the police.",
        "non-trigger 4: that the vehicle is autonomous, or which company operates it"),
    Perturbation(
        "add_av_tow", "same",
        _append("The subject vehicle was subsequently towed from the scene."),
        "A bare AV tow is fleet recovery and establishes nothing about damage. "
        "This is the failure mode that produced a 99.6% reportable share in the "
        "first rubric version.",
        "non-trigger 1: the subject AV being towed, on its own"),
    Perturbation(
        "add_sensor_damage", "same",
        _append("Damage was limited to the vehicle's lidar sensor housing."),
        "Exposed sensing hardware is expensive and irrelevant to reportability.",
        "non-trigger 3: damage confined to exposed sensing hardware"),
    Perturbation(
        "add_sgo_boilerplate", "same",
        _append("This report is submitted pursuant to Standing General Order "
                "2021-01."),
        "Every narrative in the corpus was filed; the filing duty is far broader "
        "than police reportability.",
        "non-trigger 5: that this report was filed with NHTSA"),
    Perturbation(
        "add_bare_damage", "same",
        _append("Both vehicles sustained damage."),
        "Undifferentiated damage boilerplate establishes only that contact "
        "occurred, which the reader already knows.",
        "non-trigger 2: an undifferentiated statement of damage"),
    Perturbation(
        "remove_tow", "same", _remove_tow,
        "Deleting a tow mention must not LOWER the label, because the tow was "
        "not supposed to be carrying it in the first place.",
        "non-trigger 1, tested in the other direction"),

    # --- sensitivity: the triggers the rubric names -----------------------
    Perturbation(
        "add_injury", "up",
        _append("A passenger in the subject vehicle complained of neck pain at "
                "the scene."),
        "Any complaint of pain is trigger (a).",
        "trigger a: injury, complaint of pain, ambulance, hospital"),
    Perturbation(
        "add_partner_airbag", "up",
        _append("The other vehicle's airbags deployed."),
        "Airbag deployment in any vehicle is trigger (b).",
        "trigger b: an airbag deployed in any vehicle"),
    Perturbation(
        "add_hit_and_run", "up",
        _append("The other driver left the scene without exchanging "
                "information."),
        "Leaving without exchanging information is trigger (e).",
        "trigger e: a party left the scene"),
    Perturbation(
        "add_police", "up",
        _append("Police responded to the scene and took a report."),
        "Police attendance is trigger (f).",
        "trigger f: police attended or a report was made"),
    Perturbation(
        "add_partner_tow", "up",
        _append("The other vehicle was disabled and had to be towed from the "
                "scene."),
        "A partner tow needs no corroboration; nobody recovers a member of the "
        "public's car as fleet policy.",
        "trigger c: a vehicle towed because it could not be driven"),

    # --- sensitivity downward ---------------------------------------------
    Perturbation(
        "no_damage", "down", _strip_damage,
        "An explicit statement that nothing was damaged removes every damage-"
        "based path to reportability.",
        "the 'no' default: contact without established extent"),
]


def check(before: str, after: str, direction: str) -> Optional[bool]:
    """Did the label move as the rubric requires? None if either is unusable."""
    b, a = ORDER.get(str(before).lower()), ORDER.get(str(after).lower())
    if b is None or a is None:
        return None
    if direction == "same":
        return a == b
    if direction == "up":
        # Already at the ceiling: the edit cannot raise it, so the case carries
        # no information and is excluded rather than scored as a pass.
        return None if b == ORDER["yes"] else a > b
    if direction == "down":
        return None if b == ORDER["no"] else a < b
    raise ValueError(direction)


def build_cases(incidents, extractions: dict, n: int = 120, seed: int = 11) -> list:
    """Sample narratives and generate every applicable perturbation of each.

    Stratified on the model's own baseline label so that "up" tests have
    headroom and "down" tests have somewhere to fall -- an all-"yes" sample
    would make every upward test uninformative.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    d = incidents[incidents["Report ID"].astype(str).isin(extractions)].copy()
    d["_label"] = d["Report ID"].astype(str).map(
        lambda r: str(extractions[r].get("police_reportable", "")).lower())
    d = d[d["_label"].isin(ORDER)]

    per = max(n // 3, 1)
    picks = []
    for lab in ("no", "maybe", "yes"):
        sub = d[d["_label"] == lab]
        if sub.empty:
            continue
        take = min(per, len(sub))
        picks.append(sub.iloc[rng.choice(len(sub), take, replace=False)])
    samp = pd.concat(picks) if picks else d.iloc[0:0]

    cases = []
    for _, r in samp.iterrows():
        narr = str(r["Narrative"])
        for p in PERTURBATIONS:
            edited = p.apply(narr)
            if edited is None or edited.strip() == narr.strip():
                continue          # perturbation not applicable to this narrative
            cases.append({
                "report_id": str(r["Report ID"]),
                "entity": str(r["Reporting Entity"]),
                "perturbation": p.name, "direction": p.direction,
                "clause": p.clause,
                "baseline_label": r["_label"],
                "narrative": edited,
            })
    return cases


def summarize(results: list) -> dict:
    """Pass rate per perturbation, and the failures worth reading."""
    by: dict[str, dict] = {}
    for r in results:
        ok = r.get("passed")
        p = r["perturbation"]
        e = by.setdefault(p, {"direction": r["direction"], "clause": r["clause"],
                              "n_scored": 0, "n_pass": 0, "n_uninformative": 0,
                              "failures": []})
        if ok is None:
            e["n_uninformative"] += 1
            continue
        e["n_scored"] += 1
        e["n_pass"] += int(ok)
        if not ok and len(e["failures"]) < 5:
            e["failures"].append({"report_id": r["report_id"],
                                  "before": r["baseline_label"],
                                  "after": r["new_label"],
                                  "evidence": r.get("new_evidence")})
    for e in by.values():
        e["pass_rate"] = (e["n_pass"] / e["n_scored"]) if e["n_scored"] else None
    return by


def main():
    import argparse
    import pandas as pd
    from extract.composition_extract import (AnthropicCompositionBackend,
                                             extract_one, summarize_cost)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser(
        description="Metamorphic validation of the reportability rubric.")
    ap.add_argument("--incidents",
                    default=os.path.join("data", "interim", "ads_incidents.parquet"))
    ap.add_argument("--extractions",
                    default=os.path.join("data", "processed",
                                         "composition_extractions.jsonl"))
    ap.add_argument("--n", type=int, default=120,
                    help="Narratives to sample. Each yields up to 12 "
                         "perturbations, so calls ~= 8x this.")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the cases and report how many calls it would "
                         "take, without calling the API.")
    ap.add_argument("--out", default=os.path.join("data", "processed",
                                                  "metamorphic_rubric.json"))
    a = ap.parse_args()

    ext = {}
    with open(a.extractions) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") and r.get("extraction"):
                ext[str(r["report_id"])] = r["extraction"]

    cases = build_cases(pd.read_parquet(a.incidents), ext, n=a.n)
    print(f"[meta] {len(cases):,} perturbed cases from {a.n} narratives")
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["perturbation"]] = counts.get(c["perturbation"], 0) + 1
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"[meta]   {k:24s} {v:5d}")
    if a.dry_run:
        print(f"[meta] dry run: would make {len(cases):,} calls "
              f"(~${len(cases) * 0.011:.2f} at the measured rate)")
        return

    be = AnthropicCompositionBackend(model=a.model)
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(extract_one, be, c["narrative"], c["entity"],
                            c["report_id"], "meta"): c for c in cases}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            res = fut.result()
            new = (res.extraction or {}).get("police_reportable")
            results.append({
                **{k: c[k] for k in ("report_id", "perturbation", "direction",
                                     "clause", "baseline_label")},
                "new_label": new,
                "new_evidence": (res.extraction or {}).get("reportable_evidence"),
                "ok": res.ok,
                "passed": check(c["baseline_label"], new, c["direction"])
                if res.ok else None,
                "usage": res.usage,
            })
            if i % 100 == 0 or i == len(futs):
                print(f"[meta] {i:,}/{len(futs):,}", flush=True)

    by = summarize(results)

    class _U:
        pass
    us = []
    for r in results:
        u = _U()
        u.usage = r.get("usage")
        us.append(u)
    cost = summarize_cost(us, be.name)

    rep = {"n_narratives": a.n, "n_cases": len(cases), "model": a.model,
           "by_perturbation": by, "cost": cost, "results": results}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    print(f"\n{'perturbation':24s} {'dir':>5s} {'n':>5s} {'pass':>7s}  clause")
    for k, e in sorted(by.items(), key=lambda kv: (kv[1]["direction"],
                                                   kv[1]["pass_rate"] or 0)):
        pr = f"{e['pass_rate']:.3f}" if e["pass_rate"] is not None else "   -  "
        print(f"{k:24s} {e['direction']:>5s} {e['n_scored']:5d} {pr:>7s}  "
              f"{e['clause'][:52]}")
    if cost.get("cost_usd") is not None:
        print(f"\n[meta] measured cost: ${cost['cost_usd']:.2f}")
    print(f"[meta] wrote {a.out}")


if __name__ == "__main__":
    main()
