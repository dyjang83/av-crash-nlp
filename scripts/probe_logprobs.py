"""Smoke test: does Ollama return usable token logprobs under schema constraint?

This gates the whole token-probability analysis, because the answer determines
what the numbers MEAN, and there is no way to tell from the values alone:

  (A) logprobs returned, top_logprobs contains schema-ILLEGAL tokens
      -> pre-mask. We are seeing the model's raw next-token belief, and a
         renormalization over the legal enum members is both meaningful and
         necessary to get a categorical posterior.
  (B) logprobs returned, only schema-LEGAL tokens, mass ~= 1.0
      -> post-mask. The number already IS the constrained decode probability;
         renormalizing is a no-op and the paper must say so.
  (C) logprobs absent when response_format is set
      -> the OpenAI-compat layer drops them under structured outputs; fall back
         to the native /api/chat endpoint, then to forced-choice enum scoring.

Run:  ollama serve &
      PYTHONPATH=src python scripts/probe_logprobs.py --model qwen2.5:7b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema.schema import json_schema, CrashExtraction  # noqa: E402
from extract.prompts import SYSTEM_PROMPT, build_messages  # noqa: E402

# Enum members per field, used to decide whether a candidate token is a legal
# continuation. Taken from the pydantic model so it can never drift from schema.py.
def enum_values() -> dict[str, list[str]]:
    out = {}
    for name, field in CrashExtraction.model_fields.items():
        ann = field.annotation
        members = getattr(ann, "__members__", None)
        if members:
            out[name] = [m.value for m in members.values()]
    return out


def probe(model: str, base_url: str, narrative: str, top_k: int = 20) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + \
           build_messages(narrative, "probe", "probe-0001")
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=0.0, max_tokens=1024,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "crash_extraction",
                                         "schema": json_schema()}},
        logprobs=True, top_logprobs=top_k,
    )
    choice = resp.choices[0]
    content = choice.message.content
    lp = getattr(choice, "logprobs", None)
    return {"content": content, "logprobs": lp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--narrative", default=None)
    a = ap.parse_args()

    narrative = a.narrative
    if narrative is None:
        # Use a real gold-set narrative rather than a toy string: token
        # boundaries on synthetic text are not representative.
        with open(os.path.join("data", "gold", "gold.jsonl")) as f:
            narrative = json.loads(f.readline())["narrative"]

    print(f"[probe] model={a.model} top_k={a.top_k}")
    print(f"[probe] narrative[:120]={narrative[:120]!r}\n")

    try:
        res = probe(a.model, a.base_url, narrative, a.top_k)
    except Exception as e:
        print(f"[probe] REQUEST FAILED: {type(e).__name__}: {e}")
        print("[probe] -> is `ollama serve` running? is the model pulled?")
        sys.exit(2)

    content, lp = res["content"], res["logprobs"]
    print(f"[probe] content: {content}\n")

    if lp is None or not getattr(lp, "content", None):
        print("[probe] VERDICT (C): no logprobs returned under response_format.")
        print("[probe] -> fall back to native /api/chat, then forced-choice scoring.")
        sys.exit(3)

    toks = lp.content
    print(f"[probe] VERDICT: logprobs ARE returned. {len(toks)} tokens.\n")

    # Find a categorical value position: the first token after `"weather": "`
    enums = enum_values()
    text, offsets = "", []
    for t in toks:
        offsets.append((len(text), len(text) + len(t.token)))
        text += t.token

    verdicts = []
    for field in ("weather", "lighting", "collision_type"):
        needle = f'"{field}"'
        k = text.find(needle)
        if k < 0:
            continue
        # first token whose span starts at/after the opening quote of the value
        vq = text.find('"', text.find(":", k) ) + 1
        ti = next((i for i, (s, e) in enumerate(offsets) if s <= vq < e), None)
        if ti is None:
            continue
        tok = toks[ti]
        legal = enums.get(field, [])
        cands = getattr(tok, "top_logprobs", []) or []
        rows, legal_mass, total_mass = [], 0.0, 0.0
        for c in cands:
            p = math.exp(c.logprob)
            total_mass += p
            ok = any(v.startswith(c.token.strip('"')) and c.token.strip('"')
                     for v in legal)
            legal_mass += p if ok else 0.0
            rows.append((c.token, p, ok))
        print(f"--- field {field!r}: chosen={tok.token!r} p={math.exp(tok.logprob):.4f}")
        for t_, p_, ok_ in rows[:8]:
            print(f"      {'LEGAL  ' if ok_ else 'illegal'} {t_!r:24s} p={p_:.5f}")
        n_illegal = sum(1 for _, _, ok_ in rows if not ok_)
        print(f"      top-{len(rows)} mass={total_mass:.4f}  legal mass={legal_mass:.4f}  "
              f"illegal candidates={n_illegal}")
        verdicts.append(n_illegal)
        print()

    if not verdicts:
        print("[probe] could not locate a categorical value position -- inspect manually.")
        sys.exit(4)

    if max(verdicts) > 0:
        print("[probe] VERDICT (A): PRE-MASK. top_logprobs contains schema-illegal "
              "tokens, so these are raw model beliefs.")
        print("[probe] -> renormalize over legal enum members to get p_renorm.")
    else:
        print("[probe] VERDICT (B): POST-MASK. Only legal tokens appear and the mass "
              "is already normalized over them.")
        print("[probe] -> p_renorm == p_first; say so explicitly in the paper.")


if __name__ == "__main__":
    main()
