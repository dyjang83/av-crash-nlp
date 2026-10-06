"""Run the CRSS-compatible composition extraction over the ADS incident corpus.

Reuses the backend machinery in `extract.llm_extract` rather than reimplementing
it -- the retry policy, the logprobs-rejection fallback and the OpenAI-compat
server handling are all load-bearing and already tested. What changes is the
schema being enforced, the system prompt, and the unit of work: ONE ROW PER
DISTINCT ADS INCIDENT (from `corpus.sgo_ads`), not one per narrative in the
pooled corpus.

CONCURRENCY. The v1 runner is serial with an exclusive file lock, which is
correct for a resumable single-process run but takes about eighty minutes over
2,353 incidents. This module uses a thread pool and serialises only the writes.
The lock is kept for the same reason v1 has it: two concurrent invocations
against one output file would each read the same stale `done` set and duplicate
rows.

DETERMINISM CAVEAT, STATED. Thread-pool completion order is nondeterministic,
so the JSONL line order varies between runs. Nothing downstream depends on line
order -- every consumer keys on `report_id` -- but a byte-level diff of two runs
will differ, and that is a property of this runner rather than of the model.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.composition_schema import CompositionExtraction, json_schema  # noqa: E402
from extract.composition_prompts import SYSTEM_PROMPT, build_messages  # noqa: E402
from extract.llm_extract import _is_logprobs_rejection  # noqa: E402
from extract.token_probs import extract_field_probs  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

INCIDENTS = os.path.join("data", "interim", "ads_incidents.parquet")
OUT = os.path.join("data", "processed", "composition_extractions.jsonl")

TOOL_NAME = "record_composition_extraction"


@dataclass
class Result:
    report_id: str
    incident_key: str
    entity: str
    model: str
    ok: bool
    extraction: Optional[dict]
    latency_s: float
    error: Optional[str] = None
    token_probs: Optional[dict] = None
    # Per-call token usage, kept so the run's cost is a measured quantity rather
    # than an estimate, and so a silently broken cache is visible: if
    # `cache_read` stays at zero across calls, the prefix is being re-billed in
    # full every time and nobody would otherwise notice.
    usage: Optional[dict] = None


class AnthropicCompositionBackend:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 2048,
                 timeout: float = 90.0):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "paste your key in, or export it directly.")
        # EXPLICIT TIMEOUT. The SDK default is ten minutes, which is a sensible
        # ceiling for one long generation and a disastrous one for a thread
        # pool: a single hung connection parks a worker for ten minutes, and
        # when every worker is parked the run looks alive -- process running,
        # no errors logged -- while producing nothing. That is exactly how a
        # full-corpus run stalled at 755 of 2,338 with no diagnostic. These
        # calls return in about a second, so 90s is generous; a request that
        # exceeds it is hung, and failing fast lets tenacity retry it.
        # SDK retries are kept (default 2) rather than delegated entirely to
        # tenacity: the SDK honours the `retry-after` header on a 429, which
        # tenacity's blind exponential backoff does not. Tenacity remains the
        # outer net for everything the SDK gives up on. Worst case per call is
        # bounded at roughly 3 x timeout instead of unbounded.
        self.client = anthropic.Anthropic(api_key=key, timeout=timeout)
        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self._tool = {
            "name": TOOL_NAME,
            "description": "Record the CRSS-compatible coding of the narrative.",
            "input_schema": json_schema(),
        }

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def extract_raw(self, narrative: str, entity: str, report_id: str):
        # PROMPT CACHING. The request renders as tools -> system -> messages,
        # and the first two are byte-identical on every one of the ~2,300
        # calls: a ~4,000-token system prompt plus tool schema against ~240
        # tokens of actual narrative. Marking the system block caches
        # everything up to and including it, so the constant prefix is read at
        # a tenth of the input price and only the narrative is billed in full.
        # The breakpoint deliberately sits BEFORE `messages` -- putting it
        # after would fold the varying narrative into the cached prefix and
        # invalidate the cache on every call.
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[self._tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=build_messages(narrative, entity, report_id),
        )
        u = resp.usage
        usage = {
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
        }
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input), None, None, usage
        raise ValueError("No tool_use block returned by model.")


class OpenAICompatCompositionBackend:
    """Open-weight backend, for the §7 coder-swap robustness check."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 max_tokens: int = 2048, server_type: str = "vllm",
                 logprobs: bool = True, top_logprobs: int = 20):
        from openai import OpenAI
        if server_type not in ("vllm", "ollama"):
            raise ValueError(f"server_type must be 'vllm' or 'ollama', got {server_type!r}")
        self.client = OpenAI(base_url=base_url,
                             api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self.server_type = server_type
        self._schema = json_schema()
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self._lp_lock = threading.Lock()

    def _kwargs(self, narrative, entity, report_id, with_logprobs):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + \
               build_messages(narrative, entity, report_id)
        k = dict(model=self.model, max_tokens=self.max_tokens, messages=msgs,
                 temperature=0.0)
        if self.server_type == "vllm":
            k["extra_body"] = {"guided_json": self._schema}
        else:
            k["response_format"] = {"type": "json_schema",
                                    "json_schema": {"name": "composition_extraction",
                                                    "schema": self._schema}}
        if with_logprobs:
            k["logprobs"] = True
            k["top_logprobs"] = self.top_logprobs
        return k

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def extract_raw(self, narrative: str, entity: str, report_id: str):
        try:
            resp = self.client.chat.completions.create(
                **self._kwargs(narrative, entity, report_id, self.logprobs))
        except Exception as e:
            if not (self.logprobs and _is_logprobs_rejection(e)):
                raise
            # Guarded: several workers can hit the rejection at once, and the
            # flag must flip exactly once rather than racing.
            with self._lp_lock:
                if self.logprobs:
                    print(f"[composition] server rejected logprobs "
                          f"({type(e).__name__}); continuing without them")
                    self.logprobs = False
            resp = self.client.chat.completions.create(
                **self._kwargs(narrative, entity, report_id, False))
        choice = resp.choices[0]
        text = choice.message.content
        lp = getattr(choice, "logprobs", None)
        u = getattr(resp, "usage", None)
        usage = ({"input_tokens": getattr(u, "prompt_tokens", None),
                  "output_tokens": getattr(u, "completion_tokens", None)}
                 if u else None)
        return (json.loads(text),
                (getattr(lp, "content", None) if lp else None), text, usage)


def extract_one(backend, narrative: str, entity: str, report_id: str,
                incident_key: str) -> Result:
    t0 = time.time()
    try:
        raw, lp_content, raw_text, usage = backend.extract_raw(
            narrative, entity, report_id)
        obj = CompositionExtraction.model_validate(raw)
        tp = None
        if lp_content:
            try:
                tp = extract_field_probs(lp_content, content=raw_text)
            except Exception as e:
                tp = {"fields": {}, "meta": {"error": f"{type(e).__name__}: {e}"}}
        return Result(report_id, incident_key, entity, backend.name, True,
                      obj.model_dump(mode="json"), time.time() - t0,
                      token_probs=tp, usage=usage)
    except (ValidationError, ValueError, KeyError) as e:
        return Result(report_id, incident_key, entity, backend.name, False, None,
                      time.time() - t0, error=f"{type(e).__name__}: {e}")


# Published per-MTok rates, for turning measured token counts into a measured
# dollar figure. Kept as data so a price change is a one-line edit and the
# number in the run log is never a recollection.
PRICING = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


def summarize_cost(results: list, model: str) -> dict:
    """Measured token usage and cost for a run.

    Reported rather than estimated: the pre-run estimate for this corpus came
    from a characters-per-token heuristic and is worth about +/-20%, which is
    fine for deciding whether to run and useless for reporting what a run cost.
    """
    tot = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    n = 0
    for r in results:
        u = r.usage or {}
        if not u:
            continue
        n += 1
        tot["input"] += u.get("input_tokens") or 0
        tot["output"] += u.get("output_tokens") or 0
        tot["cache_write"] += u.get("cache_creation_input_tokens") or 0
        tot["cache_read"] += u.get("cache_read_input_tokens") or 0

    pi, po = PRICING.get(model, (None, None))
    cost = None
    if pi is not None:
        cost = ((tot["input"] + tot["cache_write"] * CACHE_WRITE_MULT
                 + tot["cache_read"] * CACHE_READ_MULT) / 1e6 * pi
                + tot["output"] / 1e6 * po)
    cacheable = tot["cache_read"] + tot["cache_write"]
    return {
        "n_calls_with_usage": n,
        "tokens": tot,
        "cache_hit_rate": (tot["cache_read"] / cacheable) if cacheable else 0.0,
        "cost_usd": cost,
        "cost_per_call_usd": (cost / n) if (cost is not None and n) else None,
    }


def run(incidents: str, output: str, backend, limit: Optional[int] = None,
        workers: int = 8, min_words: int = 8,
        only_ids: Optional[set] = None) -> dict:
    """Extract over the ADS incident frame. Resumable; skips completed ids.

    `only_ids` restricts the run to a given set of Report IDs -- used to
    extract exactly the human double-coding sample before committing to the
    full corpus, which is step 5 of the execution sequence: validate the
    reportability rubric against human coding, and only then scale up.
    """
    df = pd.read_parquet(incidents)
    if only_ids is not None:
        before = len(df)
        df = df[df["Report ID"].astype(str).isin(only_ids)]
        print(f"[composition] restricted to {len(df):,} of {before:,} incidents "
              f"by --only-ids")

    done = set()
    if os.path.exists(output):
        with open(output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("model") == backend.name and r.get("ok"):
                    done.add(str(r["report_id"]))

    todo = []
    skipped_short = 0
    for _, row in df.iterrows():
        rid = str(row["Report ID"])
        if rid in done:
            continue
        narr = str(row.get("Narrative") or "").strip()
        # A filing whose narrative is fully redacted is still a crash and still
        # belongs in the structured denominator, but there is nothing to extract
        # from it. Skipped here and counted, so the shortfall between the
        # incident count and the extraction count is explained rather than
        # discovered later as a silent join loss.
        if len(narr.split()) < min_words:
            skipped_short += 1
            continue
        todo.append((rid, str(row["incident_key"]), str(row["Reporting Entity"]), narr))
        if limit and len(todo) >= limit:
            break

    print(f"[composition] {len(df):,} incidents; {len(done):,} already done; "
          f"{skipped_short:,} without extractable narrative; {len(todo):,} to run")
    if not todo:
        return {"n_new": 0, "skipped_short": skipped_short, "n_done": len(done)}

    write_lock = threading.Lock()
    n_ok = n_fail = 0
    collected: list = []
    t0 = time.time()

    with open(output, "a") as fout:
        try:
            fcntl.flock(fout, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(
                f"{output} is locked by another running extraction. Wait for it "
                "to finish, or point --output at a different file.")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(extract_one, backend, narr, ent, rid, ik): rid
                    for rid, ik, ent, narr in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                with write_lock:
                    fout.write(json.dumps(asdict(res)) + "\n")
                    fout.flush()
                collected.append(res)
                n_ok += int(res.ok)
                n_fail += int(not res.ok)
                if i % 50 == 0 or i == len(futs):
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"[composition] {i:,}/{len(futs):,} "
                          f"({n_fail} failed) {rate:.1f}/s", flush=True)

    cost = summarize_cost(collected, backend.name)
    return {"n_new": n_ok + n_fail, "n_ok": n_ok, "n_failed": n_fail,
            "skipped_short": skipped_short, "n_done_before": len(done),
            "elapsed_s": time.time() - t0, "cost": cost}


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="CRSS-compatible composition extraction over ADS incidents.")
    ap.add_argument("--incidents", default=INCIDENTS)
    ap.add_argument("--output", default=OUT)
    ap.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--server-type", choices=["vllm", "ollama"], default="vllm")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-logprobs", action="store_true")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="Per-request timeout in seconds. The SDK default of "
                         "600 lets one hung connection park a pool worker for "
                         "ten minutes; these calls return in about one.")
    ap.add_argument("--only-sample", default=None,
                    help="Path to a coder JSONL from annotate.make_composition_"
                         "sample. Restricts the run to exactly those report_ids "
                         "-- the validation pass that precedes the full corpus.")
    a = ap.parse_args()

    only = None
    if a.only_sample:
        with open(a.only_sample) as f:
            only = {str(json.loads(l)["report_id"]) for l in f if l.strip()}

    if a.backend == "anthropic":
        be = AnthropicCompositionBackend(model=a.model, timeout=a.timeout)
    else:
        url = a.base_url or ("http://localhost:11434/v1" if a.server_type == "ollama"
                             else "http://localhost:8000/v1")
        be = OpenAICompatCompositionBackend(model=a.model, base_url=url,
                                            server_type=a.server_type,
                                            logprobs=not a.no_logprobs)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    rep = run(a.incidents, a.output, be, limit=a.limit, workers=a.workers,
              only_ids=only)

    print(f"\n[composition] {rep['n_ok']:,} ok, {rep['n_failed']:,} failed, "
          f"{rep['elapsed_s']:.0f}s")
    c = rep.get("cost") or {}
    if c.get("n_calls_with_usage"):
        t = c["tokens"]
        print(f"[composition] tokens: {t['input']:,} input, "
              f"{t['cache_write']:,} cache-write, {t['cache_read']:,} cache-read, "
              f"{t['output']:,} output")
        print(f"[composition] cache hit rate: {c['cache_hit_rate']:.1%}")
        if c["cache_hit_rate"] < 0.5 and t["cache_read"] + t["cache_write"] > 0:
            print("[composition] WARNING: low cache hit rate -- the constant "
                  "prefix is being re-billed. Check that the system prompt and "
                  "tool schema are byte-identical across calls.")
        if c.get("cost_usd") is not None:
            print(f"[composition] measured cost: ${c['cost_usd']:.2f} "
                  f"(${c['cost_per_call_usd']:.4f}/call)")


if __name__ == "__main__":
    main()
