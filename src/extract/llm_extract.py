"""Schema-constrained extraction engine.

Two interchangeable backends so the paper can benchmark a closed model against
open-weight models on identical inputs:

  - AnthropicBackend: uses tool-calling to force the CrashExtraction schema.
    Models: claude-sonnet-4-6 (default), claude-haiku-4-5-20251001 (cheaper).
  - OpenAICompatBackend: targets an OpenAI-compatible server (e.g. vLLM serving
    Qwen2.5 / Llama-3.x). Uses guided JSON decoding via the schema.

Both return a validated CrashExtraction plus metadata (latency, raw json,
n_repaired). Outputs are written as JSONL keyed by (source, report_id, model).

This module makes NETWORK CALLS to the configured model provider. It does not
fabricate extractions. Run it against real narratives produced by the parse step.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.schema import CrashExtraction, json_schema  # noqa: E402
from extract.prompts import SYSTEM_PROMPT, build_messages  # noqa: E402

# Load ANTHROPIC_API_KEY (and OPENAI_API_KEY, if using the open-weight backend)
# from a .env file in the project root, so you don't have to `export` it in
# every new terminal session. Copy .env.example to .env and fill in your key;
# .env itself is gitignored and never committed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed -- fall back to whatever's already in the shell env


@dataclass
class ExtractionResult:
    source: str
    report_id: str
    model: str
    ok: bool
    extraction: Optional[dict]
    latency_s: float
    error: Optional[str] = None


class Backend:
    name = "base"

    def extract_raw(self, narrative: str, source: str, report_id: str) -> dict:
        raise NotImplementedError


class AnthropicBackend(Backend):
    """Closed-model backend using Anthropic tool-calling for constrained output."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        import anthropic  # local import so the module loads without the dep
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env in "
                "the project root and paste your key in (get one at "
                "platform.claude.com -> Settings -> API keys), or export it "
                "directly: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.name = model
        self._tool = {
            "name": "record_crash_extraction",
            "description": "Record the structured coding of the collision narrative.",
            "input_schema": json_schema(),
        }

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def extract_raw(self, narrative: str, source: str, report_id: str) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[self._tool],
            tool_choice={"type": "tool", "name": "record_crash_extraction"},
            messages=build_messages(narrative, source, report_id),
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("No tool_use block returned by model.")


class OpenAICompatBackend(Backend):
    """Open-weight backend via an OpenAI-compatible server.

    Two server types are supported, since they take the JSON-schema constraint
    differently even though both expose an OpenAI-compatible /v1 endpoint:

      - "vllm":   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
                  Constrained via `extra_body={"guided_json": schema}`.
      - "ollama": ollama pull qwen2.5:7b   (serves on :11434 automatically)
                  Constrained via the standard `response_format` structured-
                  outputs parameter (OpenAI-style json_schema), which is what
                  Ollama's OpenAI-compat layer understands -- it does not read
                  `guided_json` at all and would otherwise silently ignore the
                  schema and return free-form text.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 max_tokens: int = 1024, server_type: str = "vllm"):
        from openai import OpenAI
        if server_type not in ("vllm", "ollama"):
            raise ValueError(f"server_type must be 'vllm' or 'ollama', got {server_type!r}")
        self.client = OpenAI(base_url=base_url,
                             api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model = model
        self.max_tokens = max_tokens
        self.name = model
        self.server_type = server_type
        self._schema = json_schema()

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def extract_raw(self, narrative: str, source: str, report_id: str) -> dict:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + \
               build_messages(narrative, source, report_id)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                     messages=msgs, temperature=0.0)
        if self.server_type == "vllm":
            kwargs["extra_body"] = {"guided_json": self._schema}  # vLLM guided decoding
        else:  # ollama
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "crash_extraction", "schema": self._schema},
            }
        resp = self.client.chat.completions.create(**kwargs)
        return json.loads(resp.choices[0].message.content)


def extract_one(backend: Backend, narrative: str, source: str,
                report_id: str) -> ExtractionResult:
    t0 = time.time()
    try:
        raw = backend.extract_raw(narrative, source, report_id)
        obj = CrashExtraction.model_validate(raw)  # schema enforcement
        return ExtractionResult(source, report_id, backend.name, True,
                                obj.model_dump(mode="json"), time.time() - t0)
    except (ValidationError, ValueError, KeyError) as e:
        return ExtractionResult(source, report_id, backend.name, False, None,
                                time.time() - t0, error=f"{type(e).__name__}: {e}")


def run(input_jsonl: str, output_jsonl: str, backend: Backend,
        text_field: str = "narrative", id_field: str = "report_id",
        source_field: str = "source", limit: Optional[int] = None) -> None:
    """Extract over a JSONL of narratives, writing one ExtractionResult per line.

    input rows must contain at least {id_field, text_field, source_field}.
    Resumable: skips ids already present in output_jsonl for this model.
    """
    done = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                r = json.loads(line)
                if r.get("model") == backend.name:
                    done.add((r["source"], r["report_id"]))

    n = 0
    with open(input_jsonl) as fin, open(output_jsonl, "a") as fout:
        # Exclusive, non-blocking lock on the output file for the life of this
        # run. Two concurrent invocations against the same output_jsonl would
        # otherwise both read the same (stale) `done` set above and both
        # append results for the same report_id, silently duplicating rows
        # and corrupting downstream indexed lookups (score_agreement.py).
        try:
            fcntl.flock(fout, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(
                f"{output_jsonl} is already locked by another running "
                "extraction process. Wait for it to finish, or point "
                "--output at a different file."
            )
        for line in fin:
            row = json.loads(line)
            key = (row[source_field], str(row[id_field]))
            if key in done:
                continue
            narrative = (row.get(text_field) or "").strip()
            if not narrative:
                continue
            res = extract_one(backend, narrative, row[source_field],
                              str(row[id_field]))
            fout.write(json.dumps(asdict(res)) + "\n")
            fout.flush()
            n += 1
            if limit and n >= limit:
                break
    print(f"[extract] wrote {n} new extractions to {output_jsonl} via {backend.name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run schema-constrained extraction.")
    ap.add_argument("--input", required=True, help="JSONL of narratives")
    ap.add_argument("--output", required=True, help="JSONL of extractions")
    ap.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--base-url", default=None,
                    help="Defaults to the standard port for --server-type "
                    "(vllm: http://localhost:8000/v1, "
                    "ollama: http://localhost:11434/v1)")
    ap.add_argument("--server-type", choices=["vllm", "ollama"], default="vllm",
                    help="Which OpenAI-compatible server is serving the "
                    "open-weight model. Controls how the JSON-schema "
                    "constraint is sent (guided_json for vLLM, "
                    "response_format for Ollama).")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.backend == "anthropic":
        be = AnthropicBackend(model=a.model)
    else:
        base_url = a.base_url or (
            "http://localhost:11434/v1" if a.server_type == "ollama"
            else "http://localhost:8000/v1")
        be = OpenAICompatBackend(model=a.model, base_url=base_url,
                                 server_type=a.server_type)
    run(a.input, a.output, be, limit=a.limit)