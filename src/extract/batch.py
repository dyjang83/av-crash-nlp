"""Resumable Message Batches runner shared by the span tagger and re-extraction.

A batch is submitted once and its id is written to a state file before anything
else happens, so an interrupted run (closed laptop, killed process) resumes by
polling the batch it already paid for instead of submitting a second copy.

This module makes NETWORK CALLS and spends money. It never fabricates results:
records whose request errored or expired are returned as failures, and callers
write them as ok=False rows.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _client():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
    return anthropic.Anthropic()


def run_batch(name: str, requests: Iterable[tuple[str, dict]], state_dir: str,
              poll_s: int = 60) -> dict[str, dict]:
    """Submit (or resume) a batch and return {key: outcome}.

    requests: (key, params) pairs, where params is a Messages API request body.
    Keys may be any string; custom_ids are positional ("r000123") and the
    key<->custom_id map is stored in the state file, because the API restricts
    custom_id to [A-Za-z0-9_-]{1,64}.

    outcome = {"ok": True, "message": <Message as dict>}
            | {"ok": False, "error": "<type>: <detail>"}
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, f"{name}.batch.json")
    client = _client()

    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"[batch] resuming {name}: {state['batch_id']} "
              f"({len(state['keys'])} requests)")
    else:
        reqs = list(requests)
        if not reqs:
            return {}
        keys = [k for k, _ in reqs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate request keys")
        batch = client.messages.batches.create(requests=[
            Request(custom_id=f"r{i:06d}",
                    params=MessageCreateParamsNonStreaming(**params))
            for i, (_, params) in enumerate(reqs)])
        state = {"batch_id": batch.id, "keys": keys, "created": time.time()}
        with open(state_path, "w") as f:
            json.dump(state, f)
        print(f"[batch] submitted {name}: {batch.id} ({len(keys)} requests)")

    while True:
        b = client.messages.batches.retrieve(state["batch_id"])
        if b.processing_status == "ended":
            break
        rc = b.request_counts
        print(f"[batch] {name}: {b.processing_status} "
              f"processing={rc.processing} succeeded={rc.succeeded} "
              f"errored={rc.errored}", flush=True)
        time.sleep(poll_s)

    out: dict[str, dict] = {}
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    for res in client.messages.batches.results(state["batch_id"]):
        key = state["keys"][int(res.custom_id[1:])]
        r = res.result
        if r.type == "succeeded":
            msg = r.message.model_dump(mode="json")
            for k in usage:
                usage[k] += int((msg.get("usage") or {}).get(k) or 0)
            out[key] = {"ok": True, "message": msg}
        elif r.type == "errored":
            out[key] = {"ok": False, "error": f"errored: {r.error.model_dump_json()}"}
        else:
            out[key] = {"ok": False, "error": r.type}
    n_ok = sum(o["ok"] for o in out.values())
    print(f"[batch] {name}: {n_ok}/{len(out)} succeeded; usage {usage}")
    state["usage"] = usage
    with open(state_path, "w") as f:
        json.dump(state, f)
    return out


def tool_input(message: dict, tool_name: str) -> dict:
    """The input of the named tool_use block, or ValueError if absent."""
    if message.get("stop_reason") == "max_tokens":
        raise ValueError("response truncated at max_tokens")
    for block in message.get("content") or []:
        if block.get("type") == "tool_use" and block.get("name") == tool_name:
            return dict(block["input"])
    raise ValueError(f"no {tool_name} tool_use block returned")
