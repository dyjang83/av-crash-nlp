"""Per-field decode-time token probabilities from an OpenAI-compatible logprobs payload.

The extraction backends return a self-reported `confidence` float that the model
writes into the JSON itself (schema.py). That is one confidence signal. An
open-weight model served locally exposes a second, independent one: the actual
next-token distribution at decode time. The closed tool-calling backend does not
expose this, which is exactly why the comparison is only possible on the open
models.

The API hands back a FLAT token list for the whole JSON object. To turn that into
a probability per SCHEMA FIELD we:

  1. align_tokens()  -- concatenate token strings, accumulating character offsets,
     so every token knows which slice of the emitted JSON it produced;
  2. field_spans()   -- locate each field's VALUE span in that same string. The
     extraction schema is a flat 13-key object (schema.py:CrashExtraction), so a
     per-key regex is exact here; this would not be safe on a nested schema;
  3. field_token_prob() -- aggregate the logprobs of the tokens covering that span.

Four aggregations are reported per field, because they answer different questions:

  p_first  probability of the first value token (usually decisive for enums)
  p_seq    exp(sum logprobs) -- joint probability of the whole value string
  p_min    the bottleneck token, i.e. the model's least-certain moment
  p_renorm probability of the value renormalized over the still-LEGAL enum
           members at each position (chain rule restricted to the schema).
           This is the headline quantity: a posterior over CATEGORIES rather
           than over token strings.

Why p_renorm rather than p_seq: p_seq is diluted by tokenization. A value that is
split into four tokens is not less certain than a one-token value, but its p_seq
is smaller. p_renorm conditions on the schema constraint at every step, so
`intersection` vs `intersection_related` -- which share a long prefix and only
diverge at the end -- is scored at the position where the ambiguity actually
lives, not at the first token where there is none.

Truncation is tracked, never assumed away, and is distinguished from schema
determinism. A position where the prefix already fixes the value -- `parking` can
only continue to `parking_lot_private` -- has no alternatives by construction, and
its renormalized factor of 1.0 is correct. A position that still permits several
values but shows none of them in the top-k is different: there p_renorm collapses
to 1.0 for lack of anything to divide against. Only the latter sets `truncated`,
and only when the chosen token was not itself near-certain. Truncated fields must
be excluded from calibration statistics rather than read as confident.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field as _dcfield
from typing import Any, Iterable, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema.schema import CrashExtraction  # noqa: E402


# ---------------------------------------------------------------------------
# Accessors: work with both the OpenAI SDK objects and plain dicts, so the
# alignment logic is unit-testable without a server or the SDK installed.
# ---------------------------------------------------------------------------
def _g(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class Tok:
    text: str
    logprob: float
    start: int
    end: int
    top: list[tuple[str, float]] = _dcfield(default_factory=list)

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


def align_tokens(logprob_content: Iterable[Any]) -> tuple[list[Tok], str]:
    """Attach character offsets to each token; return (tokens, reconstructed_text).

    The reconstructed text is what the offsets index into. It should equal the
    message content; callers that have both should check, because a mismatch means
    the tokenizer round-trip is lossy (byte-level BPE on non-ASCII) and every span
    downstream would be off.
    """
    toks: list[Tok] = []
    pos = 0
    for t in logprob_content:
        s = _g(t, "token", "") or ""
        top = []
        for c in (_g(t, "top_logprobs", []) or []):
            top.append((_g(c, "token", "") or "", float(_g(c, "logprob", -math.inf))))
        toks.append(Tok(text=s, logprob=float(_g(t, "logprob", -math.inf)),
                        start=pos, end=pos + len(s), top=top))
        pos += len(s)
    return toks, "".join(t.text for t in toks)


# ---------------------------------------------------------------------------
# Field value spans in the emitted JSON
# ---------------------------------------------------------------------------
def enum_values() -> dict[str, list[str]]:
    """Legal values per categorical field, read from the pydantic model.

    Derived from CrashExtraction rather than restated, so it cannot drift from
    the schema that constrained the decode.
    """
    out: dict[str, list[str]] = {}
    for name, f in CrashExtraction.model_fields.items():
        members = getattr(f.annotation, "__members__", None)
        if members:
            out[name] = [m.value for m in members.values()]
    return out


BOOL_FIELDS = ["other_party_present", "av_moving"]
# Free text and the self-reported signal itself: no categorical posterior applies.
SKIP_FIELDS = {"contributory_evidence", "confidence"}

# Group 1 is the value: a JSON string (with escapes), a literal, or a number.
_VALUE = r'("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'


def field_spans(json_str: str, fields: Optional[Iterable[str]] = None
                ) -> dict[str, tuple[int, int, str]]:
    """Locate each field's value span. Returns {field: (start, end, raw_value)}.

    Spans cover the raw JSON token INCLUDING surrounding quotes for strings;
    field_token_prob() narrows to the interior where that matters.
    """
    if fields is None:
        fields = [f for f in CrashExtraction.model_fields if f not in SKIP_FIELDS]
    out = {}
    for name in fields:
        m = re.search(rf'"{re.escape(name)}"\s*:\s*{_VALUE}', json_str)
        if m:
            out[name] = (m.start(1), m.end(1), m.group(1))
    return out


# ---------------------------------------------------------------------------
# Per-field probability
# ---------------------------------------------------------------------------
def _legal_at(prefix: str, cand: str, legal: list[str], quoted: bool) -> bool:
    """Can `prefix + cand` still lead to a legal value?

    `quoted` marks a string field, where the closing quote is itself a legal
    continuation once `prefix` exactly equals some enum member -- that is the
    position at which `intersection` and `intersection_related` finally diverge.
    """
    if not cand:
        return False
    if quoted and cand.startswith('"'):
        return prefix in legal
    hyp = prefix + cand
    return any(v.startswith(hyp) for v in legal)


def field_token_prob(toks: list[Tok], span: tuple[int, int, str],
                     legal: Optional[list[str]]) -> dict:
    """Aggregate token probabilities over one field's value span."""
    start, end, raw = span
    quoted = raw.startswith('"')
    # Interior of the value: enum text lives between the quotes.
    v_start, v_end = (start + 1, end - 1) if quoted else (start, end)

    covering = [t for t in toks if t.end > v_start and t.start < v_end]
    if not covering:
        return {"value": raw.strip('"'), "n_tokens": 0, "p_first": None,
                "p_seq": None, "p_min": None, "p_renorm": None,
                "topk_mass": None, "n_legal_alts": None, "truncated": True}

    probs = [t.prob for t in covering]
    logsum = sum(t.logprob for t in covering)

    out = {
        "value": raw.strip('"'),
        "n_tokens": len(covering),
        "p_first": probs[0],
        "p_seq": math.exp(logsum),
        "p_min": min(probs),
        "p_renorm": None,
        "topk_mass": None,
        "truncated": False,
    }
    if not legal:
        out["n_legal_alts"] = None
        return out

    # Chain rule restricted to the legal set: at each value token position,
    # renormalize the chosen token's probability over the candidates that could
    # still complete a legal enum member.
    text = "".join(t.text for t in toks)
    log_renorm = 0.0
    min_topk_mass = 1.0
    min_legal_alts = None
    truncated = False
    for t in covering:
        prefix = text[v_start:max(t.start, v_start)]

        # How many DISTINCT continuations the schema still permits here. Once the
        # prefix uniquely determines the value -- `parking` can only become
        # `parking_lot_private` -- the next token is forced, its renormalized
        # factor is legitimately 1.0, and the absence of alternatives in the
        # top-k says nothing about truncation. Only positions that still carry a
        # real choice can be truncated.
        extensions = [v for v in legal if v.startswith(prefix) and v != prefix]
        n_possible = len(extensions) + (1 if (quoted and prefix in legal) else 0)

        legal_mass, topk_mass, chosen_seen, n_alts = 0.0, 0.0, False, 0
        for cand_text, cand_lp in t.top:
            p = math.exp(cand_lp)
            topk_mass += p
            if _legal_at(prefix, cand_text, legal, quoted):
                legal_mass += p
                if cand_text == t.text:
                    chosen_seen = True
                else:
                    n_alts += 1
        # Some servers omit the sampled token from top_logprobs; it is legal by
        # construction (the emitted value validated against the schema), so add
        # it rather than dividing by a mass that excludes it.
        if not chosen_seen:
            legal_mass += t.prob
            topk_mass += t.prob
        log_renorm += math.log(min(t.prob / legal_mass, 1.0)) if legal_mass > 0 else 0.0
        min_topk_mass = min(min_topk_mass, topk_mass)

        if n_possible > 1:
            min_legal_alts = n_alts if min_legal_alts is None else min(min_legal_alts, n_alts)
            # A real choice existed, no legal alternative surfaced in the top-k,
            # and the chosen token was not itself near-certain: p_renorm has
            # collapsed to 1.0 for lack of anything to divide against. That is a
            # top-k artifact, not model certainty, and must not be read as one.
            if n_alts == 0 and t.prob < 0.999:
                truncated = True

    out["p_renorm"] = math.exp(log_renorm)
    out["topk_mass"] = min_topk_mass
    # Legal alternatives visible in the top-k at the tightest position that
    # actually carried a choice. None means the value was schema-determined
    # throughout, which is informative rather than defective.
    out["n_legal_alts"] = min_legal_alts
    out["truncated"] = truncated
    return out


def extract_field_probs(logprob_content: Iterable[Any],
                        content: Optional[str] = None) -> dict:
    """Top-level entry: logprobs payload -> {field: {p_first, p_seq, p_min, p_renorm, ...}}.

    `content` is the message string, if the caller has it. When supplied it is
    checked against the token round-trip; a mismatch is recorded rather than
    raised, because one lossy record should not abort a corpus run -- but it must
    be visible, since every span in that record is then untrustworthy.
    """
    toks, text = align_tokens(logprob_content)
    enums = enum_values()
    spans = field_spans(text)

    fields = {}
    for name, span in spans.items():
        legal = enums.get(name) or (["true", "false"] if name in BOOL_FIELDS else None)
        fields[name] = field_token_prob(toks, span, legal)

    meta = {"n_tokens": len(toks), "n_fields": len(fields)}
    if content is not None and content != text:
        meta["roundtrip_mismatch"] = True
    return {"fields": fields, "meta": meta}
