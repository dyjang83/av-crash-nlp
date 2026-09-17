"""Unit tests for per-field token-probability alignment.

SYNTHETIC TOKEN FIXTURES ON PURPOSE. These verify the alignment and
renormalization arithmetic against hand-computed values; they are not results and
no number here enters the paper. Real probabilities come from running
extract.llm_extract against a local Ollama server.
"""
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract.token_probs import (align_tokens, field_spans, field_token_prob,
                                 extract_field_probs, enum_values)


def mk(spec):
    """spec: [(token_text, prob, [(cand_text, cand_prob), ...]), ...] -> API-shaped dicts."""
    return [{"token": t, "logprob": math.log(p),
             "top_logprobs": [{"token": c, "logprob": math.log(q)} for c, q in top]}
            for t, p, top in spec]


# --------------------------------------------------------------------------
def test_align_offsets_and_roundtrip():
    toks, text = align_tokens(mk([('{"a": ', 0.9, []), ("1", 0.8, []), ("}", 0.99, [])]))
    assert text == '{"a": 1}'
    assert [(t.start, t.end) for t in toks] == [(0, 6), (6, 7), (7, 8)]
    assert abs(toks[1].prob - 0.8) < 1e-12


def test_field_spans_flat_schema():
    js = '{"weather": "clear", "av_moving": true, "locality": "segment"}'
    sp = field_spans(js, ["weather", "av_moving", "locality"])
    assert js[sp["weather"][0]:sp["weather"][1]] == '"clear"'
    assert js[sp["av_moving"][0]:sp["av_moving"][1]] == "true"
    assert sp["locality"][2] == '"segment"'


def test_renorm_strips_illegal_mass():
    """Pre-mask case: illegal candidates are excluded, so p_renorm > p_first."""
    spec = [('{"weather": "', 0.99, []),
            ("clear", 0.6, [("clear", 0.6), ("sunny", 0.3), ("rain", 0.1)]),
            ('"}', 0.99, [])]
    toks, text = align_tokens(mk(spec))
    sp = field_spans(text, ["weather"])["weather"]
    r = field_token_prob(toks, sp, enum_values()["weather"])
    assert r["value"] == "clear"
    assert r["n_tokens"] == 1
    assert abs(r["p_first"] - 0.6) < 1e-12
    # "sunny" is not a prefix of any weather enum; "rain" is. 0.6 / (0.6+0.1)
    assert abs(r["p_renorm"] - 0.6 / 0.7) < 1e-12
    assert r["n_legal_alts"] == 1
    assert not r["truncated"]


def test_prefix_sharing_ambiguity_is_caught_at_divergence():
    """`intersection` vs `intersection_related` share a prefix.

    p_first is ~1.0 and says nothing; the real ambiguity lives at the token where
    the value either closes or continues. p_renorm must see it.
    """
    spec = [('{"locality": "', 0.99, []),
            ("intersection", 0.9, [("intersection", 0.9), ("segment", 0.05),
                                   ("drive", 0.05)]),
            ("_related", 0.55, [("_related", 0.55), ('"', 0.45)]),
            ('"}', 0.99, [])]
    toks, text = align_tokens(mk(spec))
    sp = field_spans(text, ["locality"])["locality"]
    r = field_token_prob(toks, sp, enum_values()["locality"])
    assert r["value"] == "intersection_related"
    assert r["n_tokens"] == 2
    assert abs(r["p_first"] - 0.9) < 1e-12
    # all candidates legal at both positions -> 0.9 * 0.55
    assert abs(r["p_renorm"] - 0.495) < 1e-9
    assert r["p_renorm"] < r["p_first"]      # the point of the test
    assert abs(r["p_min"] - 0.55) < 1e-12


def test_multi_token_enum_pseq_vs_prenorm():
    """p_seq is diluted by tokenization; p_renorm is not."""
    spec = [('{"subject_pre_crash_maneuver": "', 0.99, []),
            ("proceeding", 0.8, [("proceeding", 0.8), ("stopped", 0.2)]),
            ("_str", 0.9, [("_str", 0.9)]),
            ("aight", 0.95, [("aight", 0.95)]),
            ('"}', 0.99, [])]
    toks, text = align_tokens(mk(spec))
    sp = field_spans(text, ["subject_pre_crash_maneuver"])["subject_pre_crash_maneuver"]
    r = field_token_prob(toks, sp, enum_values()["subject_pre_crash_maneuver"])
    assert r["value"] == "proceeding_straight"
    assert r["n_tokens"] == 3
    assert abs(r["p_seq"] - 0.8 * 0.9 * 0.95) < 1e-12
    # positions 2 and 3 have no legal alternative, so they contribute 1.0
    assert abs(r["p_renorm"] - 0.8) < 1e-12
    assert r["p_renorm"] > r["p_seq"]
    # ... and that absence is SCHEMA DETERMINISM, not top-k truncation: once
    # `proceeding` is emitted, no other maneuver remains reachable. Flagging it
    # would discard a perfectly good extraction.
    assert r["truncated"] is False
    assert r["n_legal_alts"] == 1     # only position 1 carried a real choice


def test_schema_determined_tail_is_not_truncation():
    """A long unique value has no alternatives after its prefix diverges."""
    spec = [('{"road_class": "', 0.99, []),
            ("parking", 0.82, [("parking", 0.82), ("high", 0.10), ("surface", 0.08)]),
            ("_lot", 0.99, [("_lot", 0.99)]),         # forced: only one value fits
            ("_private", 0.99, [("_private", 0.99)]),  # forced
            ('"}', 0.99, [])]
    toks, text = align_tokens(mk(spec))
    sp = field_spans(text, ["road_class"])["road_class"]
    r = field_token_prob(toks, sp, enum_values()["road_class"])
    assert r["value"] == "parking_lot_private"
    assert r["truncated"] is False
    assert abs(r["p_renorm"] - 0.82) < 1e-9


def test_near_certain_token_is_not_flagged_truncated():
    """p=1.0 pushes every alternative below top-k precision. Not truncation."""
    spec = [('{"collision_type": "', 0.99, []),
            ("v", 1.0, [("v", 1.0)]), ("ru", 1.0, [("ru", 1.0)]),
            ('"}', 0.99, [])]
    toks, text = align_tokens(mk(spec))
    sp = field_spans(text, ["collision_type"])["collision_type"]
    r = field_token_prob(toks, sp, enum_values()["collision_type"])
    assert r["value"] == "vru"
    assert r["truncated"] is False


def test_truncated_topk_is_flagged_not_silently_confident():
    """No legal alternative in top-k -> p_renorm collapses to 1.0. Must be flagged."""
    spec = [('{"weather": "', 0.99, []),
            ("clear", 0.6, [("sunny", 0.25), ("foggy", 0.15)]),   # chosen absent, all illegal
            ('"}', 0.99, [])]
    toks, text = align_tokens(mk(spec))
    sp = field_spans(text, ["weather"])["weather"]
    r = field_token_prob(toks, sp, enum_values()["weather"])
    assert abs(r["p_renorm"] - 1.0) < 1e-12   # spuriously confident ...
    assert r["n_legal_alts"] == 0             # ... and the reason is visible
    assert r["truncated"] is True


def test_boolean_field():
    spec = [('{"av_moving": ', 0.99, []),
            ("true", 0.7, [("true", 0.7), ("false", 0.3)]),
            ("}", 0.99, [])]
    toks, text = align_tokens(mk(spec))
    sp = field_spans(text, ["av_moving"])["av_moving"]
    r = field_token_prob(toks, sp, ["true", "false"])
    assert r["value"] == "true"
    assert abs(r["p_renorm"] - 0.7) < 1e-12
    assert r["n_legal_alts"] == 1


def test_extract_field_probs_skips_freetext_and_confidence():
    spec = [('{"weather": "', 0.99, []), ("clear", 0.6, [("clear", 0.6), ("rain", 0.4)]),
            ('", "contributory_evidence": "', 0.9, []), ("a car", 0.5, []),
            ('", "confidence": ', 0.9, []), ("0.82", 0.4, []), ("}", 0.99, [])]
    out = extract_field_probs(mk(spec))
    assert "weather" in out["fields"]
    assert "contributory_evidence" not in out["fields"]
    assert "confidence" not in out["fields"]
    assert out["meta"]["n_tokens"] == 7


def test_roundtrip_mismatch_is_recorded():
    spec = [('{"weather": "', 0.99, []), ("clear", 0.6, []), ('"}', 0.99, [])]
    out = extract_field_probs(mk(spec), content='{"weather": "sunny"}')
    assert out["meta"].get("roundtrip_mismatch") is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("OK: token-probability alignment tests passed (synthetic fixtures)")
