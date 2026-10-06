"""Tests for the outcome-statement masker (synthetic sentences; no paper numbers)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from leakage.outcome_lexicon import (mask, residual_hits, split_sentences,
                                     sentence_categories, spans_to_sentences)

CTX = "The AV was stopped at a red light on Mission St. at approximately 3 PM."
HIT = "A passenger car traveling behind the AV made contact with its rear bumper."


@pytest.mark.parametrize("sentence,cat", [
    ("The passenger complained of neck pain.", "O1"),
    ("No injuries were reported.", "O1"),              # absence leaks too
    ("There were no reported injuries and the AV was towed.", "O1"),
    ("The cyclist was transported to a hospital by ambulance.", "O2"),
    ("The driver declined medical attention at the scene.", "O2"),
    ("The pedestrian was pronounced dead at the scene.", "O3"),
    ("Police responded to the scene.", "O4"),
    ("Claimant alleges the vehicle failed to brake.", "O4"),
    ("Waymo is reporting this crash under Request No. 2 of Standing General "
     "Order 2021-01.", "O5"),
])
def test_categories_detected(sentence, cat):
    assert cat in sentence_categories(sentence)


@pytest.mark.parametrize("sentence", [
    CTX, HIT,
    "The Waymo AV was in reverse on a dead-end road at 2 MPH.",
    "The AV was proceeding straight through the intersection at 25 mph.",
])
def test_crash_context_not_flagged(sentence):
    assert not (sentence_categories(sentence) - {"O4"})


def test_outcome_mask_deletes_whole_sentence_without_placeholder():
    text = f"{CTX} {HIT} The passenger was transported to a hospital with minor injuries."
    r = mask(text, "outcome")
    assert "hospital" not in r.text and "injur" not in r.text
    assert CTX in r.text and HIT in r.text
    assert "[" not in r.text.replace("[XXX]", "")     # no mask token inserted
    assert r.removed == [2]
    assert residual_hits(r.text, "outcome") == 0


def test_outcome_tier_keeps_o4_strict_removes_it():
    text = f"{CTX} Police responded to the scene."
    assert "Police" in mask(text, "outcome").text
    assert "Police" not in mask(text, "strict").text


def test_preimpact_truncates_after_contact_and_ignores_template_opening():
    opening = ("On June 3, 2024 a Waymo AV operating in San Francisco was in a "
               "collision involving a passenger car.")
    after = "The AV then pulled over and the other driver exited the vehicle."
    r = mask(f"{opening} {CTX} {HIT} {after}", "preimpact")
    assert opening in r.text and HIT in r.text
    assert after not in r.text
    assert r.truncated_after == 2


def test_idempotent():
    text = f"{CTX} No injuries were reported. {HIT}"
    once = mask(text, "strict").text
    assert mask(once, "strict").text == once


def test_empty_result_is_kept_not_dropped():
    r = mask("No injuries were reported.", "outcome")
    assert r.text == "" and r.n_sentences == 1


def test_llm_spans_union_and_category_filter():
    s3 = "The occupant later sought care for a stiff neck."   # lexicon misses
    text = f"{CTX} {HIT} {s3}"
    assert mask(text, "outcome").removed == []
    r = mask(text, "outcome", llm_spans=["sought care for a stiff neck"],
             llm_categories=["O1"])
    assert r.removed == [2] and r.removed_llm == [2]
    # an O4-labelled tagger span is ignored by the outcome tier
    r = mask(text, "outcome", llm_spans=["sought care"], llm_categories=["O4"])
    assert r.removed == []


def test_splitter_abbreviations_and_offsets():
    text = "The AV was on Mission St. near 5th. It stopped.\n\nA car hit it."
    sents = split_sentences(text)
    pieces = [text[a:b] for a, b in sents]
    assert pieces[0].startswith("The AV was on Mission St. near 5th")
    assert pieces[-1] == "A car hit it."
    assert spans_to_sentences(text, sents, ["car  hit"]) == {len(sents) - 1}


def test_tagger_span_coercion():
    from leakage.tag_outcome_spans import coerce_spans
    assert coerce_spans([{"text": "a", "category": "O1"}]) == [{"text": "a", "category": "O1"}]
    assert coerce_spans('[{"text": "b", "category": "O2"}]') == [{"text": "b", "category": "O2"}]
    bad = '[{"text": "filed a Report (the "Report") today", "category": "O5"}]'
    assert coerce_spans(bad) == [{"text": 'filed a Report (the "Report") today', "category": "O5"}]
    assert coerce_spans(None) == [] and coerce_spans(["c"]) == [{"text": "c", "category": None}]
