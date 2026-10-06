"""Identify and remove outcome statements from crash narratives.

The severity lift test asks whether the narrative adds information about injury
severity beyond the structured fields. Dropping the LLM-extracted
narrative_injury_severity field does not answer that on its own: the TF-IDF or
SBERT features are built from the narrative itself, and a narrative that says
"the passenger was transported to hospital" hands the outcome to any model that
can read. This module defines what counts as an outcome statement, detects
those statements with a transparent lexicon, and deletes the sentences that
carry them.

Categories (see src/annotate/outcome_masking_guide.md for the coder version):
  O1 injury status      -- injury, pain, complaint of pain, AND absence
                           statements ("no injuries were reported"): an explicit
                           "no injury" is as informative about the target as an
                           explicit injury.
  O2 medical response   -- EMS, ambulance, transport to care, hospital,
                           treatment, medical evaluation, declined treatment.
  O3 fatality           -- death, fatal, killed.
  O4 post-crash response proxies -- police/fire attendance, claimant/claim
                           language, allegations, attorneys. Not outcomes in
                           themselves, but their presence is driven by the
                           outcome (injury crashes get police and claims).
  O5 reportability statements -- "reporting this crash under Request No. 1 of
                           Standing General Order 2021-01 because ...". The SGO
                           request number is itself determined by the outcome
                           (Request No. 1 is triggered by hospital-treated
                           injury, fatality, VRU involvement, airbag deployment
                           or tow-away), so even the bare request number leaks.
                           Also covers report-maintenance sentences that name
                           the severity field ("to correct ... the HIGHEST
                           INJURY SEVERITY field").

Tiers:
  outcome   = O1-O3, O5 the primary mask
  strict    = O1-O5     robustness: also removes response proxies
  preimpact = strict mask applied to the text up to and including the first
              sentence describing contact; everything after contact is dropped.
              A conservative lower bound -- it also removes legitimate
              post-impact crash dynamics.

Design choices that matter for the leakage argument:
  - Whole sentences are deleted, not spans. A sentence reporting an outcome
    usually carries it in more than the trigger word ("was taken by ambulance
    to UCSF with a neck complaint").
  - Nothing is put in their place. A [MASK] token would itself be a feature:
    injury narratives would carry more of them.
  - Records are never dropped, even if masking empties them. Dropping would
    make the sample depend on the outcome.
  - The lexicon is intentionally recall-oriented; its precision and recall
    against two human coders are measured by annotate.score_outcome_mask, and
    the LLM tagger (leakage.tag_outcome_spans) is unioned with it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# --------------------------------------------------------------------------
# Lexicon. Printed verbatim in the paper's appendix; keep each entry readable.
# --------------------------------------------------------------------------
_O1 = [
    r"injur\w*",                  # injury, injuries, injured, uninjured
    r"hurt\b",
    r"unharmed",
    r"pain\b|pains\b|painful",
    r"\bache[sd]?\b|\baching\b",
    r"\bsore(ness)?\b",
    r"complain\w*\s+of",
    r"bruis\w*",
    r"laceration\w*|abrasion\w*|contusion\w*",
    r"concussion\w*",
    r"fractur\w*",
    r"broken\s+(bone|arm|leg|wrist|ankle|rib|nose|hand|foot|collarbone)\w*",
    r"whiplash",
    r"sprain\w*",
    r"bleed\w*|\bblood\w*",
    r"unconscious\w*|unresponsive|lost\s+consciousness",
    r"\bwound\w*",
    r"casualt\w*",
    r"no\s+one\s+was\s+(hurt|harmed)",
]
_O2 = [
    r"ambulance\w*",
    r"\bEMS\b|\bEMTs?\b",
    r"paramedic\w*",
    r"hospital\w*",
    r"\bmedical\w*|\bmedic\b|\bmedics\b",
    r"transport(ed|ing)\s+(to|by|via|from)",
    r"taken\s+to\s+(the\s+|a\s+)?(hospital|emergency|clinic|medical|urgent)",
    r"first\s+aid",
    r"emergency\s+room|\bER\b|urgent\s+care",
    r"\bclinic\b",
    r"\bdoctor\w*|physician\w*",
    r"x-?rays?\b",
    r"treat(ed|ment)\b",
    r"(was|were|be|being)\s+(medically\s+)?(evaluated|examined|assessed|checked)",
    r"(declined|refused)\s+(medical|treatment|transport|care|an\s+ambulance)",
]
_O3 = [
    r"fatal\w*",
    r"\bdied\b|\bdies\b|\bdying\b",
    r"\bdeaths?\b|\bdead\b(?![-\s]end)",
    r"deceased",
    r"\bkilled\b",
    r"passed\s+away|succumbed",
    r"life[-\s]threatening",
]
_O4 = [
    r"\bpolice\b|\bofficers?\b|law\s+enforcement|\bPD\b",
    r"\bsheriff\w*|\btroopers?\b|highway\s+patrol|\bCHP\b",
    r"fire\s+(department|dept|fighter|crew|truck)\w*|firefighter\w*",
    r"first\s+responder\w*",
    r"emergency\s+(personnel|responder\w*|services|vehicle\w*|crew\w*)",
    r"\b911\b",
    r"claimant\w*|\bclaim(s|ed)?\b",
    r"alleg\w*",
    r"attorney\w*|lawyer\w*|lawsuit\w*|litigation",
    r"\bcit(ed|ation)\b",
]

_O5 = [
    r"request\s+no\.?\s*\d",
    r"standing\s+general\s+order|\bSGO\b",
    r"reporting\s+(this|the)\s+(crash|incident|collision)\s+(under|pursuant|because|as)",
    r"reportab\w*",
    r"injury\s+severity",
]

LEXICON: dict[str, list[str]] = {"O1": _O1, "O2": _O2, "O3": _O3, "O4": _O4,
                                 "O5": _O5}
CATEGORY_RE: dict[str, re.Pattern] = {
    c: re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    for c, pats in LEXICON.items()
}
TIERS: dict[str, tuple[str, ...]] = {
    "outcome": ("O1", "O2", "O3", "O5"),
    "strict": ("O1", "O2", "O3", "O4", "O5"),
    "preimpact": ("O1", "O2", "O3", "O4", "O5"),
}

# Contact sentence for the preimpact tier. The generic noun "collision" is
# deliberately absent: templated openings such as "a Waymo AV ... was in a
# collision involving a passenger car" would otherwise truncate every such
# narrative to its first sentence, before any crash dynamics are described.
CONTACT_RE = re.compile(
    r"made\s+contact|\bcontact(ed|ing)?\s+(the|with|its)\b|\bcollided\b|"
    r"\bstruck\b|\bstrikes?\b|\bstriking\b|\bhit\b|\bhits\b|\bimpact(ed|ing|s)?\b|"
    r"rear[-\s]ended|sideswip\w*|\bclipped\b|\bbumped\b|\bran\s+into\b",
    re.IGNORECASE)

# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------
_ABBREV = re.compile(
    r"(\b(St|Ave|Blvd|Rd|Dr|Hwy|Mr|Mrs|Ms|No|approx|Approx|vs|etc|Inc|Co|Corp|"
    r"Jr|Sr|Lt|Sgt|Mt|Ft|Ln|Pkwy|Ct|Pl|Sq|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|"
    r"Oct|Nov|Dec)\.|\b[A-Z]\.|\b(e\.g|i\.e|a\.m|p\.m|U\.S)\.)$")
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n\s*\n|\n(?=\s*[A-Z\[(\"'])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets of each sentence in text.

    Offsets rather than strings, so detectors that report character spans (the
    LLM tagger) and detectors that work per sentence (the lexicon) flag the
    same units. Empty and whitespace-only pieces are not sentences.
    """
    text = text or ""
    cuts, pos = [], 0
    for m in _BOUNDARY.finditer(text):
        piece = text[pos:m.start()].rstrip()
        # A boundary after a known abbreviation ("Mission St. The AV ...") is
        # not a sentence end, except when the next token is clearly a new
        # sentence start after a street suffix -- we accept that small
        # over-merge: merging two sentences can only make the mask remove more.
        if _ABBREV.search(piece):
            continue
        cuts.append((pos, m.start()))
        pos = m.end()
    cuts.append((pos, len(text)))
    out = []
    for s, e in cuts:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def sentence_categories(sentence: str) -> set[str]:
    return {c for c, rx in CATEGORY_RE.items() if rx.search(sentence)}


def spans_to_sentences(text: str, sentences: list[tuple[int, int]],
                       spans: Iterable[str]) -> set[int]:
    """Indices of sentences overlapped by any verbatim span.

    The LLM tagger returns verbatim substrings. Every occurrence is located
    (whitespace-insensitively) and every sentence it overlaps is flagged; a
    span that cannot be found in the text is ignored here and counted by the
    caller as unaligned.
    """
    flagged: set[int] = set()
    for sp in spans:
        sp = (sp or "").strip()
        if not sp:
            continue
        pat = re.compile(r"\s+".join(re.escape(w) for w in sp.split()),
                         re.IGNORECASE)
        for m in pat.finditer(text):
            for i, (s, e) in enumerate(sentences):
                if m.start() < e and m.end() > s:
                    flagged.add(i)
    return flagged


def span_found(text: str, span: str) -> bool:
    sp = (span or "").strip()
    if not sp:
        return False
    pat = re.compile(r"\s+".join(re.escape(w) for w in sp.split()), re.IGNORECASE)
    return pat.search(text or "") is not None


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
@dataclass
class MaskResult:
    text: str
    n_sentences: int
    removed: list[int] = field(default_factory=list)      # sentence indices
    removed_lexicon: list[int] = field(default_factory=list)
    removed_llm: list[int] = field(default_factory=list)
    truncated_after: Optional[int] = None                 # preimpact only
    categories: list[str] = field(default_factory=list)
    chars_removed: int = 0


def mask(text: str, tier: str = "outcome",
         llm_spans: Optional[Iterable[str]] = None,
         llm_categories: Optional[Iterable[str]] = None) -> MaskResult:
    """Delete outcome sentences (and, for preimpact, everything after contact).

    llm_spans: verbatim outcome spans from the LLM tagger for this record. They
    are unioned with the lexicon. llm_categories, parallel to llm_spans, lets
    the outcome tier ignore tagger spans the tagger itself labelled O4.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {list(TIERS)}")
    text = text or ""
    cats_in_tier = set(TIERS[tier])
    sents = split_sentences(text)

    lex_flag, cats = set(), set()
    for i, (s, e) in enumerate(sents):
        hit = sentence_categories(text[s:e]) & cats_in_tier
        if hit:
            lex_flag.add(i)
            cats |= hit

    llm_flag: set[int] = set()
    if llm_spans is not None:
        spans = list(llm_spans)
        lc = list(llm_categories) if llm_categories is not None else [None] * len(spans)
        keep = [sp for sp, c in zip(spans, lc) if c is None or c in cats_in_tier]
        llm_flag = spans_to_sentences(text, sents, keep)
        cats |= {c for c in lc if c in cats_in_tier}

    drop = lex_flag | llm_flag
    truncated_after = None
    if tier == "preimpact":
        for i, (s, e) in enumerate(sents):
            if CONTACT_RE.search(text[s:e]):
                truncated_after = i
                break
        if truncated_after is not None:
            drop |= set(range(truncated_after + 1, len(sents)))

    kept = [text[s:e] for i, (s, e) in enumerate(sents) if i not in drop]
    out = " ".join(kept)
    return MaskResult(
        text=out, n_sentences=len(sents), removed=sorted(drop),
        removed_lexicon=sorted(lex_flag), removed_llm=sorted(llm_flag),
        truncated_after=truncated_after, categories=sorted(cats),
        chars_removed=len(text) - len(out))


def residual_hits(text: str, tier: str = "outcome") -> int:
    """Number of sentences in (already masked) text that still hit the lexicon.

    Must be zero on any masked corpus; build_masked_corpus asserts it.
    """
    cats = set(TIERS[tier])
    return sum(1 for s, e in split_sentences(text)
               if sentence_categories(text[s:e]) & cats)


def lexicon_appendix_rows() -> list[tuple[str, str]]:
    """(category, pattern) pairs for the paper appendix."""
    return [(c, p) for c, pats in LEXICON.items() for p in pats]


CATEGORY_NAMES = {"O1": "injury status", "O2": "medical response",
                  "O3": "fatality", "O4": "response proxies",
                  "O5": "reportability"}


def write_lexicon_table(path: str) -> None:
    """LaTeX appendix table of the lexicon, generated from LEXICON itself so the
    paper can never drift from the code that did the masking."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    esc = lambda p: (p.replace("\\", r"\textbackslash{}").replace("_", r"\_")
                      .replace("&", r"\&").replace("#", r"\#").replace("$", r"\$")
                      .replace("{", r"\{").replace("}", r"\}")
                      .replace(r"\textbackslash\{\}", r"\textbackslash{}")
                      .replace("^", r"\^{}").replace("~", r"\~{}"))
    lines = [r"\begin{tabular}{p{0.17\linewidth}p{0.75\linewidth}}", r"\toprule",
             r"Category & Patterns (case-insensitive; a sentence matching any is flagged) \\",
             r"\midrule"]
    for c, pats in LEXICON.items():
        tiers = ", ".join(t for t, cs in TIERS.items() if c in cs)
        # Break opportunities after alternations so long patterns wrap in the column.
        body = r" \quad ".join(r"\texttt{" + esc(p).replace("|", r"|\allowbreak{}") + "}"
                               for p in pats)
        lines.append(f"{c} {CATEGORY_NAMES[c]} ({tiers}) & {body} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))
