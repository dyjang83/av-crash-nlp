# AV Crash Narratives → Structured Liability & Severity

A schema-constrained LLM pipeline that converts public autonomous-vehicle (AV)
collision narratives (NHTSA SGO 2021-01 and CA DMV OL 316) into structured fields
for insurance triage. On top of that sit three analyses:

- a **leakage-controlled lift test** — does narrative text add predictive value
  for injury severity beyond the structured fields insurers already collect?
- a **two-signal calibration study** — the model's self-reported confidence
  versus the decoder's actual token probabilities, which only open-weight
  backends expose;
- a **human-crash comparison** — where AV crashes differ structurally from
  police-reported human crashes (NHTSA CRSS), once the two reporting regimes are
  harmonized to a common population.

## What is in this repo, and what is *not*

This repository contains the **complete, runnable code**. It does **not** contain
results, and the results are not invented.

- Everything under `data/` (raw, interim, processed) is gitignored. The numbers
  come from running the pipeline below on the **real** public data.
- Generated LaTeX tables and figures are not committed either. Every results
  table is `\input` from `paper/tables/`; if a table has not been generated, a
  placeholder renders in its place. Nothing is typed in by hand.
- Two steps cannot be automated away: (a) downloading the data requires reaching
  `nhtsa.gov` and `dmv.ca.gov`; (b) the gold evaluation set requires **two humans**
  to code ~250 narratives.

This is deliberate. The work's value depends on real data and a real
human-validated gold set; fabricating either would make it useless.

## Layout

```
src/
  fetch/      fetch_sgo.py, fetch_dmv_ol316.py, parse_ol316.py   # acquire + parse
              fetch_crss.py                                      # human comparator
  schema/     schema.py            extraction schema (pydantic, drives constrained decoding)
              distant_map.py       structured codes -> schema enums (distant key)
              crss_map.py          CRSS codes  <-> schema enums (human comparison)
  extract/    prompts.py, llm_extract.py                         # constrained extraction
              token_probs.py       per-field decode-time probabilities
  annotate/   annotation_guide.md, make_gold_sample.py, adjudicate.py
              score_agreement.py   IAA, model-vs-gold, calibration
              score_distant.py     corpus-scale accuracy vs the distant key
              score_confidence.py  the two confidence signals
  features/   build_features.py    S / T / S+T, optional confidence gating
  models/     lift_test.py         the core lift test
              transfer_eval.py     OL 316 -> SGO transfer
              av_vs_human.py       SGO vs CRSS, threshold-harmonized
              severity_dist.py     severity distribution + dominance tests
  utils/      config.py            reads config.yaml (canonical model, etc.)
tests/        93 tests, all synthetic fixtures -- no number here enters the paper
scripts/      run_all.sh, probe_logprobs.py
config.yaml   requirements.txt
```

`paper/` (manuscript sources) is not currently published to this repository.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # for the closed-model extraction backend
export PYTHONPATH=src
```

## Run order (`scripts/run_all.sh` covers steps 1–8)

1. **Fetch** (needs open internet): `python3 -m fetch.fetch_sgo` and
   `python3 -m fetch.fetch_dmv_ol316`. If your environment blocks these domains,
   run this step elsewhere and copy `data/raw/` over.
2. **Parse**: `python3 -m fetch.parse_ol316` → `data/interim/narratives.jsonl`.
3. **Sample gold**: `python3 -m annotate.make_gold_sample --n 250 --seed 11`.
   Then **annotate by hand** (`data/gold/annotator_A.jsonl`, `_B.jsonl`) per
   `src/annotate/annotation_guide.md`, and adjudicate into `data/gold/gold.jsonl`.
4. **Extract**: `python3 -m extract.llm_extract --input data/interim/narratives.jsonl
   --output data/processed/extractions.jsonl --backend anthropic
   --model claude-sonnet-4-6`.
   Open-weight models go through `--backend openai` against vLLM
   (`--server-type vllm`) or Ollama (`--server-type ollama`). Only these return
   token logprobs; see *Two confidence signals* below.
5. **Score**: `python3 -m annotate.score_agreement` → IAA, extraction and
   calibration tables.
6. **Score at corpus scale**: `python3 -m annotate.score_distant` → accuracy
   against the distant key, with the abstention decomposition.
7. **Transfer**: `python3 -m models.transfer_eval --model claude-sonnet-4-6`.
8. **Lift test**: `python3 -m models.lift_test --embed tfidf` →
   `paper/tables/tab_lift.tex` and `data/processed/lift_test.json`.
9. **Confidence signals** (needs an open-weight run from step 4, with logprobs):
   `python3 -m annotate.score_confidence --extractions <that file>
   --model qwen2.5:7b --committed-only`.
10. **Human comparison**: `python3 -m fetch.fetch_crss` then
    `python3 -m models.av_vs_human --n-boot 500` and
    `python3 -m models.severity_dist --threshold any_injury`.
11. **Build paper**: `cd paper && latexmk -pdf main.tex`.

## Verify the analysis code now (no data needed)

```bash
python3 -m pytest tests/ -q       # 93 tests, synthetic fixtures only
```

## Integrity design choices baked into the code

- **No fault prediction.** `contributory_party` is validated as an extraction
  target only; it is dropped from the lift-test features (`DROP_FROM_T` in
  `build_features.py`) because predicting a text-derived field from text is
  circular.
- **No outcome leakage.** `narrative_injury_severity` (text-derived) is dropped
  from the predictors; the lift-test target is the independent *structured* SGO
  severity field.
- **One canonical extraction model, enforced.** `build_features.load()` filters
  to `extraction.canonical_model` from `config.yaml` before deduplicating.
  Without the filter, a report_id extracted by several models resolved by JSONL
  append order — and because the contested rows were exactly those where the
  closed model's extraction failed schema validation, an open model silently
  backfilled the *hardest* records.
- **Entity-grouped CV.** Folds are grouped by reporting entity so models cannot
  exploit operator-specific phrasing.
- **Effect size with the p-value**, never a p-value alone.
- **Nested hyperparameter selection.** `C` (logreg) and depth/learning-rate/L2
  (gbm) are chosen per representation by an inner GroupKFold search run on the
  training portion of each outer fold only, so the held-out rows used for the
  reported lift never inform the search that produced them.
- **Estimator variance is reported, not hidden.** Gradient boosting draws its
  early-stopping validation split from the training fold; with 14 fatalities in
  3,272 records that split lands differently per seed and moves QWK by up to
  ~0.10 — wider than the bootstrap CI, which resamples examples for a *single*
  fitted model and cannot see it. `lift_test.py` averages over seeds and reports
  the spread alongside the CI.
- **Requested-but-unmeasurable is an error, not a no-op.** Asking for confidence
  gating against extractions that carry no token probabilities raises instead of
  silently completing and writing plausible numbers.

## Distant supervision

Extraction accuracy is evaluated primarily against a **distant-supervision key**:
the structured codes the reporting entities file alongside each narrative (SGO
incident CSVs; OL 316 checkbox groups). This supplies **31,204 supervised field
decisions across ten fields** at no annotation cost, far more than the
hand-annotated sample provides.

- `src/schema/distant_map.py` — value-level mapping from structured codes to
  schema enums. Every field defines a *reducer* that projects the key and the
  prediction into the coarsest space both can express, or drops the row when a
  mapping would require inventing a convention. Read the module docstring before
  changing any mapping.
- `src/annotate/score_distant.py` — scoring, coverage, abstention decomposition,
  and the human-vs-distant validity check. Emits `tab_distant.tex` and
  `tab_distant_cov.tex`.
- `tests/test_distant.py` — pins the specific ways the key could be silently
  wrong (NaN leakage, coarsening direction, the weather indicator columns, the
  two collision-type binaries staying separate).

`contributory_party` has no structured counterpart in either corpus and remains
doubly hand-annotated. Distant supervision replaces the convention-driven
fields, not the judgment-driven one.

**Abstention is not error.** Where the narrative does not state a condition the
extractor answers `unknown`; against a key that always committed, that scores as
a miss. The reported figures separate abstention rate from accuracy on the
committed subset, and the distinction matters far beyond presentation — see below.

## Two confidence signals

The schema asks the model to write a `confidence` float into its own output.
That is a *self-report*. An open-weight model served locally exposes a second,
independent quantity: the **decode-time probability** of each field's value. The
Anthropic tool-calling API does not, which is why this analysis covers only the
open-weight backends.

- `src/extract/token_probs.py` reconstructs the emitted JSON from its tokens,
  locates each field's value span by character offset, and renormalizes over the
  *legal* enum members at each position — a chain rule restricted to the schema,
  giving a posterior over categories rather than token strings. It distinguishes
  top-k truncation from schema determinism: once `parking` is emitted, only
  `parking_lot_private` remains reachable, and having no alternatives there is
  correct rather than defective.
- `scripts/probe_logprobs.py` settles, before any run, whether a server returns
  probabilities from before or after the schema constraint is applied. The
  answer changes what every downstream number means and cannot be read off the
  values.
- `src/annotate/score_confidence.py` scores both signals on ECE (calibration),
  AUROC (discrimination), out-of-fold isotonic recalibration, and a
  risk-coverage curve. It shares its ECE binning with `score_agreement.py` so
  the figures stay comparable to previously published ones.

Score with `--committed-only`. On the full sample the token signal's AUROC
*inverts* on high-abstention fields, because `unknown` is a sharply-peaked safe
continuation that the distant key always marks wrong — the field ranking becomes
abstention rate rather than signal quality.

## Human comparison (CRSS)

`src/models/av_vs_human.py` compares SGO against NHTSA CRSS, the nationally
representative sample of police-reported crashes.

SGO's mandatory-filing threshold sits far below the bar at which a human crash
generates a police report, so the raw corpora are not comparable and no
reweighting makes them so — the populations differ in what they *contain*. Both
sides are therefore restricted to a common **outcome** bar (any-injury, and
separately tow-away), with retained-N reported on each side as the measurement
of the gap. Crash *rates* are deliberately not computed: no exposure denominator
exists here, which is also why no underreporting correction is needed.

CRSS is a complex survey sample. Every human-side estimate is `WEIGHT`-ed and
variance comes from resampling PSUs within (year, stratum) cells; an iid
bootstrap would ignore the clustering and report intervals far too narrow.
`crss_map.py` projects each side independently into a shared space — there is no
pairing, because an SGO crash and a CRSS crash are different crashes — using the
coarsest space both can express *totally*, so projection drops only genuinely
missing values.

`src/models/severity_dist.py` tests the severity distributions for first-order
stochastic dominance in both directions and locates CDF crossings, rather than
collapsing to a single directional claim.

### Dependencies

`pypdf` is **required**, not optional. OL 316 narratives live in AcroForm text
fields, not in the page content stream — without pypdf every one of the 868
PDFs falls through to the page-text path, finds only the blank form template,
and reports "no narrative found". The parser raises a clear error instead of
failing silently. The legacy `PyPDF2` package is not a substitute.

`sentence-transformers` is left commented out (it pulls in torch) and is needed
only for `lift_test.py --embed sbert`, the representation-robustness check.

### SGO schema generations

The SGO third amendment (June 2025) restructured the incident CSVs. The parser
covers both generations, but note:

- `Lighting` was **dropped** with no replacement, so the current files supply no
  lighting supervision (2,270 of 3,503 SGO rows). OL 316 checkboxes fill much of
  this gap.
- `ADS Equipped?` → `Engagement Status`, with graded values
  (`Verified Engaged`, `Alleged Engaged`, ...). Alleged values are dropped.
- Weather gained `Partly Cloudy`, `Dust Storm`, `Severe Hurricane` and
  `Structure-Indoor`; `Fog/Smoke` became `Fog/Smoke/Haze`.

The same hazard applies to CRSS, which renumbers codes between annual releases.
`crss_map.py` therefore matches on the decoded `*NAME` companion columns rather
than the integer codes, so a silently shifted code cannot corrupt an estimate.

### OL 316 checkbox decoding

AcroForm names follow `<GROUP> <LETTER> <VEHICLE>` (e.g. `WEATHER A 1`,
`MOVEMENT  B 2` — the double space is in the form). Letters index the printed
option list; the trailing digit is the **vehicle number**, where vehicle 1 is
the AV and vehicle 2 is the other party. `MOVEMENT` and `TYPE` are read for
vehicle 1 only. The form's `ROADWAY` group is roadway *surface* (dry/wet/icy),
not a road classification — OL 316 supplies no road-class or locality
supervision. Verified against all 868 filings with
`python3 -m fetch.parse_ol316 --audit-checkboxes`.
