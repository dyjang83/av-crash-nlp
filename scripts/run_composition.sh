#!/usr/bin/env bash
# The composition study: given a crash, how does an ADS crash differ from a
# human one? Runs on top of the raw data fetched by run_all.sh steps 1-2.
#
# This is a SEPARATE pipeline from run_all.sh on purpose. run_all.sh produces
# the extraction-pipeline paper (lift test, calibration, distant supervision)
# against a FROZEN schema; adding fields to that schema would invalidate the
# decode-time probabilities the calibration study measures. The composition
# study therefore runs its own schema, its own extraction pass, and its own
# corpus (ADS incidents, not pooled narratives).
#
# Run from the repo root. Steps needing money or humans are flagged.
set -euo pipefail
export PYTHONPATH="src:${PYTHONPATH:-}"

DRIVERLESS="${DRIVERLESS:---driverless-only}"
YEARS="${YEARS:-2021 2022 2023 2024}"

echo "== 1. ADS corpus: exclusions, version collapse, driverless tagging =="
# Same Incident ID already links co-filed reports across entities, so the
# fuzzy layer runs first only to MEASURE the residual and hand back merges.
python -m corpus.dedup_audit
python -m corpus.sgo_ads --merges data/processed/dedup_audit.json
echo "   -> HUMAN STEP (optional): verify data/gold/dedup_audit_sample.jsonl"

echo "== 2. Human comparator: CRSS vehicle-level extract, domain-restricted =="
# Builds TWO frames: the vehicle-level one for composition, and a crash-level
# one for quasi-induced exposure. They are not interchangeable -- a
# vehicle-level filter destroys the striking/struck pairing QIE depends on.
python -m human.crss_extract --years ${YEARS}

echo "== 3. Draw the human double-coding sample (blinded) =="
python -m annotate.make_composition_sample --n 300 --seed 11
echo "   -> HUMAN STEP: two coders fill data/gold/composition_coder_{A,B}.jsonl."
echo "      Six fields have no structured counterpart anywhere; police_reportable"
echo "      is the one every Tier B result rests on."

echo "== 4. Composition extraction (COSTS MONEY: ~\$25 for the full corpus) =="
# Validate on the 300-row sample first, then scale. Prompt caching makes the
# constant ~4k-token prefix cost a tenth of list price; check the reported
# cache hit rate is near 100%.
echo "   validation pass first:"
echo "   python -m extract.composition_extract --only-sample data/gold/composition_coder_A.jsonl"
python -m extract.composition_extract --workers 8

echo "== 5. Reportability tiers A and B + validation checks =="
# Also emits the Tier A tow-definition variants. Tier A's tow leg carries ~90%
# of its records and 672 of those are subject-AV-only tows, which are fleet
# recovery rather than damage -- so every Tier A result is reported under
# any_tow / cp_tow / no_tow and is only as strong as its stability across them.
#
# There is no Tier C. The post-amendment population is empty in the 2021-24
# comparison window (0 of 829), so the regime split is a validation check
# (regime_check) rather than a way of cutting the ADS side down to the human one.
python -m report.reportability

echo "== 5a. Reportability validation that needs NO human coding =="
# Three checks, in the order they were built. The first two are free and run
# off data already on disk; the third costs about \$6 at n=50.
#   - Tier A sufficiency: a structured-injury or airbag crash labelled 'no' is
#     an unambiguous compliance failure.
#   - narrative-incompleteness decomposition: separates rubric error (<1%) from
#     the narrative omitting a fact the structured field records (8-46%). This
#     is the one human coding cannot supply -- two humans reading the same text
#     would make the same 'errors'.
#   - metamorphic tests: twelve controlled edits whose correct DIRECTION is
#     known by construction, closing the false-positive gap the other two leave.
python -m annotate.metamorphic_rubric --n 50 --dry-run
echo "   -> drop --dry-run to actually run it (~\$6 at n=50)"

echo "== 5b. Test-retest stability across two extraction runs (if both exist) =="
# The rubric revision changed only the REPORTABILITY section of the prompt, so
# every other field should be near-identical between runs. Where it is not,
# that is prompt contamination -- and it found some: tow coverage fell 54% with
# accuracy going DOWN, i.e. the model abstained on cases it had been getting right.
python -m annotate.score_composition_retest || \
  echo "   (skipped: needs composition_extractions.rubric_v1.jsonl)"

echo "== 6.1/6.2 Crash-type composition and the multinomial model =="
python -m models.composition ${DRIVERLESS} --years ${YEARS}

echo "== 6.3 Quasi-induced exposure (Tier A only, crash-level human frame) =="
python -m models.qie ${DRIVERLESS} --years ${YEARS} --role-source structured
python -m models.qie ${DRIVERLESS} --years ${YEARS} --role-source extracted \
  --out data/processed/qie_extracted.json

echo "== 6.4 Conditional pre-crash sequences =="
python -m models.sequences ${DRIVERLESS} --years ${YEARS} --source extracted

echo "== 5b. Reweighting: entropy-balance CRSS onto the ADS domain =="
# Speed limit exists only for pre-amendment ADS filings -- the third amendment
# dropped the column and narratives do not state it (298/298 extracted
# 'unknown'). The balanced run is therefore pre-amendment by construction.
python -m human.balance ${DRIVERLESS} --years ${YEARS}

echo "== 7. Robustness =="
echo "   -- supervised-inclusive:"
python -m models.composition --years ${YEARS} --out data/processed/composition_allsup.json
echo "   -- Waymo-excluded (Waymo files 79-100% of the driverless corpus):"
python -m models.composition ${DRIVERLESS} --years ${YEARS} --exclude-entities waymo \
  --out data/processed/composition_nowaymo.json
python -m models.qie ${DRIVERLESS} --years ${YEARS} --exclude-entities waymo \
  --out data/processed/qie_nowaymo.json
echo "   -- driverless definition sensitivity (blank-only vs blank-or-remote):"
echo "      python -m corpus.sgo_ads --driverless-def strict   # then re-run the above"
echo "   -- coder swap (open-weight model, needs a local server):"
echo "      ollama serve && python -m extract.composition_extract \\"
echo "        --backend openai --server-type ollama --model qwen2.5:7b \\"
echo "        --output data/processed/composition_extractions_qwen.jsonl"

echo "Done. Results in data/processed/{ads_corpus,dedup_audit,crss_extract,"
echo "reportability,composition,qie,sequences,balance}.json"
