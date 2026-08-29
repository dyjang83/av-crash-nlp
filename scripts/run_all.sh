#!/usr/bin/env bash
# End-to-end pipeline. Steps that need the open internet (fetch) or a GPU
# (open-weight extraction) or humans (annotation) are flagged. Run from repo root.
set -euo pipefail
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "== 1. Fetch raw data (needs nhtsa.gov + dmv.ca.gov reachable) =="
python -m fetch.fetch_sgo
python -m fetch.fetch_dmv_ol316

echo "== 2. Parse into unified narratives.jsonl =="
python -m fetch.parse_ol316

echo "== 2b. Verify the OL 316 checkbox name map against the actual filings =="
# The AcroForm field names are generated from the printed labels but are not
# stable across form revisions. An unrecognised revision must surface here as a
# long "unmapped_ticked" list, not as quiet missing coverage in the results.
python -m fetch.parse_ol316 --audit-checkboxes

echo "== 3. Draw stratified gold sample =="
# Distant supervision (step 5b) now carries the fields with a structured
# counterpart. Hand annotation is needed only for contributory_party, which has
# none -- so this sample is far cheaper to code than it was, but it is still a
# human step and still requires two independent coders.
python -m annotate.make_gold_sample --n 250 --seed 11
echo "   -> HUMAN STEP: code contributory_party in data/gold/annotator_A.jsonl"
echo "      and annotator_B.jsonl, then adjudicate into data/gold/gold.jsonl."
echo "   -> Re-run from step 4 after."

echo "== 4. Run extraction (closed model; set ANTHROPIC_API_KEY) =="
python -m extract.llm_extract \
  --input data/interim/narratives.jsonl \
  --output data/processed/extractions.jsonl \
  --backend anthropic --model claude-sonnet-4-6

echo "   (optional) open-weight models via a running vLLM server:"
echo "   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000"
echo "   python -m extract.llm_extract --input data/interim/narratives.jsonl \\"
echo "       --output data/processed/extractions.jsonl --backend openai \\"
echo "       --model Qwen/Qwen2.5-7B-Instruct"

echo "== 5. Score extraction: IAA, model-vs-gold, calibration -> paper/tables =="
python -m annotate.score_agreement

echo "== 5b. Score against the distant-supervision key (corpus scale) -> paper/tables =="
python -m annotate.score_distant

echo "== 6. Cross-source transfer (DMV -> SGO) -> paper/tables =="
python -m models.transfer_eval --model claude-sonnet-4-6

echo "== 7. Severity lift test -> paper/tables/tab_lift.tex, data/processed/lift_test.json =="
python -m models.lift_test --embed tfidf

echo "== 7b. (optional) Representation-robustness check with SBERT embeddings =="
echo "   requires: pip install sentence-transformers (pulls in torch)"
echo "   writes to tab_lift_sbert.tex / lift_test_sbert.json -- never touches the"
echo "   canonical tfidf files, so it's safe to skip or rerun independently."
# python -m models.lift_test --embed sbert

echo "== 8. Build the paper =="
( cd paper && latexmk -pdf -interaction=nonstopmode main.tex )

echo "Done. Populate any remaining \\TODO macros in the paper from the JSON/tables."
