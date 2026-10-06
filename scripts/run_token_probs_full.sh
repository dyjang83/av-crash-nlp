#!/usr/bin/env bash
# Open-weight extraction with token logprobs over the full corpus (4,367 records).
#
# Sequential on purpose: on an 8 GB machine Ollama cannot keep both models
# resident, so running them concurrently would reload a model on every request.
#
#   Qwen2.5-7B  resumes data/processed/extractions_qwen_full.jsonl (2,973 done;
#               re-extraction under Ollama 0.32.1 reproduced stored rows exactly).
#               Pre-resume copy: data/processed/archive_run250/extractions_qwen_full.n2973.jsonl
#   Llama-3.1-8B runs fresh into extractions_llama_full.jsonl: the current server
#               reproduces its extractions but shifts p_renorm by up to ~0.01, so
#               the 245-record extractions_llama_gold.jsonl is not reused. Gold
#               records come first (narratives_goldfirst.jsonl).
#
# Both runs are resumable: rerun this script after an interruption.
# Usage: nohup caffeinate -i scripts/run_token_probs_full.sh > data/processed/logs/token_probs_full.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../src"

OLLAMA="--backend openai --server-type ollama"
P=../data/processed

echo "== $(date) Qwen2.5-7B (resume) =="
python -m extract.llm_extract $OLLAMA --model qwen2.5:7b \
  --input ../data/interim/narratives.jsonl --output $P/extractions_qwen_full.jsonl

echo "== $(date) Llama-3.1-8B (full) =="
python -m extract.llm_extract $OLLAMA --model llama3.1:8b \
  --input ../data/interim/narratives_goldfirst.jsonl --output $P/extractions_llama_full.jsonl

echo "== $(date) done =="
