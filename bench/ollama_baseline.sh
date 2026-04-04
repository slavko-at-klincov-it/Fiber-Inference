#!/bin/bash
# ollama_baseline.sh — Reproduzierbarer Benchmark fuer Ollama
# Misst Prefill (TTFT) und Decode (tok/s) fuer verschiedene Modelle
# Ausfuehren: bash bench/ollama_baseline.sh

set -e

echo "=== Ollama Baseline Benchmark ==="
echo "Date: $(date)"
echo "Ollama: $(ollama --version 2>&1)"
echo ""

# Benchmark function: misst TTFT und Decode tok/s
benchmark() {
    local model=$1
    local prompt=$2
    local label=$3

    echo "--- $label: $model ---"
    echo "Prompt: \"$prompt\""

    # Warmup (first run loads model)
    ollama run "$model" "$prompt" --verbose </dev/null 2>/dev/null >/dev/null || true
    sleep 2

    # 3 timed runs
    for run in 1 2 3; do
        result=$(ollama run "$model" "$prompt /no_think" --verbose </dev/null 2>&1 | grep -E "eval rate|prompt eval rate" || true)
        prefill_rate=$(echo "$result" | grep "prompt eval rate" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        decode_rate=$(echo "$result" | grep -v "prompt" | grep "eval rate" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "  Run $run: prefill=${prefill_rate:-?} tok/s, decode=${decode_rate:-?} tok/s"
    done
    echo ""
}

# Test 1: Qwen3-0.6B (dim=1024 — our ANE sweet spot)
benchmark "qwen3:0.6b" "The capital of France is" "Qwen3-0.6B (dim=1024)"

# Test 2: Qwen3-0.6B longer prompt (RAG-like)
benchmark "qwen3:0.6b" "Given the following context about European capitals: Paris is the capital of France, located on the Seine river. It has a population of about 2.1 million in the city proper. Berlin is the capital of Germany. London is the capital of the United Kingdom. Based on this context, what is the capital of France and what river is it on?" "Qwen3-0.6B RAG-like"

# Test 3: Qwen3-4B (if downloaded)
if ollama list 2>/dev/null | grep -q "qwen3:4b"; then
    benchmark "qwen3:4b" "The capital of France is" "Qwen3-4B (dim=2560)"
fi

echo "=== BASELINE COMPLETE ==="
