#!/bin/bash
# Run all codeword analysis scripts in order.
# Usage: bash scripts/run_all_analysis.sh
#
# Pipeline:
#   1) analyze_codewords.py             -- basic per-split summaries  (train, 5k-sample quick scan)
#   2) deep_analyze_train_codewords.py  -- full-data deep analysis (train, 100k samples per model)
#   3) summarize_codeword_analysis.py   -- cross-split markdown summary (fast)
#   4) comprehensive_lora_analysis.py   -- first-pass LoRA-conditioning analysis (train)
#   5) enhanced_lora_analysis.py        -- enhanced per-decoder analysis (train)
#   6) consolidated_analysis.py         -- consolidated report + figures
#
# Output: exps/seed42/COST2100/in/codeword_analysis/

set -euo pipefail

PYTHON=/home/hujiacong/zxd/.envs/miniconda3/envs/torch/bin/python

EXP_ROOT="exps/seed42/COST2100/in"
TRAINING_RESULTS="${EXP_ROOT}/training_results.csv"
ANALYSIS_ROOT="${EXP_ROOT}/codeword_analysis"

echo "============================================"
echo " Codeword Analysis Pipeline (train only)"
echo " Target : ${EXP_ROOT}"
echo " Output : ${ANALYSIS_ROOT}"
echo "============================================"
echo ""

skip_if_done() {
    local marker="$1"
    if [ -f "$marker" ]; then
        echo "  [SKIP] marker $marker exists, already done."
        return 0
    fi
    return 1
}

# =========================================================================
# Step 1 — basic per-split analysis (train, 5k-sample quick scan)
# =========================================================================
marker="${ANALYSIS_ROOT}/train/report.md"
if ! skip_if_done "$marker"; then
    echo "[1] analyze_codewords.py ..."
    $PYTHON scripts/analyze_codewords.py \
        --exp_root "$EXP_ROOT" \
        --split train \
        --out_dir "${ANALYSIS_ROOT}/train" \
        --max_samples 10000 \
        --plot_samples 10000 || echo "  [WARN] step 1 failed, continuing..."
    echo ""
fi

# =========================================================================
# Step 2 — deep full-data analysis (train, 100k samples per model)
# =========================================================================
marker="${ANALYSIS_ROOT}/deep_train_full/full_summary.csv"
if ! skip_if_done "$marker"; then
    echo "[2] deep_analyze_train_codewords.py ..."
    $PYTHON scripts/deep_analyze_train_codewords.py \
        --exp_root "$EXP_ROOT" \
        --training_results "$TRAINING_RESULTS" \
        --out_dir "${ANALYSIS_ROOT}/deep_train_full" || echo "  [WARN] step 2 failed (OOM?), continuing..."
    echo ""
fi

# =========================================================================
# Step 3 — cross-split summary from step 1 output
# =========================================================================
marker="${ANALYSIS_ROOT}/summary_report.md"
if ! skip_if_done "$marker"; then
    echo "[3] summarize_codeword_analysis.py ..."
    $PYTHON scripts/summarize_codeword_analysis.py \
        --analysis_root "$ANALYSIS_ROOT" \
        --out "$marker" || echo "  [WARN] step 3 failed, continuing..."
    echo ""
fi

# =========================================================================
# Step 4 — comprehensive LoRA-conditioning analysis (train)
# =========================================================================
marker="${ANALYSIS_ROOT}/comprehensive_lora/report.md"
if ! skip_if_done "$marker"; then
    echo "[4] comprehensive_lora_analysis.py ..."
    $PYTHON scripts/comprehensive_lora_analysis.py \
        --exp_root "$EXP_ROOT" \
        --training_results "$TRAINING_RESULTS" \
        --out_dir "${ANALYSIS_ROOT}/comprehensive_lora" \
        --n_sampling_models 6 || echo "  [WARN] step 4 failed, continuing..."
    echo ""
fi

# =========================================================================
# Step 5 — enhanced per-decoder LoRA analysis (train)
# =========================================================================
marker="${ANALYSIS_ROOT}/enhanced_lora/report.md"
if ! skip_if_done "$marker"; then
    echo "[5] enhanced_lora_analysis.py ..."
    $PYTHON scripts/enhanced_lora_analysis.py \
        --exp_root "$EXP_ROOT" \
        --training_results "$TRAINING_RESULTS" \
        --out_dir "${ANALYSIS_ROOT}/enhanced_lora" \
        --n_sampling_models 9 || echo "  [WARN] step 5 failed, continuing..."
    echo ""
fi

# =========================================================================
# Step 6 — consolidated analysis
# =========================================================================
marker="${ANALYSIS_ROOT}/consolidated/consolidated_report.md"
if ! skip_if_done "$marker"; then
    echo "[6] consolidated_analysis.py ..."
    $PYTHON scripts/consolidated_analysis.py \
        --exp_root "$EXP_ROOT" \
        --training_results "$TRAINING_RESULTS" \
        --out_dir "${ANALYSIS_ROOT}/consolidated" || echo "  [WARN] step 6 failed, continuing..."
    echo ""
fi

# =========================================================================
echo "============================================"
echo " Pipeline complete."
echo " Master report: ${ANALYSIS_ROOT}/consolidated/consolidated_report.md"
echo "============================================"
