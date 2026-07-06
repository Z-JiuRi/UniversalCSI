#!/usr/bin/env bash
# Run sampling + NMSE evaluation for all completed (lr=5e-5) experiments
# Evaluates on both training set (100k paired codes) and test set (20k generated codes)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

PYTHON="conda run -n torch python"
GPU="${GPU:-0}"
TARGET_EXP="exps/COST2100/in/seed42/transnet_transnet"
DECODER_ARGS_JSON="${TARGET_EXP}/args.json"
TRAIN_CODE_PATH="${TARGET_EXP}/codewords/train_code.pt"
TEST_CODE_PATH="${TARGET_EXP}/codewords/test_code.pt"
TRAIN_CSI_PATH="/storage/hujiacong/zxd/datasets/cost2100/in_train.pt"
TEST_CSI_PATH="/storage/hujiacong/zxd/datasets/cost2100/in_test.pt"

EXPS_BASE="decoder_param_fm/exps"

# Completed experiments (all lr=5e-5 with Finished training. + best_loss.pth)
COMPLETED=(
  "random_film_rms_tok512_h2048_lr5e-5_ep400_seed42"
  "random_film_zscore_tok512_h2048_lr5e-5_ep400_seed42"
  "random_hyper_lora_rms_tok512_h2048_lr5e-5_ep400_seed42"
  "set_transformer_cross_attention_rms_tok512_h2048_lr5e-5_ep400_seed42"
  "set_transformer_film_rms_tok512_h2048_lr5e-5_ep400_seed42"
  "set_transformer_film_zscore_tok512_h2048_lr5e-5_ep400_seed42"
  "set_transformer_hyper_lora_zscore_tok512_h2048_lr5e-5_ep400_seed42"
  "svd_cross_attention_zscore_tok512_h2048_lr5e-5_ep400_seed42"
)

echo "============================================"
echo "Batch inference + NMSE evaluation"
echo "GPU: ${GPU}"
echo "Target: ${TARGET_EXP}"
echo "============================================"

for exp_name in "${COMPLETED[@]}"; do
  exp_dir="${EXPS_BASE}/${exp_name}"
  gen_output="${exp_dir}/generated/generated_decoder.pth"
  nmse_train="${exp_dir}/generated/nmse_train.json"
  nmse_test="${exp_dir}/generated/nmse_test.json"
  gen_log="${exp_dir}/generated/sample.log"

  # Skip if already done (both train and test NMSE exist)
  if [ -f "$nmse_train" ] && [ -f "$nmse_test" ]; then
    echo "[$exp_name] Already completed. Skipping."
    continue
  fi

  echo ""
  echo "----------------------------------------"
  echo "[$exp_name] Step 1: Sampling generated decoder..."
  echo "----------------------------------------"

  mkdir -p "${exp_dir}/generated"

  $PYTHON -u decoder_param_fm/sample_param_fm.py \
    --exp_dir "$exp_dir" \
    --checkpoint "${exp_dir}/checkpoints/best_loss.pth" \
    --output "$gen_output" \
    --ode_steps 16 \
    --gpu "$GPU" \
    --max_guide_codes 0 \
    2>&1 | tee "$gen_log"

  echo ""
  echo "----------------------------------------"
  echo "[$exp_name] Step 2a: Evaluating NMSE on TRAINING set..."
  echo "----------------------------------------"

  $PYTHON -u decoder_param_fm/test_generated_nmse.py \
    --decoder_state "$gen_output" \
    --decoder_args_json "$DECODER_ARGS_JSON" \
    --code_path "$TRAIN_CODE_PATH" \
    --csi_path "$TRAIN_CSI_PATH" \
    --batch_size 1024 \
    --max_samples 0 \
    --gpu "$GPU" \
    --output_json "$nmse_train" \
    2>&1 | tee "${exp_dir}/generated/nmse_train_eval.log"

  echo ""
  echo "----------------------------------------"
  echo "[$exp_name] Step 2b: Evaluating NMSE on TEST set..."
  echo "----------------------------------------"

  $PYTHON -u decoder_param_fm/test_generated_nmse.py \
    --decoder_state "$gen_output" \
    --decoder_args_json "$DECODER_ARGS_JSON" \
    --code_path "$TEST_CODE_PATH" \
    --csi_path "$TEST_CSI_PATH" \
    --batch_size 1024 \
    --max_samples 0 \
    --gpu "$GPU" \
    --output_json "$nmse_test" \
    2>&1 | tee "${exp_dir}/generated/nmse_test_eval.log"

  echo ""
  echo "[$exp_name] Results:"
  echo "  Train NMSE: $(python3 -c "import json; print(json.load(open('$nmse_train'))['nmse_db'])" 2>/dev/null || echo 'N/A') dB"
  echo "  Test  NMSE: $(python3 -c "import json; print(json.load(open('$nmse_test'))['nmse_db'])" 2>/dev/null || echo 'N/A') dB"
done

echo ""
echo "============================================"
echo "SUMMARY OF ALL RESULTS"
echo "============================================"
printf "%-60s %12s %12s\n" "Experiment" "Train NMSE" "Test NMSE"
printf "%-60s %12s %12s\n" "----------" "----------" "---------"
for exp_name in "${COMPLETED[@]}"; do
  nmse_train_file="${EXPS_BASE}/${exp_name}/generated/nmse_train.json"
  nmse_test_file="${EXPS_BASE}/${exp_name}/generated/nmse_test.json"
  train_nmse=$(python3 -c "import json; print(json.load(open('$nmse_train_file'))['nmse_db'])" 2>/dev/null || echo "N/A")
  test_nmse=$(python3 -c "import json; print(json.load(open('$nmse_test_file'))['nmse_db'])" 2>/dev/null || echo "N/A")
  printf "%-60s %12s %12s\n" "$exp_name" "$train_nmse" "$test_nmse"
done
