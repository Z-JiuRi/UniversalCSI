#!/bin/bash

# 等待 canonical encoder 实验保存 codeword 后，自动为每个已完成实验启动 adapter。
#
# 当前默认服务于这组实验：
#   exps/COST2100/in/encoder_canonical/aux_pca_1e-2_code_mean1e-4_cov1e-4
#
# 规则：
# - decoder 固定为同一 scheme 下的 seed42_transnet_transnet。
# - encoder 使用 canonical_root 下每个 seed*_ENCODER_DECODER 子目录自己的 seed 和 encoder 架构。
# - 当 encoder 实验和固定 decoder 实验都同时具备：
#     checkpoints/best_nmse.pth
#     codewords/train_code.pt
#   就认为 ready，并启动 adapter。
# - 默认跳过 seed42_transnet_transnet 自己，因为它就是固定 decoder/teacher。
#
# 常用用法：
#   bash scripts/run_canonical_adapter_when_ready.sh
#
# 指定 scheme：
#   schemes="aux_pca_1e-2_code_mean1e-4_cov1e-4" \
#     bash scripts/run_canonical_adapter_when_ready.sh
#
# 指定 adapter 训练 GPU/轮数：
#   gpu=1 epochs=400 bash scripts/run_canonical_adapter_when_ready.sh
#
# 只跑某些 encoder 架构：
#   encoder_filter="csinet clnet crnet" bash scripts/run_canonical_adapter_when_ready.sh
#
# 只跑某些 seed：
#   seed_filter="2026 3407" bash scripts/run_canonical_adapter_when_ready.sh
#
# 包含 seed42_transnet_transnet 自身：
#   skip_decoder_self=0 bash scripts/run_canonical_adapter_when_ready.sh
#
# 已启动过的 adapter 会写入 .adapter_launched marker。设置 force=1 可重新启动。

set -euo pipefail

schemes=${schemes:-"aux_pca_1e-2_code_mean1e-4_cov1e-4"}
canonical_base=${canonical_base:-exps/COST2100/in/encoder_canonical}
adapter_base=${adapter_base:-COST2100/in/encoder_canonical/adapter}

decoder_seed=${decoder_seed:-42}
decoder_encoder=${decoder_encoder:-transnet}
decoder_decoder=${decoder_decoder:-transnet}
skip_decoder_self=${skip_decoder_self:-1}

encoder_filter=${encoder_filter:-}
seed_filter=${seed_filter:-}

gpu=${gpu:-6}
epochs=${epochs:-100}
poll_seconds=${poll_seconds:-120}
force=${force:-0}

adapter=${adapter:-gated_lowrank_affine_mlp}
adapter_rank=${adapter_rank:-32}
adapter_hidden_dim=${adapter_hidden_dim:-2048}
adapter_gate_init=${adapter_gate_init:-0.1}
lambda_recon=${lambda_recon:-1.0}
lambda_code=${lambda_code:-1e-3}
lambda_fc=${lambda_fc:-1e-2}
lambda_recT=${lambda_recT:-0.0}
lr_init=${lr_init:-5e-4}

contains_word() {
  local list=$1
  local word=$2
  [ -z "${list}" ] && return 0
  local item
  for item in ${list}; do
    [ "${item}" = "${word}" ] && return 0
  done
  return 1
}

parse_exp_name() {
  local name=$1
  if [[ ! "${name}" =~ ^seed([0-9]+)_(.+)_([^_]+)$ ]]; then
    return 1
  fi
  parsed_seed="${BASH_REMATCH[1]}"
  parsed_decoder="${BASH_REMATCH[3]}"
  parsed_encoder="${BASH_REMATCH[2]}"
  return 0
}

is_complete_exp() {
  local exp_dir=$1
  [ -f "${exp_dir}/checkpoints/best_nmse.pth" ] && \
  [ -f "${exp_dir}/codewords/train_code.pt" ]
}

adapter_exp_name() {
  local scheme=$1
  local enc_seed=$2
  local enc_encoder=$3
  local enc_decoder=$4

  echo "${adapter_base}/${scheme}/${adapter}/rank${adapter_rank}_hidden${adapter_hidden_dim}_gate${adapter_gate_init}_code${lambda_code}_fc${lambda_fc}_lr${lr_init}_ep${epochs}/enc_seed${enc_seed}_${enc_encoder}_${enc_decoder}_dec_seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}"
}

launch_adapter() {
  local scheme=$1
  local enc_seed=$2
  local enc_encoder=$3
  local enc_decoder=$4
  local enc_dir=$5
  local dec_dir=$6

  local exp_name
  exp_name=$(adapter_exp_name "${scheme}" "${enc_seed}" "${enc_encoder}" "${enc_decoder}")
  local exp_dir="exps/${exp_name}"
  local marker="${exp_dir}/.adapter_launched"

  if [ "${force}" != "1" ] && [ -f "${marker}" ]; then
    echo "[skip] ${scheme}/${enc_dir##*/}: adapter already launched"
    return 0
  fi

  mkdir -p "${exp_dir}"
  date '+%Y-%m-%d %H:%M:%S' > "${marker}"

  echo "[launch] ${scheme}/${enc_dir##*/} -> seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}"
  echo "         exp_name: ${exp_name}"

  canonical_scheme="${scheme}" \
  encoder="${enc_encoder}" decoder="${enc_decoder}" \
  seed="${enc_seed}" decoder_seed="${decoder_seed}" gpu="${gpu}" epochs="${epochs}" \
  pretrained_encoder="${enc_dir}/checkpoints/best_nmse.pth" \
  pretrained_decoder="${dec_dir}/checkpoints/best_nmse.pth" \
  teacher_code="${dec_dir}/codewords/train_code.pt" \
  adapter="${adapter}" adapter_rank="${adapter_rank}" \
  adapter_hidden_dim="${adapter_hidden_dim}" \
  adapter_gate_init="${adapter_gate_init}" \
  lambda_recon="${lambda_recon}" lambda_code="${lambda_code}" \
  lambda_fc="${lambda_fc}" lambda_recT="${lambda_recT}" \
  lr_init="${lr_init}" exp_name="${exp_name}" \
  bash scripts/train_canonical_adapter.sh
}

pending_count() {
  local count=0
  local scheme scheme_dir dec_dir enc_dir name exp_name marker

  for scheme in ${schemes}; do
    scheme_dir="${canonical_base}/${scheme}"
    dec_dir="${scheme_dir}/seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}"
    [ -d "${scheme_dir}" ] || continue
    [ -d "${dec_dir}" ] || continue

    for enc_dir in "${scheme_dir}"/seed*_*; do
      [ -d "${enc_dir}" ] || continue
      name=${enc_dir##*/}
      parse_exp_name "${name}" || continue

      contains_word "${encoder_filter}" "${parsed_encoder}" || continue
      contains_word "${seed_filter}" "${parsed_seed}" || continue

      if [ "${skip_decoder_self}" = "1" ] && \
         [ "${parsed_seed}" = "${decoder_seed}" ] && \
         [ "${parsed_encoder}" = "${decoder_encoder}" ] && \
         [ "${parsed_decoder}" = "${decoder_decoder}" ]; then
        continue
      fi

      exp_name=$(adapter_exp_name "${scheme}" "${parsed_seed}" "${parsed_encoder}" "${parsed_decoder}")
      marker="exps/${exp_name}/.adapter_launched"
      if [ "${force}" = "1" ] || [ ! -f "${marker}" ]; then
        count=$((count + 1))
      fi
    done
  done

  echo "${count}"
}

echo "Waiting for canonical codewords, then launching adapter experiments."
echo "  schemes: ${schemes}"
echo "  fixed decoder: seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}"
echo "  encoder_filter: ${encoder_filter:-<all>}"
echo "  seed_filter: ${seed_filter:-<all>}"
echo "  adapter: ${adapter}, epochs=${epochs}, gpu=${gpu}"
echo "  poll_seconds: ${poll_seconds}"

while [ "$(pending_count)" -gt 0 ]; do
  launched_any=0

  for scheme in ${schemes}; do
    scheme_dir="${canonical_base}/${scheme}"
    dec_dir="${scheme_dir}/seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}"

    if [ ! -d "${scheme_dir}" ]; then
      echo "[wait] missing scheme dir: ${scheme_dir}"
      continue
    fi

    if ! is_complete_exp "${dec_dir}"; then
      echo "[wait] fixed decoder not ready: ${dec_dir}"
      continue
    fi

    for enc_dir in "${scheme_dir}"/seed*_*; do
      [ -d "${enc_dir}" ] || continue
      name=${enc_dir##*/}
      parse_exp_name "${name}" || continue

      contains_word "${encoder_filter}" "${parsed_encoder}" || continue
      contains_word "${seed_filter}" "${parsed_seed}" || continue

      if [ "${skip_decoder_self}" = "1" ] && \
         [ "${parsed_seed}" = "${decoder_seed}" ] && \
         [ "${parsed_encoder}" = "${decoder_encoder}" ] && \
         [ "${parsed_decoder}" = "${decoder_decoder}" ]; then
        continue
      fi

      exp_name=$(adapter_exp_name "${scheme}" "${parsed_seed}" "${parsed_encoder}" "${parsed_decoder}")
      marker="exps/${exp_name}/.adapter_launched"
      if [ "${force}" != "1" ] && [ -f "${marker}" ]; then
        continue
      fi

      if is_complete_exp "${enc_dir}"; then
        launch_adapter "${scheme}" "${parsed_seed}" "${parsed_encoder}" "${parsed_decoder}" "${enc_dir}" "${dec_dir}"
        launched_any=1
      else
        echo "[wait] encoder not ready: ${enc_dir}"
      fi
    done
  done

  if [ "$(pending_count)" -eq 0 ]; then
    break
  fi

  if [ "${launched_any}" -eq 0 ]; then
    sleep "${poll_seconds}"
  fi
done

echo "All requested adapter experiments have been launched."
