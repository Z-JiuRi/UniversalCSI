#!/bin/bash

# 批量启动小参数 code-only flow matching 实验。
#
# 目的：
#   不改 flow matching 架构代码，只把 hidden_dim / num_blocks / time_dim
#   降到接近或小于 seed42 TransNet decoder 的参数量，验证 142.9M 大模型
#   是否严重过参。
#
# 默认覆盖五个 source：
#   seed2026_transnet_transnet, seed3407_transnet_transnet,
#   seed2026_clnet_transnet, seed2026_crnet_transnet, seed2026_csinet_transnet
#
# 默认小模型配置：
#   h256_b2_t64  约 1.66M 参数，和 decoder 同量级
#   h256_b1_t64  约 1.13M 参数
#   h192_b3_t64  约 1.33M 参数
#   h128_b4_t64  约 0.82M 参数
#
# 用法：
#   bash flow_matching/scripts/run_flow_matching_small.sh
#   gpus="0 1 4 6 7" epochs=400 background=1 bash flow_matching/scripts/run_flow_matching_small.sh
#   lr=1e-4 eta_min=1e-5 bash flow_matching/scripts/run_flow_matching_small.sh
#   configs="h256_b2_t64 h256_b1_t64" bash flow_matching/scripts/run_flow_matching_small.sh
#   exp_root=flow_matching/exps/my_small_test bash flow_matching/scripts/run_flow_matching_small.sh
#   eval_decoder_every=20 eval_decoder_max_samples=10000 bash flow_matching/scripts/run_flow_matching_small.sh
#
# 如果只想跑单个 source：
#   sources="seed2026_transnet_transnet:exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt" \
#     bash flow_matching/scripts/run_flow_matching_small.sh

set -euo pipefail

target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
epochs=${epochs:-400}
batch_size=${batch_size:-512}
lr=${lr:-5e-4}
eta_min=${eta_min:-1e-4}
scheduler=${scheduler:-cosine}
ode_steps=${ode_steps:-16}
ode_method=${ode_method:-euler}
condition=${condition:-source_start}
lambda_endpoint=${lambda_endpoint:-0.0}
eval_ode_every=${eval_ode_every:-20}
eval_decoder_every=${eval_decoder_every:-20}
eval_decoder_max_samples=${eval_decoder_max_samples:-0}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
align_mode=${align_mode:-affine}
background=${background:-1}
gpus=${gpus:-"0 1 4 6 7"}
configs=${configs:-"h256_b2_t64 h256_b1_t64 h192_b3_t64 h128_b4_t64"}
sources=${sources:-}
exp_root=${exp_root:-flow_matching/exps/code_only_small}

DEFAULT_SOURCES=(
  "seed2026_transnet_transnet:exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt"
  "seed3407_transnet_transnet:exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt"
  "seed2026_clnet_transnet:exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt"
  "seed2026_crnet_transnet:exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt"
  "seed2026_csinet_transnet:exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt"
)

if [ -n "${sources}" ]; then
  read -r -a SOURCE_LIST <<< "${sources}"
else
  SOURCE_LIST=("${DEFAULT_SOURCES[@]}")
fi

read -r -a GPU_LIST <<< "${gpus}"
read -r -a CONFIG_LIST <<< "${configs}"
gpu_idx=0

for config in "${CONFIG_LIST[@]}"; do
  case "${config}" in
    h256_b2_t64)
      hidden_dim=256
      num_blocks=2
      time_dim=64
      ;;
    h256_b1_t64)
      hidden_dim=256
      num_blocks=1
      time_dim=64
      ;;
    h192_b3_t64)
      hidden_dim=192
      num_blocks=3
      time_dim=64
      ;;
    h128_b4_t64)
      hidden_dim=128
      num_blocks=4
      time_dim=64
      ;;
    h128_b2_t64)
      hidden_dim=128
      num_blocks=2
      time_dim=64
      ;;
    *)
      echo "Unknown config: ${config}" >&2
      echo "Supported: h256_b2_t64 h256_b1_t64 h192_b3_t64 h128_b4_t64 h128_b2_t64" >&2
      exit 1
      ;;
  esac

  for item in "${SOURCE_LIST[@]}"; do
    source_name=${item%%:*}
    source_code=${item#*:}
    gpu=${GPU_LIST[$((gpu_idx % ${#GPU_LIST[@]}))]}
    gpu_idx=$((gpu_idx + 1))

    run_exp_dir="${exp_root}/${config}/align${align_mode}_cond${condition}_end${lambda_endpoint}_ode${ode_steps}_${ode_method}/${source_name}_to_seed42_transnet_lr${lr}_ep${epochs}"

    source_name="${source_name}" \
    source_code="${source_code}" \
    target_code="${target_code}" \
    align_mode="${align_mode}" \
    condition="${condition}" \
    lambda_endpoint="${lambda_endpoint}" \
    epochs="${epochs}" \
    batch_size="${batch_size}" \
    lr="${lr}" \
    eta_min="${eta_min}" \
    scheduler="${scheduler}" \
    hidden_dim="${hidden_dim}" \
    num_blocks="${num_blocks}" \
    time_dim="${time_dim}" \
    ode_steps="${ode_steps}" \
    ode_method="${ode_method}" \
    eval_ode_every="${eval_ode_every}" \
    eval_decoder_every="${eval_decoder_every}" \
    eval_decoder_max_samples="${eval_decoder_max_samples}" \
    decoder_checkpoint="${decoder_checkpoint}" \
    decoder_args_json="${decoder_args_json}" \
    csi_path="${csi_path}" \
    gpu="${gpu}" \
    background="${background}" \
    exp_dir="${run_exp_dir}" \
      bash flow_matching/scripts/train_flow_matching.sh
  done
done
