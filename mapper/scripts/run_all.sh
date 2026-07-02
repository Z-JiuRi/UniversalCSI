#!/bin/bash
set -euo pipefail

gpus=${gpus:-0,1,4,6,7}
wait_by_source=${wait_by_source:-0}
wait_existing=${wait_existing:-1}
wait_seconds=${wait_seconds:-600}
scheduler=${scheduler:-cosine}
eta_min=${eta_min:-5e-5}
mapper_list=${mapper_list:-delta_mlp}
residual_mapping=${residual_mapping:-1}
align_mode=${align_mode:-affine}
align_ridge=${align_ridge:-1e-4}
residual_condition=${residual_condition:-source_start}
residual_scale=${residual_scale:-1.0}

IFS=',' read -r -a MAPPER_LIST <<< "${mapper_list}"

for mapper in "${MAPPER_LIST[@]}"; do
mapper="${mapper}" gpus="${gpus}" wait_existing="${wait_existing}" wait_seconds="${wait_seconds}" wait_by_source="${wait_by_source}" \
    scheduler="${scheduler}" eta_min="${eta_min}" \
    residual_mapping="${residual_mapping}" align_mode="${align_mode}" align_ridge="${align_ridge}" \
    residual_condition="${residual_condition}" residual_scale="${residual_scale}" \
    bash mapper/scripts/run_mapper.sh

mapper="${mapper}" gpus="${gpus}" wait_existing="${wait_existing}" wait_seconds="${wait_seconds}" wait_by_source="${wait_by_source}" \
    scheduler="${scheduler}" eta_min="${eta_min}" \
    residual_mapping="${residual_mapping}" align_mode="${align_mode}" align_ridge="${align_ridge}" \
    residual_condition="${residual_condition}" residual_scale="${residual_scale}" \
    bash mapper/scripts/run_mapper_decoder_aware.sh

mapper="${mapper}" gpus="${gpus}" wait_existing="${wait_existing}" wait_seconds="${wait_seconds}" wait_by_source="${wait_by_source}" \
    scheduler="${scheduler}" eta_min="${eta_min}" \
    residual_mapping="${residual_mapping}" align_mode="${align_mode}" align_ridge="${align_ridge}" \
    residual_condition="${residual_condition}" residual_scale="${residual_scale}" \
    bash mapper/scripts/run_mapper_combined_losses.sh
done
