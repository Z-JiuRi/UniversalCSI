#!/bin/bash
set -euo pipefail

gpus=${gpus:-0,1,4,6,7}
wait_by_source=${wait_by_source:-0}
wait_existing=${wait_existing:-1}
wait_seconds=${wait_seconds:-600}

for mapper in hybrid mlp; do
mapper="${mapper}" gpus="${gpus}" wait_existing="${wait_existing}" wait_seconds="${wait_seconds}" wait_by_source="${wait_by_source}" \
    bash mapper/scripts/run_mapper.sh

mapper="${mapper}" gpus="${gpus}" wait_existing="${wait_existing}" wait_seconds="${wait_seconds}" wait_by_source="${wait_by_source}" \
    bash mapper/scripts/run_mapper_decoder_aware.sh

mapper="${mapper}" gpus="${gpus}" wait_existing="${wait_existing}" wait_seconds="${wait_seconds}" wait_by_source="${wait_by_source}" \
    bash mapper/scripts/run_mapper_combined_losses.sh
done