#!/bin/bash

# aux_pca_1e-3：当前重建性能最好的 canonical autoencoder。
canonical_scheme=aux_pca_1e-3 \
seed=2026 decoder_seed=42 gpu=1 \
lambda_recon=1.0 lambda_code=1e-3 lambda_fc=0.0 lambda_recT=0.0 lr_init=2e-4 \
exp_name=COST2100/in/encoder_canonical/adapter/aux_pca_1e-3/mlp/best_code1e-3_lr2e-4/enc_seed2026_dec_seed42 \
bash scripts/train_canonical_adapter.sh

# aux_pca_1e-3 + decoder-aware fc loss。
canonical_scheme=aux_pca_1e-3 \
seed=2026 decoder_seed=42 gpu=4 \
lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-3 lambda_recT=0.0 lr_init=2e-4 \
exp_name=COST2100/in/encoder_canonical/adapter/aux_pca_1e-3/mlp/code1e-3_fc1e-3_lr2e-4/enc_seed2026_dec_seed42 \
bash scripts/train_canonical_adapter.sh

# aux_pca_1e-3 + teacher reconstruction consistency。
canonical_scheme=aux_pca_1e-3 \
seed=2026 decoder_seed=42 gpu=1 \
lambda_recon=1.0 lambda_code=1e-3 lambda_fc=0.0 lambda_recT=1.0 lr_init=2e-4 \
exp_name=COST2100/in/encoder_canonical/adapter/aux_pca_1e-3/mlp/code1e-3_recT1_lr2e-4/enc_seed2026_dec_seed42 \
bash scripts/train_canonical_adapter.sh

# aux_pca_1e-3 + fc loss + teacher reconstruction consistency。
canonical_scheme=aux_pca_1e-3 \
seed=2026 decoder_seed=42 gpu=6 \
lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-3 lambda_recT=1.0 lr_init=2e-4 \
exp_name=COST2100/in/encoder_canonical/adapter/aux_pca_1e-3/mlp/code1e-3_fc1e-3_recT1_lr2e-4/enc_seed2026_dec_seed42 \
bash scripts/train_canonical_adapter.sh

# aux_pca_5e-3_code_reg：code_reg 组里重建最好的方案，用作“更规范 code”对照。
canonical_scheme=aux_pca_5e-3_code_reg \
seed=2026 decoder_seed=42 gpu=4 \
lambda_recon=1.0 lambda_code=1e-3 lambda_fc=0.0 lambda_recT=0.0 lr_init=2e-4 \
exp_name=COST2100/in/encoder_canonical/adapter/aux_pca_5e-3_code_reg/mlp/best_code1e-3_lr2e-4/enc_seed2026_dec_seed42 \
bash scripts/train_canonical_adapter.sh
