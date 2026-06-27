# #!/bin/bash

# # Launch the canonical encoder experiments.
# #
# # Keep canonical_anchor_seed fixed across all training seeds. It defines the
# # shared random Q/codebook anchor and must not follow the training seed.

# seed=${seed:-3407}

# # 0. Baseline through the canonical script, no canonical constraint.
# scheme=none seed=${seed} gpu=1 \
# exp_name=COST2100/in/encoder_canonical/baseline/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh

# # 1. Fixed random row-orthogonal Q only:
# #    features(2048) -> fixed Q(anchor_seed=0) -> code(512)
# scheme=fixed_q_lowrank canonical_lowrank_rank=0 canonical_anchor_seed=0 \
# seed=${seed} gpu=1 \
# exp_name=COST2100/in/encoder_canonical/fixed_q/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh

# # 2. Fixed Q + low-rank residual:
# #    code = features @ Q.T + residual_scale * features @ U @ V
# scheme=fixed_q_lowrank canonical_anchor_seed=0 \
# canonical_lowrank_rank=16 canonical_lowrank_scale=0.05 \
# seed=${seed} gpu=1 \
# exp_name=COST2100/in/encoder_canonical/fixed_q_rank16/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh

# # 3. Fixed random codebook:
# #    features -> assignment logits -> softmax -> fixed codebook convex combination
# scheme=codebook canonical_anchor_seed=0 \
# canonical_codebook_size=1024 canonical_codebook_temperature=1.0 \
# seed=${seed} gpu=4 \
# exp_name=COST2100/in/encoder_canonical/codebook1024/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh

# # 4. Raw-CSI PCA auxiliary target:
# #    z_anchor = PCA(raw CSI flatten), loss += lambda_anchor * dist(z, z_anchor)
# scheme=aux_pca lambda_anchor=1e-3 anchor_loss=mse \
# seed=${seed} gpu=4 \
# exp_name=COST2100/in/encoder_canonical/aux_pca_1e-3/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh

# # 5. Raw-CSI DCT auxiliary target:
# #    z_anchor = fixed 2D-DCT(raw CSI), loss += lambda_anchor * dist(z, z_anchor)
# scheme=aux_dct lambda_anchor=1e-3 anchor_loss=mse \
# seed=${seed} gpu=4 \
# exp_name=COST2100/in/encoder_canonical/aux_dct_1e-3/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh

# # 6. Fixed Q + low-rank residual + code statistics regularization.
# scheme=fixed_q_lowrank canonical_anchor_seed=0 \
# canonical_lowrank_rank=16 canonical_lowrank_scale=0.05 \
# lambda_code_mean=1e-4 lambda_code_var=1e-4 lambda_code_cov=1e-4 \
# code_var_tau=256 seed=${seed} gpu=6 \
# exp_name=COST2100/in/encoder_canonical/fixed_q_rank16_code_reg/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh

# # 7. Fixed Q + low-rank residual + PCA auxiliary target.
# scheme=fixed_q_lowrank canonical_anchor_seed=0 \
# canonical_lowrank_rank=16 canonical_lowrank_scale=0.05 \
# anchor_target=pca lambda_anchor=1e-3 anchor_loss=mse \
# seed=${seed} gpu=6 \
# exp_name=COST2100/in/encoder_canonical/fixed_q_rank16_pca/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh

# # 8. Fixed codebook + DCT auxiliary target + weak code regularization.
# scheme=codebook canonical_anchor_seed=0 canonical_codebook_size=1024 \
# anchor_target=dct lambda_anchor=1e-3 anchor_loss=mse \
# lambda_code_mean=1e-4 lambda_code_cov=1e-4 \
# seed=${seed} gpu=6 \
# exp_name=COST2100/in/encoder_canonical/codebook1024_dct_reg/seed${seed}_transnet_transnet \
# bash scripts/train_encoder_canonical.sh


scheme=aux_pca \
lambda_anchor=1e-2 \
lambda_code_mean=1e-3 \
lambda_code_var=1e-3 \
lambda_code_cov=1e-4 \
seed=3407 \
exp_name=COST2100/in/encoder_canonical/aux_pca_${lambda_anchor}_code_reg/seed${seed}_transnet_transnet \
gpu=1 \
bash scripts/train_encoder_canonical.sh

scheme=aux_pca \
lambda_anchor=1e-3 \
lambda_code_mean=1e-3 \
lambda_code_var=1e-3 \
lambda_code_cov=1e-4 \
seed=3407 \
exp_name=COST2100/in/encoder_canonical/aux_pca_${lambda_anchor}_code_reg/seed${seed}_transnet_transnet \
gpu=1 \
bash scripts/train_encoder_canonical.sh

scheme=aux_pca \
lambda_anchor=5e-3 \
lambda_code_mean=1e-3 \
lambda_code_var=1e-3 \
lambda_code_cov=1e-4 \
seed=3407 \
exp_name=COST2100/in/encoder_canonical/aux_pca_${lambda_anchor}_code_reg/seed${seed}_transnet_transnet \
gpu=1 \
bash scripts/train_encoder_canonical.sh