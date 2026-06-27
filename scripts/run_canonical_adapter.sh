  scheme=aux_pca \
  lambda_anchor=1e-3 \
  lambda_code_mean=1e-3 \
  lambda_code_var=1e-3 \
  lambda_code_cov=1e-4 \
  exp_name=COST2100/in/encoder_canonical/aux_pca_1e-3_code_reg/seed2026_transnet_transnet \
  seed=2026 gpu=0 \
  bash scripts/train_encoder_canonical.sh