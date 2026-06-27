# COST2100 多seed adapter训练命令示例：
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
nt=32 nc=32 lr_init=3e-4 epochs=400 weight_decay=0 batch_size=256 \
encoder=transnet decoder=transnet gpu=1 seed=2026 \
pretrained_encoder=exps/COST2100/in/seed${seed}/${encoder}_${decoder}/checkpoints/best_nmse.pth \
pretrained_decoder=exps/COST2100/in/seed42/${encoder}_${decoder}/checkpoints/best_nmse.pth \
teacher_code=exps/COST2100/in/seed42/${encoder}_${decoder}/codewords/train_code.pt \
adapter=mlp adapter_hidden_dim=2048 lambda_recon=1.0 lambda_code=1e-2 \
exp_name=COST2100/in/adapter/${adapter}/seed${seed}_recon${lambda_recon}_code${lambda_code}_lr${lr_init} \
bash scripts/train_adapter.sh

# train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
# val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
# test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
# nt=32 nc=32 lr_init=3e-4 epochs=400 weight_decay=0 batch_size=256 \
# encoder=transnet decoder=transnet gpu=1 seed=2026 \
# pretrained_encoder=exps/COST2100/in/seed${seed}/${encoder}_${decoder}/checkpoints/best_nmse.pth \
# pretrained_decoder=exps/COST2100/in/seed42/${encoder}_${decoder}/checkpoints/best_nmse.pth \
# teacher_code=exps/COST2100/in/seed42/${encoder}_${decoder}/codewords/train_code.pt \
# adapter=mlp adapter_hidden_dim=2048 lambda_recon=1.0 lambda_code=1e-3 \
# exp_name=COST2100/in/adapter/${adapter}/seed${seed}_recon${lambda_recon}_code${lambda_code}_lr${lr_init} \
# bash scripts/train_adapter.sh

# train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
# val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
# test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
# nt=32 nc=32 lr_init=3e-4 epochs=400 weight_decay=0 batch_size=256 \
# encoder=transnet decoder=transnet gpu=1 seed=2026 \
# pretrained_encoder=exps/COST2100/in/seed${seed}/${encoder}_${decoder}/checkpoints/best_nmse.pth \
# pretrained_decoder=exps/COST2100/in/seed42/${encoder}_${decoder}/checkpoints/best_nmse.pth \
# teacher_code=exps/COST2100/in/seed42/${encoder}_${decoder}/codewords/train_code.pt \
# adapter=mlp adapter_hidden_dim=2048 lambda_recon=1.0 lambda_code=0.0 \
# exp_name=COST2100/in/adapter/${adapter}/seed${seed}_recon${lambda_recon}_code${lambda_code}_lr${lr_init} \
# bash scripts/train_adapter.sh


# # COST2100 多架构 adapter训练命令示例：
# train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
# val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
# test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
# nt=32 nc=32 lr_init=3e-4 epochs=400 weight_decay=0 batch_size=256 \
# encoder=cbam_cnn decoder=hybrid gpu=4 seed=42 \
# pretrained_encoder=exps/COST2100/in/seed${seed}/${encoder}_${decoder}/checkpoints/best_nmse.pth \
# pretrained_decoder=exps/COST2100/in/seed${seed}/transnet_hybrid/checkpoints/best_nmse.pth \
# teacher_code=exps/COST2100/in/seed${seed}/transnet_hybrid/codewords/train_code.pt \
# adapter=mlp adapter_hidden_dim=2048 lambda_recon=1.0 lambda_code=1e-3 \
# exp_name=COST2100/in/adapter/${adapter}/${encoder}_${decoder}/seed${seed}/recon${lambda_recon}_code${lambda_code}_lr${lr_init} \
# bash scripts/train_adapter.sh


# # WAIRD 多seed adapter训练命令示例：
# train_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/train.pt \
# val_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/test.pt \
# test_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/test.pt \
# nt=64 nc=64 lr_init=3e-4 epochs=1000 weight_decay=0 batch_size=256 \
# encoder=transnet decoder=hybrid gpu=1 seed=3407 \
# pretrained_encoder=exps/WAIRD/seed3407/transnet_transnet/checkpoints/best_nmse.pth \
# pretrained_decoder=exps/WAIRD/seed42/transnet_transnet/checkpoints/best_nmse.pth \
# teacher_code=exps/WAIRD/seed42/transnet_transnet/codewords/train_code.pt \
# adapter=mlp adapter_hidden_dim=2048 lambda_recon=1.0 lambda_code=0.0 \
# exp_name=WAIRD/adapter/${adapter}/seed${seed}_lambda_recon${lambda_recon}_lambda_code${lambda_code}_lr${lr_init} \
# bash scripts/train_adapter.sh