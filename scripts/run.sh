exp_name=COST2100/out/seed3407_1e-4/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/out_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/out_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/out_test.pt \
batch_size=512 \
lr_init=1e-4 \
gpu=2 \
seed=3407 \
./scripts/train.sh > out_3407.log 2>&1 &

exp_name=COST2100/out/seed42_1e-4/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/out_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/out_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/out_test.pt \
batch_size=512 \
lr_init=1e-4 \
gpu=2 \
seed=42 \
./scripts/train.sh > out_42.log 2>&1 &

exp_name=COST2100/out/seed2026_1e-4/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/out_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/out_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/out_test.pt \
batch_size=512 \
lr_init=1e-4 \
gpu=2 \
seed=2026 \
./scripts/train.sh > out_2026.log 2>&1 &

# exp_name=COST2100/in/seed42/base \
# train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
# val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
# test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
# batch_size=512 \
# gpu=0 \
# seed=42 \
# ./scripts/train.sh > in_42.log 2>&1 &

# exp_name=COST2100/in/seed3407/base \
# train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
# val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
# test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
# batch_size=512 \
# gpu=0 \
# seed=3407 \
# ./scripts/train.sh > in_3407.log 2>&1 &

# exp_name=COST2100/in/seed2026/base \
# train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
# val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
# test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
# batch_size=512 \
# gpu=0 \
# seed=2026 \
# ./scripts/train.sh > in_2026.log 2>&1 &








