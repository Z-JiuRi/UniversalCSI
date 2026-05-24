exp_name=COST2100/in/transnet_transnet/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=transnet \
decoder=transnet \
code_adapter=false \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=2 \
seed=42 \
./scripts/train.sh > in_transnet_transnet.log 2>&1 &

exp_name=COST2100/in/clnet_transnet/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=clnet \
decoder=transnet \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=3 \
seed=42 \
./scripts/train.sh > in_clnet_transnet.log 2>&1 &

exp_name=COST2100/in/csinet_transnet/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=csinet \
decoder=transnet \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=4 \
seed=42 \
./scripts/train.sh > in_csinet_transnet.log 2>&1 &

exp_name=COST2100/in/crnet_transnet/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=crnet \
decoder=transnet \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=5 \
seed=42 \
./scripts/train.sh > in_crnet_transnet.log 2>&1 &


############################################################
exp_name=COST2100/in/transnet_cnn_residual/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=transnet \
decoder=cnn_residual \
code_adapter=false \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=2 \
seed=42 \
./scripts/train.sh > in_transnet_cnn_residual.log 2>&1 &

exp_name=COST2100/in/clnet_cnn_residual/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=clnet \
decoder=cnn_residual \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=3 \
seed=42 \
./scripts/train.sh > in_clnet_cnn_residual.log 2>&1 &

exp_name=COST2100/in/csinet_cnn_residual/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=csinet \
decoder=cnn_residual \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=4 \
seed=42 \
./scripts/train.sh > in_csinet_cnn_residual.log 2>&1 &

exp_name=COST2100/in/crnet_cnn_residual/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=crnet \
decoder=cnn_residual \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=5 \
seed=42 \
./scripts/train.sh > in_crnet_cnn_residual.log 2>&1 &


############################################################
exp_name=COST2100/in/transnet_hybrid/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=transnet \
decoder=hybrid \
code_adapter=false \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=2 \
seed=42 \
./scripts/train.sh > in_transnet_hybrid.log 2>&1 &

exp_name=COST2100/in/clnet_hybrid/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=clnet \
decoder=hybrid \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=3 \
seed=42 \
./scripts/train.sh > in_clnet_hybrid.log 2>&1 &

exp_name=COST2100/in/csinet_hybrid/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=csinet \
decoder=hybrid \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=4 \
seed=42 \
./scripts/train.sh > in_csinet_hybrid.log 2>&1 &

exp_name=COST2100/in/crnet_hybrid/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
encoder=crnet \
decoder=hybrid \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=5 \
seed=42 \
./scripts/train.sh > in_crnet_hybrid.log 2>&1 &


############################################################