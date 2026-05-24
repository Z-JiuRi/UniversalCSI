exp_name=COST2100/in/transnet/base \
train_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_train.pt \
val_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_val.pt \
test_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_test.pt \
encoder=transnet \
code_adapter=false \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=0 \
seed=42 \
./scripts/train.sh > in_transnet.log 2>&1 &

exp_name=COST2100/in/clnet/base \
train_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_train.pt \
val_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_val.pt \
test_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_test.pt \
encoder=clnet \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=0 \
seed=42 \
./scripts/train.sh > in_clnet.log 2>&1 &

exp_name=COST2100/in/csinet/base \
train_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_train.pt \
val_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_val.pt \
test_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_test.pt \
encoder=csinet \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=0 \
seed=42 \
./scripts/train.sh > in_csinet.log 2>&1 &

exp_name=COST2100/in/crnet/base \
train_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_train.pt \
val_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_val.pt \
test_path=/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_test.pt \
encoder=crnet \
code_adapter=true \
batch_size=512 \
epochs=400 \
lr_init=2e-4 \
gpu=0 \
seed=42 \
./scripts/train.sh > in_crnet.log 2>&1 &