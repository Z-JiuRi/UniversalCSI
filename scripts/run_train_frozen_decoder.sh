# encoder=csinet decoder=hybrid seed=3407 gpu=5 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=cnn decoder=hybrid seed=3407 gpu=5 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=cbam_cnn decoder=hybrid seed=3407 gpu=5 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=crnet decoder=hybrid seed=3407 gpu=5 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=clnet decoder=hybrid seed=3407 gpu=0 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=resnet decoder=hybrid seed=3407 gpu=0 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=dscnn decoder=hybrid seed=3407 gpu=0 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=convnext decoder=hybrid seed=3407 gpu=0 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=mlp_mixer decoder=hybrid seed=3407 gpu=3 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=attention_cnn decoder=hybrid seed=3407 gpu=3 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=swin decoder=hybrid seed=3407 gpu=3 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=mlp_ae decoder=hybrid seed=3407 gpu=3 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=sparse_resnet decoder=hybrid seed=3407 gpu=0 bash scripts/train_frozen_decoder.sh
# sleep 5


# encoder=csinet decoder=hybrid seed=2026 gpu=0 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=cnn decoder=hybrid seed=2026 gpu=0 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=cbam_cnn decoder=hybrid seed=2026 gpu=5 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=crnet decoder=hybrid seed=2026 gpu=5 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=clnet decoder=hybrid seed=2026 gpu=5 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=resnet decoder=hybrid seed=2026 gpu=3 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=dscnn decoder=hybrid seed=2026 gpu=3 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=convnext decoder=hybrid seed=2026 gpu=3 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=mlp_mixer decoder=hybrid seed=2026 gpu=3 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=attention_cnn decoder=hybrid seed=2026 gpu=0 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=swin decoder=hybrid seed=2026 gpu=0 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=mlp_ae decoder=hybrid seed=2026 gpu=5 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5
# encoder=sparse_resnet decoder=hybrid seed=2026 gpu=5 epochs=200 bash scripts/train_frozen_decoder.sh
# sleep 5


# Frozen-decoder baselines: fixed seed42 decoder, random-init transnet encoder.
# These compare frozen-decoder training against the original joint-training runs
# under exps/COST2100/in/seed42/transnet_{transnet,hybrid}/base.
encoder=transnet decoder=transnet seed=42 gpu=3 epochs=200 exp_name=COST2100/in/frozen_decoder/seed42/transnet_transnet_ep200 bash scripts/train_frozen_decoder.sh
sleep 5
encoder=transnet decoder=transnet seed=42 gpu=3 epochs=400 exp_name=COST2100/in/frozen_decoder/seed42/transnet_transnet bash scripts/train_frozen_decoder.sh
sleep 5
encoder=transnet decoder=hybrid seed=42 gpu=5 epochs=200 exp_name=COST2100/in/frozen_decoder/seed42/transnet_hybrid_ep200 bash scripts/train_frozen_decoder.sh
sleep 5
encoder=transnet decoder=hybrid seed=42 gpu=5 epochs=400 exp_name=COST2100/in/frozen_decoder/seed42/transnet_hybrid bash scripts/train_frozen_decoder.sh
