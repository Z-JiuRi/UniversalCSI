encoder=transnet decoder=transnet batch_size=256 epochs=400 gpu=6 lr_init=1e-3 seed=3407 \
pretrained_encoder=exps/COST2100/in/seed${seed}/${encoder}_${decoder}/checkpoints/best_nmse.pth \
pretrained_decoder=exps/COST2100/in/seed42/${encoder}_${decoder}/checkpoints/best_nmse.pth \
exp_name=COST2100/in/unfreeze_fc_decoder/seed${seed}/${encoder}_${decoder}_${lr_init} \
bash scripts/train.sh