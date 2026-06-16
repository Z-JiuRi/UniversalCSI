import json
import math
import os

import torch
import torch.nn as nn
from utils.parser import args
from utils import logger, setup_logging, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataloader import MyDataLoader


def main():
    exp_dir = os.path.join(os.getcwd(), "exps", args.exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    tensorboard_dir = os.path.join(exp_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    setup_logging(exp_dir)

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    logger.info(f'=> Experiment directory: {exp_dir}')
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader

    data_builder = MyDataLoader(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        channel=args.channel,
        nt=args.nt,
        nc=args.nc,
        return_indices=True)
    train_loader, val_loader, test_loader = data_builder()

    # Define model

    model = init_model(args)
    model.to(device)
    adapter_training = (
        args.code_adapter
        and args.pretrained_encoder is not None
        and args.pretrained_decoder is not None
        and args.lora_component is None
    )
    fc_decoder_training = (
        args.train_fc_decoder
        and args.pretrained_encoder is not None
        and args.pretrained_decoder is not None
        and args.lora_component is None
    )
    if args.train_fc_decoder and args.decoder != 'transnet':
        raise ValueError('--train_fc_decoder requires --decoder transnet')
    if args.train_fc_decoder and args.code_adapter:
        raise ValueError('--train_fc_decoder should not be used with --code_adapter')
    if args.teacher_code is not None and not (adapter_training or fc_decoder_training):
        raise ValueError('--teacher_code is only supported for adapter/fc_decoder training')
    if args.code_loss_lambda is not None and args.teacher_code is None:
        raise ValueError('--code_loss_lambda requires --teacher_code')
    if args.code_loss_only and args.teacher_code is None:
        raise ValueError('--code_loss_only requires --teacher_code')

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion)(test_loader)
        trainer = Trainer(model=model,
                          device=device,
                          optimizer=None,
                          criterion=criterion,
                          scheduler=None,
                          save_path=checkpoint_dir,
                          tensorboard_dir=tensorboard_dir)
        trainer.save_all_encoder_outputs(
            {"train": train_loader},
            os.path.join(exp_dir, "codewords"))
        return

    if args.lora_component is not None or adapter_training or fc_decoder_training:
        loss, nmse = Tester(model, device, criterion)(test_loader)
        if adapter_training:
            mode = "adapter"
        elif fc_decoder_training:
            mode = "fc_decoder"
        else:
            mode = "LoRA"
        logger.info(f'\n=> Before {mode} training: '
                    f'loss: {loss:.4e}    NMSE: {nmse:.4e}\n')

    # Define optimizer and scheduler

    learnable_code_loss_lambda = None
    if (args.teacher_code is not None and args.code_loss_lambda is None
            and not args.code_loss_only):
        init_lambda = 1.0
        raw_value = math.log(math.exp(init_lambda) - 1.0)
        learnable_code_loss_lambda = nn.Parameter(
            torch.tensor(raw_value, dtype=torch.float32, device=device))
        logger.info('=> code_loss_lambda is learnable; '
                    f'initial value={init_lambda:.4e}')

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if learnable_code_loss_lambda is not None:
        no_decay_params.append(learnable_code_loss_lambda)

    if not decay_params and not no_decay_params:
        raise ValueError("No trainable parameters found for optimizer")

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr_init,
    )

    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)

    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=0.1 * args.epochs * len(train_loader),
                                            eta_min=5e-5)

    # Define the training pipeline

    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=args.resume,
                      save_path=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir,
                      lora_training=args.lora_component is not None,
                      test_every_epoch=(
                          args.lora_component is not None
                          or adapter_training
                          or fc_decoder_training),
                      teacher_code=args.teacher_code,
                      code_loss_lambda=args.code_loss_lambda,
                      teacher_code_size=len(data_builder.train_dataset),
                      code_loss_raw_lambda=learnable_code_loss_lambda,
                      code_loss_only=args.code_loss_only,
                      train_fc_decoder=args.train_fc_decoder)

    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader)

    trainer.save_all_encoder_outputs(
        {"train": train_loader},
        os.path.join(exp_dir, "codewords"))

    # Final testing
    loss, nmse = Tester(model, device, criterion)(test_loader)
    logger.info(f'\n=! Final test loss: {loss:.4e}'
                f'\n         test NMSE: {nmse:.4e}\n')

    # Create images for loss and nmse



if __name__ == "__main__":
    main()
