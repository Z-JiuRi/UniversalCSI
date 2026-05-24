import json
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

    train_loader, val_loader, test_loader = MyDataLoader(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        channel=args.channel,
        nt=args.nt,
        nc=args.nc)()

    # Define model

    model = init_model(args)
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion)(test_loader)
        return

    # Define optimizer and scheduler

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

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
                      tensorboard_dir=tensorboard_dir)

    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader)

    encoder_output_path = os.path.join(exp_dir, "encoder_output.pth")
    trainer.save_encoder_outputs(train_loader, encoder_output_path)

    # Final testing
    loss, nmse = Tester(model, device, criterion)(test_loader)
    logger.info(f'\n=! Final test loss: {loss:.4e}'
                f'\n         test NMSE: {nmse:.4e}\n')

    # Create images for loss and nmse



if __name__ == "__main__":
    main()
