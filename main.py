import json
import os

import torch
import torch.nn as nn
from utils.parser import args
from utils import (
    logger,
    setup_logging,
    Trainer,
    Tester,
    log_experiment_header,
)
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from utils.init import _load_clean_state_dict
from dataloader import MyDataLoader


def load_checkpoint_ignoring_thop_stats(model, checkpoint_path):
    result = model.load_state_dict(
        _load_clean_state_dict(checkpoint_path),
        strict=False)
    ignored_suffixes = ("total_ops", "total_params")
    missing_keys = [
        key for key in result.missing_keys
        if not key.endswith(ignored_suffixes)
    ]
    unexpected_keys = [
        key for key in result.unexpected_keys
        if not key.endswith(ignored_suffixes)
    ]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint state_dict mismatch after ignoring THOP stats: "
            f"missing={missing_keys}, unexpected={unexpected_keys}")
    return result


def main():
    exp_dir = os.path.join(os.getcwd(), "exps", args.exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    tensorboard_dir = os.path.join(exp_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    setup_logging(exp_dir)

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    log_experiment_header(args, exp_dir=exp_dir, target_logger=logger)
    logger.info(f'=> Checkpoint directory: {checkpoint_dir}')
    logger.info(f'=> TensorBoard directory: {tensorboard_dir}')
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
    logger.info(
        "=> Dataset sizes: train=%d, val=%d, test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset))
    logger.info(
        "=> Loader config: batch_size=%d, workers=%d, pin_memory=%s",
        args.batch_size,
        args.workers,
        pin_memory)

    # Define model

    model = init_model(args)
    model.to(device)
    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Before-train evaluation (adapter mode only)
    if args.adapter:
        logger.info("=> Before-adapter-train evaluation "
                    "(frozen encoder + frozen decoder + identity adapter):")
        before_loss, before_nmse = Tester(model, device, criterion)(test_loader)
        logger.info(f"=> Before adapter train loss: {before_loss:.4e}"
                    f"    NMSE: {before_nmse:.4e}\n")

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
        trainer.save_codewords(
            train_loader,
            os.path.join(exp_dir, "codewords", "train_code.pt"))
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

    if not decay_params and not no_decay_params:
        raise ValueError("No trainable parameters found for optimizer")

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr_init,
    )
    logger.info(
        "=> Optimizer: AdamW lr=%s weight_decay=%s "
        "decay_params=%d no_decay_params=%d",
        args.lr_init,
        args.weight_decay,
        sum(param.numel() for param in decay_params),
        sum(param.numel() for param in no_decay_params))

    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
        logger.info("=> Scheduler: const")

    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=0.1 * args.epochs * len(train_loader),
                                            eta_min=5e-5)
        logger.info(
            "=> Scheduler: cosine T_max=%s T_warmup=%s eta_min=%s",
            args.epochs * len(train_loader),
            0.1 * args.epochs * len(train_loader),
            5e-5)

    # Define the training pipeline

    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=args.resume,
                      save_path=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir,
                      test_every_epoch=False,
                      teacher_code_path=args.teacher_code,
                      lambda_recon=args.lambda_recon,
                      lambda_code=args.lambda_code,
                      lambda_fc=args.lambda_fc,
                      lambda_recT=args.lambda_recT,
                      lambda_teacher_pca=args.lambda_teacher_pca,
                      lambda_teacher_whiten=args.lambda_teacher_whiten,
                      teacher_pca_dim=args.teacher_pca_dim,
                      anchor_target=args.anchor_target,
                      lambda_anchor=args.lambda_anchor,
                      anchor_loss=args.anchor_loss,
                      train_path=args.train_path,
                      channel=args.channel,
                      nt=args.nt,
                      nc=args.nc,
                      cr=args.cr,
                      lambda_code_mean=args.lambda_code_mean,
                      lambda_code_var=args.lambda_code_var,
                      lambda_code_cov=args.lambda_code_cov,
                      lambda_code_l1=args.lambda_code_l1,
                      code_var_tau=args.code_var_tau)

    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader)

    best_checkpoint = os.path.join(checkpoint_dir, "best_nmse.pth")
    if os.path.isfile(best_checkpoint):
        logger.info(f'=> Loading best checkpoint before codeword export: '
                    f'{best_checkpoint}')
        result = load_checkpoint_ignoring_thop_stats(model, best_checkpoint)
        if result.missing_keys or result.unexpected_keys:
            logger.info(
                '=> Ignored THOP statistic keys while loading best checkpoint: '
                f'missing={len(result.missing_keys)}, '
                f'unexpected={len(result.unexpected_keys)}')
    else:
        logger.warning('=> No best_nmse.pth found before codeword export; '
                       'using current model state')

    trainer.save_codewords(
        train_loader,
        os.path.join(exp_dir, "codewords", "train_code.pt"))

    # Final testing
    loss, nmse = Tester(model, device, criterion)(test_loader)
    logger.info(f'\n=! Final test loss: {loss:.4e}'
                f'\n         test NMSE: {nmse:.4e}\n')

    # Create images for loss and nmse



if __name__ == "__main__":
    main()
