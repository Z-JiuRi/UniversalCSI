import os
import random
import thop
import torch

from models import multi_seed_adapter_csi, universal_csi
from utils import logger, line_seg

__all__ = ["seed_everything", "init_device", "init_model", "show_parameter"]


def seed_everything(seed):
    logger.info(f"Random seed set to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_parameter(model):
    params = [
        (name, str(param.requires_grad), str(tuple(param.shape)))
        for name, param in model.named_parameters()
    ]
    shape_width = max([len(shape) for _, _, shape in params] or [0])
    fmt_str = "{:<65} {:<8} {:>{shape_width}}"
    lines = [
        fmt_str.format(name, requires_grad, shape, shape_width=shape_width)
        for name, requires_grad, shape in params
    ]
    lines.append(line_seg)
    logger.info("\n" + "\n".join(lines))


def _load_clean_state_dict(checkpoint_path):
    assert os.path.isfile(checkpoint_path)
    state_dict = torch.load(checkpoint_path,
                            weights_only=True,
                            map_location=torch.device('cpu'))['state_dict']
    # Strip thop profiling keys (total_ops, total_params) that were saved
    # during training but don't exist in the freshly-created model.
    for key in list(state_dict.keys()):
        if key.endswith('total_ops') or key.endswith('total_params'):
            del state_dict[key]
    return state_dict


def _load_decoder_state_dict(checkpoint_path):
    state_dict = _load_clean_state_dict(checkpoint_path)
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        raise ValueError(
            f"No decoder.* parameters found in checkpoint: {checkpoint_path}")
    return decoder_state


def _load_encoder_state_dict(checkpoint_path):
    state_dict = _load_clean_state_dict(checkpoint_path)
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(
            f"No encoder.* parameters found in checkpoint: {checkpoint_path}")
    return encoder_state


def _unfreeze_decoder_adapters(decoder):
    """Re-enable adapter params inside a frozen decoder."""
    for name, param in decoder.named_parameters():
        if name.startswith("sp_adapter.") or \
           name.startswith("tp_adapter.") or \
           name.startswith("tm_adapter."):
            param.requires_grad = True


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        seed_everything(seed)

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        logger.info("Running on GPU %d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading
    if args.encoder_seeds is not None:
        model = multi_seed_adapter_csi(
            encoder_name=args.encoder,
            decoder_name=args.decoder,
            reduction=args.cr,
            d_model=args.d_model,
            channel=args.channel,
            nt=args.nt,
            nc=args.nc,
            dim_feedforward=args.dim_feedforward,
            hidden=args.hidden,
            num_blocks=args.num_blocks,
            encoder_seeds=args.encoder_seeds,
            decoder_seed=args.decoder_seed,
            adapter_positions=args.adapter_pos,
            adapter_hidden_dim=args.adapter_hidden_dim)
    else:
        model = universal_csi(encoder_name=args.encoder,
                              decoder_name=args.decoder,
                              reduction=args.cr,
                              d_model=args.d_model,
                              channel=args.channel,
                              nt=args.nt,
                              nc=args.nc,
                              dim_feedforward=args.dim_feedforward,
                              hidden=args.hidden,
                              num_blocks=args.num_blocks,
                              adapter_positions=args.adapter_pos,
                              adapter_hidden_dim=args.adapter_hidden_dim)

    if args.pretrained is not None:
        state_dict = _load_clean_state_dict(args.pretrained)
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    if args.pretrained_decoder is not None:
        decoder_state = _load_decoder_state_dict(args.pretrained_decoder)
        model.decoder.load_state_dict(decoder_state)
        for param in model.decoder.parameters():
            param.requires_grad = False
        # Re-enable internal adapters so they stay trainable
        _unfreeze_decoder_adapters(model.decoder)
        logger.info("pretrained decoder loaded and frozen from {}".format(
            args.pretrained_decoder))

    if args.pretrained_encoder is not None:
        encoder_state = _load_encoder_state_dict(args.pretrained_encoder)
        if hasattr(model, 'encoders'):
            # Multi-encoder model: load into each encoder
            for key, encoder in model.encoders.items():
                encoder.load_state_dict(encoder_state)
                for param in encoder.parameters():
                    param.requires_grad = False
        else:
            # Single-encoder model
            model.encoder.load_state_dict(encoder_state)
            for param in model.encoder.parameters():
                param.requires_grad = False
        logger.info("pretrained encoder loaded and frozen from {}".format(
            args.pretrained_encoder))

    # Model flops and params counting
    H_a = torch.randn([1, args.channel, args.nt, args.nc])
    try:
        flops, params = thop.profile(model, inputs=(H_a,), verbose=False)
        flops, params = thop.clever_format([flops, params], "%.4e")
    except Exception as exc:
        flops, params = "unavailable", "unavailable"
        logger.warning(f"=> Model profiling skipped: {exc}")

    # Model info logging
    model_name = "MultiSeedEncoderAdapterCSI" if args.encoder_seeds else "UniversalCSI"
    logger.info(f'=> Model Name: {model_name} [pretrained: {args.pretrained}; '
                f'pretrained_decoder: {args.pretrained_decoder}; '
                f'pretrained_encoder: {args.pretrained_encoder}; '
                f'adapter_pos: {args.adapter_pos}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}; '
                f'encoder={args.encoder}; '
                f'decoder={args.decoder}; '
                f'encoder_seeds={args.encoder_seeds}; '
                f'decoder_seed={args.decoder_seed}; '
                f'input shape=({args.channel}, {args.nt}, {args.nc}); '
                f'input dim={args.channel * args.nt * args.nc}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'\n{line_seg}\n{model}\n{line_seg}\n')

    show_parameter(model)

    return model
