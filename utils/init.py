import os
import random
import thop
import torch

from models import universal_csi
from models.lora import apply_decoder_lora
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


def _load_decoder_only(model, checkpoint_path):
    state_dict = _load_clean_state_dict(checkpoint_path)
    model_state = model.state_dict()
    copied = 0
    for key, value in state_dict.items():
        if not key.startswith('decoder.'):
            continue
        if key not in model_state:
            raise KeyError(f'{key} from {checkpoint_path} is not in model')
        if tuple(model_state[key].shape) != tuple(value.shape):
            raise ValueError(
                f'{key} shape mismatch: model={tuple(model_state[key].shape)} '
                f'checkpoint={tuple(value.shape)}')
        model_state[key] = value
        copied += 1

    if copied == 0:
        raise ValueError(f'No decoder.* parameters found in {checkpoint_path}')
    model.load_state_dict(model_state, strict=True)
    if hasattr(model, 'freeze_decoder'):
        model.freeze_decoder()
    else:
        for param in model.decoder.parameters():
            param.requires_grad = False
        model.decoder.eval()
    return copied


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
    model = universal_csi(encoder_name=args.encoder,
                          decoder_name=args.decoder,
                          reduction=args.cr,
                          d_model=args.d_model,
                          channel=args.channel,
                          nt=args.nt,
                          nc=args.nc,
                          dim_feedforward=args.dim_feedforward,
                          code_adapter=args.code_adapter,
                          hidden=args.hidden,
                          num_blocks=args.num_blocks)

    if args.pretrained is not None and args.pretrained_decoder is not None:
        raise ValueError('--pretrained and --pretrained_decoder are mutually exclusive')

    if args.pretrained is not None:
        state_dict = _load_clean_state_dict(args.pretrained)
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    if args.pretrained_decoder is not None:
        copied = _load_decoder_only(model, args.pretrained_decoder)
        logger.info("pretrained decoder loaded from {} ({} tensors); "
                    "decoder frozen".format(args.pretrained_decoder, copied))

    if args.lora_component is not None:
        trainable = apply_decoder_lora(model,
                                       component=args.lora_component,
                                       rank=args.lora_rank,
                                       alpha=args.lora_alpha)
        logger.info("LoRA enabled on decoder.{}; rank={}; alpha={}; "
                    "trainable LoRA params={}".format(
                        args.lora_component, args.lora_rank,
                        args.lora_alpha, trainable))

    # Model flops and params counting
    H_a = torch.randn([1, args.channel, args.nt, args.nc])
    flops, params = thop.profile(model, inputs=(H_a,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.4e")

    # Model info logging
    logger.info(f'=> Model Name: UniversalCSI [pretrained: {args.pretrained}; '
                f'pretrained_decoder: {args.pretrained_decoder}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}; '
                f'encoder={args.encoder}; '
                f'decoder={args.decoder}; '
                f'code_adapter={args.code_adapter}; '
                f'lora_component={args.lora_component}; '
                f'input shape=({args.channel}, {args.nt}, {args.nc}); '
                f'input dim={args.channel * args.nt * args.nc}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'\n{line_seg}\n{model}\n{line_seg}\n')

    show_parameter(model)

    return model
