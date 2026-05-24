import argparse

parser = argparse.ArgumentParser(description='CRNet PyTorch Training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--train_path', type=str, metavar='PATH', required=True,
                    help='path to training data')
parser.add_argument('--val_path', type=str, metavar='PATH', required=True,
                    help='path to validation data')
parser.add_argument('--test_path', type=str, metavar='PATH', required=True,
                    help='path to test data')
parser.add_argument('-b', '--batch_size', type=int, required=True, metavar='N',
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, metavar='N', required=True,
                    help='number of data loading workers')


# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu_affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')
parser.add_argument('--encoder', type=str, default='transnet',
                    choices=['csinet', 'crnet', 'clnet', 'transnet'],
                    help='encoder backbone to use with the shared TransNet decoder')
parser.add_argument('--code_adapter', action='store_true',
                    help='insert LayerNorm + Linear between encoder code and decoder')
parser.add_argument('--exp_name', metavar='NAME', type=str, default='exp_1',
                    help='experiment name; outputs are saved under ./exps/NAME')
parser.add_argument('--channel', type=int, default=2,
                    help='number of channels in the CSI tensor')
parser.add_argument('--nt', type=int, default=32,
                    help='number of antennas in the CSI tensor')
parser.add_argument('--nc', type=int, default=32,
                    help='number of delay/frequency bins in the CSI tensor')
parser.add_argument('-d', '--d_model', type=int, default=64, metavar='N',
                    help='number of Transformer feature dimension')
parser.add_argument('--dim_feedforward', type=int, default=2048,
                    help='hidden dimension of Transformer feed-forward layers')
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')
parser.add_argument('--lr_init', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight decay for AdamW')
parser.add_argument('--freeze_components', type=str, nargs='+', default=[],
                    choices=['encoder', 'code_adapter', 'fc_decoder',
                             'decoder_self_attn', 'decoder_cross_attn',
                             'decoder_ffn'],
                    help='freeze components during training (space-separated list)')
parser.add_argument('--lora_component', type=str, nargs='+', default=[],
                    choices=['fc_encoder', 'fc_decoder'],
                    help='apply LoRA to components (space-separated list); '
                         'leave empty to disable LoRA')
parser.add_argument('--lora_rank', type=int, default=8,
                    help='LoRA rank for fc_encoder/fc_decoder')
parser.add_argument('--lora_alpha', type=int, default=16,
                    help='LoRA alpha for fc_encoder/fc_decoder')
args = parser.parse_args()
