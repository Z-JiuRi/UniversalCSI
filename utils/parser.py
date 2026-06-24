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
parser.add_argument('--pretrained_decoder', type=str, default=None,
                    help='path to a checkpoint whose decoder weights will be loaded and frozen')
parser.add_argument('--pretrained_encoder', type=str, default=None,
                    help='path to a checkpoint whose encoder weights will be loaded and frozen')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--adapter', type=str, default=None,
                    choices=['mlp', 'mlp_direct', 'transformer'],
                    help='adapter type: mlp (residual MLP), '
                         'mlp_direct (MLP w/o residual), '
                         'transformer (self-attention)')
parser.add_argument('--adapter_hidden_dim', default=2048, type=int,
                    help='hidden dimension of the adapter MLP (default: 4 * adapter_dim)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu_affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')
parser.add_argument('--encoder', type=str, default='transnet',
                    choices=['csinet', 'cnn', 'cbam_cnn',
                             'crnet', 'clnet', 'transnet',
                             'resnet', 'dscnn', 'convnext',
                             'mlp_mixer', 'attention_cnn', 'swin',
                             'mlp_ae', 'sparse_resnet'],
                    help='encoder backbone to use')
parser.add_argument('--decoder', type=str, default='transnet',
                    choices=['transnet', 'cnn_residual', 'hybrid'], 
                    help='decoder backbone to use')
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
parser.add_argument('--hidden', type=int, default=16,
                    help='internal channel count in CNN refinement head (decoder=hybrid)')
parser.add_argument('--num_blocks', type=int, default=2,
                    help='number of ConvResidualBlock in CNN refinement head (decoder=hybrid)')
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')
parser.add_argument('--lr_init', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight decay for AdamW')

# Teacher code distillation arguments
parser.add_argument('--teacher_code', type=str, default=None, metavar='PATH',
                    help='path to precomputed teacher codewords (train_code.pt)')
parser.add_argument('--lambda_recon', type=float, default=1.0,
                    help='weight for reconstruction MSE loss (default: 1.0)')
parser.add_argument('--lambda_code', type=float, default=0.0,
                    help='weight for code-space MSE loss against teacher (default: 0.0)')
args = parser.parse_args()
