# Encoder Zoo 设计建议

## 目标和输入约束

当前 UniversalCSI 的 encoder zoo 应服务于一个明确接口：

```text
input:  sparse CSI map x = (B, 2, H, W)
        H, W in {32, 64}

output: code = (B, code_dim)
        code_dim = 2 * H * W / cr
```

其中 `2` 是实部和虚部通道，`cr` 是压缩率分母。典型设置：

```text
(2, 32, 32), cr=4 -> input_dim=2048, code_dim=512
(2, 64, 64), cr=4 -> input_dim=8192, code_dim=2048
```

所有 encoder 必须只输出一个 dense code 向量：

```text
encoder(x) -> (B, code_dim)
```

decoder 不应感知 encoder 的内部结构。这样才能公平测试：

```text
encoder zoo x decoder zoo -> reconstruction NMSE
```

## 总体推荐

对当前单帧 `(B,2,32,32)` / `(B,2,64,64)` 数据，最适合优先加入的 encoder 是：

```text
1. ResNetCsiEncoder
2. DepthwiseSeparableCsiEncoder / MobileCsiEncoder
3. ConvNeXtCsiEncoder
4. SwinCsiEncoder
5. MLPMixerCsiEncoder
6. AttentionCNNEncoder
7. CNNEncoder / MLPAEEncoder
8. SparseTransformCsiEncoder
```

不建议优先做：

```text
Temporal / LSTM encoder
Diffusion encoder
Uplink-assisted encoder
```

原因是当前数据是单帧矩阵，没有时间序列、辅助上行 CSI 或 side information。它们更适合后续扩展数据形态后加入。

## 推荐优先级表

| Encoder | 优先级 | 主要价值 | 适合 32/64 | UE 复杂度 | 与当前 decoder 兼容性 |
|---|---:|---|---|---|---|
| ResNetCsiEncoder | P0 | 强 CNN baseline | 高 | 中 | 高 |
| DepthwiseSeparableCsiEncoder | P0 | 低复杂度 UE baseline | 高 | 低 | 高 |
| ConvNeXtCsiEncoder | P1 | 现代 CNN backbone | 高 | 中 | 高 |
| SwinCsiEncoder | P1 | 局部窗口 attention | 高 | 中高 | 高 |
| MLPMixerCsiEncoder | P1 | 无卷积无 attention 的全局混合消融 | 高 | 中 | 高 |
| AttentionCNNEncoder | P1 | CNN + 通道/空间注意力 | 高 | 中 | 高 |
| CNNEncoder | P1 | 标准卷积自编码压缩 baseline | 高 | 中 | 高 |
| MLPAEEncoder | P1 | 纯全连接自编码 baseline | 中 | 中高 | 高 |
| SparseTransformCsiEncoder | P2 | 物理/信息先验驱动 | 高 | 中 | 高 |

## 两类 Encoder Zoo

建议把 encoder zoo 分成两条线，避免把不同问题混在一起比较。

### 结构型 encoder

结构型 encoder 主要比较 backbone：

```text
resnet
dscnn / mobile
convnext
swin
mlp_mixer
attention_cnn
```

它们的训练目标不变：

```text
MSE(reconstruction, sparse_gt)
```

输出也是普通 dense code：

```text
(B, 2, H, W) -> (B, code_dim)
```

### Autoencoder 风格 / 训练机制型 encoder

AE 风格不是单一网络结构，而是一组压缩反馈范式：

```text
mlp_ae
cnn
vae / beta_vae
masked_ae_pretrain
```

它们的重点通常是：

```text
code 分布
预训练方式
```

对于当前项目，最值得纳入第一轮或第二轮的是：

```text
cnn
mlp_ae
```

其中 `cnn` 是最重要的 AE family baseline，`mlp_ae` 用作无空间先验的下限消融。

## 共同实现契约

建议所有 encoder 都继承同一套形状逻辑：

```python
input_dim = channel * nt * nc
code_dim = input_dim // reduction
assert input_dim % reduction == 0
```

构造函数统一为：

```python
def __init__(self, reduction=4, channel=2, nt=32, nc=32, ...):
```

forward 统一为：

```python
def forward(self, x):
    # x: (B, channel, nt, nc)
    return code  # (B, code_dim)
```

为了同时支持 `32x32` 和 `64x64`，不要把中间 feature map 尺寸写死。优先使用：

```text
AdaptiveAvgPool2d
flatten(1)
Linear(dynamic_feature_dim, code_dim)  # feature_dim 在 __init__ 中由 nt/nc 推导
```

或使用固定下采样次数，并通过 `nt // stride_total`、`nc // stride_total` 推导 Linear 输入维度。

## P0-1：ResNetCsiEncoder

### 适用性

这是最推荐先加的强 CNN baseline。它比 CsiNet 更深，比 CRNet 更规整，训练稳定，能回答一个重要问题：

```text
CSI encoder 是否需要复杂多分支，还是一个标准 residual CNN 就足够强？
```

### 架构设计

推荐结构：

```text
input: (B, 2, H, W)

stem:
  Conv2d(2, width, 3, padding=1)
  BatchNorm2d
  LeakyReLU

stage 1:
  ResidualBlock(width) * n1

stage 2:
  Conv2d(width, 2*width, 3, stride=2, padding=1)
  ResidualBlock(2*width) * n2

stage 3:
  Conv2d(2*width, 4*width, 3, stride=2, padding=1)
  ResidualBlock(4*width) * n3

head:
  flatten
  Linear(4*width * H/4 * W/4, code_dim)

output: (B, code_dim)
```

默认参数建议：

```text
width = 16
n1, n2, n3 = 1, 1, 1  # small
或 2, 2, 2            # strong
```

尺寸流：

```text
32x32 -> 16x16 -> 8x8
64x64 -> 32x32 -> 16x16
```

### 设计要点

- 使用 residual block 保证深层 CNN 可训练。
- 不建议一开始使用 `stride=4` 大幅下采样，CSI 稀疏结构可能被过早破坏。
- 输出前不加 sigmoid，直接线性 code。

### 推荐实验名

```text
--encoder resnet
```

## P0-2：DepthwiseSeparableCsiEncoder / MobileCsiEncoder

### 适用性

UE 侧 encoder 实际部署时更关注计算量、参数量和功耗。Depthwise separable convolution 是最直接的轻量化 baseline。

该 encoder 的价值不是一定取得最好 NMSE，而是建立：

```text
低复杂度 UE encoder 在相同 decoder 下能损失多少 NMSE？
```

### 架构设计

推荐结构：

```text
input: (B, 2, H, W)

stem:
  Conv2d(2, width, 3, padding=1)
  BatchNorm2d
  LeakyReLU

mobile block 1:
  DepthwiseConv2d(width, width, 3, padding=1, groups=width)
  PointwiseConv2d(width, width)
  BatchNorm2d
  LeakyReLU

mobile block 2:
  DepthwiseConv2d(width, width, 3, stride=2, padding=1, groups=width)
  PointwiseConv2d(width, 2*width)
  BatchNorm2d
  LeakyReLU

mobile block 3:
  DepthwiseConv2d(2*width, 2*width, 3, stride=2, padding=1, groups=2*width)
  PointwiseConv2d(2*width, 4*width)
  BatchNorm2d
  LeakyReLU

head:
  flatten
  Linear(4*width * H/4 * W/4, code_dim)
```

默认参数建议：

```text
width = 8 or 16
num_blocks = 3 to 5
```

### 设计要点

- 这是 mobile/vendor 异构场景中很重要的 encoder 类型。
- 如果 NMSE 较差，可以把 head 改成：

```text
flatten -> Linear(hidden) -> LeakyReLU -> Linear(code_dim)
```

但第一版建议保持单 Linear，便于比较。

### 推荐实验名

```text
--encoder dscnn
--encoder mobile
```

## P1-1：ConvNeXtCsiEncoder

### 适用性

ConvNeXt 是现代 CNN 风格：大核 depthwise conv、LayerNorm、pointwise MLP。它适合验证：

```text
现代卷积结构是否比传统 BN+小卷积更适合 CSI map？
```

相比 ResNet，它有更大的局部感受野，适合角延迟域中路径簇和结构连续性。

### 架构设计

推荐结构：

```text
input: (B, 2, H, W)

stem:
  Conv2d(2, width, 3, padding=1)

ConvNeXtBlock * n1:
  depthwise Conv2d(width, width, kernel=7, padding=3, groups=width)
  LayerNorm over channel
  Linear/1x1 Conv width -> 4*width
  GELU
  Linear/1x1 Conv 4*width -> width
  residual add

downsample:
  Conv2d(width, 2*width, 2, stride=2)

ConvNeXtBlock * n2

downsample:
  Conv2d(2*width, 4*width, 2, stride=2)

ConvNeXtBlock * n3

head:
  flatten
  LayerNorm
  Linear(..., code_dim)
```

默认参数建议：

```text
width = 16
n1, n2, n3 = 1, 1, 1
kernel_size = 7
```

### 设计要点

- 对 PyTorch 3.8 兼容时，LayerNorm 需要注意输入格式；可用 `GroupNorm(1, C)` 简化。
- 参数量会高于 Mobile encoder，但通常低于 full Transformer。

### 推荐实验名

```text
--encoder convnext
```

## P1-2：SwinCsiEncoder

### 适用性

TransNet 将 `2*H*W` 直接切成 token 序列，token 的空间含义较弱。Swin 风格 encoder 更适合 CSI map：

```text
局部窗口 attention 捕获邻域结构；
shifted window 允许跨窗口信息交换；
下采样逐步扩大感受野。
```

这比全局 Transformer 更贴近图像/矩阵结构，也比纯 CNN 有更强的动态建模能力。

### 架构设计

推荐简化版，不必完整复刻 Swin Transformer：

```text
input: (B, 2, H, W)

patch embedding:
  Conv2d(2, embed_dim, kernel_size=patch, stride=patch)
  -> (B, embed_dim, H/patch, W/patch)

stage 1:
  WindowAttentionBlock(embed_dim, window=4) * n1

patch merge:
  Conv2d(embed_dim, 2*embed_dim, kernel=2, stride=2)

stage 2:
  WindowAttentionBlock(2*embed_dim, window=4) * n2

head:
  flatten
  Linear(..., code_dim)
```

默认参数建议：

```text
patch = 2
embed_dim = 32
window = 4
n1, n2 = 1, 1
nhead = 2 or 4
```

尺寸流：

```text
32x32, patch=2 -> 16x16 tokens -> 8x8 after merge
64x64, patch=2 -> 32x32 tokens -> 16x16 after merge
```

### 简化实现建议

第一版可以不做 shifted window，只做 window self-attention：

```text
partition windows -> MultiheadAttention -> reverse windows
```

后续再加 shifted window 做消融。

### 设计要点

- 要加 2D positional embedding 或相对位置 bias，否则窗口内位置信息弱。
- 对 `64x64` 更有优势，因为全局 Transformer 的 token 数会变大。

### 推荐实验名

```text
--encoder swin
```

## P1-3：MLPMixerCsiEncoder

### 适用性

MLP-Mixer 是很好的结构消融。它没有 convolution，也没有 attention，但能做 token mixing 和 channel mixing。

它回答的问题是：

```text
TransNet 的收益来自 attention 本身，还是来自把 CSI 展成 token 后进行全局混合？
```

### 架构设计

推荐结构：

```text
input: (B, 2, H, W)

patch embedding:
  Conv2d(2, d_model, kernel_size=patch, stride=patch)
  -> flatten spatial -> (B, num_tokens, d_model)

MixerBlock * depth:
  token mixing:
    transpose -> Linear(num_tokens, hidden_tokens)
    GELU
    Linear(hidden_tokens, num_tokens)
    residual
  channel mixing:
    Linear(d_model, hidden_channels)
    GELU
    Linear(hidden_channels, d_model)
    residual

head:
  flatten or mean pool
  Linear(..., code_dim)
```

默认参数建议：

```text
patch = 2 or 4
d_model = 64
depth = 2
hidden_tokens = 2 * num_tokens
hidden_channels = 4 * d_model
```

尺寸流：

```text
32x32, patch=4 -> 8*8=64 tokens
64x64, patch=4 -> 16*16=256 tokens
```

如果 `64x64` 上 token mixing 的 Linear 太大，使用 `patch=8` 或先用 stride conv 下采样。

### 设计要点

- 这是很干净的 baseline，结构简单，容易排查 bug。
- 不适合做最轻量 UE encoder，因为 token mixing MLP 在高分辨率时代价明显。

### 推荐实验名

```text
--encoder mlp_mixer
```

## P1-4：AttentionCNNEncoder

### 适用性

你已有 CLNet，里面有空间注意力和通道注意力。但可以再做一个更规整的 AttentionCNN encoder，作为独立 baseline：

```text
CNN backbone + SE / CBAM / coordinate attention -> code
```

它适合验证注意力模块本身的贡献，而不和 CLNet 的具体多分支设计绑定。

### 架构设计

推荐结构：

```text
input: (B, 2, H, W)

ConvBlock(2 -> width)
SEBlock(width)

ConvBlock(width -> 2*width, stride=2)
CBAMBlock(2*width)

ConvBlock(2*width -> 4*width, stride=2)
SEBlock(4*width)

flatten
Linear(..., code_dim)
```

默认参数建议：

```text
width = 16
attention = se or cbam
```

### 设计要点

- SE 关注“哪些特征通道重要”。
- Spatial attention 关注“哪些角延迟位置重要”。
- 对 sparse CSI map 通常有意义，但过多 attention 会增加小数据过拟合风险。

### 推荐实验名

```text
--encoder attention_cnn
```

## P1-5：CNNEncoder

### 适用性

当前 CSI 压缩反馈任务本质就是 autoencoder：

```text
CSI -> encoder -> code -> decoder -> reconstructed CSI
```

`CNNEncoder` 是最标准、最干净的卷积自编码 encoder baseline。它和 CsiNet / CRNet / CLNet 同属 convolutional autoencoder family，但结构更规整，不依赖多分支或复杂注意力。

它适合回答：

```text
一个标准 ConvAE encoder 能否接近已有专用 CSI encoder？
```

### 架构设计

推荐结构：

```text
input: (B, 2, H, W)

analysis transform:
  Conv2d(2, width, 3, padding=1)
  BatchNorm2d
  LeakyReLU

  Conv2d(width, 2*width, 3, stride=2, padding=1)
  BatchNorm2d
  LeakyReLU

  Conv2d(2*width, 4*width, 3, stride=2, padding=1)
  BatchNorm2d
  LeakyReLU

  Conv2d(4*width, 4*width, 3, padding=1)
  BatchNorm2d
  LeakyReLU

head:
  flatten
  Linear(4*width * H/4 * W/4, code_dim)

output: (B, code_dim)
```

默认参数建议：

```text
width = 16
downsample stages = 2
```

### 设计要点

- 比 ResNet 更简单，适合作为 AE family 的基础参照。
- 如果 decoder 是 `cnn_residual` 或 `hybrid`，CNNEncoder 通常会有比较自然的空间结构匹配。
- 不建议一开始加过深 bottleneck，否则很难区分收益来自 encoder 还是参数量。

### 推荐实验名

```text
--encoder cnn
```

## P1-6：MLPAEEncoder

### 适用性

`MLPAEEncoder` 是最原始的 dense autoencoder encoder：

```text
flatten CSI -> MLP -> code
```

它不使用卷积、attention 或空间先验，适合作为下限 baseline。

### 架构设计

推荐结构：

```text
input: (B, 2, H, W)
flatten: (B, input_dim)

MLP:
  Linear(input_dim, hidden)
  LayerNorm(hidden)
  GELU / LeakyReLU
  Linear(hidden, code_dim)

output: (B, code_dim)
```

默认参数建议：

```text
hidden = min(4096, input_dim)
```

对于 `64x64`：

```text
input_dim = 8192
code_dim = 2048 when cr=4
```

纯 MLP 参数量会较高，因此不建议作为强 baseline，只建议用于消融。

### 设计要点

- 对 `32x32` 可接受。
- 对 `64x64` 参数量明显增大，训练可能更容易过拟合。
- 没有局部结构偏置，NMSE 弱于 CNN/Transformer 属于正常结果。

### 推荐实验名

```text
--encoder mlp_ae
```

## P2-0：VAE / BetaVAE Encoder

### 适用性

Variational autoencoder 让 encoder 输出分布参数：

```text
encoder(x) -> mu, logvar
z = mu + exp(0.5 * logvar) * eps
decoder(z) -> x_hat
```

它适合学习平滑 latent manifold，也适合后续做 domain embedding、latent interpolation 或生成式 decoder。

但对 CSI 压缩反馈的纯 NMSE 目标，它不一定有优势，因为随机采样会损伤精确重建。

### 推荐实现

为了兼容当前 decoder 接口，建议训练时内部采样，测试时使用 deterministic code：

```text
train:
  code = mu + std * eps
  loss = MSE + beta * KL

test:
  code = mu
```

默认参数建议：

```text
beta = 1e-4, 1e-3, 1e-2
```

### 设计要点

- 当前 `Trainer` 需要支持 KL aux loss。
- 如果只关心 NMSE，VAE 不应排在第一批。
- 如果后续做生成式 UniversalCSI 或 domain latent modeling，VAE 有研究价值。

### 推荐实验名

```text
--encoder vae
--encoder beta_vae
```

## P2-1：MaskedAE Pretraining

### 适用性

Masked autoencoder 更适合作为 encoder 预训练方式，而不是直接作为压缩反馈模型。

训练阶段：

```text
patchify CSI
mask 部分 patch
encoder visible patches
decoder reconstruct full CSI
```

微调阶段：

```text
pretrained encoder -> Linear(code_dim) -> UniversalCSI decoder
```

### 设计要点

- 对 `64x64` 更有价值，因为 patch 数更多，mask 预训练更稳定。
- 当前项目第一阶段不建议做，因为它需要额外 pretrain/fine-tune pipeline。
- 如果数据量很大，MAE pretraining 可以提升 encoder 泛化性。

### 推荐实验名

```text
pretrain: --pretrain_task mae
finetune: --encoder mae_resnet / mae_swin
```

## P2-2：SparseTransformCsiEncoder

### 适用性

CSI 在角延迟域本来就具有稀疏结构。除了直接 CNN/Transformer，也可以加一个显式或可学习稀疏变换：

```text
CSI map -> sparse/info domain -> neural encoder -> code
```

这类 encoder 适合提升论文的物理解释性。

### 架构设计 A：固定 DCT / FFT 前处理

如果输入还不是最终稀疏域，可以做：

```text
input: (B, 2, H, W)
fixed transform over spatial dims
top-k or soft threshold
CNN encoder
Linear(code_dim)
```

但如果当前数据已经是预处理后的 sparse angle-delay domain，则不建议重复 FFT/DCT。

### 架构设计 B：Learned Sparse Transform

推荐结构：

```text
input: (B, 2, H, W)

learned transform:
  Conv2d(2, hidden, 1)
  DepthwiseConv2d(hidden, hidden, 3)
  soft threshold / shrinkage

encoder:
  ResNet / Mobile blocks

head:
  Linear(..., code_dim)
```

soft threshold 可写成：

```text
shrink(x, lambda) = sign(x) * relu(abs(x) - lambda)
```

其中 `lambda` 是可学习参数。

### 设计要点

- 适合做可解释 baseline。
- 训练可能比普通 CNN 更敏感，建议放在 P2。

### 推荐实验名

```text
--encoder sparse_resnet
```

## 暂不优先的 Encoder

### TemporalCsiEncoder

适合输入：

```text
(B, T, 2, H, W)
```

当前数据是：

```text
(B, 2, H, W)
```

没有时间维时，LSTM/GRU 只能人为制造 token 序列，意义不如 TransNet / MLP-Mixer 清晰。因此不建议优先加入。

### UplinkAssistedEncoder

适合输入：

```text
downlink CSI + uplink CSI / statistics / side information
```

当前数据没有辅助信息，不适合做主实验。

### DiffusionEncoder

diffusion 更适合作为 decoder/generative reconstruction 或 codebook-conditioned reconstruction，而不是单纯 encoder zoo 的第一批成员。当前接口要求：

```text
encoder -> (B, code_dim)
decoder -> NMSE
```

diffusion 会显著改变 decoder 和训练流程，应放在生成式压缩阶段。

## 推荐实施顺序

第一批，直接兼容当前训练脚本：

```text
1. resnet
2. dscnn
3. cnn
4. convnext
5. mlp_mixer
```

第二批，增强结构多样性：

```text
6. attention_cnn
7. swin
8. mlp_ae
```

第三批，研究型扩展：

```text
9. vae
10. beta_vae
11. sparse_resnet
```

## 建议实验矩阵

先固定 decoder，比较 encoder：

```text
decoder = transnet

encoder in:
  csinet
  crnet
  clnet
  transnet
  resnet
  dscnn
  convnext
  cnn
  mlp_ae
  mlp_mixer
  attention_cnn
  swin
```

然后选择 top-k encoder 进入 decoder zoo：

```text
top encoders x decoder in:
  transnet
  cnn_residual
  hybrid
```

对 `32x32` 和 `64x64` 分开记录：

```text
input shape
cr
code_dim
NMSE
MSE loss
params
FLOPs
train time per epoch
GPU memory
```

## 推荐命名和 parser choices

建议未来 `--encoder` 支持：

```text
csinet
crnet
clnet
transnet
resnet
dscnn
convnext
mlp_mixer
attention_cnn
swin
cnn
mlp_ae
sparse_resnet
vae
beta_vae
```

其中第一批最值得马上实现：

```text
resnet
dscnn
cnn
convnext
mlp_mixer
```

## 参考方向

以下方向可作为后续论文和实现参考：

- 深度自编码 CSI feedback 总览：https://arxiv.org/abs/2206.14383
- Transformer CSI feedback 示例和 TransNet 相关方向：https://www.mathworks.com/help/comm/ug/csi-feedback-transformer-autoencoder.html
- Self-information domain neural CSI compression：https://arxiv.org/abs/2305.07662
- Diffusion model-based CSI compression：https://www.interdigital.com/research_papers/generative-diffusion-model-based-compression-of-mimo-csi
