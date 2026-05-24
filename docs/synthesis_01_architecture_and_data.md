# 架构、数据范围与输出激活统一

## UniversalCSI 总体架构

UniversalCSI 的基本设计是将不同 CSI feedback 模型中的 encoder 解耦出来，统一接入共享 BS decoder：

```text
CSI input
  -> selectable encoder
  -> optional / future mandatory code adapter
  -> shared BS decoder
  -> reconstructed CSI
```

当前支持的 encoder：

```text
csinet
crnet
clnet
transnet
```

这套架构服务于后续多厂家 UE 泛化研究：

```text
不同厂家 UE 设备 -> 不同 encoder
BS 端 -> 固定 base decoder
适配方式 -> decoder LoRA / adapter
```

## Encoder 移植状态

Gemini 对 encoder 移植做了逐项检查，结论是整体移植较完整。

### CsiNetEncoder

来源于 `Python_CsiNet`。

原版使用：

```text
Conv2D
BatchNormalization
LeakyReLU(alpha=0.3)
Dense
```

UniversalCSI 中对应 PyTorch 实现保留了核心结构，并在 `Conv2d + BatchNorm2d` 组合中使用 `bias=False`，符合常规写法。

### CRNetEncoder

来源于 `CRNet`。

核心结构：

```text
multi-resolution convolution branches
ConvBN
Linear compression
```

UniversalCSI 中提取了 CRNet 的 encoder 部分，保留多尺度卷积分支和压缩层。

### CLNetEncoder

来源于 `CLNet`。

核心结构：

```text
CRNet-like backbone
SpatialGate
SELayer
Conv1d compression
```

UniversalCSI 中为了统一接口，将原始 `Conv1d` 输出从：

```text
(B, code_dim, 1)
```

适配为：

```text
(B, code_dim)
```

即增加 `.squeeze(2)`，这是合理的接口适配。

### TransNetEncoder

来源于 `TransNet`。

重点是修复了历史 Transformer 实现中容易出现的 `batch_first` 维度问题。统一后约定 Transformer 输入为：

```text
(B, seq_len, d_model)
```

而不是 PyTorch 旧默认的：

```text
(seq_len, B, d_model)
```

## 数据范围统一

原始 CsiNet、CRNet、CLNet 的 COST2100 `.mat` 数据通常是：

```text
HT range: [0, 1]
mean: approximately 0.5
```

当前 UniversalCSI 的 `.pt` 数据已经预中心化为：

```text
range: [-0.5, 0.5]
mean: approximately 0
```

这一点在会话中用实际数据验证过：

```text
DATA_Htrainin.mat  min≈0       max≈1       mean≈0.5
DATA_Htestin.mat   min≈0       max≈1       mean≈0.5
DATA_Htrainout.mat min≈0       max≈1       mean≈0.5
DATA_Htestout.mat  min≈0       max≈1       mean≈0.5

in_train.pt        min=-0.5    max=0.5     mean≈0
in_test.pt         min=-0.5    max=0.5     mean≈0
out_train.pt       min=-0.5    max=0.5     mean≈0
out_test.pt        min≈-0.5    max≈0.5     mean≈0
```

## 为什么 `[-0.5, 0.5]` 是合理的

CSI sparse angle-delay domain tensor 的两个 channel 表示：

```text
channel 0: real
channel 1: imaginary
```

实部和虚部本来就是以 0 为中心的物理量。原始 `[0,1]` 表示方式更像是为了配合图像式神经网络训练和 `sigmoid` 输出层，将真实中心化值整体平移了 `+0.5`。

因此当前预处理为：

```text
[-0.5, 0.5]
```

不仅可行，而且语义更直接。

## 原始项目和输出激活的绑定

原始项目普遍存在以下闭环：

```text
data: [0,1]
decoder output: [0,1] via sigmoid / hsigmoid
metric: subtract 0.5 before NMSE/rho
```

### CsiNet

最终输出：

```python
Conv2D(2, (3, 3), activation='sigmoid', padding='same')
```

指标中：

```python
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
```

### CRNet

模型中：

```python
self.sigmoid = nn.Sigmoid()
out = self.sigmoid(out)
```

指标中：

```python
sparse_gt = sparse_gt - 0.5
sparse_pred = sparse_pred - 0.5
```

### CLNet

模型中：

```python
self.hsig = hsigmoid()
out = self.hsig(out)
```

指标中同样做：

```python
sparse_gt = sparse_gt - 0.5
sparse_pred = sparse_pred - 0.5
```

## UniversalCSI 当前应采用的输出约定

在 `[-0.5,0.5]` 数据范围下，decoder 最终不应使用：

```text
sigmoid
hsigmoid
tanh
Identity placeholder
```

而应直接：

```python
return out
```

原因：

- `sigmoid/hsigmoid` 无法输出负数。
- `tanh` 范围是 `[-1,1]`，不是必要约束，还可能引入饱和。
- `Identity` 没有实际收益，只会制造结构噪声。

内部注意力门控中的 sigmoid 不需要删除，例如：

```python
scale = torch.sigmoid(...)
```

因为它控制的是 attention / gate 权重，不是最终重建值范围。

## NMSE 计算约定

当前 evaluator 应直接在中心化 sparse domain 中计算：

```python
power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
difference = sparse_gt - sparse_pred
mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
```

不再执行：

```python
sparse_gt = sparse_gt - 0.5
sparse_pred = sparse_pred - 0.5
```

坐标系统一原则：

```text
数据坐标系 == 模型输出坐标系 == 指标计算坐标系
```

当前即：

```text
数据: [-0.5, 0.5]
输出: 线性实值
指标: 不再中心化
```

## output_activation 清理结论

UniversalCSI 中曾经存在：

```bash
--output_activation none
--output_activation sigmoid
--output_activation hsigmoid
```

合并结论是：该参数应该删除。

当前代码清理目标：

```text
parser 中删除 --output_activation
init 中不再传 output_activation
scripts 中不再传 output_activation
TransNetDecoder.forward 直接 return out
README 不再推荐 output_activation
```

历史 `exps/` 里的日志和 `args.json` 可能仍包含该字段，但它们是历史产物，不应作为当前代码状态依据。

