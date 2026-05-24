# 原始项目架构对比

## 对比范围

本次会话参考了 `~/workspace/Huawei` 下几个相关项目：

```text
/home/z-jiuri/workspace/Huawei/Python_CsiNet
/home/z-jiuri/workspace/Huawei/CRNet
/home/z-jiuri/workspace/Huawei/CLNet
/home/z-jiuri/workspace/Huawei/TransNet
/home/z-jiuri/workspace/Huawei/UniversalCSI
```

重点比较：

1. 数据读取后是否为 `[0,1]`。
2. decoder 最后一层是否限制输出范围。
3. NMSE/rho 计算前是否做 `-0.5`。
4. 哪些设计需要在 UniversalCSI 的 `[-0.5,0.5]` 数据设定下修改。

## CsiNet

原始 CsiNet 是典型的 `[0,1]` 闭环。

在 `Python_CsiNet/CsiNet_train.py` 中，最终输出层是：

```python
x = Conv2D(
    2,
    (3, 3),
    activation='sigmoid',
    padding='same',
    data_format="channels_first",
)(x)
```

这意味着模型输出被限制在 `[0,1]`。

随后在测试指标中，代码将真实值和预测值都减去 `0.5`：

```python
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
```

因此 CsiNet 的原始路径是：

```text
HT: [0,1]
model output: [0,1] via sigmoid
metric domain: output - 0.5
```

### CsiNet Improvement 中的 tanh

`Python_CsiNet/Improvement/README.md` 讨论了 `sigmoid` 和 `tanh`，但该讨论主要针对 codeword 或编码向量的量化便利性，并不是说最终重建图一定应使用 `tanh`。

在 `Improvement/CsiNet_Train.py` 中，dense code 部分出现：

```python
self.dense = nn.Sequential(
    nn.Linear(img_total, encoded_dim),
    nn.Tanh(),
    nn.Linear(encoded_dim, img_total),
)
```

但最终重建输出仍是：

```python
self.conv2 = nn.Sequential(
    nn.Conv2d(img_channels, 2, kernel_size=(3, 3), padding=(1,1)),
    nn.Sigmoid(),
)
```

所以该改进仍然默认最终重建目标为 `[0,1]`。

## CRNet

CRNet 原始模型同样假设 `[0,1]` 输出。

在 `CRNet/models/crnet.py` 中：

```python
self.sigmoid = nn.Sigmoid()
```

forward 末尾：

```python
out = self.decoder_feature(out)
out = self.sigmoid(out)
return out
```

指标文件 `CRNet/utils/statics.py` 中：

```python
sparse_gt = sparse_gt - 0.5
sparse_pred = sparse_pred - 0.5
```

因此 CRNet 也是：

```text
训练目标: [0,1]
模型输出: [0,1]
NMSE: 转换到 [-0.5,0.5] 后计算
```

如果要在 UniversalCSI 的 `[-0.5,0.5]` 数据上复用 CRNet 风格结构，必须去掉最终 `sigmoid`。

## CLNet

CLNet 原始模型也默认输出 `[0,1]`。

在 `CLNet/models/clnet.py` 中，decoder 初始化里有：

```python
self.sigmoid = nn.Sigmoid()
self.hsig = hsigmoid()
```

forward 中使用的是：

```python
out = self.decoder_feature(out)
out = self.hsig(out)
return out
```

`hsigmoid` 的定义为：

```python
out = F.relu6(x + 3, inplace=True) / 6
```

它同样将输出限制在 `[0,1]`。

指标中也做：

```python
sparse_gt = sparse_gt - 0.5
sparse_pred = sparse_pred - 0.5
```

因此 CLNet 的原始设计同样不能直接用于 `[-0.5,0.5]` 标签，除非移除最终输出激活。

## TransNet

当前本地 `TransNet` 版本与 CsiNet/CRNet/CLNet 不同：

- 模型最后没有 `sigmoid`。
- `utils/statics.py` 里的 `-0.5` 已被注释。

也就是说，本地 TransNet 已经更接近 UniversalCSI 当前的 `[-0.5,0.5]` 数据设定。

TransNet decoder 主要流程：

```text
code
  -> fc_decoder
  -> reshape to Transformer sequence
  -> TransformerDecoder
  -> reshape to (B, 2, nt, nc)
```

最终输出是线性实值，不限制范围。

## UniversalCSI 当前设计

UniversalCSI 的目标是：

```text
selectable encoder
  -> shared TransNet decoder
  -> reconstructed centered CSI
```

在本次会话前，UniversalCSI 曾保留：

```bash
--output_activation none|sigmoid|hsigmoid
```

这本来是为了兼容原始 `[0,1]` 项目中的输出激活。但在 UniversalCSI 已经统一使用 `[-0.5,0.5]` 数据后，这个参数会制造误用风险。

因此本次会话中已经删除 `output_activation`，decoder 直接返回线性输出。

## 迁移原始网络时的原则

如果只迁移 encoder：

```text
CsiNet/CRNet/CLNet encoder -> UniversalCSI shared decoder
```

通常不需要迁移原始 decoder 的 `sigmoid/hsigmoid`。

如果要迁移原始 decoder 结构：

```text
Linear expand -> CNN residual blocks -> output
```

必须删除最终：

```python
nn.Sigmoid()
hsigmoid()
```

并保持：

```python
return out
```

因为当前 target 是中心化实值。

内部注意力门控里的 sigmoid 不需要删除。例如：

```python
scale = torch.sigmoid(...)
```

这类 sigmoid 是用于生成注意力权重，不是限制最终重建输出范围。

