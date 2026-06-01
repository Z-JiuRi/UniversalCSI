# 数据范围与 NMSE 计算

## 背景

本次讨论的起点是：原始 TransNet、CsiNet、CRNet、CLNet 等 CSI feedback 项目中，输入 sparse CSI 通常存为 `[0, 1]`，而当前 UniversalCSI 的预处理阶段已经将数据改为 `[-0.5, 0.5]`。

用户关心的问题是：

1. 这个改动是否可行。
2. 如果可行，原本为 `[0, 1]` 设计的网络是否需要同步修改。
3. `utils/statics.py::evaluator(sparse_pred, sparse_gt)` 中是否还应该保留 `-0.5` 中心化代码。

## 结论

将 sparse CSI 数据预处理到 `[-0.5, 0.5]` 是可行的，而且从实部/虚部的物理意义上更自然。

原因是 CSI 的实部和虚部本来就是以 0 为中心的复数分量。原始数据中的 `[0, 1]` 表示方式不是物理域本身，而是为了配合神经网络的 `sigmoid` 输出层和图像式归一化流程，将中心化复数分量整体平移了 `+0.5`。

在 UniversalCSI 中，如果数据已经是 `[-0.5, 0.5]`，则：

```python
sparse_gt = sparse_gt - 0.5
sparse_pred = sparse_pred - 0.5
```

这类代码必须删除或注释。否则会把真实数据错误地变成 `[-1.0, 0.0]`，导致 NMSE 的分子和分母都在错误坐标系下计算。

## 当前 UniversalCSI 的 evaluator 状态

当前 [utils/statics.py](../../utils/statics.py) 中的 `evaluator()` 已经适配 `[-0.5, 0.5]`：

```python
def evaluator(sparse_pred, sparse_gt):
    with torch.no_grad():
        # # De-centralize
        # sparse_gt = sparse_gt - 0.5
        # sparse_pred = sparse_pred - 0.5

        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
        return nmse
```

这与当前 `.pt` 数据范围是一致的。

## 实际数据范围验证

会话中检查了 `~/workspace/Huawei/TransNet/data/COST2100/` 下的原始 `.mat` 文件和转换后的 `.pt` 文件。

观察结果：

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

这说明：

- 原始 COST2100 `.mat` 中的 `HT` 是 `[0, 1]` 坐标系。
- 当前 `.pt` 数据已经是中心化后的 `[-0.5, 0.5]` 坐标系。

## 坐标系统一原则

后续所有模型和指标实现都应遵守同一个原则：

```text
数据坐标系 == 模型输出坐标系 == 指标计算坐标系
```

对于 UniversalCSI 当前选择：

```text
数据范围: [-0.5, 0.5]
模型输出: 线性实值输出
NMSE: 直接在当前 sparse 域计算
```

因此：

- 不应在 evaluator 中再 `-0.5`。
- 不应在 decoder 最后使用 `sigmoid` 或 `hsigmoid`。
- 不应使用原始 `[0,1]` checkpoint 直接评估当前 `[-0.5,0.5]` 数据。

## Checkpoint 兼容性

如果 checkpoint 是在 `[0, 1]` 数据上训练的，它默认学习的是：

```text
input:  [0, 1]
target: [0, 1]
metric: 评估时 -0.5
```

而当前 UniversalCSI 是：

```text
input:  [-0.5, 0.5]
target: [-0.5, 0.5]
metric: 不再 -0.5
```

两者不能直接公平比较。可选方案：

1. 使用原始 `[0,1]` 数据和原始 evaluator 评估原始 checkpoint。
2. 使用当前 `[-0.5,0.5]` 数据重新训练或微调模型。
3. 如果必须迁移旧权重，需要明确处理输入输出坐标变换，否则结果没有可解释性。

## 建议写入项目规范

当前 `AGENTS.md` 中仍有旧说明，称 evaluator 会先对 `sparse_gt` 和 `sparse_pred` 减 `0.5`。这已经和当前代码不一致。

建议后续同步更新为：

```text
当前 UniversalCSI 使用预中心化后的 sparse CSI 数据，范围为 [-0.5, 0.5]。
NMSE 直接在当前 sparse 域计算，不再执行 sparse_gt - 0.5 或 sparse_pred - 0.5。
```

