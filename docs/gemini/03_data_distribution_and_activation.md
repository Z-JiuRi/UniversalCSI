# 03. 数据分布偏移与输出激活层分析

## 概述
本文档总结了关于将 CSI 数据集的归一化范围从 `[0, 1]` 更改为 `[-0.5, 0.5]` 的讨论，以及这种改变对网络架构的影响，特别是针对 `output_activation` 参数的处理。

## 1. `[-0.5, 0.5]` 数据集偏移

用户提出更改 CSI 数据预处理，将值映射到 `[-0.5, 0.5]` 而不是 `[0, 1]`。
*   **评估:** 这是一个极其推荐的优化。零均值数据能改善反向传播时的梯度流动，防止梯度呈现“Z”字型更新，通常能带来更快、更好的收敛。
*   **Evaluator 修复:** 因此，在 `evaluator` 指标计算中移除 `sparse_gt - 0.5` 的代码是绝对正确的，因为此时数据已经是零均值的了。

## 2. 对原始网络架构的影响

原始网络与 `[0, 1]` 数据分布深度绑定：
*   **CsiNet & CRNet:** 在其解码器的最后放置了一个严格的 `nn.Sigmoid()`，强制将输出压在 `[0, 1]` 之间。
*   **CLNet:** 为了同样的目的，使用了自定义的 `HSigmoid` (`F.relu6(x + 3) / 6`)。
*   **TransNet:** 独特之处在于它*没有*使用 Sigmoid，输出的是无界的线性值（这也是为什么它最初的 evaluator 必须手动减去 0.5）。

如果在 `[-0.5, 0.5]` 的数据上单独测试，CsiNet、CRNet 和 CLNet 将会**面临灾难性的失败**，因为它们的 Sigmoid 层阻止了网络输出负数。

## 3. UniversalCSI 的优雅之处

`UniversalCSI` 的解耦特性彻底解决了这个问题。因为那些专有的解码器（及其 Sigmoid 层）被丢弃了，只保留了 Encoder，最终的输出约束完全由共享的 `TransNetDecoder` 控制。

### `output_activation` 参数

最初，这个参数（`none`, `sigmoid`, `hsigmoid`）是为了给那些期望有界输出的网络提供向后兼容性。
*   通过简单地设置 `--output_activation none`，共享的 Transformer 解码器将保持线性输出，完美适应新的 `[-0.5, 0.5]` 分布，而根本不需要去修改 CsiNet、CRNet 或 CLNet 编码器的源码。

## 4. 代码清理：消除死代码

鉴于项目永久转向零均值数据集，保留 `output_activation` 逻辑及其相关的 `nn.Identity()` 占位符就变成了不必要的技术债务。
*   **采取的行动:** 用户从框架中彻底剥离了所有与 `output_activation` 相关的逻辑（包括 Parser、Shell 脚本、`UniversalCSI.py` 初始化以及 `TransNetDecoder.forward`）。
*   **结果:** `TransNetDecoder` 现在以干净、直接的 `return out` 结束，确保了最高性能，防止了意外的配置错误（例如错误地应用了 sigmoid），并在整个项目中确立了稳固的 `[-0.5, 0.5]` 范式。