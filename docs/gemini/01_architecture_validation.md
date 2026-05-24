# 01. 架构验证与 Encoder 移植分析

## 概述
本文档总结了对 `UniversalCSI` 项目架构的初步分析，特别是验证了不同的 Encoder（CsiNet, CRNet, CLNet, TransNet）是否从其原始仓库完整且无瑕疵地移植过来。

## 发现

`UniversalCSI` 框架成功实现了解耦架构：
```text
CSI 输入 -> 可选的 encoder -> 可选的 code_adapter -> 共享的 TransNet decoder -> 重建的 CSI
```

### 详细的 Encoder 分析

1.  **CsiNetEncoder (来自 Python_CsiNet):**
    *   **原版:** 使用 Keras 实现，包含 `Conv2D(..., padding='same')`, `BatchNormalization`, `LeakyReLU(alpha=0.3)`, 和 `Dense`。
    *   **移植版:** 完美翻译成了 PyTorch。作者在 `nn.BatchNorm2d` 之前的 `nn.Conv2d` 中正确使用了 `bias=False`，并对齐了 `LeakyReLU` 的 `negative_slope=0.3`。
    *   **状态:** 完美复刻。

2.  **CRNetEncoder (来自 CRNet):**
    *   **原版:** 使用多分辨率并行卷积 (`ConvBN`)，后接线性压缩层。
    *   **移植版:** `encoder1`, `encoder2`, 和 `fc` 层的定义，以及辅助类 `ConvBN`，均与原版 PyTorch 源码一字不差。
    *   **状态:** 完整提取。

3.  **CLNetEncoder (来自 CLNet):**
    *   **原版:** 在 CRNet 主干上增加了空间注意力 (`SpatialGate`) 和通道注意力 (`SELayer`)。通过 `Conv1d` 层压缩，得到形状为 `[Batch, 2048/cr, 1]` 的张量。
    *   **移植版:** 注意力机制被完美移植。进行了一处必要的接口适配：在 `Conv1d` 压缩后增加了一个 `.squeeze(2)` 操作，以满足 `UniversalCSI` 对 `code` 的二维接口要求 `(Batch, 2048/cr)`。
    *   **状态:** 完整提取并进行了正确的接口适配。

4.  **TransNetEncoder (来自 TransNet):**
    *   **原版 Bug:** 原版 `TransNet.py` 存在一个静默的维度错位 Bug。`TransformerEncoderLayer` 默认 `batch_first=False`（期望输入 `[Seq, Batch, Embed]`），但传入的数据却是 `[Batch, Seq, Embed]`。这导致自注意力机制在 Batch 的不同样本之间计算注意力，而不是在单个样本的序列内部计算。
    *   **移植版 (修复):** `UniversalCSI` 在实例化 `TransformerEncoderLayer` 时显式传入了 `batch_first=True`，修复了注意力机制的数学逻辑。
    *   **状态:** 完整移植，并修复了一个底层的关键 Bug。

## 结论
将四种异构的 Encoder 提取并整合到 `UniversalCSI` 框架中的工作非常严谨且完美。它为 CSI 压缩反馈的对比实验提供了一个可靠的统一环境。