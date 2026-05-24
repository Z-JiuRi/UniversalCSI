# 02. TransNet 代码重构与优化

## 概述
本文档详细记录了在原始 `TransNet` 实现中发现 Bug、修复 Bug，并随后使用高度优化的 PyTorch 官方库对整个模块进行重构的协作过程。

## 1. 原始实现中的初步 Bug 修复

用户识别并修复了原始自定义 `TransNet.py` 源码中的两个关键 Bug：

1.  **维度错位 (`batch_first`):**
    *   **问题:** `Transformer` 类中默认的 `batch_first=False` 导致底层的注意力层将 `Batch_Size` 维度错误地解析为 `Seq_Len`。
    *   **修复:** 将 `Transformer.__init__` 中的默认参数改为 `batch_first=True`。
2.  **`scale_dot_attention` 中的参数顺序:**
    *   **问题:** 函数调用时使用了位置参数 `(q, k, v, attn_mask, dropout_p)`，但函数签名是 `(q, k, v, dropout_p, attn_mask)`。这导致一个浮点数和一个张量被互换传入。
    *   **修复:** 将调用改为使用显式的关键字参数：`dropout_p=dropout_p, attn_mask=attn_mask`。

## 2. 使用 PyTorch 官方 API 的全面重构

随后，用户重写了整个 `TransNet` 模块，摒弃了数百行容易出错的自定义注意力逻辑，转而使用 `torch.nn.TransformerEncoderLayer` 和 `torch.nn.TransformerDecoderLayer`。

### 审查与优化

最初的重写非常出色，代码行数从约 400 行减少到约 150 行，同时利用了 PyTorch 的后端优化（如 FlashAttention）。

在此基础上提出了进一步的优化建议并被采纳：

1.  **移除旧版本兼容代码:**
    *   最初的重写包含了复杂的反射 (`inspect.signature`) 和手动的 `transpose` 操作，以支持不支持 Transformer 层中 `batch_first` 参数的旧版 PyTorch。
    *   **优化:** 假设当前为现代 PyTorch 环境 (>= 1.9)，完全移除了这些探测代码，直接向官方层传入 `batch_first=True`，使代码极其清爽。
2.  **精简前向传播语义:**
    *   由于 CSI 反馈重建不是自回归序列生成任务（不像语言模型），因此因果掩码（`tgt_mask`, `memory_mask`）从根本上是不需要的。
    *   虽然将 `None` 传给这些参数是无害的，但讨论后决定从内部执行流中完全移除它们，以明确空间/频率信号重建的非自回归本质。

## 最终结果
最终重写的 `TransNet.py` 结构紧凑（124行）、鲁棒、数学上完全正确，并且经过了高度优化。这个更新后的文件随后被同步回 `UniversalCSI` 仓库，作为统一的 `TransNetDecoder` 使用。