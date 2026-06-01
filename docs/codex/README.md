# Codex Session Notes

本目录记录本次 Codex 会话中围绕 UniversalCSI 的技术讨论和代码调整。

## 文档列表

- [01_data_range_and_nmse.md](01_data_range_and_nmse.md)
  - 讨论 `[-0.5, 0.5]` 数据范围是否可行、NMSE 如何计算，以及为什么不应再做 `-0.5` 中心化。
- [02_original_project_architecture_comparison.md](02_original_project_architecture_comparison.md)
  - 对比 `~/workspace/Huawei` 下 CsiNet、CRNet、CLNet、TransNet 原始项目的数据范围、输出层和指标路径。
- [03_output_activation_cleanup.md](03_output_activation_cleanup.md)
  - 记录删除 `output_activation` 参数、去掉 `sigmoid/hsigmoid/Identity` 输出层，以及相关脚本和 README 的同步调整。
- [04_general_decoder_design.md](04_general_decoder_design.md)
  - 分析如何设计更通用的 BS 端 decoder，包括 CNN residual、TransNet decoder、hybrid decoder 的取舍。
- [05_multi_vendor_lora_roadmap.md](05_multi_vendor_lora_roadmap.md)
  - 面向多厂家 UE 泛化任务，整理固定 base decoder、LoRA 适配、LoRA 参数生成器、diffusion/flow-matching 的建议路线。
- [06_decoder_choice_gemini_codex_synthesis.md](06_decoder_choice_gemini_codex_synthesis.md)
  - 对比 Gemini 与 Codex 关于 decoder 选型的不同意见，并给出推荐的 HybridDecoder 结构、LoRA 插入点和实验路线。
- [07_matrix_condition_lora_route.md](07_matrix_condition_lora_route.md)
  - 分析使用整个 encoder 输出矩阵 `(N, compressed_dim)` 作为条件生成 domain-level LoRA 的合理性，参考 CCPG 项目并给出 UniversalCSI 的推荐技术路线。
- [08_diffusion_flow_and_parameter_manifold.md](08_diffusion_flow_and_parameter_manifold.md)
  - 总结 diffusion 与 flow-matching 在 decoder adapter 参数生成中的取舍，并从参数流形角度分析 LoRA 的低维性、平滑性、单峰/多模态诊断和不同厂家差异大的处理方式。

## 当前统一假设

UniversalCSI 当前采用预中心化后的 CSI sparse tensor：

```text
shape: (B, 2, 32, 32)
range: [-0.5, 0.5]
channel 0: real
channel 1: imaginary
```

因此 decoder 最终输出为线性实值重建，不再使用 `sigmoid`、`hsigmoid`、`tanh` 或 `Identity` 占位输出层。
