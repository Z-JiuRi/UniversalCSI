# UniversalCSI Synthesis Index

本组文档合并了 `docs/codex/` 与 `docs/gemini/` 中的材料，形成统一版本，保存在 `docs/` 根目录。

## 合并文档列表

- [synthesis_01_architecture_and_data.md](synthesis_01_architecture_and_data.md)
  - UniversalCSI 总体架构、encoder 移植情况、数据范围、输出激活与 NMSE 坐标系统一。
- [synthesis_02_transnet_refactor_and_model_contract.md](synthesis_02_transnet_refactor_and_model_contract.md)
  - TransNet 重构背景、官方 PyTorch Transformer API、维度契约、decoder 输出约定。
- [synthesis_03_decoder_selection_and_hybrid_design.md](synthesis_03_decoder_selection_and_hybrid_design.md)
  - 纯 Transformer、纯 CNN residual、HybridDecoder 的取舍，以及最终推荐结构。
- [synthesis_04_multi_vendor_lora_strategy.md](synthesis_04_multi_vendor_lora_strategy.md)
  - 多厂家 UE 泛化目标、固定 BS base decoder、LoRA 插入点、厂家级 LoRA 生成策略。
- [synthesis_05_experiment_roadmap.md](synthesis_05_experiment_roadmap.md)
  - 分阶段实验路线、关键消融矩阵、评价指标和成功标准。
- [synthesis_06_matrix_condition_lora_route.md](synthesis_06_matrix_condition_lora_route.md)
  - 合并 Codex 与 Gemini 关于 `(N, compressed_dim)` 整体矩阵条件生成 domain-level LoRA 的分析，给出 CCPG 经验迁移和 UniversalCSI 推荐路线。
- [synthesis_07_encoder_zoo_design.md](synthesis_07_encoder_zoo_design.md)
  - 面向 `(N,2,32,32)` / `(N,2,64,64)` 到 `(N,code_dim)` 压缩反馈接口的 encoder zoo 设计、优先级和架构说明。

## 当前统一结论

UniversalCSI 当前应采用以下核心设定：

```text
数据范围: [-0.5, 0.5]
输出层: 线性输出，直接 return out
NMSE: 不再执行 sparse_gt - 0.5 或 sparse_pred - 0.5
BS decoder: 固定 base decoder + 可生成 LoRA adapter
主推荐 decoder: HybridDecoder = Transformer 全局对齐 + CNN residual 物理细化
LoRA 条件: 优先使用厂家级 / encoder 级 domain embedding，而不是单样本 code
```

## 来源材料

Codex 材料：

```text
docs/codex/01_data_range_and_nmse.md
docs/codex/02_original_project_architecture_comparison.md
docs/codex/03_output_activation_cleanup.md
docs/codex/04_general_decoder_design.md
docs/codex/05_multi_vendor_lora_roadmap.md
docs/codex/06_decoder_choice_gemini_codex_synthesis.md
```

Gemini 材料：

```text
docs/gemini/01_architecture_validation.md
docs/gemini/02_transnet_refactoring.md
docs/gemini/03_data_distribution_and_activation.md
docs/gemini/04_future_research_multi_vendor_generalization.md
docs/gemini/05_hybrid_decoder_and_lora_strategy.md
```
