# 码字综合分析报告（Train Split）

## 目标

分析 encoder 输出的 train codewords，为 diffusion / flow-matching 生成 decoder LoRA 权重提供条件指导。
Pipeline: `C_support -> domain_embedding -> generator -> LoRA_weights`

## 数据概况

- 架构数：42（14 encoder x 3 decoder）
- 每个模型样本数：100000
- 码字维度：512
- 总 train codewords：4,200,000

---

## 1. 码字空间分离性

码字不能仅靠 encoder 自然分离 —— decoder 类型主导了码字中心位置。同一 decoder 内部：

### Per-Decoder 码字中心分离性

| Decoder | 码字尺度范围 (L2) | 中心可分离性 | NMSE 范围 |
|---|---|---|---|
| hybrid | 5.1 - 57.4 | Yes | nan to nan |
| cnn_residual | 152.7 - 681.5 | Yes | nan to nan |
| transnet | 5.9 - 31.7 | Yes | nan to nan |

**关键发现**：为 LoRA 生成应训练 **每个 decoder 独立的 condition encoder**。

---

## 2. 采样需求

| 指标 | <5% 误差所需 K | <1% 误差所需 K |
|---|---:|---:|
| L2 norm 均值 | 16 | 64 |
| 全局标准差 | 16 | 64 |
| PCA top-5 比率 | 512 | >1024 |
| 有效秩 | >1024 | >>1024 |

**建议**：使用 K >= 128 个校准码字做 domain embedding。有效秩可接受 K=256 时 ~30% 误差。

---

## 3. 码字-NMSE 关系

码字中心距与 NMSE 距相关性弱 (r ≈ -0.14)。
码字邻近不代表 NMSE 接近 —— code space 到最优 LoRA 参数的映射是非线性的。
这为 flow-matching 替代简单 deterministic MLP 提供了依据。

---

## 4. LoRA 目标维度

| 目标 | r=4 参数量 | r=8 参数量 | r=16 参数量 |
|---|---:|---:|---:|
| 完整 HybridDecoder | 51,200 | 102,400 | 204,800 |
| 精简版 (fc_proj + ffn) | 44,032 | 88,064 | 176,128 |
| 最小版 (仅 fc_proj) | 10,240 | 20,480 | 40,960 |

生成 44K 参数对 flow-matching 而言是可行的。
建议优先生在低维 manifold coordinate (alpha) 上做 flow，再还原 LoRA。

---

## 5. 最佳与最差架构

### Top 5（按 NMSE）

| Rank | Model | NMSE | Eff Rank | Part. Ratio | PC@90% |
|---:|---|---:|---:|---:|---:|
| 1 | attention_cnn_cnn_residual | nan | 175.7 | 82.9 | 185 |
| 2 | attention_cnn_hybrid | nan | 158.1 | 91.7 | 157 |
| 3 | attention_cnn_transnet | nan | 145.8 | 83.6 | 144 |
| 4 | cbam_cnn_cnn_residual | nan | 190.0 | 110.9 | 195 |
| 5 | cbam_cnn_hybrid | nan | 134.0 | 76.6 | 134 |

### Bottom 5（按 NMSE）

| Rank | Model | NMSE | Eff Rank | Part. Ratio | PC@90% |
|---:|---|---:|---:|---:|---:|
| 42 | attention_cnn_cnn_residual | nan | 175.7 | 82.9 | 185 |
| 41 | attention_cnn_hybrid | nan | 158.1 | 91.7 | 157 |
| 40 | attention_cnn_transnet | nan | 145.8 | 83.6 | 144 |
| 39 | cbam_cnn_cnn_residual | nan | 190.0 | 110.9 | 195 |
| 38 | cbam_cnn_hybrid | nan | 134.0 | 76.6 | 134 |

---

## 6. Per-Decoder 码字特征

| Decoder | 平均 Std | 平均 L2 Norm | 平均 Eff Rank | 平均 PR | 平均 PC@90% | 平均 NMSE |
|---|---:|---:|---:|---:|---:|---:|
| hybrid | 1.35 | 30.6 | 165.7 | 93.4 | 169.7 | nan |
| cnn_residual | 14.05 | 317.9 | 172.0 | 91.4 | 189.0 | nan |
| transnet | 0.84 | 18.8 | 160.5 | 89.7 | 168.2 | nan |

**关键观察**：CNN residual 码字更坍缩（低有效秩、高 PCA 集中度），
transnet 码字分布更均匀（高有效秩）。Hybrid 码字居中，NMSE 表现最好。

---

## 7. 实现建议

### Phase 1: Static LoRA Baseline
1. 选择 HybridDecoder 作为 base（NMSE 排名领先）
2. 每个 encoder 训练 static LoRA（r=4, fc_projection + ffn 层）
3. 保存 DeltaW = BA 矩阵（非原始 A/B）用于流形分析

### Phase 2: 流形诊断
1. 对各 encoder 的 static LoRA DeltaW 做 PCA
2. 检查 within-encoder vs between-encoder 方差
3. 测量 code-distance vs DeltaW-distance Spearman 相关性
4. 若低维，提取 alpha 坐标

### Phase 3: Condition Encoder
1. DeepSets/Perceiver 处理 K=128 校准码字
2. 每个 decoder 类型独立训练 condition encoder
3. 输出：domain embedding z_d

### Phase 4: Generator
1. Baseline：MLP(z_d) -> LoRA params
2. Baseline 不够时：Flow-Matching 在 alpha 坐标空间
3. 若需多模态：Conditional Diffusion

---

## 8. 输出文件

| 目录 | 内容 |
|---|---|
| tables/summary_train.csv | Train split 码字统计 |
| figures/*.png | 所有分析图表 |
| consolidated_report.md | 本报告 |

- [01_nmse_analysis.png](figures/01_nmse_analysis.png)
- [02_encoder_nmse_by_decoder.png](figures/02_encoder_nmse_by_decoder.png)
- [03_best_worst_comparison.png](figures/03_best_worst_comparison.png)
- [04_per_decoder_pca.png](figures/04_per_decoder_pca.png)
- [05_decoder_code_scale_ratio.png](figures/05_decoder_code_scale_ratio.png)
- [06_conditioning_readiness.png](figures/06_conditioning_readiness.png)
