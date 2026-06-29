# encoder_canonical 实验分析脚本说明

这一组脚本用于分析 `exps/COST2100/in/encoder_canonical` 下已经跑完或正在跑的实验。目标是先把组会汇报需要的基础数据和图自动生成出来：日志指标、adapter 指标、码字统计、码字对齐关系，以及不同 canonical 方案的整体对比。

脚本默认只读取实验目录，不修改实验结果。输出默认写到：

```bash
analysis_outputs/encoder_canonical
```

## 快速使用

完整跑一遍，码字统计尽量放到 GPU 1：

```bash
python analysis/encoder_canonical/run_all.py --device cuda --gpu 1
```

如果只想快速看日志和 adapter 指标，不加载码字：

```bash
python analysis/encoder_canonical/run_all.py --skip-codewords
```

如果码字很多，先抽样分析：

```bash
python analysis/encoder_canonical/run_all.py --sample-size 20000 --device cuda --gpu 1
```

`--device auto` 会优先使用可用 CUDA；`--device cuda --gpu N` 可以指定第 N 张 GPU。日志解析和画图主要在 CPU 上完成，GPU 主要用于 `analyze_codewords.py` 里的码字矩阵统计和两两比较。

## 输出文件

运行后主要生成：

```text
analysis_outputs/encoder_canonical/
  experiment_log_summary.csv    # 每个实验目录一行的日志和配置汇总
  scheme_log_summary.csv        # 按 scheme 聚合后的日志指标
  codeword_stats.csv            # 单个实验的码字分布统计
  codeword_pairwise.csv         # 不同实验之间的码字距离/相似度
  report_auto_summary.md        # 自动生成的简要中文结果摘要
  figures/*.png                 # 汇报用图
```

## 脚本说明

### 1. `parse_logs.py`

分析什么：

- 扫描 `encoder_canonical` 下所有包含 `run.log` 或 `args.json` 的实验目录。
- 提取每个实验的 best/final/latest Test NMSE、训练是否完成、是否有报错、是否已经保存 codeword。
- 提取训练配置，例如 seed、encoder、decoder、canonical scheme、lambda_anchor、code regularization 权重、adapter 类型、rank、hidden size、loss lambda 等。
- 对 adapter 实验额外解析 `adapter_delta_ratio`、`adapter_gate_mean`、`adapter_gate_std`、`loss_code`、`loss_fc`、`loss_recT` 等日志里出现的指标。

为什么要分析：

- 现在实验目录很多，而且有些还在跑。只看目录名很难判断哪个实验真正完成、哪个实验中途失败、哪个实验只有 checkpoint 但没保存 codeword。
- 比较 canonical 方案时，必须先确认 NMSE 是来自同一类实验、同一评估口径，而不是把未完成实验或报错实验混进来。
- adapter 效果差时，不能只看最终 NMSE，还要看 adapter 是否实际改变了 code、门控是否退化、code/fc/teacher loss 是否主导训练。

怎么分析：

- 用正则从 `run.log` 里解析 `Test NMSE`、best NMSE、epoch、loss 和 adapter 相关指标。
- 读取 `args.json` 补全命令行参数，避免只依赖目录名。
- 从相对路径推断实验族：
  - canonical encoder/decoder 训练；
  - adapter 训练；
  - 不同 canonical scheme；
  - 不同 seed、encoder、decoder 组合。
- 输出两张 CSV：
  - `experiment_log_summary.csv`：每个实验目录一行；
  - `scheme_log_summary.csv`：按 scheme 聚合，便于看平均 NMSE、最优 NMSE、完成数量。

### 2. `analyze_codewords.py`

分析什么：

- 读取每个实验保存的 `codewords/train_code.pt`。
- 统计单个实验内部的 code 分布：
  - 全局均值、标准差、RMS；
  - 样本 code 范数；
  - 每个维度的均值和方差；
  - covariance 的 off-diagonal ratio；
  - effective rank。
- 比较不同实验之间的同一样本 code：
  - cosine similarity；
  - MSE；
  - L2 distance。

为什么要分析：

- 当前问题的核心不是单个自编码器能不能重建，而是不同 seed/encoder 学到的 code 坐标系是否一致。
- 自编码器只约束 `D(E(x)) ≈ x`，理论上存在 `R E(x)` 和 `D R^{-1}` 的等价解，所以只看重建 NMSE 看不出 code 坐标系是否对齐。
- canonical 约束是否有效，需要看 code 的统计结构和跨实验同一样本 code 的相似度：
  - 如果 cosine 更高、MSE 更低，说明坐标系更接近；
  - 如果 covariance offdiag 更低，说明维度相关性更弱；
  - 如果 effective rank 太低，说明 code 可能塌缩；
  - 如果 scale 差异很大，adapter 可能先被迫做尺度校准，而不是学习真正的结构映射。

怎么分析：

- 对每个 `train_code.pt` 取前 `--sample-size` 个样本；如果想全量分析，可以把 `--sample-size` 设得大于样本数。
- 将 code reshape 成二维 `(N, code_dim)`。
- 对中心化后的 code `Xc` 计算协方差：

```text
C = Xc.T @ Xc / (N - 1)
```

- off-diagonal ratio 计算为：

```text
sum(abs(C - diag(C))) / sum(abs(C))
```

- effective rank 使用稳定的能量维度估计：

```text
effective_rank = (trace(C)^2) / trace(C^2)
```

- 两两实验比较时，对同一样本位置的 code 计算 cosine/MSE/L2。这个指标直接回答“不同 encoder 对同一 CSI 是否编码到相近坐标”的问题。

### 3. `plot_summary.py`

分析什么：

- 根据 `experiment_log_summary.csv`、`scheme_log_summary.csv`、`codeword_stats.csv`、`codeword_pairwise.csv` 生成汇报用图。
- 当前会生成：
  - scheme NMSE 排名；
  - 实验完成状态统计；
  - encoder/decoder NMSE heatmap；
  - adapter NMSE top 图；
  - adapter 指标关系图；
  - effective rank 与 NMSE 的关系；
  - covariance offdiag ratio 与 NMSE 的关系；
  - 不同 scheme 的 effective rank；
  - 不同 scheme pair 的 code cosine。

为什么要分析：

- 组会汇报不能只给一堆日志数字，需要图来回答几个关键问题：
  - 哪些 canonical 方案本身重建性能最好；
  - 哪些方案训练稳定、完成率高；
  - code 统计有没有被规范化；
  - code 对齐是否真的改善；
  - adapter 差是因为 code 未对齐、结构不够强，还是训练 loss/门控设置有问题。

怎么分析：

- 用 pandas 读取前面两个脚本输出的 CSV。
- 用 seaborn/matplotlib 画图。
- 字体设置复用 `test_fonts.py` 的思路，优先使用本机宋体和 Times New Roman：

```text
/home/hujiacong/zxd/.envs/SongTi.ttf
/home/hujiacong/zxd/.envs/TimesNewRoman.ttf
```

- 为避免 matplotlib 在受限环境下写用户 cache 失败，公共模块会把 `MPLCONFIGDIR` 指到 `/tmp/matplotlib`。

### 4. `run_all.py`

分析什么：

- 串起完整分析流程：
  1. 解析日志；
  2. 可选分析码字；
  3. 生成图和自动摘要。

为什么要分析：

- 当前实验还没全部跑完，后续会不断产生新日志和新 codeword。需要一个可重复运行的入口，每次只要重新跑一条命令就能刷新所有 CSV 和图。
- 这样可以避免手工复制日志、手工合并表格导致口径不一致。

怎么分析：

- 通过 subprocess 依次调用：

```text
parse_logs.py -> analyze_codewords.py -> plot_summary.py
```

- 如果指定 `--skip-codewords`，会跳过码字加载，只做日志和图中不依赖码字的部分。
- `--root` 可以改成其他实验根目录，`--out-dir` 可以指定新的输出目录，便于不污染已有分析结果。

### 5. `common.py`

分析什么：

- 这个文件本身不做实验分析，提供公共工具函数。

为什么要有：

- 多个脚本都需要处理路径、创建目录、读取 CSV、设置中文字体、保存图片。
- 把这些逻辑集中起来可以保证所有图的字体、尺寸、输出行为一致。

怎么实现：

- 提供实验根目录和输出目录的默认路径。
- 提供 matplotlib 中文字体初始化。
- 提供安全的数值转换、路径处理和 figure 保存函数。

## 建议的汇报前工作流

1. 先快速跑日志分析，确认哪些实验已经完成：

```bash
python analysis/encoder_canonical/run_all.py --skip-codewords
```

2. 对已经保存 codeword 的实验跑 GPU 码字分析：

```bash
python analysis/encoder_canonical/run_all.py --device cuda --gpu 1 --sample-size 20000
```

3. 打开 `analysis_outputs/encoder_canonical/report_auto_summary.md` 和 `figures/`，筛选组会中最关键的图。

4. 如果新实验又保存了 codeword，重新运行同一条 `run_all.py` 命令即可刷新结果。

## 注意事项

- `analyze_codewords.py` 默认只分析已经存在 `codewords/train_code.pt` 的实验；没有保存码字的实验会保留在日志 CSV 里，但不会进入 codeword CSV。
- 如果 adapter 码字保存逻辑已经改成 best model，后续重新导出的 codeword 才能代表 best checkpoint；旧的最后一轮 codeword 不建议混用。
- pairwise code 对比默认假设不同实验的 `train_code.pt` 样本顺序一致。如果训练集保存顺序被改变，同一样本 cosine/MSE 就没有可比性。
- GPU 只加速码字张量计算，不会明显加速日志解析和普通画图。
