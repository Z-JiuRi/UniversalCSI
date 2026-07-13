# exp34：使用 train + val + test 的 affine alignment oracle

此实验保持 exp25 `D_s20_ltc1_lc0` 的模型、损失和训练超参数不变，仅把初始
least-squares affine alignment 的拟合数据从：

```text
train: 100,000 paired source/target codes
```

改为：

```text
train + val + test: 150,000 paired source/target codes
```

随后仍以普通方式训练 mapper，并由 validation decoder NMSE 选 checkpoint。

## 重要限制

该实验拟合 affine 时读取了 validation 和 test 的 **target code**。它不是合法的
held-out 泛化评估，也不能替代 exp25 的正式结果；test NMSE 只用于诊断初始全局线性对齐的
上限。结果应始终标为 `oracle affine / test-code leakage`。

若它相对于 exp25 明显改善，说明更好的跨模型全局坐标对齐仍有价值；若几乎不改善，则 exp25
的主要瓶颈不在 affine 估计样本数。

运行：

```bash
bash adapter/scripts/run_exp34.sh
```
