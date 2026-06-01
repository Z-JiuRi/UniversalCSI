# output_activation 清理记录

## 背景

UniversalCSI 之前支持命令行参数：

```bash
--output_activation none
--output_activation sigmoid
--output_activation hsigmoid
```

对应 [models/UniversalCSI.py](../../models/UniversalCSI.py) 中的 `TransNetDecoder` 输出层：

```python
if output_activation == "sigmoid":
    self.output_activation = nn.Sigmoid()
elif output_activation == "hsigmoid":
    self.output_activation = HSigmoid()
else:
    self.output_activation = nn.Identity()
```

forward 中：

```python
return self.output_activation(out)
```

在当前项目已经统一使用 `[-0.5,0.5]` 数据范围后，这个参数不再合适。

## 为什么删除

### sigmoid/hsigmoid 不适合当前标签

`sigmoid` 和 `hsigmoid` 都会把最终输出限制在 `[0,1]`。

但当前 target 是：

```text
[-0.5, 0.5]
```

如果模型最后使用 `sigmoid`，则无法输出负数。模型会系统性偏正，负值区域只能通过接近 0 的输出来近似，导致训练和 NMSE 都异常。

### Identity 没有必要

在决定彻底删除可配置输出激活后，保留：

```python
self.output_activation = nn.Identity()
```

没有实际收益。

它只会让模型结构打印时多一个无意义节点，并让读代码的人误以为输出层仍是一个可配置设计点。

因此最终选择：

```python
return out
```

## 已执行的代码修改

### models/UniversalCSI.py

删除：

```python
class HSigmoid(nn.Module):
    ...
```

删除 `TransNetDecoder.__init__()` 中的 `output_activation` 参数。

修改前：

```python
class TransNetDecoder(nn.Module):
    def __init__(..., dim_feedforward=None, output_activation="none"):
        ...
        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "hsigmoid":
            self.output_activation = HSigmoid()
        else:
            self.output_activation = nn.Identity()
```

修改后：

```python
class TransNetDecoder(nn.Module):
    def __init__(..., dim_feedforward=None):
        ...
```

修改 forward。

修改前：

```python
out = self.decoder(memory, memory)
out = out.view(batch_size, self.channel, self.nt, self.nc)
return self.output_activation(out)
```

修改后：

```python
out = self.decoder(memory, memory)
out = out.view(batch_size, self.channel, self.nt, self.nc)
return out
```

修改 `universal_csi()` 签名。

修改前：

```python
def universal_csi(..., code_adapter=False, output_activation="none"):
```

修改后：

```python
def universal_csi(..., code_adapter=False):
```

### utils/parser.py

删除：

```python
parser.add_argument('--output_activation', type=str, default='none',
                    choices=['none', 'sigmoid', 'hsigmoid'],
                    help='optional activation after the shared TransNet decoder')
```

### utils/init.py

删除模型构造中的：

```python
output_activation=args.output_activation
```

删除日志中的：

```python
f'output_activation={args.output_activation}; '
```

### scripts/train.sh

删除默认环境变量：

```bash
output_activation=${output_activation:-none}
```

删除运行参数：

```bash
--output_activation "${output_activation}"
```

### scripts/test.sh

同样删除：

```bash
output_activation=${output_activation:-none}
--output_activation "${output_activation}"
```

### README.md

删除 `--output_activation` 相关说明。

修改文件列表说明：

```text
utils/parser.py          adds --encoder and --code_adapter
```

## 额外修复

清理后做验证时，发现 `models/UniversalCSI.py` 原本从 `.TransNet` 导入：

```python
TransformerEncoderLayer
TransformerEncoder
TransformerDecoderLayer
TransformerDecoder
```

但当前 `models/TransNet.py` 并没有导出这些名字，会导致 `main.py --help` 都无法运行。

因此同步修正为从 PyTorch 官方模块导入：

```python
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
```

同时删除了不再使用的：

```python
import torch.nn.functional as F
```

## 验证结果

执行过以下检查：

```bash
python -m py_compile models/UniversalCSI.py utils/parser.py utils/init.py
```

通过。

实例化模型：

```python
from models.UniversalCSI import universal_csi

model = universal_csi(
    encoder_name="transnet",
    reduction=4,
    d_model=64,
    channel=2,
    nt=32,
    nc=32,
    dim_feedforward=2048,
    code_adapter=False,
)

print(type(model.decoder).__name__)
print(hasattr(model.decoder, "output_activation"))
```

结果：

```text
TransNetDecoder
False
```

检查 CLI：

```bash
python main.py --help | rg "output_activation|encoder|code_adapter"
```

输出中不再出现 `output_activation`。

搜索残留：

```bash
rg -n "output_activation|HSigmoid|hsigmoid|return self\\.output_activation" . --glob '!exps/**'
```

无结果。

## 注意事项

`exps/` 下历史运行产物中仍可能存在：

```text
output_activation=none
output_activation=sigmoid
(output_activation): Identity()
```

这些是历史日志和 `args.json`，不应作为当前代码状态依据，也不应为了清理而修改实验产物。

