# CoLLM 复现项目

本项目复现论文 **CoLLM: Industrial Large-Small Model Collaboration With Fuzzy Decision-Making Agent and Self-Reflection** 的主要流程，用于 NASA CMAPSS 航空发动机剩余寿命预测，也就是 RUL 预测。

当前代码主要支持 **FD001** 和 **FD003** 两个子数据集。整体思路是：先用小模型快速预测，如果小模型比较自信，就直接采用小模型结果；如果小模型不确定，再调用 GPT-2 based One Fits All 大模型，并通过模糊决策和自反思模块决定最终使用小模型、大模型，还是两者融合的结果。

## 项目特点

- 支持 CMAPSS FD001 / FD003 数据读取、滑动窗口采样、标准化和 RUL 截断。
- 小模型 `SmallModel` 用于低成本 RUL 预测。
- 大模型 `OneFitsAllTimeSeries` 采用 GPT-2 based One Fits All 思路：时间序列 patch embedding + GPT-2 预训练骨干 + 冻结大部分参数。
- 模糊决策模块 `FuzzyDecisionAgent` 用于估计小模型置信度。
- 自反思模块 `SelfReflection` 用于估计大模型置信度。
- `CoLLM` 模块实现论文中的大模型-小模型协作路由逻辑。
- 提供完整训练脚本、测试脚本、结果图保存和当前训练好的权重。

## 当前复现结果

下面结果使用论文风格的阈值预设 **C**。

| 数据集 | Small RMSE | Large RMSE | CoLLM-C RMSE | CoLLM-C MAE | 论文 CoLLM-C RMSE | 论文 CoLLM-C MAE |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 16.617 | 16.276 | 14.656 | 10.482 | 12.33 | 8.86 |
| FD003 | 16.428 | 16.844 | 15.567 | 11.254 | 11.11 | 7.12 |

目前已经跑通完整流程，并且 CoLLM 结果优于单独的小模型和大模型。不过结果仍然弱于论文，主要原因是论文没有完全公开 One Fits All 的训练细节、预训练权重选择、置信度校准细节等。后续调参应尽量保持论文算法路线，不建议为了指标随意改掉协作机制。

## 环境配置

使用 `environment.yml` 创建环境：

```bash
conda env create -f environment.yml
conda activate 环境名
```

如果 PowerShell 识别不了 `conda`，可以先初始化：

```powershell
conda init powershell
```

然后关闭 PowerShell 重新打开。

检查 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 输出 `False`，说明当前环境没有正确使用 GPU，训练速度会明显变慢。

## 数据集

CMAPSS 数据放在：

```text
data/
  CMAPSS/
    train_FD001.txt
    test_FD001.txt
    RUL_FD001.txt
    train_FD003.txt
    test_FD003.txt
    RUL_FD003.txt
```

项目目录里也包含 FD002 / FD004 原始文件，但当前主要复现和评估的是 FD001 / FD003。

默认数据处理方式：

| 项目 | 默认值 |
|---|---:|
| 输入窗口长度 | 50 |
| 滑动步长 | 1 |
| RUL 最大截断 | 125 |
| 使用传感器通道数 | 14 |
| 测试集标准化 | 使用训练阶段保存的 `scaler_stats.npz` |

## 项目结构

```text
CoLLM/
  data/CMAPSS/                  # CMAPSS 原始数据
  datasets/
    cmapss.py                   # 训练集读取、RUL 构造、滑动窗口
    cmapss_test.py              # 官方测试集构造
  models/
    small.py                    # 小模型
    one_fits_all_ts.py          # GPT-2 based One Fits All 大模型
    fuzzy.py                    # 模糊决策代理
    reflection.py               # 自反思模块
    collm.py                    # CoLLM 推理路由逻辑
  train/
    train_all.py                # 训练入口
    small.pt                    # FD001 小模型权重
    large.pt                    # FD001 大模型权重
    fuzzy.pt                    # FD001 模糊决策权重
    reflect.pt                  # FD001 自反思权重
    scaler_stats.npz            # FD001 标准化参数
  train_fd003/
    small.pt                    # FD003 小模型权重
    large.pt                    # FD003 大模型权重
    fuzzy.pt                    # FD003 模糊决策权重
    reflect.pt                  # FD003 自反思权重
    scaler_stats.npz            # FD003 标准化参数
  results_test/                 # FD001 / FD003 测试结果图
  config.py                     # 论文阈值预设
  eval_test.py                  # 测试和画图入口
  main.py                       # 简化统一入口
```

## 快速使用

推荐从 `main.py` 开始。它只是一个简化入口，内部会转发到真正的训练脚本或测试脚本。

训练 FD001：

```bash
python main.py train --subset FD001 --save-dir train --threshold-preset C --stages all
```

测试 FD001：

```bash
python main.py eval --subset FD001 --model-dir train --save-dir results_test --threshold-preset C
```

训练 FD003：

```bash
python main.py train --subset FD003 --save-dir train_fd003 --threshold-preset C --stages all
```

测试 FD003：

```bash
python main.py eval --subset FD003 --model-dir train_fd003 --save-dir results_test --threshold-preset C
```

查看帮助：

```bash
python main.py --help
python main.py train --help
python main.py eval --help
```

## 训练说明

真正的训练脚本是：

```text
train/train_all.py
```

训练分成三个阶段：

1. `small`：训练小模型 `SmallModel`。
2. `large`：训练 GPT-2 based One Fits All 大模型 `OneFitsAllTimeSeries`。
3. `confidence`：冻结小模型和大模型，训练 `FuzzyDecisionAgent` 和 `SelfReflection`。

完整训练：

```bash
python train/train_all.py --subset FD001 --save-dir train --threshold-preset C --stages all
```

只训练某一阶段，例如只重训置信度模块：

```bash
python train/train_all.py --subset FD001 --save-dir train --threshold-preset C --stages confidence
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--subset` | `FD001` | 选择 `FD001` 或 `FD003` |
| `--threshold-preset` | `C` | 论文风格阈值预设：`A`、`B`、`C` |
| `--window-size` | `50` | 输入时间窗口长度 |
| `--patch-size` | `4` | 大模型 patch 大小 |
| `--gpt2-name` | `gpt2` | HuggingFace GPT-2 名称或本地缓存名称 |
| `--gpt-layers` | `6` | 保留 GPT-2 前几层 |
| `--gpt2-allow-download` | 默认关闭 | 允许自动下载 GPT-2 权重 |
| `--batch-size` | `64` | 批大小 |
| `--epochs-small` | `30` | 小模型训练轮数 |
| `--epochs-large` | `120` | 大模型训练轮数 |
| `--epochs-conf` | `40` | 置信度模块训练轮数 |
| `--split-mode` | `random` | 窗口随机切分或按发动机切分：`random` / `unit` |
| `--norm-scope` | `train` | 标准化范围，严格实验建议使用 `train` |

## 测试说明

真正的测试脚本是：

```text
eval_test.py
```

它会输出：

- Small / Large / CoLLM 的 RMSE
- Small / Large / CoLLM 的 MAE
- 样本最终走小模型、大模型、融合路径的比例
- RUL 预测曲线图
- 误差分布图
- 误差与真实 RUL 关系图

测试 FD001：

```bash
python eval_test.py --subset FD001 --model-dir train --save-dir results_test --threshold-preset C
```

测试 FD003：

```bash
python eval_test.py --subset FD003 --model-dir train_fd003 --save-dir results_test --threshold-preset C
```

输出文件示例：

```text
results_test/FD001_test_rul_comparison.png
results_test/FD001_test_error_distribution.png
results_test/FD001_test_error_vs_rul.png
```

## 阈值说明

论文中的 CoLLM 路由依赖两个阈值：

- `tau1`：判断小模型是否足够自信。
- `tau2`：判断大模型是否比小模型更值得相信。

阈值写在 `config.py` 中：

```python
PAPER_THRESHOLDS = {
    "FD001": {
        "A": (0.3, 0.1),
        "B": (0.4, 0.1),
        "C": (0.6, 0.05),
    },
    "FD003": {
        "A": (0.15, 0.1),
        "B": (0.4, 0.1),
        "C": (0.6, 0.05),
    },
}
```

严格复现时建议使用：

```bash
--threshold-preset A
--threshold-preset B
--threshold-preset C
```

虽然也可以手动指定：

```bash
python main.py eval --subset FD001 --model-dir train --tau1 0.6 --tau2 0.05
```


## 代码阅读建议

建议按这个顺序看：

1. `datasets/cmapss.py`：先理解数据怎么变成 `(X, y)`。
2. `models/small.py`：理解小模型如何从传感器序列预测 RUL。
3. `models/one_fits_all_ts.py`：理解 GPT-2 怎么被改造成时间序列模型。
4. `models/fuzzy.py`：理解小模型置信度 `q_s` 怎么来。
5. `models/reflection.py`：理解大模型置信度 `q_l` 怎么来。
6. `models/collm.py`：理解最终路由逻辑。
7. `train/train_all.py`：理解完整训练流程。
8. `eval_test.py`：理解测试指标和结果图。

## 当前保留文件说明

- `train/`：当前 FD001 权重。
- `train_fd003/`：当前 FD003 权重。
- `results_test/`：当前 FD001 / FD003 测试图。
- 旧实验图、临时测试脚本和中间文本已经清理掉。

## 后续可以改进的方向

- 继续细化 GPT-2 based One Fits All 的训练策略。
- 改善小模型和大模型的置信度校准。
- 多随机种子重复实验，报告均值和标准差。
- 增加推理耗时、调用大模型比例等分析，更完整地体现“大模型-小模型协作”的意义。
- 在不改变论文核心算法的前提下，继续缩小与论文结果的差距。

