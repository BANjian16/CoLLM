# CoLLM 复现项目

本项目复现论文 **CoLLM: Industrial Large-Small Model Collaboration With Fuzzy Decision-Making Agent and Self-Reflection** 中的工业剩余寿命预测框架。当前版本重点支持 NASA CMAPSS 数据集的 **FD001** 和 **FD003**，并尽量严格保留论文中的大小模型协作、模糊决策和自反思思路。

> 说明：论文已经单独汇报过，本仓库 README 主要面向代码运行、实验复现和后续继续调试。

## 项目目标

CoLLM 的核心思想是：先用小模型进行低成本预测，再根据置信度和模糊阈值判断是否交给大模型或融合分支处理，从而在精度和计算成本之间取得平衡。

本复现项目已经完成：

- CMAPSS FD001 / FD003 数据读取、滑窗、归一化和 RUL 截断。
- SmallModel、One Fits All 风格 LargeModel、FuzzyDecisionAgent、SelfReflection 与 CoLLM 路由。
- FD001 / FD003 的训练、测试和结果图输出。
- 论文阈值预设 A/B/C 的统一配置。
- 项目汇报 PPT：`ppt_work/output/CoLLM_复现项目汇报.pptx`。

## 当前结果

以下结果使用严格论文阈值预设 **C**，即默认 `--threshold-preset C`。

| 数据集 | Small RMSE | Large RMSE | CoLLM-C RMSE | CoLLM-C MAE | 论文 CoLLM-C RMSE | 论文 CoLLM-C MAE |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 15.093 | 14.111 | 14.082 | 10.580 | 12.33 | 8.86 |
| FD003 | 16.903 | 15.045 | 15.018 | 10.558 | 11.11 | 7.12 |

当前复现已经跑通完整流程，但与论文结果仍有差距。主要原因是本项目的大模型为可复现的 One Fits All 风格实现，还没有完全等价于论文中的完整预训练资源和训练细节；后续优化应继续围绕大模型能力、confidence 校准和多 seed 稳定性展开，而不是偏离论文算法路线做结果搜索。

## 环境配置

推荐使用 conda 环境 `collm_env`：

```powershell
conda env create -n collm_env -f environment.yml
conda activate collm_env
```

如果本机是 RTX 50 系列或更新 GPU，`environment.yml` 中较旧的 PyTorch CUDA 版本可能无法正常使用 GPU。可以在激活环境后安装适配当前显卡的 PyTorch 版本，例如本机调试时使用的是支持 CUDA 12.8 的 PyTorch：

```powershell
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

检查 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 数据准备

数据目录应保持如下结构：

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

项目中也保留了 FD002 / FD004 原始文件，但当前训练与评估脚本主要支持 FD001 和 FD003。

数据处理默认设置：

- 滑窗长度：`window_size=50`
- 滑窗步长：`stride=1`
- RUL 最大截断：`125`
- 传感器数量：14 个论文相关传感器
- 测试集归一化：优先使用模型目录下的 `scaler_stats.npz`

## 项目结构

```text
CoLLM/
  data/CMAPSS/                  # CMAPSS 原始数据
  datasets/
    cmapss.py                   # 训练集预处理与滑窗
    cmapss_test.py              # 官方测试集构造
  models/
    small.py                    # 小模型
    one_fits_all_ts.py          # One Fits All 风格大模型
    fuzzy.py                    # 模糊决策主体
    reflection.py               # 自反思模块
    collm.py                    # CoLLM 推理路由
  train/
    train_all.py                # 统一训练入口
    train_conf.py               # 早期 confidence 训练脚本
    optimize_thresholds.py      # 阈值搜索辅助脚本，不作为默认论文复现实验
    small.pt / large.pt / fuzzy.pt / reflect.pt
    scaler_stats.npz            # FD001 当前权重与归一化统计
  train_fd003/
    small.pt / large.pt / fuzzy.pt / reflect.pt
    scaler_stats.npz            # FD003 当前权重与归一化统计
  results/                      # 训练或早期实验图
  results_test/                 # FD001 / FD003 官方测试图
  ppt_work/
    output/CoLLM_复现项目汇报.pptx
    src/build_deck.mjs
  config.py                     # 论文阈值 A/B/C
  eval_test.py                  # 官方测试集评估入口
  main.py                       # 原始入口脚本
  CLEANUP_PLAN.md               # 项目清理记录
```

## 训练

统一训练脚本为 `train/train_all.py`。训练分为三个阶段：

1. `small`：训练 SmallModel。
2. `large`：训练 LargeModel。
3. `confidence`：冻结 small / large，训练 FuzzyDecisionAgent 和 SelfReflection。

### FD001 训练

```powershell
python train/train_all.py `
  --subset FD001 `
  --save-dir train `
  --threshold-preset C `
  --stages all
```

### FD003 训练

```powershell
python train/train_all.py `
  --subset FD003 `
  --save-dir train_fd003 `
  --threshold-preset C `
  --stages all
```

### 只重训某个阶段

例如只重训 confidence：

```powershell
python train/train_all.py `
  --subset FD003 `
  --save-dir train_fd003 `
  --threshold-preset C `
  --stages confidence
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--subset` | `FD001` | 可选 `FD001` / `FD003` |
| `--threshold-preset` | `C` | 论文阈值预设 `A` / `B` / `C` |
| `--window-size` | `50` | 输入窗口长度 |
| `--patch-size` | `4` | LargeModel patch 大小 |
| `--epochs-small` | `30` | 小模型训练轮数 |
| `--epochs-large` | `120` | 大模型训练轮数 |
| `--epochs-conf` | `40` | confidence 分支训练轮数 |
| `--split-mode` | `random` | 训练/验证划分方式，可选 `random` / `unit` |
| `--norm-scope` | `train` | 归一化统计来源，可选 `train` / `combined` |

## 测试与出图

评估入口为 `eval_test.py`。脚本会输出 Small、Large、CoLLM 的 RMSE / MAE 和路由比例，并保存三类图：

- RUL 预测对比图
- 误差分布图
- 误差与真实 RUL 关系图

### FD001 测试

```powershell
python eval_test.py `
  --subset FD001 `
  --model-dir train `
  --save-dir results_test `
  --threshold-preset C
```

输出文件示例：

```text
results_test/FD001_test_rul_comparison.png
results_test/FD001_test_error_distribution.png
results_test/FD001_test_error_vs_rul.png
```

### FD003 测试

```powershell
python eval_test.py `
  --subset FD003 `
  --model-dir train_fd003 `
  --save-dir results_test `
  --threshold-preset C
```

输出文件示例：

```text
results_test/FD003_test_rul_comparison.png
results_test/FD003_test_error_distribution.png
results_test/FD003_test_error_vs_rul.png
```

## 阈值配置

论文阈值写在 `config.py`：

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

默认使用 `C`。如果要临时覆盖阈值，可以在训练或测试时传入：

```powershell
python eval_test.py --subset FD001 --model-dir train --tau1 0.6 --tau2 0.05
```

注意：为了严格遵守论文思路，正式汇报结果建议使用 `--threshold-preset A/B/C`，不要用私自搜索出的阈值替代论文阈值。

## 汇报材料

已生成一份用于向老师汇报复现项目的 PPT：

```text
ppt_work/output/CoLLM_复现项目汇报.pptx
```

PPT 源脚本：

```text
ppt_work/src/build_deck.mjs
```

如需重新生成 PPT，需要恢复 `ppt_work/node_modules` 中对 `@oai/artifact-tool` 的链接，或使用 Codex/Artifact Tool 环境重新运行生成脚本。

## 后续改进方向

建议后续按以下顺序继续迭代：

1. 补齐更接近论文的 LargeModel / One Fits All / 预训练 backbone。
2. 保持论文 A/B/C 阈值思想，继续优化 confidence 训练与验证方式。
3. 固定 seed，多次训练，报告均值、方差和路由比例。
4. 补充推理耗时、分流比例和计算成本，体现大小模型协作价值。
5. 在不违背论文算法的前提下，继续缩小 FD001 / FD003 与论文指标的差距。

## 注意事项

- `train/` 和 `train_fd003/` 是当前正式权重目录，不要随意删除。
- `results_test/FD001_*` 和 `results_test/FD003_*` 是当前正式测试结果。
- `optimize_thresholds.py` 仅作为分析工具，不能替代严格论文阈值实验。
- 如果重新清理项目，可参考 `CLEANUP_PLAN.md`。
