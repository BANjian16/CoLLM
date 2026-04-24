# CoLLM 复现项目（工业 RUL 预测）

本项目复现论文 CoLLM: Industrial Large-Small Model Collaboration With Fuzzy Decision-Making Agent and Self-Reflection 的核心思想，用于 NASA CMAPSS 数据集上的剩余寿命（RUL）预测。

论文原文（本地）：

- D:\CoLLM\CoLLM_Industrial_LargeSmall_Model_Collaboration_With_Fuzzy_Decision-Making_Agent_and_Self-Reflection.pdf

---

## 1. 论文方法（简述）

CoLLM 的核心是样本级动态路由：

1. 先由 Small Model 预测，得到 y_s 和特征 phi_s。
2. Fuzzy Decision Agent 输出小模型置信度 Q_s。
3. 若 Q_s >= tau1，直接采用小模型结果。
4. 若 Q_s < tau1，调用 Large Model 得到 y_l, phi_l。
5. Self-Reflection 输出大模型置信度 Q_l，计算 Delta = Q_s - Q_l。
6. 若 Delta <= tau2，采用 y_l；否则采用融合结果 0.5 * (y_s + y_l)。

论文强调三部分：

- 模糊决策（FNN）用于不确定性建模
- 自反思机制用于抑制大模型不可靠输出
- 三阶段训练与分层冻结，减少梯度干扰

---

## 2. 本仓库实现范围

本仓库是工程化复现，不是作者官方训练流水线逐行复刻。

当前实现包含：

- Small Model（轻量时序编码器）
- Large Model（One Fits All 风格封装）
- Fuzzy Decision Agent
- Self-Reflection
- CoLLM 样本级协作路由

默认路由阈值：

- tau1 = 0.7
- tau2 = -0.2

---

## 3. 与论文的主要差异

论文描述中 LM 架构与实验设置更广（含 GPT-2、One Fits All 等多种 LM 对比），本仓库聚焦在可运行的复现基线上，主要差异包括：

- 代码中的 LM 为 OneFitsAll 风格时序模型封装
- 训练与评估脚本为工程化版本（便于快速复现与调试）
- 结果会受到随机种子、数据划分、训练预算和阈值设置影响

因此，仓库结果与论文表格不保证数值一一对齐，但方法路径一致。

---

## 4. 数据与预处理

默认使用 CMAPSS FD001：

- 训练数据：data/CMAPSS/train_FD001.txt
- 测试数据：data/CMAPSS/test_FD001.txt
- 测试标签：data/CMAPSS/RUL_FD001.txt

实现中的关键预处理：

- 使用 14 个有效传感器特征
- 标准化（使用训练集统计量）
- 滑窗：window_size = 50, stride = 1
- RUL 截断上限：125

---

## 5. 快速开始

### 5.1 环境

```bash
conda env create -f environment.yml
conda activate collm_env
```

### 5.2 训练

```bash
python train/train_all.py
```

### 5.3 评估

```bash
python eval_test.py
```

也可以显式指定阈值：

```bash
python eval_test.py --tau1 0.7 --tau2 -0.2
```

---

## 6. 当前测试结果（FD001）

在当前默认阈值（tau1 = 0.7, tau2 = -0.2）下：

```yaml
RMSE Small : 17.820
RMSE Large : 16.676
RMSE CoLLM : 16.103
```

该结果说明协作路由在当前权重下优于单独小模型和单独大模型。

---

## 7. 项目结构

```text
CoLLM/
├── data/
│   └── CMAPSS/
├── datasets/
│   ├── cmapss.py
│   └── cmapss_test.py
├── models/
│   ├── small.py
│   ├── one_fits_all_ts.py
│   ├── fuzzy.py
│   ├── reflection.py
│   └── collm.py
├── train/
│   ├── train_all.py
│   ├── small.pt
│   ├── large.pt
│   ├── fuzzy.pt
│   └── reflect.pt
├── main.py
├── eval_test.py
├── environment.yml
└── README.md
```

---

## 8. 引用

```bibtex
@article{wang2026collm,
  title={CoLLM: Industrial Large-Small Model Collaboration With Fuzzy Decision-Making Agent and Self-Reflection},
  author={Wang, Haiteng and Ren, Lei and Zhao, Tuo and Jiao, Lu},
  journal={IEEE Transactions on Fuzzy Systems},
  volume={34},
  number={4},
  pages={1120--1133},
  year={2026}
}
```
