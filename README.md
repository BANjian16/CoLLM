# CoLLM Industrial RUL: A Practical Large-Small Collaboration Baseline

## 项目定位

这个仓库面向工业剩余寿命预测（RUL）场景，核心目标不是“单模型刷分”，而是让模型在精度、稳定性和推理开销之间取得可控平衡。

当前实现围绕 CoLLM 思路组织：

- Small Model：负责高吞吐、低延迟预测
- Large Model（One Fits All 风格）：处理更复杂样本
- Fuzzy Decision Agent：估计样本级置信度
- Self-Reflection：评估大模型输出可靠性

数据集默认使用 NASA CMAPSS FD001。

---

## 核心思想

对每个样本，系统先走小模型，再根据置信度决定是否调用大模型：

1. 小模型先给出预测 $y_s$ 和特征 $\phi_s$
2. 模糊决策器输出小模型置信度 $Q_s$
3. 若 $Q_s \ge \tau_1$，直接采用小模型输出
4. 若 $Q_s < \tau_1$，调用大模型得到 $y_l, \phi_l$
5. 自反思模块给出大模型置信度 $Q_l$
6. 用 $\Delta = Q_s - Q_l$ 与 $\tau_2$ 比较，决定“直接用大模型”还是“大小模型融合”

默认阈值固定为：

- tau1 = 0.7
- tau2 = -0.2

---

## 当前实现说明

### Large Model

- 使用 One Fits All 风格时序骨干（项目内封装）
- 输入为 patch 化时间序列后做线性投影
- 主干支持冻结训练，仅训练投影层/头部以提高稳定性

### 训练流程

训练采用三阶段：

1. Stage 1：训练 Small Model 回归能力
2. Stage 2：训练 Large Model（适配层和回归头）
3. Stage 3：固定 S/L，训练 Fuzzy + Reflection

这样做的目的是降低模块间梯度干扰，便于单独定位性能瓶颈。

---

## 快速开始

### 1) 环境

推荐使用 conda 环境：

```bash
conda activate collm_env
```

如需从头创建环境，可参考 environment.yml。

### 2) 训练

```bash
python train/train_all.py
```

### 3) 推理与评估

```bash
python main.py
python eval_test.py
```

如果要显式指定阈值：

```bash
python eval_test.py --tau1 0.7 --tau2 -0.2
```

---

## 当前结果（FD001）

以下结果来自当前默认路由阈值（tau1=0.7, tau2=-0.2）：

```yaml
RMSE Small : 17.820
RMSE Large : 16.676
RMSE CoLLM : 16.103
```

可见在当前权重下，协作路由优于单独小模型与大模型。

---

## 目录结构

```text
CoLLM/
├── data/
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
├── eval_test.py
├── main.py
├── environment.yml
└── README.md
```

---

## 说明与边界

- 本仓库是工程化复现与改造版本，不等价于论文作者原始训练流水线。
- 若要严格对齐论文数字，需同时对齐模型版本、数据划分、训练预算与阈值策略。
- 建议将本项目作为可复现基线，再逐项替换模块做对照实验。




