# CoLLM: 工业大小模型协作框架（复现）

> 论文：*CoLLM: Industrial Large–Small Model Collaboration With Fuzzy Decision-Making Agent and Self-Reflection*  
> IEEE TRANSACTIONS ON FUZZY SYSTEMS, VOL. 34, NO. 4, APRIL 2026

本项目为上述论文的工程化复现，面向工业剩余使用寿命（RUL）预测场景，核心目标是在预测精度与推理开销之间取得动态平衡。

---

## 1. 核心思想

工业样本的复杂程度差异显著：简单样本可由小模型快速处理，复杂样本则需要大模型的深度推理能力。CoLLM 提出一种**样本级（sample-level）**动态路由策略，而非传统的任务级（task-level）静态分配：

- 小模型（SM）先对全部样本进行快速推理；
- 模糊决策智能体（Fuzzy Agent）评估 SM 的置信度；
- 高置信度样本直接输出，低置信度样本激活大模型（LM）；
- 自反思机制（Self-Reflection）进一步评估 LM 输出的可靠性，必要时融合大小模型结果以降低风险。

该策略在保持精度不降甚至提升的同时，可将大模型计算开销降低 **1.26× ~ 14.54×**。

---

## 2. 方法框架

### 2.1 模型组成

| 模块 | 说明 | 论文对应 |
|------|------|---------|
| **Small Model (S)** | 轻量 Transformer 编码器，负责快速特征提取与初始预测 | Eq. (1) |
| **Fuzzy Decision Agent (F)** | 基于模糊神经网络（FNN）的决策智能体，输出 SM 置信度 $Q_s$ | Eq. (2)–(3), (8)–(11) |
| **Large Model (L)** | 基于预训练 GPT-2，冻结主干，仅训练 Patch Embedding 与回归头 | Eq. (4)–(5) |
| **Self-Reflection (R)** | 全连接网络，评估 LM 特征空间的预测偏差，输出 LM 置信度 $Q_l$ | Eq. (5)–(6), (12)–(14) |

### 2.2 推理流程（级联决策）

```
输入 x
  │
  ▼
┌─────────────────┐
│  Small Model S  │──→ 预测 ys, 特征 φs
└─────────────────┘
  │
  ▼
┌──────────────────────┐
│ Fuzzy Agent F(φs)    │──→ 置信度 Qs ∈ [0,1]
└──────────────────────┘
  │
  ├─ Qs ≥ τ1 ──→ 直接输出 ys  （SM 独占区）
  │
  └─ Qs < τ1 ──→ 激活 Large Model
                      │
                      ▼
              ┌─────────────────┐
              │ Large Model L   │──→ 预测 yl, 特征 φl
              └─────────────────┘
                      │
                      ▼
              ┌──────────────────────┐
              │ Self-Reflection R(φl) │──→ 置信度 Ql ∈ [0,1]
              └──────────────────────┘
                      │
                      ▼
              计算 Δ = Qs − Ql
                      │
          ├─ Δ ≤ τ2 ──→ 输出 yl       （LM 可靠）
          │
          └─ Δ > τ2 ──→ 输出 (ys + yl)/2  （融合修正）
```

### 2.3 模糊决策智能体（FNN）

- 采用**高斯隶属度函数**将 SM 提取的时序特征映射到模糊空间，均值 $\mu$ 与方差 $\sigma^2$ 均为可学习参数；
- 相比确定性神经网络，FNN 对工业数据中的噪声和边界模糊性更具鲁棒性；
- 置信度监督信号：$Q_s^* = 1 - \tanh(|y_s - y^*| / \alpha)$。

### 2.4 自反思机制

- 输入为 LM 最后一层隐状态 $\phi_l$；
- 通过单层全连接投影得到 LM 置信度 $Q_l$；
- 监督信号：$Q_l^* = 1 - \tanh(|y_l - y^*| / \alpha)$；
- 当 LM 置信度显著低于 SM（$\Delta > \tau_2$）时，触发大小模型融合，防止大模型过拟合导致的灾难性错误。

---

## 3. 训练策略

采用**三阶段渐进训练 + 分层参数冻结**，避免模块间梯度干扰：

| 阶段 | 训练模块 | 冻结模块 | 目的 |
|------|---------|---------|------|
| **Stage 1** | Small Model (S) | 其余全部 | 建立轻量时序特征表示 |
| **Stage 2** | Large Model (L) 的投影层与回归头 | SM 及 GPT-2 主干 | 适配预训练 LM 到工业时序数据 |
| **Stage 3** | Fuzzy Agent (F) + Self-Reflection (R) | SM、LM 全部冻结 | 学习置信度驱动的动态路由 |

论文使用 Adam 优化器，训练 **100 epochs** 并配合 Early Stopping，验证集占原始数据 20%。

---

## 4. 数据集与实验设置

- **数据集**：NASA CMAPSS（FD001 / FD003）
- **传感器筛选**：剔除恒值传感器（1, 5, 6, 10, 16, 18, 19），保留 **14 维有效传感器**
- **归一化**：$x_f = (x_f - \mu_f) / \sigma_f$，基于训练集全局统计量
- **滑窗处理**：窗口大小 `window_size = 50`，步长 `stride = 1`
- **RUL 标签**：分段线性，上限截断 `MAX_RUL = 125`
- **评价指标**：RMSE、MAE、FLOPs 加速比

### 论文主要结果

| 数据集 | 配置 | RMSE | MAE | FLOPs 降低 |
|--------|------|------|-----|-----------|
| FD001 | CoLLM-A [0.30, 0.10] | 12.45 | 9.13 | **3.88×** |
| FD001 | CoLLM-C [0.60, 0.05] | 12.33 | 8.86 | 1.26× |
| FD003 | CoLLM-A [0.15, 0.10] | 11.26 | 7.42 | **14.54×** |
| FD003 | CoLLM-C [0.60, 0.05] | 11.11 | 7.12 | 1.57× |

> 注：括号内为 $[\tau_1, \tau_2]$ 阈值配置。$\tau_1$ 越高，进入 LM 的样本越少，加速比越大。

---

## 5. 快速开始

### 环境

```bash
conda env create -f environment.yml
conda activate collm_env
```

### 训练

三阶段一键训练：

```bash
python train/train_all.py
```

默认配置：
- Small Model：`epochs=20`, `lr=1e-3`
- Large Model：`epochs=40`, `lr=2e-4`
- Fuzzy + Reflection：`epochs=20`, `lr=1e-3`

权重保存在 `train/` 目录下。

### 推理与可视化

```bash
python main.py        # 验证集推理 + 可视化
python eval_test.py   # 官方测试集（Last Cycle Window）评估
```

### 阈值网格搜索

搜索最优 $[\tau_1, \tau_2]$ 组合，输出 Pareto 前沿与最佳阈值：

```bash
python train/grid_search_tau.py
```

结果保存在 `results/tau_search/`。

---

## 6. 目录结构

```text
CoLLM/
├── data/CMAPSS/              # NASA CMAPSS 数据集
├── datasets/
│   ├── cmapss.py             # 滑窗数据集（训练/验证）
│   └── cmapss_test.py        # 官方测试集（Last Cycle Window）
├── models/
│   ├── small.py              # Small Model (轻量 Transformer)
│   ├── gpt2_ts.py            # Large Model (GPT-2 + Patch Embedding)
│   ├── fuzzy.py              # Fuzzy Decision Agent (FNN)
│   ├── reflection.py         # Self-Reflection (FCN)
│   └── collm.py              # 协作推理框架封装
├── train/
│   ├── train_all.py          # 三阶段训练脚本
│   ├── grid_search_tau.py    # τ1/τ2 网格搜索
│   ├── small.pt              # 预训练权重
│   ├── large.pt
│   ├── fuzzy.pt
│   └── reflect.pt
├── results/                  # 可视化输出
├── main.py                   # 验证集推理与绘图
├── eval_test.py              # 官方测试集评估
├── environment.yml
└── README.md
```

---

## 7. 说明与边界

- 本仓库为论文的工程化复现与改造版本，部分实现细节（如 LM 从 One Fits All 替换为 GPT-2、训练 epoch 数等）可能与原始训练流水线存在差异。
- 若要严格对齐论文数字，需同时复现：模型版本、数据划分策略、训练预算（100 epochs + early stopping）以及阈值搜索策略。
- 建议将本项目作为**可复现基线**，逐项替换模块进行消融实验。

---

## Citation

```bibtex
@article{wang2026collm,
  title={CoLLM: Industrial Large–Small Model Collaboration With Fuzzy Decision-Making Agent and Self-Reflection},
  author={Wang, Haiteng and Ren, Lei and Zhao, Tuo and Jiao, Lu},
  journal={IEEE Transactions on Fuzzy Systems},
  volume={34},
  number={4},
  pages={1120--1133},
  year={2026},
  publisher={IEEE}
}
```
