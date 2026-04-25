import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.collm import CoLLM
from models.small import SmallModel
from models.gpt2_ts import GPT2TimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection
from datasets.cmapss import CMAPSSDataset
from config import TAU1, TAU2


# ============================================================
# Config
# ============================================================

# 本脚本用于在训练集 FD001 上复现论文中的协同推理流程，
# 并保存若干对比图，帮助观察：
# 1. 小模型 / 大模型 / CoLLM 的预测差异
# 2. 协同推理是否缩小了误差分布
# 3. Q_s 与 Q_l 的路由行为
DEVICE = 'cpu'
DATA_PATH = './data/CMAPSS/train_FD001.txt'
BATCH_SIZE = 64

# tau1 对应论文中的第一层阈值：
# 若 Q_s >= tau1，则样本在小模型处分流并提前退出。
#
# tau2 对应第二层阈值：
# 若 Delta = Q_s - Q_l <= tau2，则直接采用大模型输出；
# 否则触发“小模型辅助修正”，即与小模型结果做融合。
SAVE_DIR = './results'
DPI = 600

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Load Models
# ============================================================

# 加载论文中的小模型 S：
# 面向低不确定性、较简单样本的快速推理分支。
S = SmallModel().to(DEVICE)
S.load_state_dict(torch.load('./train/small.pt', map_location=DEVICE))
S.eval()

# 加载论文中的大模型 L：
# 面向高不确定性、复杂样本的深层推理分支。
L = GPT2TimeSeries().to(DEVICE)
L.load_state_dict(torch.load('./train/large.pt', map_location=DEVICE))
L.eval()

# 模糊决策代理 F：
# 根据小模型中间特征预测 Q_s，决定是否继续调用大模型。
Fz = FuzzyDecisionAgent(32, 50).to(DEVICE)
Fz.load_state_dict(torch.load('./train/fuzzy.pt', map_location=DEVICE))
Fz.eval()

# 自反思模块 R：
# 根据大模型中间特征预测 Q_l，用于判断大模型当前结果是否可靠。
Rf = SelfReflection(768, 13).to(DEVICE)
Rf.load_state_dict(torch.load('./train/reflect.pt', map_location=DEVICE))
Rf.eval()

# CoLLM 将四个子模块串联成论文中的完整协同推理管线。
model = CoLLM(S, L, Fz, Rf)


# ============================================================
# Load Dataset
# ============================================================

# 这里直接在训练集上做可视化分析，
# 主要目的是观察三条预测路径在逐样本层面的差异。
dataset = CMAPSSDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# Inference
# ============================================================

ys_list, yl_list, yc_list, ytrue_list = [], [], [], []
Qs_list, Ql_list = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)

        # 单独运行小模型与大模型，便于后续与协同结果做对比。
        ys, phi_s = S(x)
        yl, phi_l = L(x)
        # 协同预测主流程：
        # S -> Q_s -> 是否提前退出
        #          -> L -> Q_l -> 是否触发小模型辅助融合
        yc = model.inference(x, TAU1, TAU2)

        # 显式计算 Q_s 与 Q_l，后续可直接画出路由分布图。
        Qs = Fz(phi_s)
        Ql = Rf(phi_l)

        ys_list.append(ys.cpu().numpy())
        yl_list.append(yl.cpu().numpy())
        yc_list.append(yc.cpu().numpy())
        ytrue_list.append(y.numpy())

        Qs_list.append(Qs.cpu().numpy())
        Ql_list.append(Ql.cpu().numpy())


ys = np.concatenate(ys_list)
yl = np.concatenate(yl_list)
yc = np.concatenate(yc_list)
ytrue = np.concatenate(ytrue_list)
Qs = np.concatenate(Qs_list)
Ql = np.concatenate(Ql_list)


# ============================================================
# Metrics
# ============================================================

def rmse(p, y):
    # RMSE 是论文在 RUL 回归任务中使用的核心评价指标之一。
    return np.sqrt(np.mean((p - y) ** 2))

print(f'RMSE Small : {rmse(ys, ytrue):.3f}')
print(f'RMSE Large : {rmse(yl, ytrue):.3f}')
print(f'RMSE CoLLM : {rmse(yc, ytrue):.3f}')


# ============================================================
# Plot 1: RUL Prediction Comparison
# ============================================================

N_SHOW = 300

# 对前 N_SHOW 个样本绘图，
# 直观看 Ground Truth、小模型、大模型与最终 CoLLM 输出的差别。
plt.figure(figsize=(10, 4))
plt.plot(ytrue[:N_SHOW], label='Ground Truth', linewidth=2)
plt.plot(ys[:N_SHOW], '--', label='Small Model')
plt.plot(yl[:N_SHOW], ':', label='Large Model')
plt.plot(yc[:N_SHOW], label='CoLLM', linewidth=2)

plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.title('RUL Prediction Comparison (FD001)')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/rul_comparison.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/rul_comparison.pdf', dpi=DPI)
plt.close()


# ============================================================
# Plot 2: Error Distribution
# ============================================================

# 对比三种方法的残差分布。
# 若 CoLLM 的误差分布更集中、尾部更短，说明协同机制抑制大误差的能力更强。
err_s = ys - ytrue
err_l = yl - ytrue
err_c = yc - ytrue

plt.figure(figsize=(6, 4))
plt.hist(err_s, bins=50, alpha=0.5, label='Small')
plt.hist(err_l, bins=50, alpha=0.5, label='Large')
plt.hist(err_c, bins=50, alpha=0.7, label='CoLLM')

plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/error_distribution.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/error_distribution.pdf', dpi=DPI)
plt.close()


# ============================================================
# Plot 3: Confidence Routing Behavior
# ============================================================

# 绘制 Q_s 与 Q_l 的散点分布，观察论文中的置信度路由机制如何工作。
# 红色竖线 tau1 表示小模型提前退出阈值；
# 对角虚线仅作为参考线，用来粗略判断两种置信度谁更高。
plt.figure(figsize=(6, 5))
plt.scatter(Qs, Ql, s=5, alpha=0.5)
plt.axvline(TAU1, color='r', linestyle='--', label=r'$\tau_1$')
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')

plt.xlabel(r'$Q_s$ (Small Confidence)')
plt.ylabel(r'$Q_l$ (Large Confidence)')
plt.title('Confidence Routing Behavior')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/confidence_routing.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/confidence_routing.pdf', dpi=DPI)
plt.close()


print(f'\nAll figures saved to ./{SAVE_DIR}/ (PNG + PDF, dpi={DPI})')
