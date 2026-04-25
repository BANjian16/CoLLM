import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from models.collm import CoLLM
from models.small import SmallModel
from models.gpt2_ts import GPT2TimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection
from datasets.cmapss_test import CMAPSSTestDataset
from config import TAU1, TAU2


# ============================================================
# Config
# ============================================================

# 本脚本在官方测试集 FD001 上评估 CoLLM。
# 与 main.py 偏可视化训练集行为不同，这里更接近论文中的正式测试流程。
DEVICE = 'cpu'
DATA_ROOT = 'data/CMAPSS'
BATCH_SIZE = 64
DPI = 600
SAVE_DIR = './results_test'
N_SHOW = 300

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Load models
# ============================================================

# 恢复协同框架中的四个核心模块。
# 注意：推理阶段只做前向传播，不再更新任何参数。
S = SmallModel().to(DEVICE)
S.load_state_dict(torch.load('./train/small.pt', map_location=DEVICE))
S.eval()

L = GPT2TimeSeries().to(DEVICE)
L.load_state_dict(torch.load('./train/large.pt', map_location=DEVICE))
L.eval()

Fz = FuzzyDecisionAgent(32, 50).to(DEVICE)
Fz.load_state_dict(torch.load('./train/fuzzy.pt', map_location=DEVICE))
Fz.eval()

Rf = SelfReflection(768, 13).to(DEVICE)
Rf.load_state_dict(torch.load('./train/reflect.pt', map_location=DEVICE))
Rf.eval()

model = CoLLM(S, L, Fz, Rf)


# ============================================================
# Load TEST dataset
# ============================================================

# 测试集中每台发动机只贡献一个最终窗口，
# 标签来自官方提供的真实剩余寿命 RUL。
dataset = CMAPSSTestDataset(
    f'{DATA_ROOT}/test_FD001.txt',
    f'{DATA_ROOT}/RUL_FD001.txt',
    stats_path='./train/scaler_stats.npz'
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# Evaluation
# ============================================================

ys, yl, yc, ytrue = [], [], [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)

        # 单独的小模型预测，作为对照基线之一。
        y_s, _ = S(x)

        # 单独的大模型预测，作为另一条对照基线。
        y_l, _ = L(x)

        # 协同预测。
        # 此处直接使用 CoLLM.inference 的默认阈值 tau1=0.6、tau2=0.05，
        # 与论文中常见的一组阈值配置一致。
        y_c = model.inference(x, TAU1, TAU2)

        ys.append(y_s.numpy())
        yl.append(y_l.numpy())
        yc.append(y_c.numpy())
        ytrue.append(y.numpy())

ys = np.concatenate(ys)
yl = np.concatenate(yl)
yc = np.concatenate(yc)
ytrue = np.concatenate(ytrue)


def rmse(p, y):
    # 测试集同样使用 RMSE 评价预测误差。
    return np.sqrt(np.mean((p - y) ** 2))


print(f'RMSE Small : {rmse(ys, ytrue):.3f}')
print(f'RMSE Large : {rmse(yl, ytrue):.3f}')
print(f'RMSE CoLLM : {rmse(yc, ytrue):.3f}')


# ============================================================
# Plot 1: RUL Prediction Comparison (Test)
# ============================================================

# 在留出测试样本上比较三条预测路径与真实标签的关系。
plt.figure(figsize=(10, 4))
plt.plot(ytrue[:N_SHOW], label='Ground Truth', linewidth=2)
plt.plot(ys[:N_SHOW], '--', label='Small Model')
plt.plot(yl[:N_SHOW], ':', label='Large Model')
plt.plot(yc[:N_SHOW], label='CoLLM', linewidth=2)

plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.title('RUL Prediction on Test Set (FD001)')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/test_rul_comparison.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/test_rul_comparison.pdf', dpi=DPI)
plt.close()


# ============================================================
# Plot 2: Error Distribution (Test)
# ============================================================

# 误差直方图用于观察协同推理是否减小了测试集上的误差离散程度。
err_s = ys - ytrue
err_l = yl - ytrue
err_c = yc - ytrue

plt.figure(figsize=(6, 4))
plt.hist(err_s, bins=50, alpha=0.5, label='Small')
plt.hist(err_l, bins=50, alpha=0.5, label='Large')
plt.hist(err_c, bins=50, alpha=0.7, label='CoLLM')

plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution on Test Set (FD001)')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/test_error_distribution.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/test_error_distribution.pdf', dpi=DPI)
plt.close()


# ============================================================
# Plot 3: Error vs RUL (Test)
# ============================================================

# 用“真实 RUL - 预测误差”散点图观察：
# 不同寿命阶段是否会让某个分支更容易出错。
plt.figure(figsize=(6, 4))
plt.scatter(ytrue, err_s, s=5, alpha=0.3, label='Small')
plt.scatter(ytrue, err_l, s=5, alpha=0.3, label='Large')
plt.scatter(ytrue, err_c, s=5, alpha=0.4, label='CoLLM')
plt.axhline(0, linestyle='--')

plt.xlabel('Ground Truth RUL')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs RUL (Test Set)')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/test_error_vs_rul.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/test_error_vs_rul.pdf', dpi=DPI)
plt.close()
