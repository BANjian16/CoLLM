import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.collm import CoLLM
from models.small import SmallModel
from models.one_fits_all_ts import OneFitsAllTimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection
from datasets.cmapss import CMAPSSDataset


# ============================================================
# Config
# ============================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = './data/CMAPSS/train_FD001.txt'
BATCH_SIZE = 64
WINDOW_SIZE = 50
PATCH_SIZE = 4

TAU1 = 0.7
TAU2 = -0.2

SAVE_DIR = './results'
DPI = 600

os.makedirs(SAVE_DIR, exist_ok=True)


def load_ckpt_or_raise(model, ckpt_path, device, tag):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Missing checkpoint: {ckpt_path}. '\
            f'Please run train/train_all.py to generate {tag} weights first.'
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))


# ============================================================
# Load Models
# ============================================================

S = SmallModel().to(DEVICE)
load_ckpt_or_raise(S, './train/small.pt', DEVICE, 'small')
S.eval()

L = OneFitsAllTimeSeries(patch_size=PATCH_SIZE).to(DEVICE)
load_ckpt_or_raise(L, './train/large.pt', DEVICE, 'large')
L.eval()

Fz = FuzzyDecisionAgent(32, WINDOW_SIZE).to(DEVICE)
load_ckpt_or_raise(Fz, './train/fuzzy.pt', DEVICE, 'fuzzy')
Fz.eval()

n_patches = math.ceil(WINDOW_SIZE / PATCH_SIZE)
Rf = SelfReflection(L.hidden_size, n_patches).to(DEVICE)
load_ckpt_or_raise(Rf, './train/reflect.pt', DEVICE, 'reflection')
Rf.eval()

model = CoLLM(S, L, Fz, Rf)


# ============================================================
# Load Dataset
# ============================================================

dataset = CMAPSSDataset(DATA_PATH, window_size=WINDOW_SIZE)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE == 'cuda')
)


# ============================================================
# Inference
# ============================================================

ys_list, yl_list, yc_list, ytrue_list = [], [], [], []
Qs_list, Ql_list = [], []

with torch.no_grad():
    for x, y in loader:
        # x shape: [B, T, d], y shape: [B]
        x = x.to(DEVICE, non_blocking=(DEVICE == 'cuda'))

        # ys shape: [B], phi_s shape: [B, T, ds]
        ys, phi_s = S(x)
        # yl shape: [B], phi_l shape: [B, K, dl]
        yl, phi_l = L(x)
        # [Paper Eq. (3),(6),(7)] 样本级路由后的协作输出
        # yc shape: [B]
        yc, details = model.inference(x, TAU1, TAU2, return_details=True)
        # Qs/Ql shape: [B]
        Qs = details['Qs']
        Ql = details['Ql']

        ys_list.append(ys.detach().cpu().numpy())
        yl_list.append(yl.detach().cpu().numpy())
        yc_list.append(yc.detach().cpu().numpy())
        ytrue_list.append(y.detach().cpu().numpy())

        Qs_list.append(Qs.detach().cpu().numpy())
        Ql_list.append(Ql.detach().cpu().numpy())


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
    # RMSE = sqrt(mean((p-y)^2))
    return np.sqrt(np.mean((p - y) ** 2))

print(f'RMSE Small : {rmse(ys, ytrue):.3f}')
print(f'RMSE Large : {rmse(yl, ytrue):.3f}')
print(f'RMSE CoLLM : {rmse(yc, ytrue):.3f}')


# ============================================================
# Plot 1: RUL Prediction Comparison
# ============================================================

N_SHOW = 300

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

# 仅保留实际触发大模型的样本，避免 Ql 占位值（未推理样本）造成统计误导。
need_lm = Qs < TAU1
Qs_lm = Qs[need_lm]
Ql_lm = Ql[need_lm]

plt.figure(figsize=(6, 5))
plt.scatter(Qs_lm, Ql_lm, s=5, alpha=0.5)
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
