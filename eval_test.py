import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import argparse

from models.collm import CoLLM
from models.small import SmallModel
from models.one_fits_all_ts import OneFitsAllTimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection
from datasets.cmapss_test import CMAPSSTestDataset


# ============================================================
# Config
# ============================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = 'data/CMAPSS'
BATCH_SIZE = 64
WINDOW_SIZE = 50
PATCH_SIZE = 4
DPI = 600
SAVE_DIR = './results_test'
N_SHOW = 300

os.makedirs(SAVE_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CoLLM on CMAPSS FD001 test set.')
    parser.add_argument('--tau1', type=float, default=0.7, help='SM confidence threshold.')
    parser.add_argument('--tau2', type=float, default=-0.2, help='Confidence gap threshold.')
    return parser.parse_args()


ARGS = parse_args()


def load_ckpt_or_raise(model, ckpt_path, device, tag):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Missing checkpoint: {ckpt_path}. '\
            f'Please run train/train_all.py to generate {tag} weights first.'
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))


# ============================================================
# Load models
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
# Load TEST dataset
# ============================================================

dataset = CMAPSSTestDataset(
    f'{DATA_ROOT}/test_FD001.txt',
    f'{DATA_ROOT}/RUL_FD001.txt',
    window_size=WINDOW_SIZE,
    train_path=f'{DATA_ROOT}/train_FD001.txt'
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE == 'cuda')
)


# ============================================================
# Evaluation
# ============================================================

ys, yl, yc, ytrue = [], [], [], []

with torch.no_grad():
    for x, y in loader:
        # x shape: [B, T, d], y shape: [B]
        x = x.to(DEVICE, non_blocking=(DEVICE == 'cuda'))

        # Small
        # y_s shape: [B]
        y_s, _ = S(x)

        # Large
        # y_l shape: [B]
        y_l, _ = L(x)

        # CoLLM
        # [Paper Eq. (3),(6),(7)] 协作路由输出, y_c shape: [B]
        y_c = model.inference(x, tau1=ARGS.tau1, tau2=ARGS.tau2)

        ys.append(y_s.detach().cpu().numpy())
        yl.append(y_l.detach().cpu().numpy())
        yc.append(y_c.detach().cpu().numpy())
        ytrue.append(y.detach().cpu().numpy())

ys = np.concatenate(ys)
yl = np.concatenate(yl)
yc = np.concatenate(yc)
ytrue = np.concatenate(ytrue)


def rmse(p, y):
    # RMSE = sqrt(mean((p-y)^2))
    return np.sqrt(np.mean((p - y) ** 2))


print(f'RMSE Small : {rmse(ys, ytrue):.3f}')
print(f'RMSE Large : {rmse(yl, ytrue):.3f}')
print(f'RMSE CoLLM : {rmse(yc, ytrue):.3f}')
print(f'Thresholds  : tau1={ARGS.tau1:.3f}, tau2={ARGS.tau2:.3f}')


# ============================================================
# Plot 1: RUL Prediction Comparison (Test)
# ============================================================

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
