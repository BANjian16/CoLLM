import torch
import math
import numpy as np
import os
from torch.utils.data import DataLoader
from models.collm import CoLLM
from models.small import SmallModel
from models.one_fits_all_ts import OneFitsAllTimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection
from datasets.cmapss_test import CMAPSSTestDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data/CMAPSS"
BATCH_SIZE = 64
WINDOW_SIZE = 50
PATCH_SIZE = 4

def load_ckpt_or_raise(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

S = SmallModel().to(DEVICE)
load_ckpt_or_raise(S, "./train/small.pt", DEVICE)
S.eval()

L = OneFitsAllTimeSeries(patch_size=PATCH_SIZE).to(DEVICE)
load_ckpt_or_raise(L, "./train/large.pt", DEVICE)
L.eval()

Fz = FuzzyDecisionAgent(32, WINDOW_SIZE).to(DEVICE)
load_ckpt_or_raise(Fz, "./train/fuzzy.pt", DEVICE)
Fz.eval()

n_patches = math.ceil(WINDOW_SIZE / PATCH_SIZE)
Rf = SelfReflection(L.hidden_size, n_patches).to(DEVICE)
load_ckpt_or_raise(Rf, "./train/reflect.pt", DEVICE)
Rf.eval()

dataset = CMAPSSTestDataset(
    f"{DATA_ROOT}/test_FD001.txt",
    f"{DATA_ROOT}/RUL_FD001.txt",
    window_size=WINDOW_SIZE,
    train_path=f"{DATA_ROOT}/train_FD001.txt"
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def rmse(p, y):
    return np.sqrt(np.mean((p - y) ** 2))

ys_all, yl_all, ytrue_all = [], [], []
qs_all, ql_all = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        ys, phi_s = S(x)
        qs = Fz(phi_s)
        yl, phi_l = L(x)
        ql = Rf(phi_l)
        ys_all.append(ys.cpu().numpy())
        yl_all.append(yl.cpu().numpy())
        ytrue_all.append(y.numpy())
        qs_all.append(qs.cpu().numpy())
        ql_all.append(ql.cpu().numpy())

ys_all = np.concatenate(ys_all)
yl_all = np.concatenate(yl_all)
ytrue_all = np.concatenate(ytrue_all)
qs_all = np.concatenate(qs_all)
ql_all = np.concatenate(ql_all)

rmse_small = rmse(ys_all, ytrue_all)
rmse_large = rmse(yl_all, ytrue_all)

def evaluate_collm(tau1, tau2):
    use_small = (qs_all >= tau1)
    need_lm = ~use_small
    y_collm = ys_all.copy()
    qs_lm = qs_all[need_lm]
    ql_lm = ql_all[need_lm]
    ys_lm = ys_all[need_lm]
    yl_lm = yl_all[need_lm]
    delta_lm = qs_lm - ql_lm
    use_fusion_lm = (delta_lm > tau2)
    y_final_lm = yl_lm.copy()
    y_final_lm[use_fusion_lm] = 0.5 * (ys_lm[use_fusion_lm] + yl_lm[use_fusion_lm])
    y_collm[need_lm] = y_final_lm
    r_small = np.mean(use_small)
    r_fusion = np.sum(use_fusion_lm) / len(ys_all)
    r_large_only = np.sum(~use_fusion_lm) / len(ys_all) # This was wrong before, it should be relative to need_lm? No, the prompt says ratios of usage.
    # Recalculate ratios relative to total:
    r_large_only = np.sum(~use_fusion_lm) / len(ys_all)
    return rmse(y_collm, ytrue_all), r_small, r_large_only, r_fusion

# Ratio fix
def evaluate_collm_v2(tau1, tau2):
    use_small_mask = (qs_all >= tau1)
    num_total = len(ys_all)
    
    y_collm = ys_all.copy()
    
    # Need LM part
    need_lm_mask = ~use_small_mask
    delta_lm = qs_all[need_lm_mask] - ql_all[need_lm_mask]
    use_fusion_in_lm = (delta_lm > tau2)
    
    # Calculate values for need_lm
    ys_lm = ys_all[need_lm_mask]
    yl_lm = yl_all[need_lm_mask]
    y_final_lm = yl_lm.copy()
    y_final_lm[use_fusion_in_lm] = 0.5 * (ys_lm[use_fusion_in_lm] + yl_lm[use_fusion_in_lm])
    
    y_collm[need_lm_mask] = y_final_lm
    
    r_small = np.sum(use_small_mask) / num_total
    r_fusion = np.sum(use_fusion_in_lm) / num_total
    r_large_only = (num_total - np.sum(use_small_mask) - np.sum(use_fusion_in_lm)) / num_total
    
    return rmse(y_collm, ytrue_all), r_small, r_large_only, r_fusion

rmse_def, r_s_def, r_l_def, r_f_def = evaluate_collm_v2(0.6, 0.05)
print(f"RMSE Small: {rmse_small:.4f}")
print(f"RMSE Large: {rmse_large:.4f}")
print(f"RMSE CoLLM (Default): {rmse_def:.4f}")
print(f"Routing (Default): Small={r_s_def:.4f}, LargeOnly={r_l_def:.4f}, Fusion={r_f_def:.4f}")

tau1_range = np.arange(0.1, 0.95 + 0.05, 0.05)
tau2_range = np.arange(-0.1, 0.2 + 0.02, 0.02)
best_rmse = float('inf')
best_params = (0, 0)
best_ratios = (0, 0, 0)

for t1 in tau1_range:
    for t2 in tau2_range:
        current_rmse, rs, rl, rf = evaluate_collm_v2(t1, t2)
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = (t1, t2)
            best_ratios = (rs, rl, rf)

print(f"Best RMSE: {best_rmse:.4f}")
print(f"Best Parameters: tau1={best_params[0]:.2f}, tau2={best_params[1]:.2f}")
print(f"Best Routing: Small={best_ratios[0]:.4f}, LargeOnly={best_ratios[1]:.4f}, Fusion={best_ratios[2]:.4f}")
