import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cmapss import CMAPSSDataset
from models.fuzzy import FuzzyDecisionAgent
from models.gpt2_ts import GPT2TimeSeries
from models.reflection import SelfReflection
from models.small import SmallModel


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search optimal CoLLM thresholds tau1 and tau2.")
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "CMAPSS" / "train_FD001.txt"))
    parser.add_argument("--model-dir", type=str, default=str(ROOT / "train"))
    parser.add_argument("--config-path", type=str, default=str(ROOT / "config.py"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau1-min", type=float, default=0.0)
    parser.add_argument("--tau1-max", type=float, default=1.0)
    parser.add_argument("--tau1-step", type=float, default=0.01)
    parser.add_argument("--tau2-min", type=float, default=-0.5)
    parser.add_argument("--tau2-max", type=float, default=0.5)
    parser.add_argument("--tau2-step", type=float, default=0.01)
    parser.add_argument("--min-large-rate", type=float, default=0.05)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--random-gpt2", action="store_true")
    parser.add_argument("--gpt2-name", type=str, default="gpt2")
    return parser.parse_args()


def make_grid(start, stop, step):
    count = int(round((stop - start) / step)) + 1
    values = start + np.arange(count) * step
    return np.round(values, 10)


def num_patches(window_size, patch_size):
    return int(np.ceil(window_size / patch_size))


def split_by_unit(dataset, val_ratio, seed):
    units = np.unique(dataset.sample_unit_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    val_unit_count = max(1, int(round(len(units) * val_ratio)))
    val_units = set(units[:val_unit_count])

    train_indices, val_indices = [], []
    for idx, unit_id in enumerate(dataset.sample_unit_ids):
        if unit_id in val_units:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_loader(args):
    dataset = CMAPSSDataset(args.data, window_size=args.window_size, stride=args.stride)

    if args.split == "all":
        selected = dataset
    else:
        train_set, val_set = split_by_unit(dataset, args.val_ratio, args.seed)
        selected = val_set if args.split == "val" else train_set

    return DataLoader(selected, batch_size=args.batch_size, shuffle=False)


def load_models(args, device):
    model_dir = Path(args.model_dir)

    S = SmallModel().to(device)
    L = GPT2TimeSeries(
        patch_size=args.patch_size,
        pretrained_name=args.gpt2_name,
        use_pretrained=not args.random_gpt2,
        local_files_only=args.local_files_only,
    ).to(device)
    Fz = FuzzyDecisionAgent(32, args.window_size).to(device)
    Rf = SelfReflection(L.gpt.config.hidden_size, num_patches(args.window_size, args.patch_size)).to(device)

    S.load_state_dict(torch.load(model_dir / "small.pt", map_location=device))
    L.load_state_dict(torch.load(model_dir / "large.pt", map_location=device))
    Fz.load_state_dict(torch.load(model_dir / "fuzzy.pt", map_location=device))
    Rf.load_state_dict(torch.load(model_dir / "reflect.pt", map_location=device))

    S.eval()
    L.eval()
    Fz.eval()
    Rf.eval()
    return S, L, Fz, Rf


def collect_outputs(S, L, Fz, Rf, loader, device):
    ys_all, yl_all, q_s_all, q_l_all, y_all = [], [], [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            ys, phi_s = S(x)
            yl, phi_l = L(x)
            q_s = Fz(phi_s)
            q_l = Rf(phi_l)

            ys_all.append(ys.cpu().numpy())
            yl_all.append(yl.cpu().numpy())
            q_s_all.append(q_s.cpu().numpy())
            q_l_all.append(q_l.cpu().numpy())
            y_all.append(y.numpy())

    return (
        np.concatenate(ys_all),
        np.concatenate(yl_all),
        np.concatenate(q_s_all),
        np.concatenate(q_l_all),
        np.concatenate(y_all),
    )


def evaluate_thresholds(ys, yl, q_s, q_l, y_true, tau1, tau2):
    small_mask = q_s >= tau1
    large_direct_mask = (q_s - q_l) <= tau2
    fused = 0.5 * (ys + yl)
    pred = np.where(small_mask, ys, np.where(large_direct_mask, yl, fused))
    rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
    large_rate = float(np.mean(~small_mask))
    fusion_rate = float(np.mean((~small_mask) & (~large_direct_mask)))
    return rmse, large_rate, fusion_rate


def search(ys, yl, q_s, q_l, y_true, tau1_values, tau2_values, min_large_rate):
    best = None
    fallback = None

    for tau1 in tau1_values:
        for tau2 in tau2_values:
            rmse, large_rate, fusion_rate = evaluate_thresholds(ys, yl, q_s, q_l, y_true, tau1, tau2)
            candidate = {
                "tau1": float(tau1),
                "tau2": float(tau2),
                "rmse": rmse,
                "large_rate": large_rate,
                "fusion_rate": fusion_rate,
            }
            if fallback is None or candidate["rmse"] < fallback["rmse"]:
                fallback = candidate

            if large_rate < min_large_rate:
                continue

            if best is None:
                best = candidate
                continue

            better_rmse = candidate["rmse"] < best["rmse"] - 1e-12
            same_rmse_lower_cost = abs(candidate["rmse"] - best["rmse"]) <= 1e-12 and candidate["large_rate"] < best["large_rate"]
            if better_rmse or same_rmse_lower_cost:
                best = candidate

    return best if best is not None else fallback


def write_config(config_path, tau1, tau2):
    config_path = Path(config_path)
    text = f"TAU1 = {tau1:.10g}\nTAU2 = {tau2:.10g}\n"
    config_path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    loader = build_loader(args)
    S, L, Fz, Rf = load_models(args, device)
    ys, yl, q_s, q_l, y_true = collect_outputs(S, L, Fz, Rf, loader, device)

    tau1_values = make_grid(args.tau1_min, args.tau1_max, args.tau1_step)
    tau2_values = make_grid(args.tau2_min, args.tau2_max, args.tau2_step)
    best = search(ys, yl, q_s, q_l, y_true, tau1_values, tau2_values, args.min_large_rate)
    write_config(args.config_path, best["tau1"], best["tau2"])

    small_rmse = float(np.sqrt(np.mean((ys - y_true) ** 2)))
    large_rmse = float(np.sqrt(np.mean((yl - y_true) ** 2)))
    fusion_rmse = float(np.sqrt(np.mean((0.5 * (ys + yl) - y_true) ** 2)))
    print("Baselines:")
    print(f"  small only : {small_rmse:.6f}")
    print(f"  large only : {large_rmse:.6f}")
    print(f"  avg fusion : {fusion_rmse:.6f}")
    print("Best thresholds found:")
    print(f"  tau1       : {best['tau1']:.6f}")
    print(f"  tau2       : {best['tau2']:.6f}")
    print(f"  RMSE       : {best['rmse']:.6f}")
    print(f"  large rate : {best['large_rate']:.2%}")
    print(f"  fusion rate: {best['fusion_rate']:.2%}")
    print(f"  min large rate constraint: {args.min_large_rate:.2%}")
    print(f"Updated config: {args.config_path}")


if __name__ == "__main__":
    main()
