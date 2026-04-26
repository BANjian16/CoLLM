import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cmapss import CMAPSSDataset, load_cmapss_sensors
from datasets.cmapss_test import CMAPSSTestDataset
from config import get_thresholds
from models.collm import CoLLM
from models.fuzzy import FuzzyDecisionAgent
from models.one_fits_all_ts import OneFitsAllTimeSeries
from models.reflection import SelfReflection
from models.small import SmallModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train CoLLM on CMAPSS FD001/FD003.")
    parser.add_argument("--data-root", type=str, default=str(ROOT / "data" / "CMAPSS"))
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "train_repro"))
    parser.add_argument("--subset", choices=["FD001", "FD003"], default="FD001")
    parser.add_argument("--threshold-preset", choices=["A", "B", "C"], default="C")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-small", type=int, default=30)
    parser.add_argument("--epochs-large", type=int, default=120)
    parser.add_argument("--epochs-conf", type=int, default=40)
    parser.add_argument("--lr-small", type=float, default=1e-3)
    parser.add_argument("--lr-large", type=float, default=1e-4)
    parser.add_argument("--lr-conf", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--tau1", type=float, default=None)
    parser.add_argument("--tau2", type=float, default=None)
    parser.add_argument("--freeze-large-backbone", action="store_true")
    parser.add_argument("--split-mode", choices=["unit", "random"], default="random")
    parser.add_argument("--norm-scope", choices=["train", "combined"], default="train")
    parser.add_argument("--stages", nargs="+", choices=["small", "large", "confidence", "all"], default=["all"])
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def split_units(data_path, seed):
    unit_ids = np.loadtxt(data_path, usecols=[0]).astype(int)
    units = np.unique(unit_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_train = max(1, int(len(units) * 0.70))
    n_val = max(1, int(len(units) * 0.15))
    train_units = units[:n_train]
    val_units = units[n_train:n_train + n_val]
    conf_units = units[n_train + n_val:]
    if len(conf_units) == 0:
        conf_units = val_units[-1:]
        val_units = val_units[:-1]
    return train_units, val_units, conf_units


def normalization_stats(args):
    if args.norm_scope == "train":
        return None, None
    train_path = os.path.join(args.data_root, f"train_{args.subset}.txt")
    test_path = os.path.join(args.data_root, f"test_{args.subset}.txt")
    _, _, train_sensors = load_cmapss_sensors(train_path)
    _, _, test_sensors = load_cmapss_sensors(test_path)
    sensors = np.concatenate([train_sensors, test_sensors], axis=0)
    return sensors.mean(0), sensors.std(0) + 1e-6


def make_loaders(args):
    train_path = os.path.join(args.data_root, f"train_{args.subset}.txt")
    norm_mean, norm_std = normalization_stats(args)
    if args.split_mode == "random":
        dataset = CMAPSSDataset(
            train_path,
            window_size=args.window_size,
            stride=args.stride,
            mean=norm_mean,
            std=norm_std,
        )
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
        conf_val_size = max(1, int(len(val_ds) * 0.5))
        val_main_size = len(val_ds) - conf_val_size
        val_ds, conf_ds = random_split(val_ds, [val_main_size, conf_val_size], generator=generator)
        print(f"Window split(random) -> train={len(train_ds)}, val={len(val_ds)}, conf={len(conf_ds)}")
        kwargs = {"num_workers": 0, "pin_memory": args.device == "cuda"}
        return (
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs),
            DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kwargs),
            DataLoader(conf_ds, batch_size=args.batch_size, shuffle=False, **kwargs),
            dataset,
        )

    train_units, val_units, conf_units = split_units(train_path, args.seed)

    train_ds = CMAPSSDataset(
        train_path,
        window_size=args.window_size,
        stride=args.stride,
        mean=norm_mean,
        std=norm_std,
        valid_unit_ids=train_units,
    )
    val_ds = CMAPSSDataset(
        train_path,
        window_size=args.window_size,
        stride=args.stride,
        mean=train_ds.mean,
        std=train_ds.std,
        valid_unit_ids=val_units,
    )
    conf_ds = CMAPSSDataset(
        train_path,
        window_size=args.window_size,
        stride=args.stride,
        mean=train_ds.mean,
        std=train_ds.std,
        valid_unit_ids=conf_units,
    )

    print(f"Unit split -> train={len(train_units)}, val={len(val_units)}, conf={len(conf_units)}")
    print(f"Window split -> train={len(train_ds)}, val={len(val_ds)}, conf={len(conf_ds)}")

    kwargs = {"num_workers": 0, "pin_memory": args.device == "cuda"}
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kwargs),
        DataLoader(conf_ds, batch_size=args.batch_size, shuffle=False, **kwargs),
        train_ds,
    )


def evaluate_regressor(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred, _ = model(x)
            preds.append(pred)
            targets.append(y)
    return rmse(torch.cat(preds), torch.cat(targets))


def train_regressor(model, train_loader, val_loader, device, epochs, lr, weight_decay, tag, save_path):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    criterion = nn.MSELoss()
    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred, _ = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_rmse = evaluate_regressor(model, val_loader, device)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
        scheduler.step()
        print(f"[{tag}] {epoch:03d}/{epochs} train_mse={np.mean(losses):.4f} val_rmse={val_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{tag}] best_val_rmse={best_rmse:.4f} saved={save_path}")
    return best_rmse


def train_confidence(small, large, fuzzy, reflect, train_loader, val_loader, device, epochs, lr, alpha, save_dir):
    small.eval()
    large.eval()
    for param in small.parameters():
        param.requires_grad = False
    for param in large.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(list(fuzzy.parameters()) + list(reflect.parameters()), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_fuzzy, best_reflect = None, None

    for epoch in range(1, epochs + 1):
        fuzzy.train()
        reflect.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                ys, phi_s = small(x)
                yl, phi_l = large(x)
                q_s_star = 1.0 - torch.tanh(torch.abs(ys - y) / alpha)
                q_l_star = 1.0 - torch.tanh(torch.abs(yl - y) / alpha)

            loss = criterion(fuzzy(phi_s), q_s_star) + criterion(reflect(phi_l), q_l_star)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_loss = evaluate_confidence(small, large, fuzzy, reflect, val_loader, device, alpha)
        if val_loss < best_loss:
            best_loss = val_loss
            best_fuzzy = {k: v.detach().cpu().clone() for k, v in fuzzy.state_dict().items()}
            best_reflect = {k: v.detach().cpu().clone() for k, v in reflect.state_dict().items()}
            torch.save(best_fuzzy, os.path.join(save_dir, "fuzzy.pt"))
            torch.save(best_reflect, os.path.join(save_dir, "reflect.pt"))
        print(f"[confidence] {epoch:03d}/{epochs} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}")

    if best_fuzzy is not None:
        fuzzy.load_state_dict(best_fuzzy)
    if best_reflect is not None:
        reflect.load_state_dict(best_reflect)
    print(f"[confidence] best_val_loss={best_loss:.6f}")


def evaluate_confidence(small, large, fuzzy, reflect, loader, device, alpha):
    fuzzy.eval()
    reflect.eval()
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ys, phi_s = small(x)
            yl, phi_l = large(x)
            q_s_star = 1.0 - torch.tanh(torch.abs(ys - y) / alpha)
            q_l_star = 1.0 - torch.tanh(torch.abs(yl - y) / alpha)
            losses.append((criterion(fuzzy(phi_s), q_s_star) + criterion(reflect(phi_l), q_l_star)).item())
    return float(np.mean(losses))


def evaluate_official(args, small, large, fuzzy, reflect, device):
    stats_path = os.path.join(args.save_dir, "scaler_stats.npz")
    dataset = CMAPSSTestDataset(
        os.path.join(args.data_root, f"test_{args.subset}.txt"),
        os.path.join(args.data_root, f"RUL_{args.subset}.txt"),
        window_size=args.window_size,
        stats_path=stats_path,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    collm = CoLLM(small, large, fuzzy, reflect)
    ys_all, yl_all, yc_all, y_all = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            ys, _ = small(x)
            yl, _ = large(x)
            yc = collm.inference(x, args.tau1, args.tau2)
            ys_all.append(ys)
            yl_all.append(yl)
            yc_all.append(yc)
            y_all.append(y)
    y = torch.cat(y_all)
    return {
        "small": rmse(torch.cat(ys_all), y),
        "large": rmse(torch.cat(yl_all), y),
        "collm": rmse(torch.cat(yc_all), y),
    }


def main():
    args = parse_args()
    paper_tau1, paper_tau2 = get_thresholds(args.subset, args.threshold_preset)
    if args.tau1 is None:
        args.tau1 = paper_tau1
    if args.tau2 is None:
        args.tau2 = paper_tau2
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    train_loader, val_loader, conf_loader, train_ds = make_loaders(args)
    np.savez(
        os.path.join(args.save_dir, "scaler_stats.npz"),
        sensor_mean=train_ds.sensor_mean,
        sensor_std=train_ds.sensor_std,
    )

    requested = set(args.stages)
    if "all" in requested:
        requested = {"small", "large", "confidence"}

    small = SmallModel().to(device)
    large = OneFitsAllTimeSeries(
        patch_size=args.patch_size,
        freeze_backbone=args.freeze_large_backbone,
    ).to(device)
    n_patches = math.ceil(args.window_size / args.patch_size)
    fuzzy = FuzzyDecisionAgent(32, args.window_size).to(device)
    reflect = SelfReflection(large.hidden_size, n_patches).to(device)

    small_path = os.path.join(args.save_dir, "small.pt")
    large_path = os.path.join(args.save_dir, "large.pt")

    if "small" in requested:
        train_regressor(small, train_loader, val_loader, device, args.epochs_small, args.lr_small, args.weight_decay, "small", small_path)
    else:
        small.load_state_dict(torch.load(small_path, map_location=device))

    if "large" in requested:
        train_regressor(large, train_loader, val_loader, device, args.epochs_large, args.lr_large, args.weight_decay, "large", large_path)
    else:
        large.load_state_dict(torch.load(large_path, map_location=device))

    if "confidence" in requested:
        small.load_state_dict(torch.load(small_path, map_location=device))
        large.load_state_dict(torch.load(large_path, map_location=device))
        train_confidence(small, large, fuzzy, reflect, train_loader, conf_loader, device, args.epochs_conf, args.lr_conf, args.alpha, args.save_dir)
    else:
        fuzzy.load_state_dict(torch.load(os.path.join(args.save_dir, "fuzzy.pt"), map_location=device))
        reflect.load_state_dict(torch.load(os.path.join(args.save_dir, "reflect.pt"), map_location=device))

    report = evaluate_official(args, small, large, fuzzy, reflect, device)
    print(f"=== Official {args.subset} Test | CoLLM-{args.threshold_preset} tau=({args.tau1}, {args.tau2}) ===")
    print(f"RMSE Small : {report['small']:.4f}")
    print(f"RMSE Large : {report['large']:.4f}")
    print(f"RMSE CoLLM : {report['collm']:.4f}")


if __name__ == "__main__":
    main()
