import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cmapss import CMAPSSDataset
from models.fuzzy import FuzzyDecisionAgent
from models.gpt2_ts import GPT2TimeSeries
from models.reflection import SelfReflection
from models.small import SmallModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train CoLLM in three stages.")
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "CMAPSS" / "train_FD001.txt"))
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "train"))
    parser.add_argument("--stages", nargs="+", choices=["small", "large", "confidence", "all"], default=["all"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--gpt2-name", type=str, default="gpt2")
    parser.add_argument("--random-gpt2", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs-small", type=int, default=100)
    parser.add_argument("--epochs-large", type=int, default=100)
    parser.add_argument("--epochs-confidence", type=int, default=100)
    parser.add_argument("--lr-small", type=float, default=1e-3)
    parser.add_argument("--lr-large", type=float, default=1e-3)
    parser.add_argument("--lr-confidence", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def qstar(pred, target, alpha):
    # 论文公式：Q* = 1 - tanh(|y_pred - y_true| / alpha)。
    return 1 - torch.tanh(torch.abs(pred - target) / alpha)


def rmse_from_loss(mse_loss):
    return mse_loss ** 0.5


def build_loaders(args):
    dataset = CMAPSSDataset(args.data, window_size=args.window_size, stride=args.stride)
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate_regressor(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred, _ = model(x)
            loss = F.mse_loss(pred, y, reduction="sum")
            total_loss += loss.item()
            total_count += y.numel()
    return rmse_from_loss(total_loss / max(total_count, 1))


def train_regressor(model, train_loader, val_loader, optimizer, epochs, device, name, save_path):
    best_rmse = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred, _ = model(x)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_count += y.numel()

        train_rmse = rmse_from_loss(total_loss / max(total_count, 1))
        val_rmse = evaluate_regressor(model, val_loader, device)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), save_path)

        print(f"[{name}] Epoch {epoch:03d}/{epochs} | train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}")

    print(f"[{name}] Best val RMSE: {best_rmse:.4f}; saved to {save_path}")


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def evaluate_confidence(S, L, Fz, Rf, loader, device, alpha):
    Fz.eval()
    Rf.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            ys, phi_s = S(x)
            yl, phi_l = L(x)
            loss_s = F.mse_loss(Fz(phi_s), qstar(ys, y, alpha), reduction="sum")
            loss_l = F.mse_loss(Rf(phi_l), qstar(yl, y, alpha), reduction="sum")
            total_loss += (loss_s + loss_l).item()
            total_count += y.numel()
    return total_loss / max(total_count, 1)


def train_confidence(S, L, Fz, Rf, train_loader, val_loader, optimizer, epochs, device, alpha, save_dir):
    freeze_model(S)
    freeze_model(L)

    best_loss = float("inf")
    fuzzy_path = save_dir / "fuzzy.pt"
    reflect_path = save_dir / "reflect.pt"

    for epoch in range(1, epochs + 1):
        Fz.train()
        Rf.train()
        total_loss = 0.0
        total_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                ys, phi_s = S(x)
                yl, phi_l = L(x)

            loss_s = F.mse_loss(Fz(phi_s), qstar(ys, y, alpha))
            loss_l = F.mse_loss(Rf(phi_l), qstar(yl, y, alpha))
            loss = loss_s + loss_l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_count += y.numel()

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate_confidence(S, L, Fz, Rf, val_loader, device, alpha)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(Fz.state_dict(), fuzzy_path)
            torch.save(Rf.state_dict(), reflect_path)

        print(
            f"[confidence] Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.6f} | val loss {val_loss:.6f}"
        )

    print(f"[confidence] Best val loss: {best_loss:.6f}; saved to {fuzzy_path} and {reflect_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device)
    train_loader, val_loader = build_loaders(args)

    requested = set(args.stages)
    if "all" in requested:
        requested = {"small", "large", "confidence"}

    small_path = save_dir / "small.pt"
    large_path = save_dir / "large.pt"

    S = SmallModel().to(device)
    L = GPT2TimeSeries(
        patch_size=args.patch_size,
        pretrained_name=args.gpt2_name,
        use_pretrained=not args.random_gpt2,
        local_files_only=args.local_files_only,
    ).to(device)

    if "small" in requested:
        optimizer_s = torch.optim.Adam(S.parameters(), lr=args.lr_small)
        train_regressor(S, train_loader, val_loader, optimizer_s, args.epochs_small, device, "small", small_path)
    elif small_path.exists():
        S.load_state_dict(torch.load(small_path, map_location=device))

    if "large" in requested:
        trainable_l = [p for p in L.parameters() if p.requires_grad]
        optimizer_l = torch.optim.Adam(trainable_l, lr=args.lr_large)
        train_regressor(L, train_loader, val_loader, optimizer_l, args.epochs_large, device, "large", large_path)
    elif large_path.exists():
        L.load_state_dict(torch.load(large_path, map_location=device))

    if "confidence" in requested:
        if not small_path.exists() or not large_path.exists():
            raise FileNotFoundError("Training confidence modules requires small.pt and large.pt.")

        S.load_state_dict(torch.load(small_path, map_location=device))
        L.load_state_dict(torch.load(large_path, map_location=device))

        num_patches = (args.window_size - args.patch_size) // args.patch_size + 1
        Fz = FuzzyDecisionAgent(feature_dim=32, T=args.window_size).to(device)
        Rf = SelfReflection(feature_dim=L.gpt.config.hidden_size, T=num_patches).to(device)
        optimizer_conf = torch.optim.Adam(
            list(Fz.parameters()) + list(Rf.parameters()),
            lr=args.lr_confidence,
        )
        train_confidence(
            S,
            L,
            Fz,
            Rf,
            train_loader,
            val_loader,
            optimizer_conf,
            args.epochs_confidence,
            device,
            args.alpha,
            save_dir,
        )


if __name__ == "__main__":
    main()
