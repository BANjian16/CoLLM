import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.cmapss import CMAPSSDataset
from datasets.cmapss_test import CMAPSSTestDataset
from models.small import SmallModel
from models.one_fits_all_ts import OneFitsAllTimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection
from models.collm import CoLLM


def rmse(pred, target):
    # pred/target shape: [N]
    # RMSE = sqrt(mean((pred-target)^2))
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def build_loaders(data_path, window_size, stride, batch_size, val_ratio, seed):
    """
    按发动机 unit_id 拆分，避免滑动窗口随机切分导致的数据穿越：
    70% train units, 15% val units(用于 Stage 1/2), 15% conf_val units(用于 Stage 3 验证)。
    """
    del val_ratio  # 保留参数兼容历史调用；当前采用固定 70/15/15。

    unit_ids = np.loadtxt(data_path, usecols=[0]).astype(int)
    unique_units = np.unique(unit_ids)
    n_units = len(unique_units)
    if n_units < 3:
        raise ValueError('Need at least 3 unique units to split into train/val/conf_val.')

    rng = np.random.default_rng(seed)
    shuffled_units = unique_units.copy()
    rng.shuffle(shuffled_units)

    n_train_units = max(1, int(n_units * 0.70))
    n_val_units = max(1, int(n_units * 0.15))
    n_conf_units = n_units - n_train_units - n_val_units

    # 保证 conf_val 至少有 1 个 unit。
    if n_conf_units < 1:
        n_conf_units = 1
        if n_val_units > 1:
            n_val_units -= 1
        else:
            n_train_units -= 1

    train_units = shuffled_units[:n_train_units]
    val_units = shuffled_units[n_train_units:n_train_units + n_val_units]
    conf_val_units = shuffled_units[n_train_units + n_val_units:]

    # 训练集统计量用于所有 split 的归一化，避免验证集信息泄漏。
    train_ds = CMAPSSDataset(
        data_path,
        window_size=window_size,
        stride=stride,
        valid_unit_ids=train_units,
    )
    val_ds = CMAPSSDataset(
        data_path,
        window_size=window_size,
        stride=stride,
        mean=train_ds.mean,
        std=train_ds.std,
        valid_unit_ids=val_units,
    )
    conf_val_ds = CMAPSSDataset(
        data_path,
        window_size=window_size,
        stride=stride,
        mean=train_ds.mean,
        std=train_ds.std,
        valid_unit_ids=conf_val_units,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    conf_val_loader = DataLoader(conf_val_ds, batch_size=batch_size, shuffle=False)

    print(
        f'Unit split -> train: {len(train_units)}, val: {len(val_units)}, conf_val: {len(conf_val_units)}'
    )
    print(
        f'Window split -> train: {len(train_ds)}, val: {len(val_ds)}, conf_val: {len(conf_val_ds)}'
    )

    return train_loader, val_loader, conf_val_loader


def train_regressor(model, train_loader, val_loader, device, epochs, lr, weight_decay, tag):
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_state = None
    best_val_rmse = float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            # x shape: [B, T, d], y shape: [B]
            x = x.to(device)
            y = y.to(device)

            # pred shape: [B]
            pred, _ = model(x)
            # MSE = (1/B) * sum_i (pred_i - y_i)^2
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred, _ = model(x)
                preds.append(pred)
                gts.append(y)

        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)
        val_rmse = rmse(preds, gts)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f'[{tag}] Epoch {ep:03d}/{epochs} | '
            f'train_mse={np.mean(train_losses):.6f} | val_rmse={val_rmse:.4f}'
        )
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_rmse


def train_confidence_heads(
    small_model,
    large_model,
    fuzzy_agent,
    reflection,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    alpha,
):
    small_model.eval()
    large_model.eval()
    fuzzy_agent.to(device)
    reflection.to(device)

    opt_f = torch.optim.AdamW(fuzzy_agent.parameters(), lr=lr)
    opt_r = torch.optim.AdamW(reflection.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state_f = None
    best_state_r = None
    best_val = float('inf')

    for ep in range(1, epochs + 1):
        fuzzy_agent.train()
        reflection.train()
        train_losses = []

        for x, y in train_loader:
            # x shape: [B, T, d], y shape: [B]
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                # ys/yl shape: [B]
                # phi_s shape: [B, T, ds]
                # phi_l shape: [B, num_patches, dl]
                ys, phi_s = small_model(x)
                yl, phi_l = large_model(x)

            # [Paper Sec 3.C, Eq. (13)/(14)] 置信度监督信号：
            # Q*_s = 1 - tanh(|ys - y| / alpha)
            # Q*_l = 1 - tanh(|yl - y| / alpha)
            # q_s_star / q_l_star shape: [B]
            q_s_star = 1.0 - torch.tanh(torch.abs(ys - y) / alpha)
            q_l_star = 1.0 - torch.tanh(torch.abs(yl - y) / alpha)

            # [Paper Eq. (11)] q_s = F(phi_s)
            # [Paper Eq. (12)] q_l = R(phi_l)
            # q_s / q_l shape: [B]
            q_s = fuzzy_agent(phi_s.detach())
            q_l = reflection(phi_l.detach())

            # [Paper Eq. (14)] L_reflect = MSE(q_l, q_l_star)
            # Fuzzy head 同理使用 MSE(q_s, q_s_star)
            loss_f = criterion(q_s, q_s_star)
            loss_r = criterion(q_l, q_l_star)
            loss = loss_f + loss_r

            opt_f.zero_grad()
            opt_r.zero_grad()
            loss.backward()
            opt_f.step()
            opt_r.step()

            train_losses.append(loss.item())

        fuzzy_agent.eval()
        reflection.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                ys, phi_s = small_model(x)
                yl, phi_l = large_model(x)

                q_s_star = 1.0 - torch.tanh(torch.abs(ys - y) / alpha)
                q_l_star = 1.0 - torch.tanh(torch.abs(yl - y) / alpha)

                q_s = fuzzy_agent(phi_s)
                q_l = reflection(phi_l)

                loss = criterion(q_s, q_s_star) + criterion(q_l, q_l_star)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state_f = {k: v.cpu().clone() for k, v in fuzzy_agent.state_dict().items()}
            best_state_r = {k: v.cpu().clone() for k, v in reflection.state_dict().items()}

        print(
            f'[Fuzzy+Reflect] Epoch {ep:03d}/{epochs} | '
            f'train_loss={np.mean(train_losses):.6f} | val_loss={val_loss:.6f}'
        )

    if best_state_f is not None:
        fuzzy_agent.load_state_dict(best_state_f)
    if best_state_r is not None:
        reflection.load_state_dict(best_state_r)

    return best_val


def evaluate_collm(collm, small_model, large_model, loader, device, tau1, tau2):
    collm_scores = []
    small_scores = []
    large_scores = []
    route_small = []
    route_fusion = []

    with torch.no_grad():
        for x, y in loader:
            # x shape: [B, T, d], y shape: [B]
            x = x.to(device)
            y = y.to(device)

            # ys / yl / yc shape: [B]
            ys, _ = small_model(x)
            yl, _ = large_model(x)
            yc, details = collm.inference(x, tau1=tau1, tau2=tau2, return_details=True)

            small_scores.append((ys, y))
            large_scores.append((yl, y))
            collm_scores.append((yc, y))

            # use_small / use_fusion shape: [B] -> batch routing ratio scalar
            route_small.append(details['use_small'].float().mean().item())
            route_fusion.append(details['use_fusion'].float().mean().item())

    def aggregate(scores):
        preds = torch.cat([p for p, _ in scores], dim=0)
        gts = torch.cat([g for _, g in scores], dim=0)
        return rmse(preds, gts)

    return {
        'rmse_small': aggregate(small_scores),
        'rmse_large': aggregate(large_scores),
        'rmse_collm': aggregate(collm_scores),
        'ratio_small': float(np.mean(route_small)),
        'ratio_fusion': float(np.mean(route_fusion)),
    }


def test_cmapss_official(
    collm,
    small_model,
    large_model,
    data_root,
    train_path,
    device,
    batch_size,
    tau1,
    tau2,
    window_size,
):
    """
    CMAPSS 官方评测：每台发动机仅使用最后一个时刻对应的滑动窗口进行预测。

    对应文件：
        - test_FD001.txt
        - RUL_FD001.txt
    """
    test_path = os.path.join(data_root, 'test_FD001.txt')
    rul_path = os.path.join(data_root, 'RUL_FD001.txt')

    dataset = CMAPSSTestDataset(
        test_path=test_path,
        rul_path=rul_path,
        window_size=window_size,
        train_path=train_path,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ys_list, yl_list, yc_list, y_list = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            # x shape: [B, window_size, 14], 每个样本对应一台发动机的 Last Cycle Window
            x = x.to(device)
            y = y.to(device)

            ys, _ = small_model(x)
            yl, _ = large_model(x)
            yc = collm.inference(x, tau1=tau1, tau2=tau2)

            ys_list.append(ys)
            yl_list.append(yl)
            yc_list.append(yc)
            y_list.append(y)

    ys_all = torch.cat(ys_list, dim=0)
    yl_all = torch.cat(yl_list, dim=0)
    yc_all = torch.cat(yc_list, dim=0)
    y_true = torch.cat(y_list, dim=0)

    return {
        'rmse_small': rmse(ys_all, y_true),
        'rmse_large': rmse(yl_all, y_true),
        'rmse_collm': rmse(yc_all, y_true),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train CoLLM checkpoints (small/large/fuzzy/reflect).')
    parser.add_argument('--data_path', type=str, default=os.path.join(ROOT_DIR, 'data', 'CMAPSS', 'train_FD001.txt'))
    parser.add_argument('--data_root', type=str, default=os.path.join(ROOT_DIR, 'data', 'CMAPSS'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(ROOT_DIR, 'train'))
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='auto: prefer CUDA when available; otherwise CPU.'
    )
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_ratio', type=float, default=0.2)

    parser.add_argument('--epochs_small', type=int, default=20)
    parser.add_argument('--epochs_large', type=int, default=100)
    parser.add_argument('--epochs_conf', type=int, default=20)

    parser.add_argument('--lr_small', type=float, default=1e-3)
    parser.add_argument('--lr_large', type=float, default=1e-4)
    parser.add_argument('--lr_conf', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--alpha', type=float, default=30.0)
    parser.add_argument('--tau1', type=float, default=0.7)
    parser.add_argument('--tau2', type=float, default=-0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Using device: {device}')

    train_loader, val_loader, conf_val_loader = build_loaders(
        data_path=args.data_path,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    small = SmallModel(input_dim=14, emb_dim=32).to(device)
    large = OneFitsAllTimeSeries(input_dim=14, patch_size=args.patch_size).to(device)

    # n_patches 对齐 OneFitsAllTimeSeries.forward 的 padding + patch 切分：
    # 若 T 不能被 patch 整除，会在时间维左侧补零到最近的整倍数。
    # 因此 num_patches = ceil(T / patch_size)。
    patch_size = large.patch
    n_patches = math.ceil(args.window_size / patch_size)
    fuzzy = FuzzyDecisionAgent(feature_dim=32, T=args.window_size).to(device)
    reflect = SelfReflection(feature_dim=large.hidden_size, T=n_patches).to(device)

    print('=== Stage 1: Train SmallModel ===')
    best_small = train_regressor(
        small,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_small,
        lr=args.lr_small,
        weight_decay=args.weight_decay,
        tag='SmallModel',
    )
    print(f'[SmallModel] best val RMSE: {best_small:.4f}')

    print('=== Stage 2: Train OneFitsAllTimeSeries (proj/head) ===')
    best_large = train_regressor(
        large,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_large,
        lr=args.lr_large,
        weight_decay=args.weight_decay,
        tag='LargeModel',
    )
    print(f'[LargeModel] best val RMSE: {best_large:.4f}')

    print('=== Stage 3: Train FuzzyDecision + SelfReflection ===')
    best_conf = train_confidence_heads(
        small,
        large,
        fuzzy,
        reflect,
        train_loader=val_loader,
        val_loader=conf_val_loader,
        device=device,
        epochs=args.epochs_conf,
        lr=args.lr_conf,
        alpha=args.alpha,
    )
    print(f'[Fuzzy+Reflect] best val loss: {best_conf:.6f}')

    collm = CoLLM(small, large, fuzzy, reflect)
    report = evaluate_collm(collm, small, large, val_loader, device, args.tau1, args.tau2)

    print('=== Validation Report ===')
    print(f"RMSE Small : {report['rmse_small']:.4f}")
    print(f"RMSE Large : {report['rmse_large']:.4f}")
    print(f"RMSE CoLLM : {report['rmse_collm']:.4f}")
    print(f"Route Small Ratio : {report['ratio_small']:.3f}")
    print(f"Route Fusion Ratio: {report['ratio_fusion']:.3f}")

    official_report = test_cmapss_official(
        collm=collm,
        small_model=small,
        large_model=large,
        data_root=args.data_root,
        train_path=args.data_path,
        device=device,
        batch_size=args.batch_size,
        tau1=args.tau1,
        tau2=args.tau2,
        window_size=args.window_size,
    )
    print('=== Official CMAPSS FD001 Test (Last Cycle Window) ===')
    print(f"RMSE Small : {official_report['rmse_small']:.4f}")
    print(f"RMSE Large : {official_report['rmse_large']:.4f}")
    print(f"RMSE CoLLM : {official_report['rmse_collm']:.4f}")

    torch.save(small.state_dict(), os.path.join(args.save_dir, 'small.pt'))
    torch.save(large.state_dict(), os.path.join(args.save_dir, 'large.pt'))
    torch.save(fuzzy.state_dict(), os.path.join(args.save_dir, 'fuzzy.pt'))
    torch.save(reflect.state_dict(), os.path.join(args.save_dir, 'reflect.pt'))
    print(f'Checkpoints saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
