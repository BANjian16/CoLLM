import argparse
import math
import os

import matplotlib

# Windows/服务器环境下可能没有图形界面，Agg 后端可以直接把图保存成文件。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_thresholds
from datasets.cmapss_test import CMAPSSTestDataset
from models.collm import CoLLM
from models.fuzzy import FuzzyDecisionAgent
from models.one_fits_all_ts import OneFitsAllTimeSeries
from models.reflection import SelfReflection
from models.small import SmallModel


def parse_args():
    """测试脚本参数。

    常用参数：
    --subset 选择 FD001 或 FD003
    --model-dir 指向训练得到的权重目录，例如 train 或 train_fd003
    --save-dir 指定测试图保存位置
    """
    parser = argparse.ArgumentParser(description="Evaluate CoLLM on CMAPSS FD001/FD003 test set.")
    parser.add_argument("--data-root", type=str, default="data/CMAPSS")
    parser.add_argument("--model-dir", type=str, default="train")
    parser.add_argument("--save-dir", type=str, default="results_test")
    parser.add_argument("--subset", choices=["FD001", "FD003"], default="FD001")
    parser.add_argument("--threshold-preset", choices=["A", "B", "C"], default="C")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--gpt2-name", type=str, default="gpt2")
    parser.add_argument("--gpt-layers", type=int, default=6)
    parser.add_argument("--gpt2-allow-download", action="store_true")
    parser.add_argument("--tau1", type=float, default=None)
    parser.add_argument("--tau2", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def rmse(pred, target):
    """均方根误差，越小越好。"""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred, target):
    """平均绝对误差，越小越好，比 RMSE 对极端误差稍微不那么敏感。"""
    return float(np.mean(np.abs(pred - target)))


def load_state(model, path, device):
    """加载模型权重，并切换到 eval 模式。"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    """官方测试集评估入口。

    流程：
    1. 根据 model-dir 加载 small、large、fuzzy、reflect 四个模块。
    2. 对测试集每台发动机取最后一个窗口预测 RUL。
    3. 分别统计 small、large、CoLLM 的 RMSE/MAE。
    4. 保存预测曲线、误差分布和误差-RUL 散点图。
    """
    args = parse_args()
    paper_tau1, paper_tau2 = get_thresholds(args.subset, args.threshold_preset)
    if args.tau1 is None:
        args.tau1 = paper_tau1
    if args.tau2 is None:
        args.tau2 = paper_tau2
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # small 是轻量 Transformer，小而快。
    small = load_state(SmallModel().to(device), os.path.join(args.model_dir, "small.pt"), device)
    # large 是 GPT-2 based One Fits All 大模型，结构必须和训练时完全一致。
    large = load_state(
        OneFitsAllTimeSeries(
            patch_size=args.patch_size,
            pretrained_name=args.gpt2_name,
            use_pretrained=True,
            local_files_only=not args.gpt2_allow_download,
            gpt_layers=args.gpt_layers,
            num_patches=math.ceil(args.window_size / args.patch_size),
        ).to(device),
        os.path.join(args.model_dir, "large.pt"),
        device,
    )
    # fuzzy 估计小模型置信度 q_s。
    fuzzy = load_state(
        FuzzyDecisionAgent(32, args.window_size).to(device),
        os.path.join(args.model_dir, "fuzzy.pt"),
        device,
    )
    n_patches = math.ceil(args.window_size / args.patch_size)
    # reflect 估计大模型置信度 q_l。
    reflect = load_state(
        SelfReflection(large.hidden_size, n_patches).to(device),
        os.path.join(args.model_dir, "reflect.pt"),
        device,
    )
    collm = CoLLM(small, large, fuzzy, reflect)

    # 测试集使用训练时保存的 scaler_stats.npz 做标准化。
    stats_path = os.path.join(args.model_dir, "scaler_stats.npz")
    dataset = CMAPSSTestDataset(
        os.path.join(args.data_root, f"test_{args.subset}.txt"),
        os.path.join(args.data_root, f"RUL_{args.subset}.txt"),
        window_size=args.window_size,
        stats_path=stats_path if os.path.exists(stats_path) else None,
        train_path=os.path.join(args.data_root, f"train_{args.subset}.txt"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ys, yl, yc, ytrue = [], [], [], []
    route_small, route_large, route_fusion = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # 分别计算小模型、大模型、协作模型输出，便于比较。
            y_s, _ = small(x)
            y_l, _ = large(x)
            y_c, details = collm.inference(x, args.tau1, args.tau2, return_details=True)

            ys.append(y_s.cpu().numpy())
            yl.append(y_l.cpu().numpy())
            yc.append(y_c.cpu().numpy())
            ytrue.append(y.numpy())
            route_small.append(details["use_small"].cpu().numpy())
            route_large.append(details["use_large"].cpu().numpy())
            route_fusion.append(details["use_fusion"].cpu().numpy())

    ys = np.concatenate(ys)
    yl = np.concatenate(yl)
    yc = np.concatenate(yc)
    ytrue = np.concatenate(ytrue)
    route_small = np.concatenate(route_small)
    route_large = np.concatenate(route_large)
    route_fusion = np.concatenate(route_fusion)

    # route_* 表示最终样本走了哪条路径，用来观察 CoLLM 是否真的在“按置信度分流”。
    print(f"RMSE Small : {rmse(ys, ytrue):.3f}")
    print(f"RMSE Large : {rmse(yl, ytrue):.3f}")
    print(f"RMSE CoLLM : {rmse(yc, ytrue):.3f}")
    print(f"MAE Small  : {mae(ys, ytrue):.3f}")
    print(f"MAE Large  : {mae(yl, ytrue):.3f}")
    print(f"MAE CoLLM  : {mae(yc, ytrue):.3f}")
    print(f"Routes     : small={route_small.mean():.2%}, large={route_large.mean():.2%}, fusion={route_fusion.mean():.2%}")
    print(f"Dataset    : {args.subset}")
    print(f"Thresholds : preset={args.threshold_preset}, tau1={args.tau1:.3f}, tau2={args.tau2:.3f}")

    n_show = min(300, len(ytrue))
    # 图 1：真实 RUL 与三种预测曲线对比。
    plt.figure(figsize=(10, 4))
    plt.plot(ytrue[:n_show], label="Ground Truth", linewidth=2)
    plt.plot(ys[:n_show], "--", label="Small Model")
    plt.plot(yl[:n_show], ":", label="Large Model")
    plt.plot(yc[:n_show], label="CoLLM", linewidth=2)
    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.title(f"RUL Prediction on Test Set ({args.subset})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"{args.subset}_test_rul_comparison.png"), dpi=args.dpi)
    plt.savefig(os.path.join(args.save_dir, f"{args.subset}_test_rul_comparison.pdf"), dpi=args.dpi)
    plt.close()

    err_s = ys - ytrue
    err_l = yl - ytrue
    err_c = yc - ytrue

    # 图 2：误差分布。分布越集中在 0 附近，说明预测越稳定。
    plt.figure(figsize=(6, 4))
    plt.hist(err_s, bins=40, alpha=0.5, label="Small")
    plt.hist(err_l, bins=40, alpha=0.5, label="Large")
    plt.hist(err_c, bins=40, alpha=0.7, label="CoLLM")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution on Test Set ({args.subset})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"{args.subset}_test_error_distribution.png"), dpi=args.dpi)
    plt.savefig(os.path.join(args.save_dir, f"{args.subset}_test_error_distribution.pdf"), dpi=args.dpi)
    plt.close()

    # 图 3：不同真实 RUL 下的误差。可以观察模型在早期/临近失效时哪里更容易错。
    plt.figure(figsize=(6, 4))
    plt.scatter(ytrue, err_s, s=8, alpha=0.3, label="Small")
    plt.scatter(ytrue, err_l, s=8, alpha=0.3, label="Large")
    plt.scatter(ytrue, err_c, s=8, alpha=0.4, label="CoLLM")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Ground Truth RUL")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error vs RUL (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"{args.subset}_test_error_vs_rul.png"), dpi=args.dpi)
    plt.savefig(os.path.join(args.save_dir, f"{args.subset}_test_error_vs_rul.pdf"), dpi=args.dpi)
    plt.close()


if __name__ == "__main__":
    main()
