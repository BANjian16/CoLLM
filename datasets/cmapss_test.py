import numpy as np
import torch
from torch.utils.data import Dataset


class CMAPSSTestDataset(Dataset):
    def __init__(self, test_path, rul_path, window_size=50, sensor_mean=None, sensor_std=None, stats_path=None):
        # CMAPSS 测试集预处理。
        # 与训练集不同，测试集每台发动机只取“最后一个可观测窗口”，
        # 标签来自官方提供的 RUL 文件。
        data = np.loadtxt(test_path)
        rul_last = np.loadtxt(rul_path)

        unit_ids = data[:, 0].astype(int)
        cycles = data[:, 1]
        SENSOR_IDX = [
            1, 2, 3, 6, 7, 8,
            10, 11, 12, 13, 14, 16,
            19, 20
        ]

        # 测试时必须与训练阶段保持完全一致的 14 维传感器选择。
        sensors = data[:, [5 + i for i in SENSOR_IDX]]
        if stats_path is not None:
            stats = np.load(stats_path)
            sensor_mean = stats["sensor_mean"]
            sensor_std = stats["sensor_std"]
        if sensor_mean is None or sensor_std is None:
            raise ValueError("CMAPSSTestDataset requires training sensor_mean/sensor_std or stats_path.")
        sensors = (sensors - np.asarray(sensor_mean)) / (np.asarray(sensor_std) + 1e-6)

        self.X, self.y = [], []

        for uid in np.unique(unit_ids):
            idx = np.where(unit_ids == uid)[0]
            if len(idx) < window_size:
                continue

            # CMAPSS 标准评测做法：
            # 对每台发动机，仅使用最后一个长度为 window_size 的观测窗口，
            # 预测该发动机在测试终点之后的剩余寿命。
            x = sensors[idx[-window_size:]]
            self.X.append(x)
            self.y.append(rul_last[uid - 1])

        self.X = torch.from_numpy(np.array(self.X)).float()
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
