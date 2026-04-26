import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.cmapss import compute_sensor_stats, load_cmapss_sensors


class CMAPSSTestDataset(Dataset):
    """测试集 Dataset。

    CMAPSS 测试集每台发动机只给出一段尚未失效的运行历史，真实 RUL 单独存在
    RUL_FDxxx.txt 中。官方评估通常对每台发动机取最后 window_size 个时间步，
    预测这台发动机“从当前时刻到失效还剩多久”。
    """

    def __init__(
        self,
        test_path,
        rul_path,
        window_size=50,
        sensor_mean=None,
        sensor_std=None,
        stats_path=None,
        train_path=None,
        mean=None,
        std=None,
        max_rul=125,
    ):
        # rul_last[i] 是第 i+1 台测试发动机最后一个观测点对应的真实 RUL。
        rul_last = np.clip(np.loadtxt(rul_path), 0, max_rul)
        unit_ids, _, sensors = load_cmapss_sensors(test_path)

        # 测试集必须使用训练时保存的均值/标准差，否则训练和测试分布不一致。
        if stats_path is not None:
            stats = np.load(stats_path)
            sensor_mean = stats["sensor_mean"]
            sensor_std = stats["sensor_std"]
        if mean is not None:
            sensor_mean = mean
        if std is not None:
            sensor_std = std
        if (sensor_mean is None or sensor_std is None) and train_path is not None:
            sensor_mean, sensor_std = compute_sensor_stats(train_path)
        if sensor_mean is None or sensor_std is None:
            raise ValueError("Provide train_path, stats_path, or training sensor statistics.")

        sensors = (sensors - np.asarray(sensor_mean)) / (np.asarray(sensor_std) + 1e-6)

        self.X, self.y = [], []
        for uid in np.unique(unit_ids):
            idx = np.where(unit_ids == uid)[0]
            if len(idx) < window_size:
                continue
            # 每台测试发动机只生成一个样本：最后 window_size 个时间步。
            self.X.append(sensors[idx[-window_size:]])
            self.y.append(rul_last[uid - 1])

        self.X = torch.from_numpy(np.asarray(self.X)).float()
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
