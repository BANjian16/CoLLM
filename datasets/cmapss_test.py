import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.cmapss import load_cmapss_sensors, compute_sensor_stats

class CMAPSSTestDataset(Dataset):
    def __init__(self, test_path, rul_path, window_size=50, train_path=None, mean=None, std=None):
        # rul_last[k] 表示第 k+1 台发动机在 test 文件末尾时刻的真实 RUL
        # rul_last shape: [num_units]
        rul_last = np.loadtxt(rul_path)

        unit_ids, cycles, sensors = load_cmapss_sensors(test_path)

        # [AI补全/优化] 论文复现建议：测试集应使用训练集统计量归一化，避免信息泄漏。
        if mean is None or std is None:
            if train_path is not None:
                mean, std = compute_sensor_stats(train_path)
            else:
                # 兼容旧用法：未提供 train_path 时回退到测试集自身统计量。
                mean = sensors.mean(0)
                std = sensors.std(0) + 1e-6

        self.mean = mean
        self.std = std
        # 归一化后 sensors shape: [N_test_rows, 14]
        sensors = (sensors - self.mean) / self.std

        self.X, self.y = [], []

        for uid in np.unique(unit_ids):
            idx = np.where(unit_ids == uid)[0]
            if len(idx) < window_size:
                continue

            # 测试阶段每台发动机仅取最后一个窗口:
            # X_u = sensors[last-window_size+1 : last] -> [window_size, 14]
            # y_u = rul_last[uid-1]                    -> scalar
            x = sensors[idx[-window_size:]]
            self.X.append(x)
            self.y.append(rul_last[uid - 1])

        self.X = torch.from_numpy(np.array(self.X)).float()
        self.y = torch.tensor(self.y, dtype=torch.float32)
        # self.X shape: [num_units_kept, window_size, 14]
        # self.y shape: [num_units_kept]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
