import numpy as np
import torch
from torch.utils.data import Dataset


class CMAPSSDataset(Dataset):
    def __init__(self, data_path, window_size=50, stride=1):
        # CMAPSS 训练集预处理。
        # 每个样本由一个定长滑动窗口构成，标签取该窗口最后一个时刻对应的 RUL。
        # 这与论文中的时间序列滑窗建样本方式一致。
        data = np.loadtxt(data_path)
        unit_ids = data[:, 0].astype(int)
        cycles = data[:, 1]
        # 按论文和 RUL 任务常见设置，仅保留 14 个信息量较高的传感器。
        # 其余传感器通常被认为常量过多或与退化相关性较弱。
        SENSOR_IDX = [
            1, 2, 3, 6, 7, 8,
            10, 11, 12, 13, 14, 16,
            19, 20
        ]

        # 原始文件前几列是 engine id、cycle 和工况变量，
        # 真实传感器列需要在此基础上偏移后取出。
        sensors = data[:, [5 + i for i in SENSOR_IDX]]

        rul = []
        for uid in np.unique(unit_ids):
            idx = unit_ids == uid
            # 训练集是完整的 run-to-failure 轨迹，
            # 因此可直接用“当前周期到该发动机最后周期的距离”计算 RUL。
            max_cycle = cycles[idx].max()
            rul.extend(max_cycle - cycles[idx])
        rul = np.array(rul)

        # 按特征维做标准化，缓解不同传感器量纲不一致的问题。
        sensors = (sensors - sensors.mean(0)) / (sensors.std(0) + 1e-6)

        self.X, self.y = [], []
        for uid in np.unique(unit_ids):
            idx = np.where(unit_ids == uid)[0]
            for i in range(0, len(idx) - window_size + 1, stride):
                # 论文采用滑动窗口切分时间序列：
                # X 是长度为 window_size 的多变量序列片段，
                # y 是窗口最后一个时间步所对齐的 RUL 标签。
                self.X.append(sensors[idx[i:i+window_size]])
                self.y.append(rul[idx[i+window_size-1]])

        self.X = torch.from_numpy(np.array(self.X)).float()
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
