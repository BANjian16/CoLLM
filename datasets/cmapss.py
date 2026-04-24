import numpy as np
import torch
from torch.utils.data import Dataset


SENSOR_IDX = [
    1, 2, 3, 6, 7, 8,
    10, 11, 12, 13, 14, 16,
    19, 20
]


def load_cmapss_sensors(data_path):
    """
    读取 CMAPSS 文本并提取论文常用的 14 个传感器特征。
    返回:
        unit_ids: [N]
        cycles: [N]
        sensors: [N, 14]

    形状定义:
        N: 该文件中所有 engine-cycle 记录总行数
        sensors[n, j]: 第 n 条记录在第 j 个传感器维度的读数
    """
    data = np.loadtxt(data_path)
    unit_ids = data[:, 0].astype(int)
    cycles = data[:, 1]
    # SENSOR_IDX 为 1-based 下标，且 Sensor 1 对应原始数据第 5 列（0-indexed）。
    # 因此列索引应为 4 + i，避免整体错位到下一个传感器。
    sensors = data[:, [4 + i for i in SENSOR_IDX]]
    return unit_ids, cycles, sensors


def compute_sensor_stats(data_path):
    """
    基于训练集计算标准化统计量，用于 train/test 一致归一化。
    """
    _, _, sensors = load_cmapss_sensors(data_path)
    mean = sensors.mean(0)
    std = sensors.std(0) + 1e-6
    return mean, std


class CMAPSSDataset(Dataset):
    def __init__(self, data_path, window_size=50, stride=1, mean=None, std=None, valid_unit_ids=None):
        unit_ids, cycles, sensors = load_cmapss_sensors(data_path)

        if valid_unit_ids is not None:
            # 使用 set 提升 membership 查询效率，确保一个数据集实例仅包含指定发动机。
            valid_unit_ids = {int(uid) for uid in valid_unit_ids}

        # [RUL Label Formula] 分段线性 RUL：
        # 先计算线性 RUL_{u,t} = max_cycle_u - cycle_{u,t}
        # 再做上限截断：RUL_{u,t} = clip(RUL_{u,t}, 0, MAX_RUL)
        # rul shape: [N]
        MAX_RUL = 125
        rul = np.zeros_like(cycles, dtype=np.float32)
        for uid in np.unique(unit_ids):
            idx = unit_ids == uid
            max_cycle = cycles[idx].max()
            unit_rul = np.clip(max_cycle - cycles[idx], a_min=0, a_max=MAX_RUL)
            rul[idx] = unit_rul


        if mean is None or std is None:
            mean = sensors.mean(0)
            std = sensors.std(0) + 1e-6

        self.mean = mean
        self.std = std
        # 标准化: x' = (x - mean) / std
        # sensors shape: [N, 14]
        sensors = (sensors - self.mean) / self.std


        self.X, self.y = [], []
        for uid in np.unique(unit_ids):
            if valid_unit_ids is not None and int(uid) not in valid_unit_ids:
                continue
            idx = np.where(unit_ids == uid)[0]
            for i in range(0, len(idx) - window_size + 1, stride):
                # 滑窗样本:
                # X_k = sensors[t : t+window_size] -> shape [window_size, 14]
                # y_k = rul[t+window_size-1]      -> scalar
                self.X.append(sensors[idx[i:i+window_size]])
                self.y.append(rul[idx[i+window_size-1]])


        self.X = torch.from_numpy(np.array(self.X)).float()
        self.y = torch.tensor(self.y, dtype=torch.float32)
        # self.X shape: [num_windows, window_size, 14]
        # self.y shape: [num_windows]


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
