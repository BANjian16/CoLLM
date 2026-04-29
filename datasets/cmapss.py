import numpy as np
import torch
from torch.utils.data import Dataset


SENSOR_IDX = [
    1, 2, 3, 6, 7, 8,
    10, 11, 12, 13, 14, 16,
    19, 20,
]


def load_cmapss_sensors(data_path):
    """读取 CMAPSS 原始 txt，并只保留论文常用的 14 个传感器通道。"""
    data = np.loadtxt(data_path)
    unit_ids = data[:, 0].astype(int)
    cycles = data[:, 1]
    # CMAPSS columns are: unit, cycle, 3 operating settings, sensors 1..21.
    # SENSOR_IDX is the zero-based offset of the retained sensors, so 1 maps to sensor 2.
    sensors = data[:, [5 + i for i in SENSOR_IDX]]
    return unit_ids, cycles, sensors


def compute_sensor_stats(data_path):
    """计算训练集传感器均值和标准差，用于标准化。"""
    _, _, sensors = load_cmapss_sensors(data_path)
    return sensors.mean(0), sensors.std(0) + 1e-6


class CMAPSSDataset(Dataset):
    """训练集 Dataset。

    CMAPSS 的每台发动机是一条从健康到失效的时间序列。训练时不能把整条序列
    一次性输入模型，所以这里用滑动窗口切成很多样本：

    X: 最近 window_size 个时间步的传感器数据
    y: 这个窗口最后一个时间步对应的 RUL
    """

    def __init__(
        self,
        data_path,
        window_size=50,
        stride=1,
        sensor_mean=None,
        sensor_std=None,
        mean=None,
        std=None,
        valid_unit_ids=None,
        max_rul=125,
    ):
        unit_ids, cycles, sensors = load_cmapss_sensors(data_path)
        if valid_unit_ids is not None:
            valid_unit_ids = {int(uid) for uid in valid_unit_ids}

        # 训练集中知道每台发动机什么时候失效。
        # RUL = 最后周期 - 当前周期，并截断到 max_rul，避免早期健康阶段 RUL 过大。
        rul = np.zeros_like(cycles, dtype=np.float32)
        for uid in np.unique(unit_ids):
            idx = unit_ids == uid
            unit_rul = cycles[idx].max() - cycles[idx]
            rul[idx] = np.clip(unit_rul, 0, max_rul)

        if mean is not None:
            sensor_mean = mean
        if std is not None:
            sensor_std = std
        self.sensor_mean = sensors.mean(0) if sensor_mean is None else np.asarray(sensor_mean)
        self.sensor_std = (sensors.std(0) + 1e-6) if sensor_std is None else np.asarray(sensor_std)
        self.mean = self.sensor_mean
        self.std = self.sensor_std

        # 标准化让不同传感器落到相近数值范围，神经网络更容易训练。
        sensors = (sensors - self.sensor_mean) / (self.sensor_std + 1e-6)

        self.X, self.y, self.sample_unit_ids = [], [], []
        for uid in np.unique(unit_ids):
            if valid_unit_ids is not None and int(uid) not in valid_unit_ids:
                continue
            idx = np.where(unit_ids == uid)[0]
            # 滑动窗口。例如 window=50、stride=1 时，第 1~50 个周期组成一个样本，
            # 第 2~51 个周期组成下一个样本。
            for i in range(0, len(idx) - window_size + 1, stride):
                self.X.append(sensors[idx[i:i + window_size]])
                self.y.append(rul[idx[i + window_size - 1]])
                self.sample_unit_ids.append(uid)

        self.X = torch.from_numpy(np.asarray(self.X)).float()
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.sample_unit_ids = np.asarray(self.sample_unit_ids)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
