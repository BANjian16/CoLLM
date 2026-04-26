import numpy as np
import torch
from torch.utils.data import Dataset


SENSOR_IDX = [
    1, 2, 3, 6, 7, 8,
    10, 11, 12, 13, 14, 16,
    19, 20,
]


def load_cmapss_sensors(data_path):
    data = np.loadtxt(data_path)
    unit_ids = data[:, 0].astype(int)
    cycles = data[:, 1]
    # CMAPSS columns are: unit, cycle, 3 operating settings, sensors 1..21.
    # SENSOR_IDX is the zero-based offset of the retained sensors, so 1 maps to sensor 2.
    sensors = data[:, [5 + i for i in SENSOR_IDX]]
    return unit_ids, cycles, sensors


def compute_sensor_stats(data_path):
    _, _, sensors = load_cmapss_sensors(data_path)
    return sensors.mean(0), sensors.std(0) + 1e-6


class CMAPSSDataset(Dataset):
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

        sensors = (sensors - self.sensor_mean) / (self.sensor_std + 1e-6)

        self.X, self.y, self.sample_unit_ids = [], [], []
        for uid in np.unique(unit_ids):
            if valid_unit_ids is not None and int(uid) not in valid_unit_ids:
                continue
            idx = np.where(unit_ids == uid)[0]
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
