import torch
import torch.nn as nn


class SmallModel(nn.Module):
    def __init__(self, input_dim=14, emb_dim=32, hidden_dim=64, n_layers=2, max_len=512):
        super().__init__()
        # 论文中的小模型 S：
        # 负责协同框架中的第一阶段快速推理，用较低计算成本先处理“简单样本”。
        # input_dim=14 对应论文在 CMAPSS 上保留的 14 个有效传感器维度。
        self.embed = nn.Linear(input_dim, emb_dim)
        # Transformer 本身不自动知道时间步顺序。
        # 对 RUL 这类退化时间序列任务，顺序信息非常关键，因此加入可学习位置嵌入。
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        enc = nn.TransformerEncoderLayer(emb_dim, 4, hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        # 回归头：把时序特征聚合后的表示映射为单个 RUL 预测值。
        self.regressor = nn.Linear(emb_dim, 1)


    def forward(self, x):
        # x 的形状为 [batch, time, sensor_dim]，
        # 对应论文中工业时间序列输入 x ∈ R^(t×d)。
        z = self.embed(x)
        z = z + self.pos_embed[:, :z.size(1), :]
        # 先将原始传感器观测投影到较低维嵌入空间，便于后续 Transformer 编码。
        phi = self.encoder(z)
        # phi 即论文中的小模型隐表示 φ_s(x)。
        # 这份表示同时承担两项职责：
        # 1. 经过池化后用于 RUL 回归；
        # 2. 送入模糊决策代理 F，估计小模型预测置信度 Q_s。
        y = self.regressor(phi.mean(1))
        # 这里对时间维做均值池化，得到序列级摘要表示，再输出最终预测。
        return y.squeeze(-1), phi
