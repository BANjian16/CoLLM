import torch.nn as nn


class SmallModel(nn.Module):
    """轻量级小模型 S。

    小模型负责快速给出一个初步 RUL 预测。它比 GPT-2 大模型小很多，
    适合处理“比较容易”的样本。论文中的协作思想就是：能让小模型解决的样本，
    就不要浪费成本调用大模型。
    """

    def __init__(self, input_dim=14, emb_dim=32, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        # 把每个时间步的 14 个传感器值映射到 emb_dim 维。
        self.embed = nn.Linear(input_dim, emb_dim)
        # 这里用一个小 Transformer Encoder 捕捉时间窗口内部的变化趋势。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        # 把整段时间序列的平均表示映射成一个 RUL 数值。
        self.regressor = nn.Linear(emb_dim, 1)

    def forward(self, x):
        # x: [batch, window_size, input_dim]
        # phi: [batch, window_size, emb_dim]，是小模型提取出的时序隐特征。
        phi = self.encoder(self.embed(x))
        # mean(1) 表示对时间维求平均，得到整个窗口的综合表示。
        y = self.regressor(phi.mean(1))
        return y.squeeze(-1), phi
