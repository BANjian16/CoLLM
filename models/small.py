import torch.nn as nn


class SmallModel(nn.Module):
    def __init__(self, input_dim=14, emb_dim=32, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.regressor = nn.Linear(emb_dim, 1)

    def forward(self, x):
        phi = self.encoder(self.embed(x))
        y = self.regressor(phi.mean(1))
        return y.squeeze(-1), phi
