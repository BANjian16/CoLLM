import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneFitsAllTimeSeries(nn.Module):
    """
    One Fits All 风格的大模型时序封装。

    说明:
        该实现保持与项目现有 LM 接口兼容:
        - forward(x) -> (yl, phi_l)
        - yl shape: [B]
        - phi_l shape: [B, num_patches, hidden_size]

        同时提供 backbone 冻结与可选预训练权重加载能力，便于在
        CoLLM 的三阶段训练流程中直接替换原 GPT2TimeSeries。
    """

    def __init__(
        self,
        input_dim=14,
        patch_size=4,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        freeze_backbone=True,
        pretrained_ckpt=None,
    ):
        super().__init__()
        self.patch = patch_size
        self.hidden_size = d_model
        self.freeze_backbone = freeze_backbone

        self.proj = nn.Linear(input_dim * patch_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

        if pretrained_ckpt is not None:
            if not os.path.exists(pretrained_ckpt):
                raise FileNotFoundError(f'Pretrained checkpoint not found: {pretrained_ckpt}')
            state = torch.load(pretrained_ckpt, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.load_state_dict(state, strict=False)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone and hasattr(self, 'backbone'):
            self.backbone.eval()
        return self

    def forward(self, x):
        # x shape: [B, T, d]
        bsz, seq_len, _ = x.shape

        remainder = seq_len % self.patch
        if remainder != 0:
            pad_len = self.patch - remainder
            x = F.pad(x, (0, 0, pad_len, 0), mode='constant', value=0.0)
            seq_len = x.shape[1]

        num_patches = seq_len // self.patch
        patches = x.reshape(bsz, num_patches, self.patch * x.shape[-1])

        emb = self.input_norm(self.proj(patches))
        phi_l = self.backbone(emb)
        yl = self.head(phi_l.mean(dim=1)).squeeze(-1)
        return yl, phi_l
