import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneFitsAllTimeSeries(nn.Module):
    """基于 GPT-2 的 One Fits All 时间序列模型。

    论文思路可以理解为：GPT-2 原本只会处理文字 token，这里先把一段传感器
    时间序列切成若干个 patch，再把每个 patch 映射成和 GPT-2 词向量同维度的
    “伪 token”。这样就可以把时间序列喂给 GPT-2 的 Transformer 层。

    默认训练策略接近 One Fits All：大部分 GPT-2 参数冻结，只训练时间序列适配层、
    少量归一化/位置参数和最后的 RUL 回归头，避免小数据集把大模型完全训坏。
    """

    def __init__(
        self,
        input_dim=14,
        patch_size=4,
        pretrained_name="gpt2",
        use_pretrained=True,
        local_files_only=True,
        freeze_backbone=True,
        gpt_layers=6,
        num_patches=13,
        tune_layer_norm=True,
        tune_position_embeddings=True,
        pretrained_ckpt=None,
    ):
        super().__init__()
        try:
            from transformers import GPT2Config, GPT2Model
        except ImportError as exc:
            raise ImportError("OneFitsAllTimeSeries requires the transformers package.") from exc

        self.patch = patch_size
        self.freeze_backbone = freeze_backbone

        # 优先加载本地 HuggingFace 缓存里的 GPT-2 预训练权重。
        # 如果 use_pretrained=False，则会创建随机初始化的 GPT-2；严格复现时不使用它。
        if use_pretrained:
            self.backbone = GPT2Model.from_pretrained(
                pretrained_name,
                local_files_only=local_files_only,
            )
        else:
            self.backbone = GPT2Model(GPT2Config())

        # 原始 GPT-2 有 12 层。论文和 One Fits All 类方法常只取前几层参与适配，
        # 这样显存更省，也能减少过拟合。这里默认取前 6 层。
        if gpt_layers is not None:
            if gpt_layers <= 0 or gpt_layers > len(self.backbone.h):
                raise ValueError(f"gpt_layers must be in [1, {len(self.backbone.h)}], got {gpt_layers}")
            self.backbone.h = nn.ModuleList(list(self.backbone.h[:gpt_layers]))
            self.backbone.config.n_layer = gpt_layers

        self.hidden_size = self.backbone.config.hidden_size
        self.num_patches = num_patches
        # 每个 patch 的原始长度是 patch_size * input_dim。
        # 例如 window=50、patch=4、input_dim=14，则一个 patch 展平后是 56 维。
        # proj 把 56 维映射到 GPT-2 hidden_size 维，作为 GPT-2 的输入 embedding。
        self.proj = nn.Linear(input_dim * patch_size, self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        # GPT-2 输出每个 patch 的隐表示。这里把所有 patch 的表示拼起来，
        # 用一个线性层回归出当前窗口末端的 RUL。
        self.head = nn.Linear(self.hidden_size * num_patches, 1)

        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                # 只放开 LayerNorm 和位置嵌入，让 GPT-2 能稍微适应时间序列分布。
                if tune_layer_norm and ".ln_" in name:
                    param.requires_grad = True
                if tune_position_embeddings and name.startswith("wpe."):
                    param.requires_grad = True

        if pretrained_ckpt is not None:
            if not os.path.exists(pretrained_ckpt):
                raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt}")
            state = torch.load(pretrained_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.load_state_dict(state, strict=False)

    def forward(self, x):
        # x: [batch, window_size, input_dim]
        # patches: [batch, num_patches, patch_size * input_dim]
        patches = patchify(x, self.patch)
        # emb: [batch, num_patches, GPT2_hidden_size]
        # 注意这里不用 GPT-2 自己的 token embedding，因为输入不是文字 token。
        emb = self.input_norm(self.proj(patches))
        # inputs_embeds 表示“我已经准备好 embedding 了”，GPT-2 直接从 Transformer 层开始算。
        phi_l = self.backbone(inputs_embeds=emb).last_hidden_state
        if phi_l.shape[1] != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {phi_l.shape[1]}")
        y_l = self.head(phi_l.flatten(1)).squeeze(-1)
        return y_l, phi_l


def patchify(x, patch_size):
    """把连续时间窗口切成不重叠 patch。

    举例：50 个时间步、patch_size=4 时，最后会补 2 个 0，得到 52 个时间步，
    再切成 13 个 patch。补 0 只是为了能整除，不代表真实传感器值。
    """
    batch_size, seq_len, input_dim = x.shape
    remainder = seq_len % patch_size
    if remainder:
        pad_len = patch_size - remainder
        x = F.pad(x, (0, 0, 0, pad_len), mode="constant", value=0.0)
        seq_len = x.shape[1]

    num_patches = seq_len // patch_size
    return x.reshape(batch_size, num_patches, patch_size * input_dim)
