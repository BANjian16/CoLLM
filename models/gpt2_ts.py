import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class GPT2TimeSeries(nn.Module):
    def __init__(
        self,
        input_dim=14,
        patch_size=4,
        pretrained_name="gpt2",
        use_pretrained=True,
        local_files_only=False,
    ):
        super().__init__()
        # 论文中的大模型 L：
        # 通过 patch embedding 将时间序列改写为“token 序列”，
        # 再复用 GPT-2 主干进行深层时序建模。
        self.patch = patch_size

        # 优先加载预训练 GPT-2 主干，这样才更接近论文中“复用大模型能力”的设定。
        # 如果本地没有缓存且无法联网，则回退到随机初始化的 GPT-2 结构，保证代码可运行。
        if use_pretrained:
            try:
                self.gpt = GPT2Model.from_pretrained(
                    pretrained_name,
                    local_files_only=local_files_only,
                )
            except OSError as exc:
                print(
                    f"[GPT2TimeSeries] Cannot load pretrained '{pretrained_name}': {exc}\n"
                    "[GPT2TimeSeries] Falling back to randomly initialized GPT-2 config."
                )
                self.gpt = GPT2Model(GPT2Config())
        else:
            self.gpt = GPT2Model(GPT2Config())

        # 按论文思路冻结 GPT-2 主体参数。
        # 这样做的目的有两点：
        # 1. 保留大模型主干已有的表示能力；
        # 2. 降低训练成本与梯度干扰，只训练轻量映射层与输出头。
        for p in self.gpt.parameters():
            p.requires_grad = False

        h = self.gpt.config.hidden_size
        # 每个 patch 把连续 patch_size 个时间步展开后拼接，
        # 再线性映射到 GPT-2 的隐藏维度，使其可以被当作 token embedding 使用。
        self.proj = nn.Linear(input_dim * patch_size, h)
        # 对输入到 GPT-2 的时间序列 patch embedding 做轻量归一化，
        # 减少传感器数值分布与语言 embedding 分布之间的偏移。
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        B, T, d = x.shape
        patches = []
        # 将时间序列按不重叠窗口切分为多个 patch。
        # 例如论文和本项目常用 T=50、patch_size=4 时，
        # 会得到 12 个完整 patch，末尾不足 4 的时间步会被忽略。
        for i in range(0, T - self.patch + 1, self.patch):
            patches.append(x[:, i:i + self.patch, :].reshape(B, -1))
        patches = torch.stack(patches, 1)

        emb = self.norm(self.proj(patches))
        # 不走词表，而是把 patch embedding 直接作为 inputs_embeds 输入 GPT-2。
        out = self.gpt(inputs_embeds=emb).last_hidden_state
        # out 即论文中的大模型隐表示 φ_l(x)。
        # 这份高维表示会送入自反思模块 R，用于估计大模型置信度 Q_l。
        y = self.head(out.mean(1))
        return y.squeeze(-1), out
