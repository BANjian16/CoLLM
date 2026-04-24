import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config


class GPT2TimeSeries(nn.Module):
    """
    [Paper Sec 3.D.1)] Large Model (LM) — 基于 GPT-2 的工业时间序列大模型
    
    论文描述：
        Large Model 利用预训练大语言模型（GPT-2）强大的表征能力，处理 Small Model
        难以应对的高不确定性（high-uncertainty）或复杂边缘样本。
        为避免从头训练大模型，本文采用"冻结主干 + 轻量适配头"的策略：
        保留预训练阶段学到的自注意力与 FFN 模块，仅重新设计输入层（Patch Embedding）
        并训练投影层（proj）和回归头（head），大幅降低训练成本。
        
    [AI补全/优化] 重要提示：
        当前代码使用 GPT2Model(config) 基于默认配置从头初始化（随机权重），
        并未加载 GPT-2 的预训练权重。论文明确提到"pre-trained GPT-2"及
        "retains the self-attention mechanism and feed-forward neural network 
        modules learned during pretraining"。若需严格复现论文，建议改为：
            self.gpt = GPT2Model.from_pretrained('gpt2')
        但当前实现因"完全离线"需求而采用随机初始化，这在工程上仍可运行，
        只是表征能力会弱于加载预训练权重的版本。
        
    架构映射（对应论文 Sec 3.D.1 及 Eq. (4)）：
        输入时间序列 x → Patch Embedding → GPT-2 (frozen) → Mean Pooling → Regressor
        yl = L(x; θl)
        其中 θl 为 LM 的可学习参数（实际仅 proj 和 head 参与训练）。
    """
    def __init__(self, input_dim=14, patch_size=4, use_pretrained=True, pretrained_name='gpt2'):
        """
        Args:
            input_dim (int): 每个时间步的传感器维度（默认 14，与 Small Model 对齐）
            patch_size (int): Patch 长度，将连续 patch_size 个时间步拼接为一个 Patch Token
                # 对应论文中"将时间序列切分为 Patch，映射到高维向量，形成输入嵌入"
        """
        super().__init__()
        self.patch = patch_size
        
        # [Paper Sec 3.D.1)] 完全离线的 GPT-2 配置（不联网下载，使用默认 config）
        # 注意：默认 GPT2Config() 的 hidden_size=768, n_layer=12, n_head=12
        # 若显存受限，建议显式传入更小的 config（如 n_layer=4, n_embd=256）
        config = GPT2Config()

        # [AI补全/优化] 论文设定为预训练 GPT-2。
        # 当 use_pretrained=True 时，若加载失败则直接报错，避免静默回退随机初始化。
        if use_pretrained:
            try:
                self.gpt = GPT2Model.from_pretrained(pretrained_name, local_files_only=False)
            except Exception as e:
                raise RuntimeError(
                    'Failed to load pre-trained GPT-2. The paper relies on pre-trained '
                    'knowledge. Please ensure weights are downloaded or path is correct. '
                    'Do not use random initialization for the large model.'
                ) from e
        else:
            self.gpt = GPT2Model(config)
        
        # [Paper Sec 3.D.1)] 冻结 GPT-2 全部参数，仅训练后续适配层
        # 这是论文提出的参数冻结策略，防止预训练知识被覆盖（若加载了预训练权重）
        for p in self.gpt.parameters():
            p.requires_grad = False
        
        # [Paper Sec 3.D.1)] 输入投影层：将 Patch 展平后的向量映射到 GPT-2 的 hidden_size
        # h = GPT-2 的隐层维度（默认 768），对应论文中的 dl
        h = self.gpt.config.hidden_size
        self.proj = nn.Linear(input_dim * patch_size, h)
        
        # [Paper Sec 3.D.1)] 回归头：将 GPT-2 输出的序列表示映射到 RUL 预测值
        self.head = nn.Linear(h, 1)

    def train(self, mode=True):
        super().train(mode)
        # 强制冻结主干模型的 Dropout 和 BatchNorm（如果有）
        if hasattr(self, 'gpt'):
            self.gpt.eval()
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入时间序列，采样自样本空间 X ⊂ R^{t×d}
                # x shape: [batch_size, seq_len, input_dim]
                #   若 seq_len 不能被 patch_size 整除，将在时间维头部做零填充
        
        Returns:
            yl (Tensor): RUL 预测值
                # yl shape: [batch_size]
                #   对应论文 Eq. (4) 中的大模型输出 yl
            phi_l (Tensor): GPT-2 最后一层隐状态，作为 Large Model 的特征表示
                # phi_l shape: [batch_size, num_patches, hidden_size]
                #   对应论文 Eq. (5) 中的 φl(x) ∈ R^{t×dl}
                #   该特征将送入 Self-Reflection (R) 评估大模型输出的可靠性
        """
        B, T, d = x.shape  # B: batch_size, T: seq_len, d: input_dim
        # 约束: 至少要有一个时间步；不足一个 patch 时会通过左侧 padding 补齐。

        # [AI补全/优化] 防止尾部时间步被截断：
        # 当 T % patch != 0 时，在时间维头部（左侧）补 0，使长度可整除 patch。
        # F.pad 对 3D 张量 [B, T, d] 的参数顺序为 (d_left, d_right, T_left, T_right)。
        remainder = T % self.patch
        if remainder != 0:
            pad_len = self.patch - remainder
            # x shape: [B, T + pad_len, d]
            x = F.pad(x, (0, 0, pad_len, 0), mode='constant', value=0.0)
            T = x.shape[1]
        
        # [Paper Sec 3.D.1)] Step 1: Patch Embedding
        # [Paper Eq. (4) 的输入重编程] 将连续 patch 的原始信号展平为 token:
        #   p_k = vec(x[:, k:k+patch, :])
        #   p_k shape: [B, patch_size * d]
        # 将时间序列切分为不重叠的 Patch，每个 Patch 包含 patch_size 个时间步。
        # 由于上一步已补齐到可整除长度，这里不会丢失尾部真实时间步。
        patches = []
        for i in range(0, T, self.patch):
            # 每个 patch shape: [batch_size, patch_size, input_dim]
            # reshape 后 shape: [batch_size, patch_size * input_dim]
            patches.append(x[:, i:i + self.patch, :].reshape(B, -1))
        
        # patches shape: [batch_size, num_patches, patch_size * input_dim]
        #   其中 num_patches = T // patch_size（整除情况下）
        patches = torch.stack(patches, 1)
        
        # [Paper Sec 3.D.1)] Step 2: 投影到 GPT-2 的嵌入空间
        # [Paper Eq. (4)] e_k = W_p p_k + b_p
        # emb shape: [batch_size, num_patches, hidden_size]
        emb = self.proj(patches)
        
        # [Paper Sec 3.D.1)] Step 3: 送入冻结的 GPT-2 提取深层特征
        # [Paper Eq. (5)] phi_l = GPT2(e)
        # phi_l shape: [batch_size, num_patches, hidden_size]
        phi_l = self.gpt(inputs_embeds=emb).last_hidden_state
        
        # [Paper Sec 3.D.1)] Step 4: 对 Patch 维度做 Mean Pooling，得到序列级表示
        # [Paper Eq. (4)] h_l = (1/K) * sum_k phi_l[:, k, :]
        # [Paper Eq. (4)] yl = W_h h_l + b_h
        # 然后通过回归头输出 RUL
        # yl shape: [batch_size, 1] → squeeze → [batch_size]
        yl = self.head(phi_l.mean(1))
        return yl.squeeze(-1), phi_l
