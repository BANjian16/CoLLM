import torch
import torch.nn as nn


class SelfReflection(nn.Module):
    """
    [Paper Sec 3.C] Self-Reflection Mechanism (R) — 大模型输出可靠性评估
    
    论文描述：
        Self-Reflection 机制用于评估 Large Model 输出的可信度（reliability），
        缓解大模型在面对边缘场景或分布外样本时可能产生的不可靠输出。
        该模块基于全连接网络（FCN），根据 Large Model 的深层特征 φl(x)
        计算其预测的置信度 Ql ∈ [0,1]。
        
        训练阶段（对应论文 Eq. (13), (14)）：
        - 置信度标签由大模型的预测误差生成：
              Q*_l = 1 - tanh( |yl - y*_l| / α )
          其中 y*_l 为真实标签，α > 0 为控制误差敏感度的缩放系数。
        - 训练目标为 MSE 损失：L_loss = (1/N) Σ (Ql - Q*_l)^2
        
        推理阶段协作逻辑（详见 collm.py / 论文 Eq. (6), (7)）：
        - 计算置信度差 Δ = Qs - Ql
        - 若 Δ ≤ τ2：大模型自信，接受 yl
        - 若 Δ > τ2：大模型不自信，触发 SM-aided 融合预测
          y_final = G(ys, yl; Δ) = (ys + yl) / 2
          
    架构映射（对应论文 Eq. (5), (12)）：
        特征 phi_l → Flatten → FC → Sigmoid → 可靠性分数 Ql
        Ql = R(φl(x); θr)
        其中 θr 为 Self-Reflection 模型的可学习参数。
    """
    def __init__(self, feature_dim, T):
        """
        Args:
            feature_dim (int): Large Model 输出特征的维度 dl（对应 GPT-2 的 hidden_size，默认 768）
            T (int): 时间序列长度（seq_len 或 patch 数量），用于计算展平后的总维度
        """
        super().__init__()
        # [Paper Sec 3.C, Eq. (12)] 单层全连接投影，将 LM 特征映射到置信度标量
        # 输入维度 feature_dim * T：将时序特征沿时间维度展平为一维向量
        self.fc = nn.Linear(feature_dim * T, 1)
    
    def forward(self, phi):
        """
        Args:
            phi (Tensor): Large Model (GPT2TimeSeries) 输出的特征 φl(x)
                # phi shape: [batch_size, seq_len_or_patches, feature_dim]
                #   即 GPT-2 的 last_hidden_state，对应论文 Eq. (5) 中的 φl(x) ∈ R^{t×dl}
        
        Returns:
            Ql (Tensor): Large Model 输出的可靠性分数（越接近 1 表示越可靠）
                # Ql shape: [batch_size]
                #   取值范围 (0, 1)，由 Sigmoid 激活函数约束
                #   对应论文 Eq. (12) 中的 Ql = R(φl(x))
        """
        # [Paper Sec 3.C, Eq. (12)] Step 1: 沿时间维度展平为静态 1-D 特征向量
        # vec(phi_l) 对应论文中将时序特征压缩为样本级描述向量。
        # flatten(1) shape: [batch_size, seq_len_or_patches * feature_dim]
        
        # [Paper Sec 3.C, Eq. (12)] Step 2: 单层全连接层映射到标量
        # logits = W_r * vec(phi_l) + b_r
        # fc 输出 shape: [batch_size, 1]
        
        # [Paper Sec 3.C, Eq. (12)] Step 3: Sigmoid 激活得到归一化可靠性分数
        # Ql = sigmoid(logits)
        # squeeze(-1) shape: [batch_size]
        return torch.sigmoid(self.fc(phi.flatten(1))).squeeze(-1)
