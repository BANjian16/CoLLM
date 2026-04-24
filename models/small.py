import torch
import torch.nn as nn


class SmallModel(nn.Module):
    """
    [Paper Sec 3.A.1)] Small Model (SM) — 轻量级工业时间序列回归模型
    
    论文描述：
        Small Model 作为 CoLLM 框架中的"快路径"，负责处理低不确定性（low-uncertainty）
        的样本。其设计目标是计算效率高、参数量小，在边缘设备上可快速推理。
        该模型首先对输入进行快速特征提取与置信度评估，从而过滤掉简单样本，
        减少 Large Model 的调用频率。
        
    架构映射（对应论文 Eq. (1)）：
        输入时间序列 x → Linear Embedding → Transformer Encoder → Mean Pooling → Regressor
        S : X → Y,   ys = S(x; θs)
        其中 θs 为 SM 的可学习参数，ys 为回归任务中的 RUL 预测值。
    """
    def __init__(self, input_dim=14, emb_dim=32, hidden_dim=64, n_layers=2):
        """
        Args:
            input_dim (int): 输入传感器维度（CMAPSS 经特征选择后默认为 14）
            emb_dim (int): 嵌入维度，对应论文中轻量模型的特征投影维度 ds
            hidden_dim (int): Transformer Feed-Forward 层维度
            n_layers (int): Transformer Encoder 层数，控制模型容量与速度的平衡
        """
        super().__init__()
        # [Paper Sec 3.A.1)] 将原始传感器信号投影到 emb_dim 维嵌入空间
        # 对应特征提取器 φs 的第一层线性变换
        self.embed = nn.Linear(input_dim, emb_dim)
        
        # [Paper Sec 3.A.1)] 使用 Transformer Encoder 捕获时序依赖关系
        # nhead=4 表示 4 头自注意力，hidden_dim 为 FFN 中间层维度
        enc = nn.TransformerEncoderLayer(emb_dim, 4, hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        
        # [Paper Sec 3.A.1), Eq. (1)] 回归头：将 pooled 特征映射到 RUL 标量
        self.regressor = nn.Linear(emb_dim, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入时间序列，采样自样本空间 X ⊂ R^{t×d}
                # x shape: [batch_size, seq_len, input_dim]
                #   batch_size: 样本批次大小 N
                #   seq_len: 时间窗口长度 t（如 CMAPSS 中的滑动窗口 T=50）
                #   input_dim: 传感器特征维度 d（默认 14）
        
        Returns:
            ys (Tensor): RUL 预测值（标量，已 squeeze）
                # ys shape: [batch_size]
                #   对应论文 Eq. (1) 中的回归输出 ys = S(x; θs)
            phi_s (Tensor): Small Model 提取的时序特征（Encoder 输出）
                # phi_s shape: [batch_size, seq_len, emb_dim]
                #   对应论文 Eq. (2) 中的特征表示 φs(x)，将送入 Fuzzy Decision Agent (F)
        """
        # [Paper Eq. (1) 的特征前半段] z_t = W_e x_t + b_e
        # Step 1: 线性嵌入，将每个时间步的原始特征映射到 emb_dim
        # z shape: [batch_size, seq_len, emb_dim]
        z = self.embed(x)
        
        # [Paper Eq. (2)] phi_s = Encoder(z)
        # Step 2: Transformer Encoder 提取上下文特征
        # phi_s shape: [batch_size, seq_len, emb_dim]
        #   对应论文中的特征提取器 φs : X → R^{t×ds}
        phi_s = self.encoder(z)
        
        # [Paper Eq. (1) 的聚合段] h_s = (1/T) * sum_t phi_s[:, t, :]
        # Step 3: 时间维度 Mean Pooling，聚合全局时序信息
        # pooled shape: [batch_size, emb_dim]
        # 对应论文中"将时序特征聚合为样本级表示"
        pooled = phi_s.mean(1)
        
        # [Paper Eq. (1) 的回归段] ys = W_r h_s + b_r
        # Step 4: 回归输出 RUL 预测值
        # ys shape: [batch_size, 1] → squeeze → [batch_size]
        ys = self.regressor(pooled)
        return ys.squeeze(-1), phi_s
