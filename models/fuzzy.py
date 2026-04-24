import torch
import torch.nn as nn


class FuzzyDecisionAgent(nn.Module):
    """
    [Paper Sec 3.B] Fuzzy Decision-Making Agent (F) — 基于模糊神经网络的不确定性量化
    
    论文描述：
        Fuzzy Decision Agent 是 CoLLM 框架的"路由大脑"，其核心任务是为 Small Model
        生成置信度分数 Qs ∈ [0,1]，从而决定该样本应由 Small Model 直接输出，还是
        需要激活 Large Model 进行深度推理。
        
        本文采用基于高斯隶属函数（Gaussian Membership Function）的模糊神经网络（FNN）：
            fd(z_t^d) = exp( -(z_t^d - μ_d)^2 / σ_d^2 )
        其中 μ_d 和 σ_d 分别为第 d 维特征的可学习模糊均值与模糊标准差。
        注意：论文文字描述称"fuzzy variance σ^2"为可学习参数（Page 5, 右侧栏），
        但公式 (8) 分母为 σ_d^2。本代码将参数视为标准差 σ，通过 σ^2 计算分母，
        数学上与论文公式等价，仅是参数化方式不同。
        
    架构映射（对应论文 Eq. (8), (9), (11)）：
        特征 phi_s → Gaussian Fuzzification → 模糊特征矩阵 M → Flatten → FC → Sigmoid → Qs
        
        输出 Qs ∈ (0,1) 表示 Small Model 对该样本的置信度：
        - Qs 高（≥ τ1）→ Small Model 足够自信，直接输出 ys
        - Qs 低（< τ1）→ Small Model 不自信，激活 Large Model L
    """
    def __init__(self, feature_dim, T, num_rules=2):
        """
        Args:
            feature_dim (int): 输入特征的维度 ds（对应 Small Model 的 emb_dim）
            T (int): 时间序列长度 t（seq_len），用于计算展平后的总维度 feature_dim * T
            num_rules (int): 每个特征维度对应的高斯隶属函数数量
                例如 d_ast=32 且总隶属函数数=64 时，num_rules=2
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_rules = num_rules

        # [Paper Sec 3.B, Eq. (8)] 可学习的高斯模糊均值（fuzzy mean）μ_d
        # 扩展为每个特征维度对应 num_rules 个不同的隶属函数分布
        # mu shape: [feature_dim, num_rules]，初始化为 0
        self.mu = nn.Parameter(torch.zeros(feature_dim, num_rules))
        
        # [Paper Sec 3.B, Eq. (8)] 可学习的高斯模糊标准差（fuzzy spread）σ_d
        # 扩展为每个特征维度的每条规则都拥有独立的 sigma
        # sigma shape: [feature_dim, num_rules]，初始化为 1
        self.sigma = nn.Parameter(torch.ones(feature_dim, num_rules))
        
        # [Paper Sec 3.B, Eq. (11)] 置信度预测层：
        # Qs = σ( W · flatten(M) + b )
        # 输入维度为 feature_dim * num_rules * T（将规则维也展平）
        self.fc = nn.Linear(feature_dim * num_rules * T, 1)

    def forward(self, phi):
        """
        Args:
            phi (Tensor): Small Model 输出的时序特征 phi_s(x)
                # phi shape: [batch_size, seq_len, feature_dim]
                #   即 [B, T, ds]，对应论文 Eq. (2) 中的 φs(x) ∈ R^{t×ds}
        
        Returns:
            Qs (Tensor): Small Model 的置信度分数（越接近 1 表示越自信）
                # Qs shape: [batch_size]
                #   取值范围 (0, 1)，由 Sigmoid 激活函数 σ(·) 约束
                #   对应论文 Eq. (11) 中的置信度预测输出
        """
        # [Paper Sec 3.B] Step 1: 调整参数形状以支持广播运算
        # 记 phi[n, t, d] 为第 n 个样本在时刻 t 的第 d 维特征。
        # mu[0,0,d,r], sigma[0,0,d,r] 与 batch/time 位置做广播匹配。
        # mu shape: [1, 1, feature_dim, num_rules]
        # sigma shape: [1, 1, feature_dim, num_rules]
        mu = self.mu.view(1, 1, self.feature_dim, self.num_rules)
        sigma = self.sigma.view(1, 1, self.feature_dim, self.num_rules) + 1e-6  # 加 epsilon 防止除零
        phi_expanded = phi.unsqueeze(-1)  # [batch_size, seq_len, feature_dim, 1]
        
        # [Paper Sec 3.B, Eq. (8)] Step 2: 逐元素高斯模糊化
        # fd(z_t^d) = exp( -(z_t^d - μ_d)^2 / σ_d^2 )
        # 扩展后: M[n,t,d,r] = exp(-((phi[n,t,d]-mu[d,r])^2)/(sigma[d,r]^2))
        # M shape: [batch_size, seq_len, feature_dim, num_rules]
        #   即论文 Eq. (9) 中的模糊特征矩阵 M
        #   每个元素表示对应特征维度在某条规则下的隶属度（0~1之间）
        M = torch.exp(-((phi_expanded - mu) ** 2) / sigma ** 2)
        
        # [Paper Sec 3.B, Eq. (11)] Step 3: 去模糊化（Defuzzification）
        # Qs = sigmoid(W * vec(M) + b)
        # 将模糊特征矩阵 M 展平为一维向量，通过全连接层聚合所有维度信息
        # flatten(1) shape: [batch_size, seq_len * feature_dim * num_rules]
        # fc 输出 shape: [batch_size, 1]
        # sigmoid 将输出压缩到 (0, 1) 区间，作为归一化的置信度分数
        # squeeze(-1) shape: [batch_size]
        return torch.sigmoid(self.fc(M.flatten(1))).squeeze(-1)
