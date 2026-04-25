import torch
import torch.nn as nn


class FuzzyDecisionAgent(nn.Module):
    def __init__(self, feature_dim, T):
        super().__init__()
        # 模糊决策代理 F：
        # 根据小模型输出的隐特征 φ_s(x) 预测置信度 Q_s，
        # 对应论文中“是否需要调用大模型”的样本级路由模块。
        #
        # mu 与 sigma 是高斯隶属函数的可学习参数，
        # 分别对应论文公式中的模糊均值 μ 和模糊方差/尺度 σ。
        self.mu = nn.Parameter(torch.zeros(feature_dim))
        self.sigma = nn.Parameter(torch.ones(feature_dim))
        # 根据论文，在得到模糊特征矩阵 M 后，
        # 将其展平并用“线性层 + Sigmoid”映射为样本级置信度分数。
        self.fc = nn.Linear(feature_dim * T, 1)


    def forward(self, phi):
        # phi 的形状为 [batch, T, feature_dim]，
        # 即论文中的小模型时序隐特征 φ_s(x)。
        mu = self.mu.view(1,1,-1)
        sigma = self.sigma.view(1,1,-1) + 1e-6
        # 论文中的高斯隶属函数：
        # M = exp(-((phi - mu)^2) / sigma^2)
        # 这里对每个时间步、每个特征维独立计算其模糊隶属度。
        M = torch.exp(-((phi - mu) ** 2) / sigma ** 2)
        # 输出 Q_s ∈ [0, 1]。
        # Q_s 越高，说明当前样本越适合直接采用小模型预测，
        # 也就是越有可能在第一阶段提前退出而不调用大模型。
        return torch.sigmoid(self.fc(M.flatten(1))).squeeze(-1)
