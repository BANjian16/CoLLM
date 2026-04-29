import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyDecisionAgent(nn.Module):
    """模糊决策代理 F。

    它的作用不是直接预测 RUL，而是根据小模型的隐表示 phi_s 判断：
    “这个样本小模型自己有没有把握？”

    输出 q_s 位于 [0, 1]：
    - q_s 越大，表示越相信小模型，可以直接用小模型结果，节省调用大模型的成本。
    - q_s 越小，表示小模型不确定，需要继续调用大模型或进行融合。
    """

    def __init__(self, feature_dim, T):
        super().__init__()
        # mu 和 sigma 是可学习的高斯隶属函数参数。
        # 直觉上：模型会学习“什么样的隐藏特征看起来可靠”。
        self.mu = nn.Parameter(torch.zeros(feature_dim))
        self.sigma = nn.Parameter(torch.ones(feature_dim))

        # 每个时间步、每个特征都会算一个隶属度 M。
        # 展平成 feature_dim * T 后，用线性层压成一个置信度分数。
        self.fc = nn.Linear(feature_dim * T, 1)

    def forward(self, phi):
        # phi: [batch, T, feature_dim]
        # T 对小模型来说是时间窗口长度，例如 50。
        mu = self.mu.view(1, 1, -1)
        sigma = F.softplus(self.sigma).view(1, 1, -1) + 1e-6

        # 高斯隶属函数：
        # M 越接近 1，说明该隐藏特征越接近模型学到的“可靠中心” mu。
        M = torch.exp(-((phi - mu) ** 2) / sigma ** 2)

        # sigmoid 把任意实数压到 0~1，作为小模型置信度 q_s。
        return torch.sigmoid(self.fc(M.flatten(1))).squeeze(-1)
