import torch
import torch.nn as nn


class SelfReflection(nn.Module):
    def __init__(self, feature_dim, T):
        super().__init__()
        # 自反思模块 R：
        # 根据大模型隐表示 φ_l(x) 估计其预测置信度 Q_l。
        # 论文这里采用的是非常直接的全连接置信度头。
        self.fc = nn.Linear(feature_dim * T, 1)
    

    def forward(self, phi):
        # 将时间/patch 维展开成一维向量，再映射为单个标量置信度。
        # 这与论文公式 Q_l = R(φ_l(x)) 的实现形式一致。
        return torch.sigmoid(self.fc(phi.flatten(1))).squeeze(-1)
