import torch
import torch.nn as nn


class SelfReflection(nn.Module):
    """自反思模块 R。

    小模型置信度由 FuzzyDecisionAgent 估计；大模型也需要一个类似的置信度。
    SelfReflection 根据大模型的隐表示 phi_l 输出 q_l，表示“大模型对自己的预测有多自信”。
    """

    def __init__(self, feature_dim, T):
        super().__init__()
        # phi_l 的形状是 [batch, patch_num, GPT_hidden]。
        # 展平后用一个全连接层输出单个置信度。
        self.fc = nn.Linear(feature_dim * T, 1)

    def forward(self, phi):
        # 输出 q_l in [0, 1]。值越大，表示越相信大模型预测。
        return torch.sigmoid(self.fc(phi.flatten(1))).squeeze(-1)
