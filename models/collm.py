import torch

from config import TAU1, TAU2


class CoLLM:
    def __init__(self, S, L, F, R):
        # CoLLM 协同推理总控器：
        # S: 小模型 Small Model
        # L: 大模型 Large Model
        # F: 模糊决策代理 Fuzzy Decision Agent
        # R: 自反思模块 Self-Reflection
        #
        # 它本身不定义新的网络层，而是把论文中的三级决策流程串起来。
        self.S, self.L, self.F, self.R = S, L, F, R


    @torch.no_grad()
    def inference(self, x, tau1=TAU1, tau2=TAU2):
        # 第一阶段：先运行小模型，获得低成本预测 ys 和隐特征 φ_s。
        ys, phi_s = self.S(x)
        Qs = self.F(phi_s)
        # 论文决策函数 D1：
        # 若 Q_s >= tau1，说明样本复杂度较低/小模型足够可靠，
        # 则直接接受小模型预测并提前退出。
        small_mask = Qs >= tau1
        y_final = ys.clone()
        uncertain_mask = ~small_mask

        if not uncertain_mask.any():
            return y_final

        # 第二阶段：若小模型置信度不足，则调用大模型进行更深层推理。
        x_uncertain = x[uncertain_mask]
        yl, phi_l = self.L(x_uncertain)
        Ql = self.R(phi_l)
        # 论文定义 Delta = Q_s - Q_l。
        # 若 Delta <= tau2，说明大模型置信度没有显著差于小模型，
        # 则直接接受大模型输出 yl。
        #
        # 若 Delta > tau2，说明大模型当前样本上的可靠性偏弱，
        # 触发“小模型辅助修正”路径。本文实现使用最简单的平均融合：
        # y_final = (ys + yl) / 2
        delta = Qs[uncertain_mask] - Ql
        large_or_fused = torch.where(delta <= tau2, yl, 0.5*(ys[uncertain_mask]+yl))
        y_final[uncertain_mask] = large_or_fused
        return y_final
