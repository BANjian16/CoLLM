import torch

from config import TAU1, TAU2


class CoLLM:
    def __init__(self, S, L, F, R):
        # S: small model，小模型
        # L: large model，大模型
        # F: fuzzy decision agent，根据小模型特征估计小模型置信度
        # R: self-reflection，根据大模型特征估计大模型置信度
        self.S, self.L, self.F, self.R = S, L, F, R

    @torch.no_grad()
    def inference(self, x, tau1=TAU1, tau2=TAU2, return_details=False):
        # 第一步：先让小模型预测。这样所有样本都会先经过低成本模型。
        ys, phi_s = self.S(x)
        q_s = self.F(phi_s)

        # 如果小模型置信度 q_s >= tau1，就直接采用小模型结果。
        # 这对应论文里的“提前退出”，可以减少大模型调用次数。
        use_small = q_s >= tau1
        y_final = ys.clone()
        uncertain = ~use_small

        use_large = torch.zeros_like(use_small, dtype=torch.bool)
        use_fusion = torch.zeros_like(use_small, dtype=torch.bool)
        q_l_full = torch.full_like(q_s, float("nan"))

        if uncertain.any():
            # 第二步：只把小模型不确定的样本送进大模型。
            yl, phi_l = self.L(x[uncertain])
            q_l = self.R(phi_l)
            # tau2 用来比较小模型和大模型的置信度差异。
            # delta 越小，说明大模型相对更值得相信。
            delta = q_s[uncertain] - q_l
            use_large_uncertain = delta <= tau2
            # 如果大模型明显更可靠，就用大模型；否则用 small/large 平均融合。
            y_uncertain = torch.where(use_large_uncertain, yl, 0.5 * (ys[uncertain] + yl))

            y_final[uncertain] = y_uncertain
            use_large[uncertain] = use_large_uncertain
            use_fusion[uncertain] = ~use_large_uncertain
            q_l_full[uncertain] = q_l

        if return_details:
            return y_final, {
                "use_small": use_small,
                "use_large": use_large,
                "use_fusion": use_fusion,
                "q_s": q_s,
                "q_l": q_l_full,
            }
        return y_final
