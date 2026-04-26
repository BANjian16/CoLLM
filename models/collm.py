import torch

from config import TAU1, TAU2


class CoLLM:
    def __init__(self, S, L, F, R):
        self.S, self.L, self.F, self.R = S, L, F, R

    @torch.no_grad()
    def inference(self, x, tau1=TAU1, tau2=TAU2, return_details=False):
        ys, phi_s = self.S(x)
        q_s = self.F(phi_s)

        use_small = q_s >= tau1
        y_final = ys.clone()
        uncertain = ~use_small

        use_large = torch.zeros_like(use_small, dtype=torch.bool)
        use_fusion = torch.zeros_like(use_small, dtype=torch.bool)
        q_l_full = torch.full_like(q_s, float("nan"))

        if uncertain.any():
            yl, phi_l = self.L(x[uncertain])
            q_l = self.R(phi_l)
            delta = q_s[uncertain] - q_l
            use_large_uncertain = delta <= tau2
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
