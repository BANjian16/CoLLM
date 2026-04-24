import torch


class CoLLM:
    """
    [Paper Sec 3.A)] CoLLM 协作推理框架 — 大小模型动态协同
    
    论文描述：
        CoLLM（Collaborative Large-Small Model）是一个工业级的大小模型协作框架，
        核心思想是"能小则小，需大则大"：
        - 简单样本（SM 高置信度）→ Small Model 快速处理，节省算力
        - 复杂样本（SM 低置信度）→ Large Model 深度推理，保证精度
        - 大模型输出不可靠时 → 融合大小模型输出，降低风险
        
    模块组成：
        S (SmallModel):  轻量 Transformer 编码器，详见 small.py  [对应论文 Eq. (1)]
        L (GPT2TimeSeries): 冻结 GPT-2 大模型，详见 gpt2_ts.py  [对应论文 Eq. (4)]
        F (FuzzyDecisionAgent): 模糊决策智能体，基于 FNN 量化样本不确定性，详见 fuzzy.py
                                [对应论文 Eq. (2), (3), (8), (9), (11)]
        R (SelfReflection): 自反思机制，评估大模型输出可靠性，详见 reflection.py
                             [对应论文 Eq. (5), (6), (12)]
        
    推理流程（对应论文 Fig. 1 及 Algorithm 1）：
        1. Small Model 前向推理，得到预测 ys 和特征 phi_s
        2. Fuzzy Agent 根据 phi_s 计算 SM 置信度分数 Qs
        3. 若 Qs ≥ τ1：SM 足够自信，直接返回 ys  [对应论文 Eq. (3) D1(Qs)=1]
        4. 若 Qs < τ1：SM 不自信，激活 Large Model  [对应论文 Eq. (3) D1(Qs)=0]
           - Large Model 前向推理，得到预测 yl 和特征 phi_l
           - Self-Reflection 根据 phi_l 计算 LM 置信度分数 Ql
           - 计算置信度差 Δ = Qs - Ql
           - 若 Δ ≤ τ2：LM 可靠，直接返回 yl  [对应论文 Eq. (6) D2(Δ)=1]
           - 若 Δ > τ2：LM 不可靠，融合返回 (ys + yl) / 2  [对应论文 Eq. (6) D2(Δ)=0, Eq. (7)]
    """
    def __init__(self, S, L, F, R):
        """
        Args:
            S (SmallModel): 小模型实例
            L (GPT2TimeSeries): 大模型实例
            F (FuzzyDecisionAgent): 模糊决策智能体实例
            R (SelfReflection): 自反思模块实例
        """
        self.S, self.L, self.F, self.R = S, L, F, R

    @torch.no_grad()
    def inference(self, x, tau1=0.7, tau2=-0.2, return_details=False):
        """
        [Paper Sec 3.A), Fig. 1, Algorithm 1] CoLLM 动态协作推理函数
        
        Args:
            x (Tensor): 输入工业时间序列样本，x ∈ X ⊂ R^{t×d}
                # x shape: [batch_size, seq_len, input_dim]
            tau1 (float): SM 置信度阈值 τ1，用于决定是否激活 Large Model
                # 论文中该阈值控制"SM 独占区"与"LM 协作区"的边界
                # 当 Qs ≥ τ1 时，SM 预测被视为可靠，直接输出 ys
                # 当 Qs < τ1 时，SM 预测不可靠，激活 LM 进行深度推理
            tau2 (float): 置信度差阈值 τ2，用于判断 LM 输出是否可信
                # Δ = Qs - Ql，当 Δ > τ2 时，说明 LM 置信度显著低于 SM，触发融合策略
        
        Returns:
            y (Tensor): RUL 预测结果
                # 基于样本级路由逐元素生成，shape [batch_size]
            details (dict, optional): 当 return_details=True 时返回路由信息
                - Qs: 小模型置信度，shape [batch_size]
                - Ql: 大模型置信度，shape [batch_size]
                - use_small: 是否直接采用小模型，bool shape [batch_size]
                - use_large_only: 是否采用大模型输出，bool shape [batch_size]
                - use_fusion: 是否触发融合，bool shape [batch_size]
        """
        # ───────────────────────────────────────────────────────────────
        # Stage 1: Small Model 快速推理
        # [Paper Sec 3.A.1), Eq. (1)] 所有样本首先经过 SM，获取初步预测 ys 和特征 phi_s
        # ys shape: [batch_size]
        # phi_s shape: [batch_size, seq_len, emb_dim]
        # ───────────────────────────────────────────────────────────────
        ys, phi_s = self.S(x)
        
        # ───────────────────────────────────────────────────────────────
        # Stage 2: 模糊决策 — 评估 SM 预测可靠性
        # [Paper Sec 3.A.2), Eq. (2)] Fuzzy Agent 基于 phi_s 计算 SM 置信度 Qs
        # Qs shape: [batch_size]，取值范围 [0, 1]
        # ───────────────────────────────────────────────────────────────
        Qs = self.F(phi_s)

        # [AI补全/优化] 修复为样本级路由（论文为 sample-level routing，不是 batch-level）。
        # use_small[i] = True 表示第 i 个样本满足 Qs_i >= tau1，直接输出 ys_i。
        # [Paper Eq. (3)] D1_i(Qs_i) = 1[Qs_i >= τ1]
        # use_small shape: [batch_size], dtype=bool
        use_small = Qs >= tau1

        # [AI补全/优化] 仅对子集执行 LM 推理：
        # need_lm[i] = True 表示该样本需进入 LM 路径。
        # need_lm shape: [batch_size], dtype=bool
        need_lm = ~use_small

        # 初始化输出与路由标记。
        # 默认全部先采用 ys，再对 need_lm 子集回填 LM / Fusion 输出。
        y = ys.clone()
        use_fusion = torch.zeros_like(need_lm)
        use_large_only = torch.zeros_like(need_lm)

        # Ql 为样本级 LM 置信度。对于未进入 LM 的样本，记为 NaN（未计算）。
        Ql = torch.full_like(Qs, float('nan'))

        if need_lm.any():
            # Boolean Indexing: 仅提取低置信度子集送入 LM。
            # x_lm shape: [num_lm, seq_len, input_dim]
            x_lm = x[need_lm]
            ys_lm = ys[need_lm]
            qs_lm = Qs[need_lm]

            # yl_lm shape: [num_lm]
            # phi_l_lm shape: [num_lm, num_patches, hidden_size]
            yl_lm, phi_l_lm = self.L(x_lm)
            # ql_lm shape: [num_lm]
            ql_lm = self.R(phi_l_lm)
            Ql[need_lm] = ql_lm

            # [Paper Eq. (6)] Δ_i = Qs_i - Ql_i (仅对 need_lm 子集计算)
            delta_lm = qs_lm - ql_lm
            use_fusion_lm = delta_lm > tau2
            use_large_only_lm = ~use_fusion_lm

            # [Paper Eq. (7)] 子集内分段融合并回填至原 batch 索引。
            y_lm = yl_lm.clone()
            y_lm[use_fusion_lm] = 0.5 * (ys_lm[use_fusion_lm] + yl_lm[use_fusion_lm])
            y[need_lm] = y_lm

            use_fusion[need_lm] = use_fusion_lm
            use_large_only[need_lm] = use_large_only_lm

        if not return_details:
            return y

        details = {
            'Qs': Qs,
            'Ql': Ql,
            'use_small': use_small,
            'use_large_only': use_large_only,
            'use_fusion': use_fusion,
        }
        return y, details
