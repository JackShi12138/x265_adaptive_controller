import math
import json
import os


class OptimizationModel:
    """
    在线自适应参数优化模型 (Online Adaptive Parameter Optimization Model)
    基于论文 Eq. (5): Δp_i = β_i * Σ (Sigmoid(...) - 0.5)
    """

    def __init__(self, initial_params: dict, hyperparams: dict, config_path=None):
        """
        :param initial_params: 用户初始 x265 参数 (P_init)
        :param hyperparams: 模型超参数 (a, b, beta_1~5)
        """
        self.initial_params = initial_params.copy()
        self.hyperparams = hyperparams

        # 加载静态配置
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "../config/model_config.json")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.modules = self.config["modules"]
        self.p_map = self.config["param_mapping"]
        self.f_map = self.config["feature_mapping"]
        self.phi = self.config["phi_matrix"]
        self.theta = self.config["theta"]
        self.h_omega = self.config["h_omega"]
        self.constraints = self.config["constraints"]

    def _sigmoid_term(self, x):
        """Sigmoid: 1 / (1 + a * exp(-b * x))"""
        a = self.hyperparams.get("a", 1.0)
        b = self.hyperparams.get("b", 1.0)

        # 防止溢出
        if -b * x > 20:
            return 0.0
        if -b * x < -20:
            return 1.0
        return 1.0 / (1.0 + a * math.exp(-b * x))

    def compute_adjustments(self, features: dict) -> dict:
        """
        根据当前 GOP 特征计算参数调整量
        :param features: 归一化特征 dict {'w1_var': 0.5, ...}
        :return: 调整后的参数 dict (key 为 x265 参数名)
                 [注意] CUTree (cutree-strength) 会返回 None 或不包含在 dict 中
        """
        new_params = {}

        for i_name in self.modules:
            param_key = self.p_map[i_name]

            # 1. 计算求和项 Sum = Σ (Sigmoid(...) - 0.5)
            sum_val = 0.0

            for j_name in self.modules:
                if i_name == j_name:
                    continue

                # 获取 j 的特征值 w_j
                w_j_key = self.f_map[j_name]
                w_j = features.get(w_j_key, 0.0)

                # 获取系数 (Theta 已根据您的要求更新为大部分为 -1)
                th_i = self.theta[i_name]
                h_j = self.h_omega[j_name]
                phi_ij = self.phi[i_name][j_name]

                # 核心项 x
                x = th_i * h_j * phi_ij * w_j

                term = self._sigmoid_term(x) - 0.5
                sum_val += term

            # 2. 计算 Delta P = Beta_i * Sum
            beta_i = self.hyperparams["beta"].get(i_name, 0.0)
            delta_p = beta_i * sum_val

            # 3. [特殊处理] CUTree 不输出实际参数
            if i_name == "CUTree":
                # 计算逻辑保留 (也许未来有用)，但这里不做输出
                new_params[param_key] = None
                continue

            # 4. 应用调整 (Base + Delta)
            base_val = self.initial_params.get(param_key)

            # 如果用户没提供初始值，对于核心优化参数，可以尝试给个默认值
            # 或者直接跳过不调
            if base_val is None:
                continue

            final_val = base_val + delta_p

            # 5. 安全性检查
            constraint = self.constraints.get(param_key)
            if constraint:
                max_step = constraint.get("max_step", 999.0)
                delta_p_clamped = max(-max_step, min(max_step, delta_p))

                final_val = base_val + delta_p_clamped

                p_min = constraint.get("min", -999.0)
                p_max = constraint.get("max", 999.0)
                final_val = max(p_min, min(p_max, final_val))

            new_params[param_key] = final_val

        return new_params
