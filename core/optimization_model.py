import math
import json
import os


class OptimizationModel:
    """
    在线自适应参数优化模型
    策略变更：
    - 自然视频：执行 CMA-ES 优化的参数调整
    - 屏幕内容(SCV)：执行旁路模式(Bypass)，强制使用初始参数，不进行微调
    """

    def __init__(self, initial_params: dict, hyperparams: dict, config_path=None):
        self.initial_params = initial_params.copy()
        self.hyperparams = hyperparams

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
        if -b * x > 20:
            return 0.0
        if -b * x < -20:
            return 1.0
        return 1.0 / (1.0 + a * math.exp(-b * x))

    def compute_adjustments(self, features: dict) -> dict:
        """
        计算参数调整量
        """
        new_params = {}

        # [策略变更] SCV 旁路模式
        # 如果检测到屏幕内容，直接使用初始参数，跳过所有优化逻辑
        is_scv = features.get("scv_flag", 0.0) > 0.5

        for i_name in self.modules:
            param_key = self.p_map[i_name]
            base_val = self.initial_params.get(param_key)
            if base_val is None:
                continue

            if is_scv:
                # SCV: 保持原样 (Delta = 0)
                new_params[param_key] = base_val
                continue

            # === 以下仅针对自然视频执行优化 ===

            # 1. 计算求和项
            sum_val = 0.0
            for j_name in self.modules:
                if i_name == j_name:
                    continue

                w_j_key = self.f_map[j_name]
                w_j = features.get(w_j_key, 0.0)

                th_i = self.theta[i_name]
                h_j = self.h_omega[j_name]
                phi_ij = self.phi[i_name][j_name]

                x = th_i * h_j * phi_ij * w_j
                term = self._sigmoid_term(x) - 0.5
                sum_val += term

            # 2. 计算 Delta P
            beta_i = self.hyperparams["beta"].get(i_name, 0.0)
            delta_p = beta_i * sum_val

            # 3. 应用调整
            final_val = base_val + delta_p

            # 4. 约束限制
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
