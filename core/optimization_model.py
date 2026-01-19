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
        (包含 SCV 特异性阻断与 Beta 解耦逻辑)
        """
        new_params = {}

        # [Step 1] 读取 SCV 标记 (由 FeatureExtractor 产生)
        is_scv = features.get("scv_flag", 0.0) > 0.5

        for i_name in self.modules:
            param_key = self.p_map[i_name]

            # [Step 2] Beta 解耦 (Split Beta)
            # 允许 CMA-ES 为 SCV 场景学习独立的权重 (例如 beta_VAQ_SCV)
            # 如果 config 中没有定义 _SCV 参数，则回退使用通用参数
            beta_key = f"{i_name}_SCV"
            if is_scv and beta_key in self.hyperparams["beta"]:
                beta_i = self.hyperparams["beta"][beta_key]
            else:
                beta_i = self.hyperparams["beta"].get(i_name, 0.0)

            # [Step 3] 计算交互项求和
            sum_val = 0.0
            for j_name in self.modules:
                if i_name == j_name:
                    continue

                w_j_key = self.f_map[j_name]
                w_j = features.get(w_j_key, 0.0)

                th_i = self.theta[i_name]
                h_j = self.h_omega[j_name]

                # 获取基础交互系数 (来自 model_config.json)
                phi_ij = self.phi[i_name][j_name]

                # === [关键手术] SCV 特异性连接阻断 ===
                if is_scv:
                    # 针对 Φ12 (VAQ ↔ CUTree):
                    # 物理实测 SlideEditing 为 0.008 (几近于0)，而基准配置为 ~0.35
                    # 必须强制切断，消除来自时域的巨大噪声干扰
                    if (i_name == "VAQ" and j_name == "CUTree") or (
                        i_name == "CUTree" and j_name == "VAQ"
                    ):
                        phi_ij = 0.0

                    # 针对 Φ15 (VAQ ↔ QComp):
                    # 物理实测从 -0.14 变为 ~0.03 (基本消失)
                    # 建议切断以减少干扰
                    if (i_name == "VAQ" and j_name == "QComp") or (
                        i_name == "QComp" and j_name == "VAQ"
                    ):
                        phi_ij = 0.0

                x = th_i * h_j * phi_ij * w_j
                term = self._sigmoid_term(x) - 0.5
                sum_val += term

            # [Step 4] 计算最终调整量
            # 注意：此处不再进行 delta_p = -delta_p 的翻转
            # 我们依靠"连接阻断"消除错误信号，依靠"Beta_SCV"学习正确幅度
            delta_p = beta_i * sum_val

            # [Step 5] 应用调整 (Clamping & Bounds)
            base_val = self.initial_params.get(param_key)
            if base_val is None:
                continue

            constraint = self.constraints.get(param_key)
            if constraint:
                max_step = constraint.get("max_step", 999.0)
                delta_p_clamped = max(-max_step, min(max_step, delta_p))
                final_val = base_val + delta_p_clamped

                p_min = constraint.get("min", -999.0)
                p_max = constraint.get("max", 999.0)
                final_val = max(p_min, min(p_max, final_val))
            else:
                final_val = base_val + delta_p

            new_params[param_key] = final_val

        return new_params
