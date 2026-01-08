import unittest
import sys
import os
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from core.optimization_model import OptimizationModel


class TestOptimizationModel(unittest.TestCase):

    def setUp(self):
        # 1. 初始参数
        self.initial_params = {
            "aq-strength": 1.0,
            "cutree-strength": 1.0,  # 即使这里给了初值
            "psy-rd": 2.0,
            "psy-rdoq": 1.0,
            "qcomp": 0.6,
        }

        # 2. 超参数
        self.hyperparams = {
            "a": 1.0,
            "b": 1.0,
            "beta": {
                "VAQ": 0.5,
                "CUTree": 0.5,  # 即使给了 Beta
                "PsyRD": 1.0,
                "PsyRDOQ": 1.0,
                "QComp": 0.2,
            },
        }

        self.model = OptimizationModel(self.initial_params, self.hyperparams)

    def test_logic(self):
        print("\n=== Testing Optimization Model (Final) ===")

        # 模拟特征
        feats = {
            "w1_var": 0.9,
            "w2_sad": 0.5,
            "w3_grad": 0.8,
            "w4_tex": 0.8,
            "w5_cplx": 0.85,
        }

        adj_params = self.model.compute_adjustments(feats)

        print("Adjusted Params:")
        for k, v in adj_params.items():
            if v is None:
                print(f"  {k}: [Ignored/None] (Expected for CUTree)")
            else:
                print(f"  {k}: {self.initial_params.get(k)} -> {v:.4f}")

        # 验证 CUTree 返回 None
        self.assertIsNone(
            adj_params.get("cutree-strength"), "CUTree output should be None"
        )

        # 验证 QComp 存在且有变化
        self.assertIsNotNone(adj_params.get("qcomp"))
        self.assertNotEqual(adj_params["qcomp"], self.initial_params["qcomp"])


if __name__ == "__main__":
    unittest.main()
