import unittest
import sys
import os

# 将上一级目录加入系统路径，以便能够导入 scripts 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scripts.compute_bd_stats import calculate_bd_score

class TestComputeBDStats(unittest.TestCase):

    def setUp(self):
        # 准备一组正常的、单调递增的模拟 PSNR 数据
        self.anchor_normal = [
            (1000, 35.0), 
            (2000, 37.0), 
            (3000, 38.5), 
            (4000, 39.5)
        ]
        # 测试组数据（相同码率下分数略高，预期 BD-Score 为正）
        self.test_normal = [
            (950, 35.5), 
            (1900, 37.8), 
            (2900, 39.0), 
            (3900, 40.2)
        ]

    def test_case1_normal_calculation(self):
        """Case 1: 基础正向测试。验证正常数据能返回正确的浮点数结果。"""
        score = calculate_bd_score(self.anchor_normal, self.test_normal, drop_threshold=1.0)
        self.assertIsInstance(score, float)
        self.assertNotEqual(score, -9999.0)
        self.assertGreater(score, 0.0)  # 测试组质量更高，预期增益大于0

    def test_case2_insufficient_points(self):
        """Case 2: 样本量不足拦截。验证点数少于4个时返回 -9999.0。"""
        anchor_short = self.anchor_normal[:3]  # 只有3个点
        score = calculate_bd_score(anchor_short, self.test_normal, drop_threshold=1.0)
        self.assertEqual(score, -9999.0)

    def test_case3_unsorted_input_robustness(self):
        """Case 3: 乱序输入鲁棒性。验证函数内部能自动按码率排序。"""
        # 故意打乱输入顺序
        anchor_unsorted = [
            (3000, 38.5), 
            (1000, 35.0), 
            (4000, 39.5), 
            (2000, 37.0)
        ]
        score_unsorted = calculate_bd_score(anchor_unsorted, self.test_normal, drop_threshold=1.0)
        score_normal = calculate_bd_score(self.anchor_normal, self.test_normal, drop_threshold=1.0)
        
        # 乱序输入的计算结果应与正常输入完全一致 (保留4位小数比较)
        self.assertAlmostEqual(score_unsorted, score_normal, places=4)

    def test_case4_identical_curves(self):
        """Case 4: 零差异校验。验证两组数据完全相同时，BD-Score 为 0.0。"""
        score = calculate_bd_score(self.anchor_normal, self.anchor_normal, drop_threshold=1.0)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_case5_sanity_check_ssim(self):
        """Case 5: SSIM 小数域的倒挂拦截。验证 drop_threshold 对 SSIM 的保护作用。"""
        ssim_anchor = [(1000, 0.90), (2000, 0.92), (3000, 0.94), (4000, 0.95)]
        # 制造一个严重的倒挂：最后一个点跌到了 0.84
        ssim_test_drop = [(1000, 0.91), (2000, 0.93), (3000, 0.94), (4000, 0.84)]
        
        # 如果使用默认的 threshold=1.0，不会触发保护 (0.84 并没有比 0.91 低 1.0)
        # 所以必须传入针对 SSIM 的小阈值，例如 0.05
        # (0.84 < 0.91 - 0.05 -> 0.84 < 0.86，条件成立)
        score = calculate_bd_score(ssim_anchor, ssim_test_drop, drop_threshold=0.05)
        self.assertEqual(score, -9999.0)

    def test_case6_mathematical_anomalies(self):
        """Case 6: 数学异常捕获。验证遇到重复 X 轴数据导致插值失败时，try-except 能成功兜底。"""
        # 构造包含重复码率 (1000) 的异常数据，这通常会导致底层的差商计算出现除以零错误
        anchor_dup = [
            (1000, 35.0), 
            (1000, 36.0), 
            (3000, 38.5), 
            (4000, 39.5)
        ]
        score = calculate_bd_score(anchor_dup, self.test_normal, drop_threshold=1.0)
        # 确保程序不会崩溃，而是优雅地返回 -9999.0
        self.assertEqual(score, -9999.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)