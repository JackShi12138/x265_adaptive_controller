import unittest
import sys
import os

# 将项目根目录添加到 sys.path，确保能导入 search 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.metric import calculate_bd_vmaf


class TestMetric(unittest.TestCase):

    def setUp(self):
        # 构造一组标准的基准数据 (Anchor)
        # 格式: (Bitrate, VMAF)
        self.anchor = [(1000, 80.0), (2000, 85.0), (3000, 88.0), (4000, 90.0)]

    def test_normal_improvement(self):
        """测试正常场景：画质提升 (BD-VMAF 应为正)"""
        # 相同码率下，VMAF 整体高 2 分
        test_better = [(1000, 82.0), (2000, 87.0), (3000, 90.0), (4000, 92.0)]
        score = calculate_bd_vmaf(self.anchor, test_better)

        print(f"\n[Test Normal] Better Score: {score}")
        self.assertTrue(score > 0, "提升画质应得到正分")
        self.assertNotEqual(score, -9999.0)

    def test_normal_degradation(self):
        """测试正常场景：画质下降 (BD-VMAF 应为负)"""
        # 相同码率下，VMAF 整体低 2 分
        test_worse = [(1000, 78.0), (2000, 83.0), (3000, 86.0), (4000, 88.0)]
        score = calculate_bd_vmaf(self.anchor, test_worse)

        print(f"[Test Normal] Worse Score: {score}")
        self.assertTrue(score < 0, "画质下降应得到负分")
        self.assertNotEqual(score, -9999.0)

    def test_identical_curves(self):
        """测试完全相同的曲线"""
        score = calculate_bd_vmaf(self.anchor, self.anchor)
        print(f"[Test Identical] Score: {score}")
        # 由于浮点计算误差，可能不绝对为 0，但应极小
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_insufficient_points(self):
        """测试数据点不足的情况"""
        # 只有 3 个点
        short_anchor = [(1000, 80), (2000, 85), (3000, 88)]
        short_test = [(1000, 82), (2000, 87), (3000, 90)]

        score = calculate_bd_vmaf(short_anchor, short_test)
        print(f"[Test Insufficient] Score: {score}")
        self.assertEqual(score, -9999.0)

    def test_unsorted_input(self):
        """测试乱序输入 (函数内部应自动排序)"""
        # 将测试数据打乱顺序
        test_unsorted = [(4000, 92.0), (1000, 82.0), (3000, 90.0), (2000, 87.0)]
        score = calculate_bd_vmaf(self.anchor, test_unsorted)
        print(f"[Test Unsorted] Score: {score}")
        self.assertTrue(score > 0)
        self.assertNotEqual(score, -9999.0)

    def test_monotonicity_violation(self):
        """测试单调性异常 (VMAF 随码率增加而大幅下降)"""
        # 模拟严重异常：码率增加，VMAF 暴跌
        test_broken = [
            (1000, 90.0),
            (2000, 85.0),
            (3000, 80.0),  # 跌幅超过阈值 (5.0)
            (4000, 70.0),
        ]
        score = calculate_bd_vmaf(self.anchor, test_broken)
        print(f"[Test Broken] Score: {score}")
        self.assertEqual(score, -9999.0, "单调性严重破坏应返回错误码")

    def test_minor_fluctuation(self):
        """测试轻微波动 (应被允许)"""
        # VMAF 轻微震荡，但在容忍阈值 (5.0) 内
        test_fluctuation = [
            (1000, 80.0),
            (2000, 82.0),
            (3000, 81.0),  # 轻微下降 1.0
            (4000, 85.0),
        ]
        score = calculate_bd_vmaf(self.anchor, test_fluctuation)
        print(f"[Test Fluctuation] Score: {score}")
        # 应该能算出数值，而不是报错
        self.assertNotEqual(score, -9999.0)


if __name__ == "__main__":
    unittest.main()
