import unittest
import sys
import os
import json
import csv
import tempfile

# 将上一级目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scripts import generate_table4
from scripts.generate_table4 import extract_rd_points


class TestGenerateTable4(unittest.TestCase):

    def setUp(self):
        """
        在测试前创建临时目录，并构造 3 种极具代表性的 Mock 序列数据。
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        # 动态劫持脚本中的全局路径常量，指向临时目录
        generate_table4.SEQ_CONFIG_PATH = os.path.join(
            self.temp_dir.name, "test_sequences.json"
        )
        generate_table4.BASELINE_JSON = os.path.join(
            self.temp_dir.name, "baseline_metrics.json"
        )
        generate_table4.OFFLINE_JSON = os.path.join(
            self.temp_dir.name, "offline_metrics.json"
        )
        generate_table4.ONLINE_JSON = os.path.join(
            self.temp_dir.name, "online_metrics.json"
        )
        generate_table4.OUTPUT_CSV = os.path.join(
            self.temp_dir.name, "Table_IV_extended.csv"
        )

        # 1. 模拟配置 (定义了输出顺序)
        seq_config = {
            "SeqA_Normal": {"class": "A"},  # 正常数据序列
            "SeqB_Missing": {"class": "B"},  # 在底层 JSON 中缺失的序列
            "SeqC_Error": {"class": "C"},  # 数据点不足4个的残缺序列
        }
        with open(generate_table4.SEQ_CONFIG_PATH, "w") as f:
            json.dump(seq_config, f)

        # 2. 模拟 Baseline 数据
        baseline = {
            "SeqA_Normal": {
                "P1": {"bitrate": 1000, "psnr": 35.0, "ssim": 0.90},
                "P2": {"bitrate": 2000, "psnr": 36.0, "ssim": 0.92},
                "P3": {"bitrate": 3000, "psnr": 37.0, "ssim": 0.94},
                "P4": {"bitrate": 4000, "psnr": 38.0, "ssim": 0.96},
            },
            "SeqC_Error": {
                "P1": {"bitrate": 1000, "psnr": 35.0, "ssim": 0.90}  # 只有1个点
            },
        }
        with open(generate_table4.BASELINE_JSON, "w") as f:
            json.dump(baseline, f)

        # 3. 模拟 Offline 数据 (相比 Baseline，同等质量下码率降低了 10%)
        offline = {
            "SeqA_Normal": {
                "P1": {"bitrate": 900, "psnr": 35.0, "ssim": 0.90},
                "P2": {"bitrate": 1800, "psnr": 36.0, "ssim": 0.92},
                "P3": {"bitrate": 2700, "psnr": 37.0, "ssim": 0.94},
                "P4": {"bitrate": 3600, "psnr": 38.0, "ssim": 0.96},
            },
            "SeqC_Error": {"P1": {"bitrate": 900, "psnr": 35.0, "ssim": 0.90}},
        }
        with open(generate_table4.OFFLINE_JSON, "w") as f:
            json.dump(offline, f)

        # 4. 模拟 Online 数据 (相比 Baseline，同等质量下码率降低了 20%)
        online = {
            "SeqA_Normal": {
                "P1": {"bitrate": 800, "psnr": 35.0, "ssim": 0.90},
                "P2": {"bitrate": 1600, "psnr": 36.0, "ssim": 0.92},
                "P3": {"bitrate": 2400, "psnr": 37.0, "ssim": 0.94},
                "P4": {"bitrate": 3200, "psnr": 38.0, "ssim": 0.96},
            },
            "SeqC_Error": {"P1": {"bitrate": 800, "psnr": 35.0, "ssim": 0.90}},
        }
        with open(generate_table4.ONLINE_JSON, "w") as f:
            json.dump(online, f)

    def test_extract_rd_points(self):
        """测试单个序列内部指标字典的降维与提取逻辑。"""
        seq_data = {
            "Very Low": {"bitrate": 1500, "psnr": 35.2, "ssim": 0.92, "vmaf": 95.0},
            "High": {"bitrate": 3500, "psnr": 38.1, "ssim": 0.97},
        }
        psnr_pts, ssim_pts = extract_rd_points(seq_data)

        self.assertEqual(len(psnr_pts), 2)
        # 验证元组组装是否正确
        self.assertEqual(psnr_pts[0], (1500, 35.2))
        self.assertEqual(ssim_pts[1], (3500, 0.97))

    def test_main_csv_generation(self):
        """
        全流程集成测试：验证生成的 CSV 是否对齐、缺失是否用 N/A 填充、异常是否用 Error 兜底。
        """
        # 调用主函数执行读取、计算与生成
        generate_table4.main()

        # 验证文件是否成功落盘
        self.assertTrue(os.path.exists(generate_table4.OUTPUT_CSV))

        with open(generate_table4.OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 应该包含 1 行表头 + 3 行数据
        self.assertEqual(len(rows), 4)

        # 验证表头是否符合最新要求
        self.assertEqual(
            rows[0],
            [
                "Sequence",
                "Offline_BD-PSNR",
                "Online_BD-PSNR",
                "Offline_BD-SSIM",
                "Online_BD-SSIM",
            ],
        )

        # --- 验证 SeqA_Normal (常规成功数据) ---
        row_a = rows[1]
        self.assertEqual(row_a[0], "SeqA_Normal")
        self.assertNotIn("Error", row_a)
        self.assertNotIn("N/A", row_a)
        # 因为我们 Mock 的 Online 数据节省了 20% 码率，Offline 节省了 10% 码率
        # 所以计算出的 BD 增益必须是正数，且 Online 的增益应该大于 Offline 的增益
        self.assertGreater(float(row_a[1]), 0.0)
        self.assertGreater(float(row_a[2]), float(row_a[1]))

        # --- 验证 SeqB_Missing (缺失数据保护) ---
        row_b = rows[2]
        self.assertEqual(row_b[0], "SeqB_Missing")
        self.assertEqual(row_b[1], "N/A")  # BD-PSNR 填入 N/A
        self.assertEqual(row_b[4], "N/A")  # BD-SSIM 填入 N/A

        # --- 验证 SeqC_Error (点数不足或异常截断保护) ---
        row_c = rows[3]
        self.assertEqual(row_c[0], "SeqC_Error")
        self.assertEqual(row_c[1], "Error")  # 底层函数返回了 -9999.0
        self.assertEqual(row_c[4], "Error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
