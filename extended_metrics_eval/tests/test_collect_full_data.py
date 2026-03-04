import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import tempfile
import concurrent.futures

# 将上一级目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scripts import collect_full_data


class TestCollectFullData(unittest.TestCase):

    def setUp(self):
        """
        在测试前创建临时沙盒目录，并接管主脚本中的所有物理路径。
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        # 动态替换脚本中的全局路径，防止污染真实工作区
        collect_full_data.eval_root = self.temp_dir.name
        collect_full_data.project_root = self.temp_dir.name
        collect_full_data.TEMP_DIR = os.path.join(self.temp_dir.name, "shm_mock")
        collect_full_data.DATASET_ROOT = (
            self.temp_dir.name
        )  # [修复] 必须接管数据集根目录
        os.makedirs(collect_full_data.TEMP_DIR, exist_ok=True)

        # 伪造配置文件的目录结构
        config_dir = os.path.join(self.temp_dir.name, "config")
        os.makedirs(config_dir, exist_ok=True)

        # 1. 伪造测试序列配置 (1个序列，2个码率档位 -> 预期产生 1x2x3=6 个并发任务)
        with open(os.path.join(config_dir, "test_sequences.json"), "w") as f:
            json.dump(
                {
                    "MockVideo_1080p": {
                        "class": "B",
                        "width": 1920,
                        "height": 1080,
                        "fps": 50,
                        "bitrates": {"Low": 2000, "High": 4000},
                    }
                },
                f,
            )

        # 2. 伪造初始参数配置
        with open(os.path.join(config_dir, "initial_params.json"), "w") as f:
            json.dump(
                {
                    "profiles": {
                        "Low": {"mode_params": {"qp": 32}, "tune_params": {}},
                        "High": {"mode_params": {"qp": 22}, "tune_params": {}},
                    }
                },
                f,
            )

        # 3. [修复] 伪造物理 YUV 文件，骗过主程序中的 os.path.exists 校验
        yuv_dir = os.path.join(self.temp_dir.name, "ClassB")
        os.makedirs(yuv_dir, exist_ok=True)
        dummy_yuv_path = os.path.join(yuv_dir, "MockVideo_1080p.yuv")
        open(dummy_yuv_path, "w").close()  # 创建一个空文件

    @patch("scripts.collect_full_data.X265Wrapper")
    @patch("scripts.collect_full_data.YUVReader")
    @patch("scripts.collect_full_data.AdaptiveController")
    @patch("scripts.collect_full_data._parse_x265_bitrate")
    @patch("scripts.collect_full_data._decode_hevc_strictly")
    @patch("scripts.collect_full_data.calculate_metrics")
    def test_worker_task_pipeline(
        self, mock_calc, mock_decode, mock_parse, mock_ctrl, mock_yuv, mock_x265
    ):
        """
        测试子进程原子任务：拦截所有底层重型调用，验证数据清洗和组装逻辑。
        """
        # 设定 Mock 函数的返回值
        mock_parse.return_value = 2100.5  # 模拟解析出的真实码率
        mock_decode.return_value = True  # 模拟解码成功
        mock_calc.return_value = {"psnr": 38.5, "ssim": 0.965}  # 模拟提取到的客观指标

        # 构造一个发给子进程的 Task Context
        task = {
            "seq_name": "MockVideo_1080p",
            "mode": "online",
            "profile": "Low",
            "target_bitrate": 2000,
            "width": 1920,
            "height": 1080,
            "fps": 50,
            "ref_yuv": "dummy.yuv",
            "mode_params": {},
            "tune_params": {},
            "offline_params": {},
        }

        # 直接调用 Worker
        res = collect_full_data.worker_task(task)

        # 验证 Worker 是否将所有数据正确打包成了我们需要的字典格式
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["seq"], "MockVideo_1080p")
        self.assertEqual(res["mode"], "online")
        self.assertEqual(res["bitrate"], 2100.5)
        self.assertEqual(res["psnr"], 38.5)

        # 因为 mode 是 online，断言它拉起的是 AdaptiveController 而不是纯 wrapper
        mock_ctrl.assert_called_once()

    @patch("scripts.collect_full_data.ProcessPoolExecutor")
    @patch("scripts.collect_full_data.worker_task")
    def test_main_scheduling_and_aggregation(self, mock_worker, mock_pool_cls):
        """
        测试主进程：验证任务展开 (Flattening)、并发结果聚合以及增量 JSON 落盘。
        """
        # 巧妙地将 ProcessPoolExecutor 替换为 ThreadPoolExecutor，以便让 Mock 对象跨越进程壁垒
        mock_pool_cls.return_value = concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        )

        # 伪造子进程的返回结果
        def fake_worker(task):
            return {
                "seq": task["seq_name"],
                "mode": task["mode"],
                "profile": task["profile"],
                "status": "success",
                "bitrate": task["target_bitrate"] + 50.0,
                "psnr": 40.0,
                "ssim": 0.98,
                "error": "",
            }

        mock_worker.side_effect = fake_worker

        # 执行主程序
        collect_full_data.main()

        # 验证总共展开了 6 个任务 (1个序列 * 2个档位 * 3种模式)
        self.assertEqual(mock_worker.call_count, 6)

        # 验证输出目录和 JSON 文件是否被成功创建
        out_dir = os.path.join(self.temp_dir.name, "results_json")
        self.assertTrue(os.path.exists(os.path.join(out_dir, "baseline_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "offline_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "online_metrics.json")))

        # 验证 JSON 的层级结构和数据对齐是否精准无误
        with open(os.path.join(out_dir, "online_metrics.json"), "r") as f:
            online_data = json.load(f)
            # 断言第一级：序列名
            self.assertIn("MockVideo_1080p", online_data)
            # 断言第二级：码率档位
            self.assertIn("Low", online_data["MockVideo_1080p"])
            self.assertIn("High", online_data["MockVideo_1080p"])
            # 断言第三级：提取的指标内容
            low_metrics = online_data["MockVideo_1080p"]["Low"]
            self.assertEqual(low_metrics["psnr"], 40.0)
            self.assertEqual(low_metrics["bitrate"], 2050.0)  # 2000 + 50.0


if __name__ == "__main__":
    unittest.main(verbosity=2)
