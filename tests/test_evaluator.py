import unittest
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.evaluator import ParallelEvaluator


class TestEvaluatorLogic(unittest.TestCase):

    def setUp(self):
        """
        测试前的准备工作：生成临时的 JSON 配置文件
        """
        # 1. 创建临时目录
        self.test_dir = tempfile.TemporaryDirectory()

        # 2. 模拟 anchor_results.json
        # 包含两个 Anchor 点：
        # - 一个接近 2000 (Very Low)
        # - 一个接近 16000 (High)
        self.anchor_data = {
            "TestSeq_1080p": [
                {
                    "bitrate": 2274.5,
                    "vmaf": 50.5,
                    "real_bitrate": 2300.0,
                },  # 应匹配 Very Low
                {
                    "bitrate": 15660.2,
                    "vmaf": 97.4,
                    "real_bitrate": 15700.0,
                },  # 应匹配 High
            ],
            "TestSeq_Missing": [  # 这个序列在 test_sequences 里不存在，应被跳过
                {"bitrate": 1000, "vmaf": 80}
            ],
        }

        # 3. 模拟 test_sequences.json
        # 定义了该序列的元数据和预设档位
        self.meta_data = {
            "TestSeq_1080p": {
                "class": "B",
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "bitrates": {
                    "Very Low": 2000,
                    "Low": 4000,
                    "Medium": 8000,
                    "High": 16000,
                },
            }
        }

        # 4. 模拟 initial_params.json
        self.init_params = {
            "profiles": {
                "Very Low": {"mode_params": {"rd": 5}},
                "High": {"mode_params": {"rd": 3}},
            }
        }

        # 写入临时文件
        self.anchor_path = os.path.join(self.test_dir.name, "anchor.json")
        self.meta_path = os.path.join(self.test_dir.name, "meta.json")
        self.init_path = os.path.join(self.test_dir.name, "init.json")

        with open(self.anchor_path, "w") as f:
            json.dump(self.anchor_data, f)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta_data, f)
        with open(self.init_path, "w") as f:
            json.dump(self.init_params, f)

        # 模拟数据集根目录
        self.dataset_root = "/mock/dataset/root"

    def tearDown(self):
        # 清理临时目录
        self.test_dir.cleanup()

    @patch("os.path.exists")
    def test_task_generation_and_profile_matching(self, mock_exists):
        """
        核心测试：验证 Profile 匹配逻辑和任务生成
        """

        # 模拟视频文件存在 (让 os.path.exists 对 .yuv 返回 True)
        # 侧面验证路径推导逻辑：它会先查 ClassB/TestSeq_1080p.yuv
        def side_effect(path):
            if path.endswith(".json") or path == "/dev/shm/x265_search_temp":
                return True
            # 模拟 Class 目录结构存在
            if "/ClassB/" in path:
                return True
            return False

        mock_exists.side_effect = side_effect

        # 初始化 Evaluator
        evaluator = ParallelEvaluator(
            anchor_json_path=self.anchor_path,
            seq_meta_json_path=self.meta_path,
            init_params_path=self.init_path,
            dataset_root=self.dataset_root,
            max_workers=4,
        )

        tasks = evaluator.tasks_metadata

        # 断言 1: 应该生成了 2 个任务 (Missing 的那个应该被跳过)
        self.assertEqual(len(tasks), 2, "应生成 2 个任务")

        # === 验证第一个任务 (Very Low 匹配) ===
        task1 = tasks[0]
        print(
            f"\n[Test] Task 1 Profile: {task1['profile']}, Target: {task1['target_bitrate']}"
        )

        self.assertEqual(task1["seq_name"], "TestSeq_1080p")
        # 核心验证：2274.5 离 2000 比离 4000 近，所以必须匹配 "Very Low"
        self.assertEqual(task1["profile"], "Very Low")
        # 核心验证：Target Bitrate 应使用定义的 2000，而不是 Anchor 的 2274.5
        self.assertEqual(task1["target_bitrate"], 2000)
        # 核心验证：Anchor Real Bitrate 应保留原始值 2300.0 (用于后续 BD-Calc)
        self.assertEqual(task1["anchor_real_bitrate"], 2300.0)

        # === 验证第二个任务 (High 匹配) ===
        task2 = tasks[1]
        print(
            f"[Test] Task 2 Profile: {task2['profile']}, Target: {task2['target_bitrate']}"
        )

        # 15660.2 离 16000 最近
        self.assertEqual(task2["profile"], "High")
        self.assertEqual(task2["target_bitrate"], 16000)

        # === 验证路径推导 ===
        # 因为我们在 meta 中定义了 class="B"，路径应包含 ClassB
        expected_path = os.path.join(self.dataset_root, "ClassB", "TestSeq_1080p.yuv")
        self.assertEqual(task1["path"], expected_path)

    @patch("os.path.exists")
    def test_path_fallback(self, mock_exists):
        """
        测试路径回退逻辑：如果 Class 目录下没有，应去根目录找
        """

        # 模拟 ClassB 目录下的文件不存在，但根目录下的存在
        def side_effect(path):
            if path.endswith(".json") or path == "/dev/shm/x265_search_temp":
                return True
            if "/ClassB/" in path:
                return False  # Class 目录不存在
            if path.endswith("TestSeq_1080p.yuv"):
                return True  # 根目录存在
            return False

        mock_exists.side_effect = side_effect

        evaluator = ParallelEvaluator(
            self.anchor_path, self.meta_path, self.init_path, self.dataset_root
        )

        task = evaluator.tasks_metadata[0]
        # 期望路径不包含 ClassB
        expected_path = os.path.join(self.dataset_root, "TestSeq_1080p.yuv")
        self.assertEqual(task["path"], expected_path)
        print(f"\n[Test] Fallback Path Verified: {task['path']}")


if __name__ == "__main__":
    unittest.main()
