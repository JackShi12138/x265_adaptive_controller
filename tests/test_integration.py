import os
import sys
import json
import logging
from unittest.mock import MagicMock

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.evaluator import ParallelEvaluator

# ==============================================================================
# [USER CONFIG] 请根据您的真实环境修改以下路径
# ==============================================================================

# 1. 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. 真实配置文件路径
ANCHOR_JSON = os.path.join(PROJECT_ROOT, "config", "anchor_results.json")
META_JSON = os.path.join(PROJECT_ROOT, "config", "test_sequences.json")
INIT_JSON = os.path.join(PROJECT_ROOT, "config", "initial_params.json")

# 3. 真实数据集根目录 (请修改为您存放 YUV 的位置)
DATASET_ROOT = "/home/shiyushen/x265_sequence/"

# 4. 指定一个用于测试的序列名称 (必须在 dataset_root 中存在对应的 .yuv 文件)
# 建议选一个小的 1080p 或 720p 视频，例如 "BasketballDrive_1920x1080_50"
TEST_SEQ_NAME = "BasketballDrive_1920x1080_50"

# 5. x265 动态库路径
LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"

# ==============================================================================


def run_real_world_smoke_test():
    print("\n=== Starting Real-World Integration Smoke Test ===")

    # 1. 检查文件是否存在
    required_files = [ANCHOR_JSON, META_JSON, INIT_JSON, LIB_PATH]
    for f in required_files:
        if not os.path.exists(f):
            print(f"[Error] Required file not found: {f}")
            return

    # 2. 初始化 Evaluator
    print(f"Loading configs from {PROJECT_ROOT}/config ...")
    try:
        evaluator = ParallelEvaluator(
            anchor_json_path=ANCHOR_JSON,
            seq_meta_json_path=META_JSON,
            init_params_path=INIT_JSON,
            dataset_root=DATASET_ROOT,
            lib_path=LIB_PATH,
            max_workers=1,  # 测试只开 1 个进程
        )
    except Exception as e:
        print(f"[Fatal] Failed to initialize Evaluator: {e}")
        return

    # 3. [关键步骤] 篡改任务列表，只保留目标测试序列
    # 我们不想跑完所有 88 个任务，只要跑通 1 个就能验证逻辑
    original_task_count = len(evaluator.tasks_metadata)
    target_tasks = [
        t for t in evaluator.tasks_metadata if t["seq_name"] == TEST_SEQ_NAME
    ]

    if not target_tasks:
        print(f"[Error] Sequence '{TEST_SEQ_NAME}' not found in metadata or task list.")
        print(
            "Available sequences (first 5):",
            [t["seq_name"] for t in evaluator.tasks_metadata[:5]],
        )
        return

    # 只取该序列的第 1 个 profile (例如 Very Low) 进行测试
    evaluator.tasks_metadata = target_tasks[:1]
    task_info = evaluator.tasks_metadata[0]

    print(f"\n[Target Task Selected]")
    print(f"  Seq: {task_info['seq_name']}")
    print(f"  Profile: {task_info['profile']}")
    print(f"  Target Bitrate: {task_info['target_bitrate']}")
    print(f"  Path: {task_info['path']}")

    if not os.path.exists(task_info["path"]):
        print(f"[Error] YUV file does not exist at: {task_info['path']}")
        return

    # 4. 构造虚拟超参数
    dummy_hyperparams = {
        "a": 1.1,
        "b": 1.5,
        "beta": {"VAQ": 0.5, "CUTree": 0.5, "PsyRD": 1.0, "PsyRDOQ": 1.0, "QComp": 0.5},
    }

    print("\n--- Running evaluate_batch (Real Encoding) ---")
    print("This may take 1-2 minutes depending on video length...")

    # 5. 执行评估
    score, details, crash_report, stats = evaluator.evaluate_batch(dummy_hyperparams)

    # 6. 结果分析
    print("\n" + "=" * 40)
    print("      TEST REPORT      ")
    print("=" * 40)

    if crash_report["count"] > 0:
        print("[FAILED] Task Crashed!")
        print("Errors:", crash_report["errors"])
        print("\nPossible Causes:")
        print("1. 'Unsupported bit depth': Params mismatch (fps passed as depth).")
        print("2. 'x265 CSV not found': Encoding failed silently or permission issue.")
    else:
        print(f"[SUCCESS] Task Completed.")
        print(f"  BD-VMAF Score: {score}")
        print(f"  VMAF Points: {details[TEST_SEQ_NAME]['test_points']}")
        print(f"  Real Bitrate: {details[TEST_SEQ_NAME]['test_points'][0][0]} kbps")

        # 验证是否真的使用了正确的参数
        # 由于是多进程，我们无法直接在这里打印 config，但如果没报错，说明 input-depth 修复生效了
        print("\nConfig Validation: PASSED (No parameter mismatch errors)")


if __name__ == "__main__":
    run_real_world_smoke_test()
