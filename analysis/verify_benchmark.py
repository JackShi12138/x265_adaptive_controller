import os
import sys
import json
import csv
import numpy as np
from collections import defaultdict

# 添加项目根目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from search.evaluator import ParallelEvaluator
from search.metric import calculate_bd_vmaf

# ==============================================================================
# [配置路径]
# ==============================================================================
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# 1. 基准数据 (Anchor): x265 Slow Default
# 请确保此文件存在，它决定了我们计算 BD-VMAF 的参照系
BASELINE_JSON = os.path.join(CONFIG_DIR, "baseline_results_91.json")

# 2. 序列元数据
META_JSON = os.path.join(CONFIG_DIR, "test_sequences.json")

# 3. 初始参数模板 (用于 Evaluator 初始化)
INIT_JSON = os.path.join(CONFIG_DIR, "initial_params.json")

# 4. 最优超参数 (Runner 的产出)
BEST_PARAMS_JSON = os.path.join(
    PROJECT_ROOT,
    "/home/shiyushen/x265_adaptive_controller/best_hyperparams/best_hyperparams_0124.json",
)

# 5. 输出报告路径
REPORT_CSV = os.path.join(PROJECT_ROOT, "analysis", "benchmark_report.csv")

# 6. 环境配置 (请按需修改)
DEFAULT_DATASET_ROOT = "/home/shiyushen/x265_sequence/"  # 您的真实 YUV 路径
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"

# ==============================================================================


def load_json(path):
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def main():
    print("=== Starting Full Benchmark Verification ===")

    # 1. 加载最优参数
    print(f"Loading best hyperparameters from: {BEST_PARAMS_JSON}")
    best_data = load_json(BEST_PARAMS_JSON)

    # 兼容两种格式：直接是参数字典，或者包含 score/timestamp 的完整字典
    if "a" in best_data and "beta" in best_data:
        # 可能是直接的参数结构
        hyperparams = {
            "a": best_data.get("a"),
            "b": best_data.get("b"),
            "beta": best_data.get("beta"),
        }
    else:
        # 可能是 Runner 输出的完整结构 (包含 score 等元数据)
        # 这里假设 Runner 输出的是扁平的或者特定结构，通常 best_hyperparams.json
        # 直接存储了 optimal params 或者在外层。
        # 根据 runner.py 的逻辑，它存的是 final_best 字典
        hyperparams = {
            "a": best_data["a"],
            "b": best_data["b"],
            "beta": best_data["beta"],
        }

    print("Hyperparams to verify:")
    print(json.dumps(hyperparams, indent=2))

    # 2. 初始化 Evaluator
    # 注意：我们将 BASELINE_JSON 传给 anchor_json_path
    # 这样 Evaluator 内部就会以 Baseline 的码率作为 Target Bitrate，
    # 并且我们可以直接获取 Anchor 的数据用于对比。
    print(f"\nInitializing Evaluator with Baseline: {BASELINE_JSON}")
    try:
        evaluator = ParallelEvaluator(
            anchor_json_path=BASELINE_JSON,
            seq_meta_json_path=META_JSON,
            init_params_path=INIT_JSON,
            dataset_root=DEFAULT_DATASET_ROOT,
            lib_path=DEFAULT_LIB_PATH,
            max_workers=10,  # 保持安全并发数
        )
    except Exception as e:
        print(f"[Fatal] Failed to init Evaluator: {e}")
        return

    # 3. 执行全量测试
    print("\nRunning evaluation on all 88 tasks...")
    print("This may take a while. Please wait...")

    # evaluate_batch 返回: score, details, crash_report, stats
    # details 结构: { "SeqName": { "test_points": [...], "anchor_points": [...] }, ... }
    # 注意：我们需要修改 evaluator.py 确保它返回 anchor_points (之前的修改已包含此逻辑)
    final_score, details, crash_report, stats = evaluator.evaluate_batch(hyperparams)

    if crash_report["count"] > 0:
        print(f"\n[Warning] {crash_report['count']} tasks crashed!")
        print("Errors:", crash_report["errors"][:5])

    # 4. 生成 CSV 报告
    print(f"\nGenerating report at: {REPORT_CSV}")

    # 加载序列元数据以获取 Class 信息
    seq_meta = load_json(META_JSON)

    csv_rows = []
    bd_vmaf_values = []

    # 对结果按序列名排序，保证输出顺序稳定
    sorted_seqs = sorted(details.keys())

    for seq_name in sorted_seqs:
        res = details[seq_name]

        # 获取基础信息
        video_class = seq_meta.get(seq_name, {}).get("class", "Unknown")

        row = {
            "Sequence": seq_name,
            "Class": video_class,
            "BD-VMAF": "",
            # 预留 4 个档位的列
            "Test_Bitrate_VeryLow": "",
            "Test_VMAF_VeryLow": "",
            "Test_Bitrate_Low": "",
            "Test_VMAF_Low": "",
            "Test_Bitrate_Medium": "",
            "Test_VMAF_Medium": "",
            "Test_Bitrate_High": "",
            "Test_VMAF_High": "",
        }

        # 检查是否有错误
        if "error" in res and "test_points" not in res:
            row["BD-VMAF"] = f"ERROR: {res['error']}"
            csv_rows.append(row)
            continue

        # 获取测试点数据
        # test_points 是 list of (bitrate, vmaf)
        # 通常 Evaluator 生成任务的顺序是按 profile 定义的 (Very Low -> High)
        # 但为了保险，我们按码率排序
        test_points = res.get("test_points", [])
        test_points.sort(key=lambda x: x[0])  # 按码率升序

        # 填充 CSV 列 (假设最多 4 个点)
        profiles = ["VeryLow", "Low", "Medium", "High"]
        for i, (br, vmaf) in enumerate(test_points):
            if i < 4:
                p_name = profiles[i]
                row[f"Test_Bitrate_{p_name}"] = round(br, 2)
                row[f"Test_VMAF_{p_name}"] = round(vmaf, 2)

        # 计算/获取 BD-VMAF
        # Evaluator 可能已经算好了存为 "bd_vmaf"，或者我们需要重算
        # 为了严谨，这里利用 details 中的 anchor_points 重算一次 (确保对比的是 Baseline)

        bd_score = -9999.0

        if (
            "anchor_points" in res
            and len(res["anchor_points"]) >= 4
            and len(test_points) >= 4
        ):
            try:
                # 再次调用 metric 库计算
                bd_score = calculate_bd_vmaf(res["anchor_points"], test_points)
            except Exception:
                bd_score = res.get("bd_vmaf", -9999.0)
        elif "bd_vmaf" in res:
            bd_score = res["bd_vmaf"]

        if bd_score != -9999.0:
            row["BD-VMAF"] = round(bd_score, 4)
            bd_vmaf_values.append(bd_score)
        else:
            row["BD-VMAF"] = "Calc Failed"

        csv_rows.append(row)

    # 计算平均分
    avg_bd_vmaf = np.mean(bd_vmaf_values) if bd_vmaf_values else 0.0
    print(f"\n[Summary] Processed {len(csv_rows)} sequences.")
    print(f"[Result] Average BD-VMAF: {avg_bd_vmaf:.4f}")

    # 添加汇总行
    summary_row = {
        "Sequence": "AVERAGE",
        "Class": "ALL",
        "BD-VMAF": round(avg_bd_vmaf, 4),
    }
    csv_rows.append(summary_row)

    # 写入 CSV
    headers = [
        "Sequence",
        "Class",
        "BD-VMAF",
        "Test_Bitrate_VeryLow",
        "Test_VMAF_VeryLow",
        "Test_Bitrate_Low",
        "Test_VMAF_Low",
        "Test_Bitrate_Medium",
        "Test_VMAF_Medium",
        "Test_Bitrate_High",
        "Test_VMAF_High",
    ]

    try:
        with open(REPORT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        print("Report saved successfully.")
    except Exception as e:
        print(f"[Error] Failed to write CSV: {e}")


if __name__ == "__main__":
    main()
