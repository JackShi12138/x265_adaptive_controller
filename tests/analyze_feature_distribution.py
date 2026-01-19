import sys
import os
import json
import argparse
import numpy as np
from tabulate import (
    tabulate,
)  # 如果没有安装，可以使用 pip install tabulate，或者我手动实现简单的对齐打印

# 将项目根目录添加到 python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from core.feature_extractor import FeatureExtractor
from utils.yuv_io import YUVReader

# 默认数据集路径 (请根据实际情况修改或通过命令行传入)
DEFAULT_DATASET_ROOT = "/home/shiyushen/x265_sequence/"


def analyze_sequence(name, meta, dataset_root, num_gops=3):
    """
    分析单个序列的特征分布
    """
    # 1. 构造文件路径
    # 尝试多种可能的路径结构
    candidates = [
        os.path.join(dataset_root, f"Class{meta.get('class', '')}", f"{name}.yuv"),
        os.path.join(dataset_root, f"{name}.yuv"),
        os.path.join(dataset_root, meta.get("path", "")),  # 兼容某些json格式
    ]

    file_path = None
    for p in candidates:
        if os.path.exists(p):
            file_path = p
            break

    if not file_path:
        return None, "File Not Found"

    # 2. 初始化提取器
    try:
        reader = YUVReader(file_path, meta["width"], meta["height"], 8, meta["fps"])
        # 使用较短的 GOP 以便快速采样
        extractor = FeatureExtractor(reader, gop_size=40, processing_width=256)
    except Exception as e:
        return None, f"Init Failed: {str(e)}"

    # 3. 提取特征
    stats = {"w1_var": [], "w2_sad": [], "w3_grad": []}

    try:
        for _ in range(num_gops):
            success, feats = extractor.get_next_gop_features()
            if not success:
                break
            stats["w1_var"].append(feats.get("w1_var", 0))
            stats["w2_sad"].append(feats.get("w2_sad", 0))
            stats["w3_grad"].append(feats.get("w3_grad", 0))
    except Exception as e:
        reader.close()
        return None, f"Extract Error: {str(e)}"

    reader.close()

    if not stats["w1_var"]:
        return None, "No Data"

    # 4. 计算统计量
    result = {}
    warnings = []

    for key, values in stats.items():
        vals = np.array(values)
        _max = np.max(vals)
        _mean = np.mean(vals)

        result[f"{key}_max"] = _max
        result[f"{key}_mean"] = _mean

        # === 自动诊断逻辑 ===
        # 1. 饱和检测 (Saturation): 过于接近 1.0
        if _mean > 0.95:
            warnings.append(f"{key} SATURATED (Mean>0.95)")
        elif _max > 0.99:
            warnings.append(f"{key} Hit Ceiling (Max>0.99)")

        # 2. 消失检测 (Vanishing): 过于接近 0.0
        # 对于 SAD，如果是 SCV (Class F) 或者是静态视频，接近 0 是正常的
        # 但如果是自然视频 (Class A/B/C/D/E)，SAD 长期 < 0.01 说明归一化分母太大了
        is_scv = meta.get("class") in ["F", "Screen"]

        if _max < 0.001:
            warnings.append(f"{key} DEAD (Max<0.001)")
        elif _max < 0.02:
            if key == "w2_sad" and is_scv:
                pass  # SCV SAD 小是正常的
            else:
                warnings.append(f"{key} VANISHING (Max<0.02)")

    return result, warnings


def main():
    parser = argparse.ArgumentParser(description="Full scale feature extraction test")
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET_ROOT, help="Path to YUV dataset root"
    )
    parser.add_argument(
        "--gops", type=int, default=3, help="Number of GOPs to scan per video"
    )
    args = parser.parse_args()

    config_path = os.path.join(project_root, "config", "test_sequences.json")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path, "r") as f:
        sequences = json.load(f)

    print(f"Loaded {len(sequences)} sequences from config.")
    print(f"Dataset Root: {args.dataset}")
    print("-" * 100)
    print(
        f"{'Seq Name':<25} | {'Class':<5} | {'Var(Max)':<8} | {'SAD(Max)':<8} | {'Grad(Max)':<8} | {'Status/Warnings'}"
    )
    print("-" * 100)

    for name, meta in sequences.items():
        # 可以在这里过滤只跑训练集，或者跑全集
        # if name not in TRAINING_SET: continue

        res, warnings = analyze_sequence(name, meta, args.dataset, args.gops)

        video_class = meta.get("class", "?")

        if res is None:
            print(
                f"{name[:25]:<25} | {video_class:<5} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | ❌ {warnings}"
            )
            continue

        # 格式化输出
        var_str = f"{res['w1_var_max']:.4f}"
        sad_str = f"{res['w2_sad_max']:.4f}"
        grad_str = f"{res['w3_grad_max']:.4f}"

        # 颜色标记 (需要终端支持ANSI)
        status = "OK"
        if warnings:
            status = "⚠️ " + ", ".join(warnings)

        print(
            f"{name[:25]:<25} | {video_class:<5} | {var_str:<8} | {sad_str:<8} | {grad_str:<8} | {status}"
        )


if __name__ == "__main__":
    main()

# python3 tests/analyze_feature_distribution.py --dataset /home/shiyushen/x265_sequence/
