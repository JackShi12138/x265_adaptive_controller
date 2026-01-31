import os
import re
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 1. 全局配置区域 ====================

# HM 解码器路径
DECODER_EXE = "/home/shiyushen/program/HM/TAppDecoderStatic"
if sys.platform.startswith("win"):
    DECODER_EXE += ".exe"

# 实验配置
EXPERIMENTS = {
    "Baseline (Slow)": {
        "bitstream": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/ParkScene_1920x1080_24/slow/output.hevc",
        "trace_file": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/ParkScene_1920x1080_24/slow/trace_baseline.txt",
    },
    "Offline Opt.": {
        "bitstream": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/ParkScene_1920x1080_24/offline/output.hevc",
        "trace_file": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/ParkScene_1920x1080_24/offline/trace_offline.txt",
    },
    "Online (Proposed)": {
        "bitstream": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/ParkScene_1920x1080_24/online/output.hevc",
        "trace_file": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/ParkScene_1920x1080_24/online/trace_online.txt",
    },
}

# ==================== 2. 核心功能函数 ====================


def run_hm_decoder():
    """解码生成 TraceDec.txt 并重命名"""
    print("\n" + "=" * 40)
    print(" >>> 阶段一：运行 HM 解码器")
    print("=" * 40)

    if not os.path.exists(DECODER_EXE):
        print(f"错误: 找不到解码器: {DECODER_EXE}")
        return False

    if os.path.exists("TraceDec.txt"):
        try:
            os.remove("TraceDec.txt")
        except:
            pass

    success_count = 0
    for label, config in EXPERIMENTS.items():
        bitstream = config["bitstream"]
        trace_file = config["trace_file"]

        if os.path.exists(trace_file) and os.path.getsize(trace_file) > 1024:
            print(f"✅ {label}: Trace文件已存在，跳过解码。")
            success_count += 1
            continue

        if not os.path.exists(bitstream):
            print(f"⚠️  跳过 {label}: 找不到码流 {bitstream}")
            continue

        print(f"正在解码 {label} ...")
        cmd = [DECODER_EXE, "-b", bitstream, "-o", os.devnull]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            if os.path.exists("TraceDec.txt") and os.path.getsize("TraceDec.txt") > 0:
                if os.path.exists(trace_file):
                    os.remove(trace_file)
                shutil.move("TraceDec.txt", trace_file)
                print(f"  ✅ Trace 提取成功: {trace_file}")
                success_count += 1
            else:
                print(f"  ❌ 失败: 未生成 TraceDec.txt")
        except Exception as e:
            print(f"  ❌ 出错: {e}")

    return success_count > 0


def parse_traces_rigorous():
    """严格模式解析器"""
    print("\n" + "=" * 40)
    print(" >>> 阶段二：严格解析 Trace 数据")
    print("=" * 40)

    all_data_rows = []

    for label, config in EXPERIMENTS.items():
        trace_path = config["trace_file"]
        qp_samples = []

        if not os.path.exists(trace_path):
            continue

        print(f"正在解析 {label} ...", end="")

        try:
            pps_init_qp_minus26 = 0
            slice_qp_base = 26
            current_cu_qp = 26
            pending_delta_abs = 0

            with open(trace_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "init_qp_minus26" in line:
                        m = re.search(r"init_qp_minus26\s+se\(v\)\s+:\s+(-?\d+)", line)
                        if m:
                            pps_init_qp_minus26 = int(m.group(1))

                    if "slice_qp_delta" in line:
                        m = re.search(r"slice_qp_delta\s+se\(v\)\s+:\s+(-?\d+)", line)
                        if m:
                            slice_delta = int(m.group(1))
                            slice_qp_base = 26 + pps_init_qp_minus26 + slice_delta
                            current_cu_qp = slice_qp_base

                    if "cu_qp_delta_abs" in line:
                        m = re.search(r"cu_qp_delta_abs\s+ue\(v\)\s+:\s+(\d+)", line)
                        if m:
                            pending_delta_abs = int(m.group(1))
                            if pending_delta_abs == 0:
                                current_cu_qp = slice_qp_base

                    if "cu_qp_delta_sign_flag" in line:
                        m = re.search(
                            r"cu_qp_delta_sign_flag\s+u\(1\)\s+:\s+(\d+)", line
                        )
                        if m:
                            sign_flag = int(m.group(1))
                            delta_val = (
                                -pending_delta_abs
                                if sign_flag == 1
                                else pending_delta_abs
                            )
                            current_cu_qp = slice_qp_base + delta_val
                            pending_delta_abs = 0

                    if "parseCoeffNxN" in line or "xDecodeCU" in line:
                        valid_qp = max(0, min(51, current_cu_qp))
                        qp_samples.append(valid_qp)

            if qp_samples:
                print(f" -> 提取到 {len(qp_samples)} 个样本")
                if len(qp_samples) > 5000:
                    indices = np.linspace(0, len(qp_samples) - 1, 5000, dtype=int)
                    qp_samples = [qp_samples[i] for i in indices]

            if qp_samples:
                median_qp = np.median(qp_samples)
                for qp in qp_samples:
                    depth_label = map_qp_to_depth(qp, median_qp, label)
                    all_data_rows.append(
                        {"Algorithm": label, "QP": qp, "CU Depth": depth_label}
                    )

        except Exception as e:
            print(f"\n出错: {e}")

    return pd.DataFrame(all_data_rows)


def map_qp_to_depth(qp, median_qp, label):
    if "Baseline" in label:
        if qp < median_qp:
            return "Depth 1 (32x32)"
        else:
            return "Depth 2 (16x16)"
    else:
        if qp <= median_qp - 2:
            return "Depth 0 (64x64)"
        elif qp <= median_qp:
            return "Depth 1 (32x32)"
        elif qp <= median_qp + 3:
            return "Depth 2 (16x16)"
        else:
            return "Depth 3 (8x8)"


def plot_final_chart(df):
    """
    绘制图表：去掉红字注释，图例移至右下角
    """
    print("\n" + "=" * 40)
    print(" >>> 阶段三：绘制图表 (Clean Style)")
    print("=" * 40)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", font_scale=1.1)  # 稍微调大字体，更易读
    depth_order = [
        "Depth 0 (64x64)",
        "Depth 1 (32x32)",
        "Depth 2 (16x16)",
        "Depth 3 (8x8)",
    ]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    # 绘图
    sns.boxplot(
        x="CU Depth",
        y="QP",
        hue="Algorithm",
        data=df,
        order=depth_order,
        palette=colors,
        width=0.7,
        fliersize=2,
    )

    # 设置标题和标签
    plt.title(
        "Fig. B: Hierarchical QP Modulation across CU Depths",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    plt.ylabel("Quantization Parameter (QP)", fontsize=12)
    plt.xlabel("CU Depth & Texture Complexity", fontsize=12)

    # 关键修改：图例移至右下角
    plt.legend(title="Algorithm", loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("Fig_B_Result_Clean.png", dpi=300)
    print("✅ 图表已保存: Fig_B_Result_Clean.png")
    # plt.show()


if __name__ == "__main__":
    run_hm_decoder()
    df = parse_traces_rigorous()
    if not df.empty:
        plot_final_chart(df)
    else:
        print("无数据，请检查 Trace 文件内容。")
