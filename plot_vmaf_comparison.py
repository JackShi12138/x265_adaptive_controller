import os
import json
import glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter

# === 配置区域 ===
# 必须与 collect_analysis_data.py 中的路径一致
DATASET_ROOT = "/home/shiyushen/x265_sequence/"
ANALYSIS_ROOT = "analysis_data"

# [请修改] VMAF 可执行文件路径 (vmafossexec 或 vmaf)
VMAF_EXEC = "vmaf"

# 绘图平滑窗口 (帧数)
SMOOTH_WINDOW = 15

# 图表保存路径
PLOT_OUTPUT_DIR = "analysis/plots"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_latest_experiment_dir():
    """自动寻找 analysis_data 下最新的时间戳目录"""
    dirs = glob.glob(os.path.join(ANALYSIS_ROOT, "*"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        return None
    # 按修改时间排序
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir


def run_vmaf_calculation(ref_yuv, dist_yuv, width, height, log_path):
    """
    [Updated] 使用官方 vmaf/vmafossexec 命令行工具计算 VMAF
    """
    if os.path.exists(log_path):
        print(f"  [Skip] VMAF log exists: {log_path}")
        return load_vmaf_log(log_path)

    print(f"  [Calc] Computing VMAF for {os.path.basename(dist_yuv)}...")

    # 构造命令 (参考您提供的代码)
    cmd_vmaf = [
        VMAF_EXEC,
        "-r",
        ref_yuv,  # Reference
        "-d",
        dist_yuv,  # Distorted
        "-w",
        str(width),
        "-h",
        str(height),
        "-p",
        "420",
        "-b",
        "8",
        "--json",
        "-o",
        log_path,
    ]

    try:
        # 使用 check=True 确保报错时抛出异常
        subprocess.run(
            cmd_vmaf, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return load_vmaf_log(log_path)
    except subprocess.CalledProcessError:
        print(f"  [Error] VMAF calculation failed for {dist_yuv}")
        # 打印一下命令以便调试
        print(f"  Command: {' '.join(cmd_vmaf)}")
        return None
    except FileNotFoundError:
        print(f"  [Error] Failed to run {VMAF_EXEC}. Is it executable?")
        return None


def load_vmaf_log(log_path):
    """解析 VMAF JSON 日志"""
    try:
        data = load_json(log_path)
        # 提取每一帧的 VMAF 分数
        # 结构通常为: { "frames": [ { "metrics": { "vmaf": 96.5 } }, ... ] }
        scores = [frame["metrics"]["vmaf"] for frame in data["frames"]]
        return np.array(scores)
    except Exception as e:
        print(f"  [Error] Failed to parse VMAF log: {e}")
        return None


def smooth_data(data, window_length):
    """滑动平均平滑"""
    if len(data) < window_length:
        return data
    window = np.ones(window_length) / window_length
    return np.convolve(data, window, mode="same")


def plot_sequence_comparison(seq_name, vmaf_data, output_path):
    """
    绘制 2x1 对比图
    """
    # 确保所有数据长度一致 (以最短的为准)
    min_len = min(len(v) for v in vmaf_data.values())
    for k in vmaf_data:
        vmaf_data[k] = vmaf_data[k][:min_len]

    frames = np.arange(min_len)

    # 平滑数据
    slow_smooth = smooth_data(vmaf_data["slow"], SMOOTH_WINDOW)
    offline_smooth = smooth_data(vmaf_data["offline"], SMOOTH_WINDOW)
    online_smooth = smooth_data(vmaf_data["online"], SMOOTH_WINDOW)

    # 计算 Delta (Online - Offline)
    delta_vmaf = online_smooth - offline_smooth

    # === 开始绘图 ===
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- 上图：绝对画质趋势 ---
    ax1.set_title(
        f"Per-Frame VMAF Comparison: {seq_name}", fontsize=16, fontweight="bold", pad=20
    )
    ax1.set_ylabel("VMAF Score", fontsize=12)

    # 绘制曲线
    ax1.plot(
        frames,
        slow_smooth,
        color="lightgray",
        linestyle="--",
        linewidth=1.5,
        label="Baseline (Slow)",
    )
    ax1.plot(
        frames,
        offline_smooth,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.8,
        label="Offline (Static Tuned)",
    )
    ax1.plot(
        frames, online_smooth, color="tab:red", linewidth=2.5, label="Online (Adaptive)"
    )

    # 标记最低点 (Worst-case)
    min_idx = np.argmin(online_smooth)
    min_val = online_smooth[min_idx]
    ax1.scatter(min_idx, min_val, color="red", zorder=5)
    ax1.annotate(
        f"Min: {min_val:.1f}",
        (min_idx, min_val - 2),
        color="red",
        ha="center",
        fontweight="bold",
    )

    ax1.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --- 下图：相对收益 Delta ---
    ax2.set_ylabel(r"$\Delta$ VMAF (Online - Offline)", fontsize=12)
    ax2.set_xlabel("Frame Index", fontsize=12)

    # 绘制参考线
    ax2.axhline(0, color="black", linewidth=1)

    # 填充面积图
    ax2.fill_between(
        frames,
        delta_vmaf,
        0,
        where=(delta_vmaf >= 0),
        color="green",
        alpha=0.3,
        interpolate=True,
        label="Gain",
    )
    ax2.fill_between(
        frames,
        delta_vmaf,
        0,
        where=(delta_vmaf < 0),
        color="tab:orange",
        alpha=0.4,
        interpolate=True,
        label="Loss",
    )

    # 绘制曲线轮廓
    ax2.plot(frames, delta_vmaf, color="darkgreen", linewidth=1, alpha=0.6)

    # 统计信息
    avg_gain = np.mean(delta_vmaf)
    ax2.text(
        0.02,
        0.9,
        f"Avg Gain: {avg_gain:+.2f}",
        transform=ax2.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        fontweight="bold",
    )

    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=10)

    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"  [Plot] Saved to {output_path}")
    plt.close()


def main():
    # 1. 寻找数据目录
    exp_dir = get_latest_experiment_dir()
    if not exp_dir:
        print("[Error] No analysis data found in analysis_data/")
        return
    print(f"=== Analyzing Experiment: {os.path.basename(exp_dir)} ===")
    print(f"Using VMAF Executable: {VMAF_EXEC}")

    # 2. 加载配置以获取分辨率
    seq_config = load_json("config/test_sequences.json")

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    # 3. 遍历所有序列文件夹
    seq_dirs = [d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)]

    for seq_path in seq_dirs:
        seq_name = os.path.basename(seq_path)
        if seq_name not in seq_config:
            # print(f"[Warn] Unknown sequence {seq_name}, skipping...")
            continue

        print(f"\nProcessing {seq_name}...")
        meta = seq_config[seq_name]
        width, height = meta["width"], meta["height"]

        # 寻找原始 YUV
        video_class = meta.get("class", "")
        class_folder = f"Class{video_class}" if video_class else ""
        ref_yuv = os.path.join(DATASET_ROOT, class_folder, f"{seq_name}.yuv")
        if not os.path.exists(ref_yuv):
            ref_yuv = os.path.join(DATASET_ROOT, f"{seq_name}.yuv")

        if not os.path.exists(ref_yuv):
            print(f"  [Error] Reference YUV not found: {ref_yuv}")
            continue

        # 收集三个模式的数据
        vmaf_data = {}
        modes = ["slow", "offline", "online"]

        for mode in modes:
            recon_yuv = os.path.join(seq_path, mode, "recon.yuv")
            log_path = os.path.join(seq_path, mode, "vmaf.json")

            if not os.path.exists(recon_yuv):
                print(f"  [Warn] Missing recon.yuv for {mode}")
                continue

            scores = run_vmaf_calculation(ref_yuv, recon_yuv, width, height, log_path)
            if scores is not None:
                vmaf_data[mode] = scores

        # 检查是否齐备
        if len(vmaf_data) == 3:
            plot_path = os.path.join(PLOT_OUTPUT_DIR, f"{seq_name}_vmaf_compare.png")
            plot_sequence_comparison(seq_name, vmaf_data, plot_path)
        else:
            print(
                f"  [Skip] Incomplete data for {seq_name}. Found modes: {list(vmaf_data.keys())}"
            )

    print(f"\nAll plots saved to {PLOT_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
