import os
import json
import glob
import subprocess
import numpy as np
import pandas as pd
import matplotlib

# 设置无头模式，必须在 import pyplot 之前
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_ROOT = "/home/shiyushen/x265_sequence/"
ANALYSIS_ROOT = "analysis_data"
VMAF_EXEC = "vmaf"
PLOT_OUTPUT_DIR = "analysis/plots_cbr_stability"

DATA_ROOT = "analysis_data"
SPECIFIC_DIR_NAME = "20260214_230748"

# [关键参数] 平滑时长 (秒)
# 作用：根据帧率自动计算平滑窗口，确保时域分析尺度一致
# 推荐：0.5 ~ 0.8 秒 (模拟人眼或 Lookahead 的感知窗口)
SMOOTH_DURATION = 0.6

# 并行工作进程数
MAX_WORKERS = max(1, os.cpu_count() // 2)

# ================= 工具函数 =================


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_target_directory(root, specific_name):
    if specific_name:
        target_path = os.path.join(root, specific_name)
        if os.path.exists(target_path):
            return target_path
        else:
            return None
    else:
        if not os.path.exists(root):
            return None
        dirs = [
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
        if not dirs:
            return None
        return max(dirs, key=os.path.getmtime)


def extract_fps_from_name(seq_name, default_fps=30):
    """从文件名尝试提取帧率，例如 RaceHorses_832x480_30 -> 30"""
    try:
        parts = seq_name.split("_")
        if len(parts) >= 3:
            # 尝试最后一部分是否为数字
            return int(parts[-1])
    except:
        pass
    return default_fps


def run_vmaf_calculation(ref_yuv, dist_yuv, width, height, log_path):
    """如果 VMAF log 不存在则运行计算"""
    if os.path.exists(log_path):
        return load_vmaf_log(log_path)

    cmd_vmaf = [
        VMAF_EXEC,
        "-r",
        ref_yuv,
        "-d",
        dist_yuv,
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
        subprocess.run(
            cmd_vmaf, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return load_vmaf_log(log_path)
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        return None


def load_vmaf_log(log_path):
    try:
        data = load_json(log_path)
        scores = [frame["metrics"]["vmaf"] for frame in data["frames"]]
        return np.array(scores)
    except Exception:
        return None


def smooth_data(data, window_length):
    """简单的滑动平均平滑"""
    if len(data) < window_length:
        return data
    window = np.ones(window_length) / window_length
    return np.convolve(data, window, mode="same")


# ================= 核心绘图逻辑 =================


def plot_cbr_comparison(seq_name, vmaf_data, output_path, fps):
    """
    绘制函数：CBR 模式下的稳定性分析 (Baseline vs Offline vs Online)
    """
    # 1. 动态计算窗口大小
    window_size = int(fps * SMOOTH_DURATION)
    window_size = max(1, window_size)  # 至少为1

    # 2. 对齐数据长度
    min_len = min(len(v) for v in vmaf_data.values())
    for k in vmaf_data:
        vmaf_data[k] = vmaf_data[k][:min_len]
    frames = np.arange(min_len)

    # 3. 数据平滑
    slow_smooth = smooth_data(vmaf_data["slow"], window_size)
    offline_smooth = smooth_data(vmaf_data["offline"], window_size)
    online_smooth = smooth_data(vmaf_data["online"], window_size)

    # 4. 计算关键指标
    # 差异：Online - Offline
    delta_vmaf = online_smooth - offline_smooth

    # 统计数据：使用原始数据计算统计量更准确
    std_offline = np.std(vmaf_data["offline"])
    std_online = np.std(vmaf_data["online"])
    # 最小值：使用平滑后的数据避免单帧噪点干扰
    min_offline = np.min(offline_smooth)
    min_online = np.min(online_smooth)

    # 5. 绘图初始化
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # === 上图：原始 VMAF 趋势 ===
    # ax1.set_title(
    #     f"Per-Frame VMAF Analysis: {seq_name} (Strict CBR)",
    #     fontsize=16,
    #     fontweight="bold",
    #     pad=15,
    # )
    ax1.set_ylabel("VMAF Score", fontsize=12)

    # Baseline (灰色背景板)
    ax1.plot(
        frames,
        slow_smooth,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Baseline (Slow)",
    )
    # Offline (蓝色参考线)
    ax1.plot(
        frames,
        offline_smooth,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.8,
        linestyle="--",
        label="Offline (Static)",
    )
    # Online (红色主角)
    ax1.plot(
        frames, online_smooth, color="#d62728", linewidth=2.5, label="Online (Adaptive)"
    )

    # 标注：最低点提升 (Quality Rescue)
    # 逻辑：在 Offline < 90 的区域里找最大的 positive diff
    low_quality_mask = offline_smooth < (np.mean(offline_smooth) + 5)
    if np.any(low_quality_mask):
        roi_indices = np.where(low_quality_mask)[0]
        if len(roi_indices) > 0:
            sub_delta = delta_vmaf[roi_indices]
            best_rescue_idx = roi_indices[np.argmax(sub_delta)]

            # 如果提升显著 (>0.5)，则标注
            if delta_vmaf[best_rescue_idx] > 0.5:
                ax1.annotate(
                    f"Min Boost: +{delta_vmaf[best_rescue_idx]:.1f}",
                    xy=(best_rescue_idx, online_smooth[best_rescue_idx]),
                    xytext=(best_rescue_idx, online_smooth[best_rescue_idx] + 5),
                    arrowprops=dict(facecolor="green", arrowstyle="->", lw=1.5),
                    color="green",
                    fontweight="bold",
                    ha="center",
                )

    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # === 下图：Delta VMAF (重分配逻辑) ===
    ax2.set_ylabel(r"$\Delta$ VMAF (Online - Offline)", fontsize=12)
    ax2.set_xlabel("Frame Index", fontsize=12)
    ax2.axhline(0, color="black", linewidth=0.8)

    # 填充颜色
    ax2.fill_between(
        frames, 0, delta_vmaf, where=(delta_vmaf >= 0), color="green", alpha=0.3
    )
    ax2.fill_between(
        frames, 0, delta_vmaf, where=(delta_vmaf < 0), color="tab:orange", alpha=0.3
    )
    ax2.plot(frames, delta_vmaf, color="#333333", linewidth=1, alpha=0.6)

    # --- 关键标注：解释削峰填谷 ---
    # 1. 找显著的绿色峰值 (Quality Boost)
    max_gain_idx = np.argmax(delta_vmaf)
    if delta_vmaf[max_gain_idx] > 0.5:
        ax2.annotate(
            "Quality Boost\n(Complex Region)",
            xy=(max_gain_idx, delta_vmaf[max_gain_idx]),
            xytext=(max_gain_idx, delta_vmaf[max_gain_idx] + 1.0),
            arrowprops=dict(facecolor="green", arrowstyle="->"),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="darkgreen",
        )

    # 2. 找显著的橙色谷底 (Bitrate Reallocation)
    min_loss_idx = np.argmin(delta_vmaf)
    if delta_vmaf[min_loss_idx] < -0.5:
        ax2.annotate(
            "Bitrate Reallocation\n(Simple Region)",
            xy=(min_loss_idx, delta_vmaf[min_loss_idx]),
            xytext=(min_loss_idx, delta_vmaf[min_loss_idx] - 1.5),
            arrowprops=dict(facecolor="chocolate", arrowstyle="->"),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="chocolate",
        )

    # --- 统计数据框 (Stability Metrics) ---
    stats_text = (
        f"Stability Metrics (CBR):\n"
        f"• Min VMAF: {min_offline:.1f} $\\rightarrow$ {min_online:.1f} "
        f"({min_online-min_offline:+.1f})\n"
        f"• Std Dev : {std_offline:.2f} $\\rightarrow$ {std_online:.2f} "
        f"({std_online-std_offline:.2f})"
    )

    ax2.text(
        0.02,
        0.92,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.9, edgecolor="gray"
        ),
    )

    # 设置 Y 轴范围，留出空间给标注
    y_max = max(abs(np.max(delta_vmaf)), abs(np.min(delta_vmaf)))
    ax2.set_ylim(-y_max * 1.5, y_max * 1.8)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# ================= 任务处理逻辑 =================


def process_single_sequence(args):
    """单个序列处理任务"""
    seq_path, seq_config = args
    seq_name = os.path.basename(seq_path)

    if seq_name not in seq_config:
        return f"[Skip] {seq_name} (Not in config)"

    meta = seq_config[seq_name]
    width, height = meta["width"], meta["height"]

    # [新增] 获取 FPS，用于计算平滑窗口
    # 优先从 config 读取，如果没有则从文件名解析，默认 30
    fps = meta.get("fps", extract_fps_from_name(seq_name))

    # 寻找 Reference YUV
    video_class = meta.get("class", "")
    class_folder = f"Class{video_class}" if video_class else ""
    ref_yuv = os.path.join(DATASET_ROOT, class_folder, f"{seq_name}.yuv")
    if not os.path.exists(ref_yuv):
        ref_yuv = os.path.join(DATASET_ROOT, f"{seq_name}.yuv")
    if not os.path.exists(ref_yuv):
        return f"[Error] {seq_name} Ref YUV not found"

    # 数据容器
    vmaf_data = {}
    modes = ["slow", "offline", "online"]

    # 读取循环
    for mode in modes:
        mode_dir = os.path.join(seq_path, mode)
        recon_yuv = os.path.join(mode_dir, "recon.yuv")
        vmaf_log = os.path.join(mode_dir, "vmaf.json")

        if os.path.exists(recon_yuv):
            scores = run_vmaf_calculation(ref_yuv, recon_yuv, width, height, vmaf_log)
            if scores is not None:
                vmaf_data[mode] = scores

    # 绘图条件：必须有 Offline 和 Online
    if "offline" in vmaf_data and "online" in vmaf_data:
        # 容错：如果 slow 缺失，用 offline 暂代（只画背景板）
        if "slow" not in vmaf_data:
            vmaf_data["slow"] = vmaf_data["offline"]

        plot_path = os.path.join(PLOT_OUTPUT_DIR, f"{seq_name}_cbr_stability.pdf")

        # [修改] 传入 fps
        plot_cbr_comparison(seq_name, vmaf_data, plot_path, fps)

        return f"[Done] {seq_name} stability plot generated (FPS={fps})."
    else:
        return f"[Skip] {seq_name} Missing Offline/Online data"


def main():
    exp_dir = get_target_directory(DATA_ROOT, SPECIFIC_DIR_NAME)
    if not exp_dir:
        print("[Error] No analysis data found.")
        return

    print(f"=== Analyzing: {os.path.basename(exp_dir)} (Strict CBR Mode) ===")

    try:
        seq_config = load_json("config/test_sequences.json")
    except FileNotFoundError:
        print("[Error] Config file not found.")
        return

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    seq_dirs = [d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)]
    tasks = [(d, seq_config) for d in seq_dirs]

    print(f"Starting processing for {len(tasks)} sequences...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_sequence, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), unit="seq"):
            pass

    print(f"\nAll plots saved to {PLOT_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
