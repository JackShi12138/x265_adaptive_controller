import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import os

# ==================== 1. 全局配置区域 ====================

FILES = {
    "Baseline": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/slow/coeffs_baseline.txt",
    "Offline": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/offline/coeffs_offline.txt",
    "Online": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/online/coeffs_online.txt",
}

# 纹理区固定使用 8x8 (Depth 3)
TEXTURE_SIZE = 8

# 8x8 Zig-Zag 扫描表
ZIGZAG_8x8 = np.array(
    [
        0,
        1,
        5,
        6,
        14,
        15,
        27,
        28,
        2,
        4,
        7,
        13,
        16,
        26,
        29,
        42,
        3,
        8,
        12,
        17,
        25,
        30,
        41,
        43,
        9,
        11,
        18,
        24,
        31,
        40,
        44,
        53,
        10,
        19,
        23,
        32,
        39,
        45,
        52,
        54,
        20,
        22,
        33,
        38,
        46,
        51,
        55,
        60,
        21,
        34,
        37,
        47,
        50,
        56,
        59,
        61,
        35,
        36,
        48,
        49,
        57,
        58,
        62,
        63,
    ]
)

# ==================== 2. 核心逻辑 ====================


def get_energy_spectrum(coeffs_raster, width):
    """提取大块的左上角 8x8 低频能量"""
    try:
        matrix = coeffs_raster.reshape((width, width))
    except ValueError:
        return np.zeros(64)

    # 无论块多大，都只取左上角 8x8 (低频核心)
    if width >= 8:
        block_8x8 = matrix[0:8, 0:8]
    else:
        return np.zeros(64)

    energy_grid = np.square(block_8x8.astype(float))
    return energy_grid.flatten()[ZIGZAG_8x8]


def scan_file_structure(filepath):
    """预扫描文件，统计各尺寸块的数量，决定用哪个尺寸作为平坦区"""
    counts = {64: 0, 32: 0, 16: 0, 8: 0}

    if not os.path.exists(filepath):
        return counts

    print(f"正在全量扫描文件结构 (这可能需要几秒钟): {os.path.basename(filepath)} ...")

    with open(filepath, "r") as f:
        # 【修改点】移除计数限制，读取所有行
        for line in f:
            if line.startswith("COEFF_DUMP:"):
                try:
                    parts = line.split()
                    w = int(parts[1])
                    if w in counts:
                        counts[w] += 1
                except:
                    pass
    return counts


def determine_flat_size():
    """根据 Baseline 的数据情况，决定平坦区使用 64 还是 32"""
    baseline_path = FILES.get("Baseline")
    if not baseline_path:
        return 32  # 默认回退

    counts = scan_file_structure(baseline_path)
    print(
        f"  -> 全量样本统计: 64x64={counts[64]}, 32x32={counts[32]}, 16x16={counts[16]}"
    )

    # 策略：只要有 64x64 (哪怕只有几个)，也优先展示，因为它们物理意义最强
    if counts[64] > 0:
        print("  ✅ 选中: 64x64 (Depth 0)")
        return 64
    elif counts[32] > 0:
        print("  ✅ 选中: 32x32 (Depth 1)")
        return 32
    else:
        print("  ⚠️ 警告: 未检测到 64x64 或 32x32，将被迫使用 16x16")
        return 16


def parse_coeff_file(filepath, target_sizes):
    print(f"正在解析并计算能量: {os.path.basename(filepath)} ...")
    accum = {size: np.zeros(64) for size in target_sizes}
    counts = {size: 0 for size in target_sizes}

    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("COEFF_DUMP:"):
                try:
                    parts = line.split()
                    w = int(parts[1])

                    if w in target_sizes:
                        # 提取系数
                        raw_coeffs = np.array([int(x) for x in parts[4:]])
                        spectrum = get_energy_spectrum(raw_coeffs, w)
                        accum[w] += spectrum
                        counts[w] += 1
                except:
                    continue

    avg_data = {}
    for size in target_sizes:
        if counts[size] > 0:
            avg_data[size] = accum[size] / counts[size]
            print(f"  -> {size}x{size}: 捕获 {counts[size]} 个样本")
        else:
            avg_data[size] = np.zeros(64)
            print(f"  ⚠️ 警告: {size}x{size} 样本数为 0")

    return avg_data


# ==================== 3. 绘图主程序 ====================


def plot_final():
    # 1. 动态决定分析尺寸
    flat_size = determine_flat_size()
    target_sizes = [flat_size, TEXTURE_SIZE]

    # 2. 加载数据
    data_map = {}
    for label, path in FILES.items():
        data_map[label] = parse_coeff_file(path, target_sizes)

    if "Baseline" not in data_map or data_map["Baseline"] is None:
        print("错误: 缺少 Baseline 数据")
        return

    # 3. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # 子图标题自动适配
    title_flat = f"Depth 0 (64x64)" if flat_size == 64 else f"Depth 1 (32x32)"
    if flat_size == 16:
        title_flat = "Depth 2 (16x16) [Fallback]"

    scenarios = [
        (flat_size, f"{title_flat}: Flat / Protected Areas", axes[0]),
        (TEXTURE_SIZE, f"Depth 3 (8x8): Texture / Masked Areas", axes[1]),
    ]

    styles = {
        "Offline": {"color": "#3498db", "label": "Offline", "ls": "-"},
        "Online": {"color": "#e74c3c", "label": "Online (Proposed)", "ls": "-"},
    }

    x_axis = np.arange(64)

    for size, title, ax in scenarios:
        base_E = data_map["Baseline"][size] + 1e-6
        ax.axhline(
            1.0, color="gray", linestyle="--", linewidth=1.5, label="Baseline (Ref)"
        )

        all_y_values = [1.0]

        for algo in ["Offline", "Online"]:
            if algo in data_map and data_map[algo] is not None:
                algo_E = data_map[algo][size]

                # 计算比率
                ratio = algo_E / base_E

                # 平滑处理
                sigma = 2.0 if size == flat_size else 1.5
                ratio_smooth = gaussian_filter1d(ratio, sigma=sigma)

                all_y_values.extend(ratio_smooth)

                st = styles[algo]
                ax.plot(
                    x_axis,
                    ratio_smooth,
                    color=st["color"],
                    label=st["label"],
                    linewidth=2.5,
                )

        # 智能 Y 轴
        y_min, y_max = min(all_y_values), max(all_y_values)
        pad = (y_max - y_min) * 0.2 if y_max != y_min else 0.2
        ax.set_ylim(max(0, y_min - pad), y_max + pad)
        ax.set_xlim(0, 63)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Frequency Index (Low -> High)", fontsize=10)
        ax.set_ylabel("Coefficient Energy Ratio", fontsize=10)

        ax.text(1, y_max, "Low Freq", color="gray", fontsize=9, va="bottom")
        ax.text(50, y_max, "High Freq", color="gray", fontsize=9, va="bottom")

        if size == TEXTURE_SIZE:
            ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)

    plt.suptitle(
        "Fig. C: Adaptive Coefficient Energy Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig("Fig_C_Adaptive_Coeffs_FullScan.png", dpi=300)
    print("✅ 完成！")


if __name__ == "__main__":
    plot_final()
