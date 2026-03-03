import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager


# ================= 配置区域 =================

# 数据根目录
DATA_ROOT = "analysis_data"

# 【修改点】指定特定的文件夹名称
# 如果留空 ""，则自动寻找 analysis_data 下最新的时间戳文件夹
SPECIFIC_DIR_NAME = "rdcurve_ssim_208"

# 输出目录
OUTPUT_DIR = "rd_plots_final"

BASELINE_JSON_PATH = (
    "/home/shiyushen/x265_adaptive_controller/config/baseline_results_91.json"
)
OFFLINE_JSON_PATH = (
    "/home/shiyushen/x265_adaptive_controller/config/offline_results_91.json"
)

# 需要绘制的序列
# 如果留空 []：自动扫描所有序列，但汇总图只取前 4 个
# 如果指定 ["SeqA", "SeqB"]：只处理这两个
TARGET_SEQUENCES = [
    "RaceHorses_832x480_30",
    "PeopleOnStreet_2560x1600_30_crop",
    "ParkScene_1920x1080_24",
    "BasketballPass_416x240_50",
]

# 清晰度档位 (x轴排序依据)
PROFILES = ["Very Low", "Low", "Medium", "High"]

# ================= [核心修改] 样式增强配置 (整体变细) =================
METHODS = {
    # Baseline: 作为背景参考，最细的虚线
    "slow": {
        "label": "Baseline (Slow)",
        "color": "#555555",
        "marker": "o",
        "markersize": 5,  # 略微减小
        "linestyle": "--",
        "linewidth": 0.4,  # [修改] 变细
        "alpha": 0.7,
        "zorder": 1,
    },
    # Offline: 中等粗细
    "offline": {
        "label": "Offline (Static)",
        "color": "#1f77b4",
        "marker": "s",
        "markersize": 6,  # 略微减小
        "linestyle": "-.",
        "linewidth": 1.1,  # [修改] 变细
        "markeredgecolor": "white",
        "markeredgewidth": 0.4,  # [修改] 描边变细
        "zorder": 2,
    },
    # Online: 主角，相对最粗，但整体比之前细
    "online": {
        "label": "Online (Adaptive)",
        "color": "#d62728",
        "marker": "^",
        "markersize": 7,  # 略微减小
        "linestyle": "-",
        "linewidth": 1.5,  # [修改] 变细
        "markeredgecolor": "white",
        "markeredgewidth": 0.4,  # [修改] 描边变细
        "zorder": 3,
    },
}

# ================= 工具函数 =================


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


def extract_fps(seq_name):
    try:
        return int(seq_name.split("_")[-1])
    except:
        return 30


def parse_x265_bitrate(csv_file, fps):
    try:
        try:
            df = pd.read_csv(csv_file, on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(csv_file, error_bad_lines=False)
        bits_col = next(
            (col for col in df.columns if col.strip() in ["Bits", "size(bits)"]), None
        )
        if bits_col:
            total_bits = df[bits_col].sum()
            duration = len(df) / fps if fps > 0 else 0
            if duration > 0:
                return (total_bits / 1000.0) / duration
    except:
        pass
    return None


def extract_online_metrics(vmaf_path, csv_path, fps):
    """读取 Online 的实时数据"""
    vmaf, bitrate = None, None
    try:
        with open(vmaf_path, "r") as f:
            data = json.load(f)
            if "pooled_metrics" in data:
                vmaf = data["pooled_metrics"]["vmaf"]["mean"]
            elif "frames" in data:
                vmaf = np.mean([f["metrics"]["vmaf"] for f in data["frames"]])
    except:
        pass
    bitrate = parse_x265_bitrate(csv_path, fps)
    return bitrate, vmaf


def load_static_json(path):
    """加载 Baseline 和 Offline 的 JSON 数据"""
    if not os.path.exists(path):
        print(f"[Warning] Static data file not found: {path}")
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load {path}: {e}")
        return {}


def collect_data(experiment_dir):
    """
    混合数据收集逻辑：
    - Online: 从 experiment_dir 扫描
    - Slow/Offline: 从 JSON 文件查找
    """
    data = {}

    # 1. 加载静态基准数据
    print("Loading static baseline/offline results...")
    baseline_db = load_static_json(BASELINE_JSON_PATH)
    offline_db = load_static_json(OFFLINE_JSON_PATH)

    # 2. 扫描实验目录获取 Online 序列
    seq_dirs = sorted(
        [
            d
            for d in os.listdir(experiment_dir)
            if os.path.isdir(os.path.join(experiment_dir, d))
        ]
    )

    for seq in seq_dirs:
        if TARGET_SEQUENCES and seq not in TARGET_SEQUENCES:
            continue

        data[seq] = {}
        fps = extract_fps(seq)

        # === Part A: 读取 Online 数据 (动态) ===
        online_points = []
        seq_path = os.path.join(experiment_dir, seq)
        for profile in PROFILES:
            # 这里的路径结构假设是 Sequence -> Profile -> online
            method_path = os.path.join(seq_path, profile, "online")
            csv_file = os.path.join(method_path, "x265_log.csv")
            vmaf_file = os.path.join(method_path, "vmaf.json")

            if os.path.exists(csv_file) and os.path.exists(vmaf_file):
                br, score = extract_online_metrics(vmaf_file, csv_file, fps)
                if br is not None and score is not None:
                    online_points.append((br, score))

        # 按码率排序
        data[seq]["online"] = sorted(online_points, key=lambda x: x[0])

        # === Part B: 读取 Slow/Offline 数据 (静态) ===
        # 定义辅助函数：从 JSON 列表转为 [(br, vmaf), ...] 格式
        def get_static_points(db, key):
            raw_list = db.get(key, [])
            if not raw_list:
                # 尝试模糊匹配？暂时只做精确匹配
                return []
            points = []
            for item in raw_list:
                if "bitrate" in item and "vmaf" in item:
                    points.append((float(item["bitrate"]), float(item["vmaf"])))
            return sorted(points, key=lambda x: x[0])

        data[seq]["slow"] = get_static_points(baseline_db, seq)
        data[seq]["offline"] = get_static_points(offline_db, seq)

        # 调试信息
        # print(f"Sequence: {seq} | Online: {len(data[seq]['online'])} pts | Offline: {len(data[seq]['offline'])} pts")

    return data


# ================= 绘图函数 (保持不变) =================


def setup_ieee_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.autolayout": True,
        }
    )


def get_smart_ylim(methods_data):
    all_vmaf = []
    for points in methods_data.values():
        for p in points:
            all_vmaf.append(p[1])
    if not all_vmaf:
        return (0, 100)
    min_v, max_v = min(all_vmaf), max(all_vmaf)
    padding = (max_v - min_v) * 0.1 if (max_v - min_v) > 5 else 2.0
    bottom = max(0, min_v - padding)
    top = min(100.5, max_v + padding)
    if top - bottom < 5:
        bottom = top - 5
    return (bottom, top)


def plot_individual_curves(data):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for seq, methods_data in data.items():
        # 只要有任意一种数据就画图
        if sum(len(pts) for pts in methods_data.values()) == 0:
            continue

        plt.figure(figsize=(5, 4))
        sorted_methods = sorted(METHODS.keys(), key=lambda m: METHODS[m]["zorder"])
        for method in sorted_methods:
            points = methods_data.get(method, [])
            if not points:
                continue
            x, y = zip(*points)
            style = METHODS[method]
            plt.plot(
                x, y, **{k: v for k, v in style.items() if k not in ["zorder", "label"]}
            )
            # 单独画图时，label 需要加回去
            plt.plot(
                [],
                [],
                label=style["label"],
                **{k: v for k, v in style.items() if k not in ["zorder", "label"]},
            )

        # 修正：上面那种写法会导致双重绘图，正确写法如下：
        plt.clf()  # 清除刚才试探性的画图
        for method in sorted_methods:
            points = methods_data.get(method, [])
            if not points:
                continue
            x, y = zip(*points)
            style = METHODS[method]
            plt.plot(x, y, **{k: v for k, v in style.items() if k != "zorder"})

        # plt.title(seq.split("_")[0])
        plt.xlabel("Bitrate (kbps)")
        plt.ylabel("VMAF Score")
        plt.grid(True, linestyle=":", alpha=0.6)
        ylim = get_smart_ylim(methods_data)
        plt.ylim(ylim)
        plt.legend(frameon=True, fancybox=True, framealpha=0.9)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{seq}_RD_Curve.pdf"))
        plt.close()


def plot_combined_grid(data):
    valid_seqs = [s for s in data.keys() if sum(len(v) for v in data[s].values()) > 0]
    if not valid_seqs:
        return
    plot_seqs = valid_seqs[:4]

    # [修改1] 稍微增加一点图片高度，从 3.5 -> 3.7，缓解拥挤
    fig, axes = plt.subplots(2, 2, figsize=(3.8, 3.7))
    axes = axes.flatten()

    sorted_methods = sorted(METHODS.keys(), key=lambda m: METHODS[m]["zorder"])

    for i, ax in enumerate(axes):
        if i >= len(plot_seqs):
            ax.axis("off")
            continue
        seq = plot_seqs[i]
        methods_data = data[seq]

        for method in sorted_methods:
            points = methods_data.get(method, [])
            if not points:
                continue
            x, y = zip(*points)
            style = METHODS[method]
            lbl = style["label"] if i == 0 else ""

            small_style = style.copy()
            small_style["markersize"] -= 1
            small_style["linewidth"] = max(0.5, small_style["linewidth"] - 0.2)

            ax.plot(
                x,
                y,
                label=lbl,
                **{
                    k: v for k, v in small_style.items() if k not in ["zorder", "label"]
                },
            )

        ax.set_title(seq.split("_")[0], fontsize=9, pad=3)
        ax.grid(True, linestyle=":", alpha=0.6)

        ylim = get_smart_ylim(methods_data)
        ax.set_ylim(ylim)

        if i >= 2:
            ax.set_xlabel("Bitrate (kbps)", fontsize=8)
        if i % 2 == 0:
            ax.set_ylabel("VMAF", fontsize=8)

    if len(plot_seqs) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        # [修改2] 调整图例位置
        # loc='upper center': 以图例的顶部中心为锚点
        # bbox_to_anchor=(0.5, 0.99): 锚点位于图的水平中心(0.5)，垂直方向最顶部往下一丢丢(0.99)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            fontsize=8,
            handlelength=2.5,
        )

    # [修改3] 调整子图布局区域
    # rect=[0, 0, 1, 0.88]: 意思是子图只占用画布高度的 0% 到 88%，留下顶部 12% 给图例
    plt.tight_layout(rect=[0, 0, 1, 0.89])

    output_path = os.path.join(OUTPUT_DIR, "Combined_RD_Grid.pdf")
    plt.savefig(output_path)
    print(f"Saved Grid: {output_path}")
    plt.close()


if __name__ == "__main__":
    setup_ieee_style()
    target_dir = get_target_directory(DATA_ROOT, SPECIFIC_DIR_NAME)
    if target_dir:
        print(f"Reading Online data from: {target_dir}")
        all_data = collect_data(target_dir)
        print("Generating individual plots...")
        plot_individual_curves(all_data)
        print("Generating combined grid plot...")
        plot_combined_grid(all_data)
        print("Done.")
    else:
        print("Error: Target directory not found.")
