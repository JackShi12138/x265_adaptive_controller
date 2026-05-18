import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
import matplotlib

# 加载自定义字体
matplotlib.font_manager.fontManager.addfont("/home/shiyushen/font/times.ttf")
plt.rcParams["font.family"] = "Times New Roman"

# ================= 配置区域 =================

# 数据根目录
DATA_ROOT = "analysis_data"

# 指定特定的文件夹名称
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
TARGET_SEQUENCES = [
    "RaceHorses_832x480_30",
    "PeopleOnStreet_2560x1600_30_crop",
    "ParkScene_1920x1080_24",
    "BasketballPass_416x240_50",
]

# 清晰度档位 (x轴排序依据)
PROFILES = ["Very Low", "Low", "Medium", "High"]

# ================= 样式配置 =================
METHODS = {
    "slow": {
        "label": "Baseline (Slow)",
        "color": "#555555",
        "marker": "o",
        "markersize": 5,
        "linestyle": "--",
        "linewidth": 0.4,
        "alpha": 0.7,
        "zorder": 1,
    },
    "ssim": {
        "label": "SSIM-based RDO",
        "color": "#9467bd",
        "marker": "D",
        "markersize": 5,
        "linestyle": "-.",
        "linewidth": 1.0,
        "markeredgecolor": "white",
        "markeredgewidth": 0.4,
        "zorder": 2,
    },
    "offline": {
        "label": "Offline (Static)",
        "color": "#1f77b4",
        "marker": "s",
        "markersize": 6,
        "linestyle": "-.",
        "linewidth": 1.1,
        "markeredgecolor": "white",
        "markeredgewidth": 0.4,
        "zorder": 3,
    },
    "online": {
        "label": "Online (Adaptive)",
        "color": "#d62728",
        "marker": "^",
        "markersize": 7,
        "linestyle": "-",
        "linewidth": 1.5,
        "markeredgecolor": "white",
        "markeredgewidth": 0.4,
        "zorder": 4,
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
    data = {}
    print("Loading static baseline/offline results...")
    baseline_db = load_static_json(BASELINE_JSON_PATH)
    offline_db = load_static_json(OFFLINE_JSON_PATH)

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
        seq_path = os.path.join(experiment_dir, seq)

        dynamic_methods = ["online", "ssim"]

        for method in dynamic_methods:
            points = []
            for profile in PROFILES:
                method_path = os.path.join(seq_path, profile, method)
                csv_file = os.path.join(method_path, "x265_log.csv")
                vmaf_file = os.path.join(method_path, "vmaf.json")

                if os.path.exists(csv_file) and os.path.exists(vmaf_file):
                    br, score = extract_online_metrics(vmaf_file, csv_file, fps)
                    if br is not None and score is not None:
                        points.append((br, score))

            if points:
                data[seq][method] = sorted(points, key=lambda x: x[0])
            else:
                data[seq][method] = []

        def get_static_points(db, key):
            raw_list = db.get(key, [])
            points = []
            for item in raw_list:
                if "bitrate" in item and "vmaf" in item:
                    points.append((float(item["bitrate"]), float(item["vmaf"])))
            return sorted(points, key=lambda x: x[0])

        data[seq]["slow"] = get_static_points(baseline_db, seq)
        data[seq]["offline"] = get_static_points(offline_db, seq)

    return data


# ================= 绘图函数 =================


def setup_ieee_style():
    # 彻底关闭自动排版干扰
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
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
    top = min(100.0, max_v + padding)
    if top - bottom < 5:
        bottom = top - 5
    return (bottom, top)


def plot_individual_curves(data):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for seq, methods_data in data.items():
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
            plt.plot(x, y, **{k: v for k, v in style.items() if k != "zorder"})

            plt.plot(
                [],
                [],
                label=style["label"],
                **{k: v for k, v in style.items() if k not in ["zorder", "label"]},
            )

        plt.clf()
        for method in sorted_methods:
            points = methods_data.get(method, [])
            if not points:
                continue
            x, y = zip(*points)
            style = METHODS[method]
            plt.plot(
                x,
                y,
                label=style["label"],
                **{k: v for k, v in style.items() if k not in ["zorder", "label"]},
            )

        plt.xlabel("Bitrate (kbps)")
        plt.ylabel("VMAF Score")
        plt.grid(True, linestyle=":", alpha=0.6)
        ylim = get_smart_ylim(methods_data)
        plt.ylim(ylim)
        plt.legend(frameon=True, fancybox=True, framealpha=0.9)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{seq}_RD_Curve.pdf"))
        plt.close()


def plot_combined_grid(data):
    ordered_seqs = [
        "PeopleOnStreet_2560x1600_30_crop",
        "ParkScene_1920x1080_24",
        "RaceHorses_832x480_30",
        "BasketballPass_416x240_50",
    ]

    plot_seqs = [seq for seq in ordered_seqs if seq in data]
    if not plot_seqs:
        return

    title_map = {
        "PeopleOnStreet_2560x1600_30_crop": "(a) PeopleOnStreet",
        "ParkScene_1920x1080_24": "(b) ParkScene",
        "RaceHorses_832x480_30": "(c) RaceHorses",
        "BasketballPass_416x240_50": "(d) BasketballPass",
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.6))

    sorted_methods = sorted(METHODS.keys(), key=lambda m: METHODS[m]["zorder"])

    for i, ax in enumerate(axes):
        seq = plot_seqs[i]
        methods_data = data[seq]

        for method in sorted_methods:
            points = methods_data.get(method, [])
            if not points:
                continue
            x, y = zip(*points)

            style = METHODS[method].copy()
            style["markersize"] = max(3.5, style.get("markersize", 5) - 1.5)
            style["linewidth"] = max(0.8, style.get("linewidth", 1.0) - 0.2)

            lbl = style["label"] if i == 0 else ""
            ax.plot(
                x,
                y,
                label=lbl,
                **{k: v for k, v in style.items() if k not in ["zorder", "label"]},
            )

        display_title = title_map.get(seq, f"({chr(97+i)}) {seq.split('_')[0]}")

        ax.set_xlabel(f"Bitrate (kbps)\n\n{display_title}", fontsize=8, labelpad=3)
        ax.grid(True, linestyle=":", alpha=0.6)

        ylim = get_smart_ylim(methods_data)
        ax.set_ylim(ylim)

        if i == 0:
            ax.set_ylabel("VMAF Score", fontsize=8, labelpad=2)

        ax.tick_params(axis="both", which="major", labelsize=7, pad=2)
        ax.locator_params(axis="x", nbins=4)

    # ================= 【终极物理布局重构】 =================
    if len(plot_seqs) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.82),  # 图例锚点设置在整个画面高度 82% 的绝对位置
            ncol=4,
            frameon=False,
            fontsize=8,
            handletextpad=0.4,
            columnspacing=1.2,
        )

    # 彻底弃用 plt.tight_layout()！
    # 使用 subplots_adjust 强行划定地盘：
    # top=0.80 意味着所有子图的顶部绝对不能超过画布 80% 的高度，正好顶在图例下方
    # bottom=0.25 给 X 轴下面的换行文字留出充足空间
    # left 和 right 控制左右贴边，wspace=0.15 保证子图间距均匀
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.25, top=0.80, wspace=0.15)

    output_path = os.path.join(OUTPUT_DIR, "Combined_RD_Grid.pdf")
    # pad_inches=0.01 将 PDF 边缘裁剪到极致
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved 1x4 Grid: {output_path}")
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
