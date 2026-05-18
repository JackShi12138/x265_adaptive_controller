import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
import matplotlib

# === 绘图配置 ===
INPUT_CSV = "frame_level_data.csv"
MODEL_CONFIG_PATH = "config/model_config.json"

TARGET_PARAM = "cutree-strength"  
TARGET_PARAM_COL = "param_cutree"  
BASE_PARAM_COL = "base_cutree"
LOOKAHEAD_WINDOW = 40

# === 样式配置 ===
COLOR_BG = "#A9A9A9"
COLOR_FG = "#8B0000"
ALPHA_BG = 0.35

plt.style.use("seaborn-v0_8-whitegrid")

# 字体配置
matplotlib.font_manager.fontManager.addfont("/home/shiyushen/font/times.ttf")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def calculate_phi_weights(model_config):
    phi = model_config["phi_matrix"]
    modules = model_config["modules"]
    feat_map = model_config["feature_mapping"]

    weights = {}
    for col_mod in modules:
        impact_sum = 0.0
        for row_mod in modules:
            val = phi.get(row_mod, {}).get(col_mod, 0.0)
            impact_sum += abs(val)

        feat_name = feat_map.get(col_mod)
        if feat_name:
            csv_name = (
                feat_name.replace("w1_", "feat_")
                .replace("w2_", "feat_")
                .replace("w3_", "feat_")
                .replace("w4_", "feat_")
                .replace("w5_", "feat_")
            )
            weights[csv_name] = impact_sum

    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total
    return weights

def main():
    print("Loading actual data...")
    df = pd.read_csv(INPUT_CSV)
    model_config = load_json(MODEL_CONFIG_PATH)

    weights = calculate_phi_weights(model_config)
    df["omega"] = 0.0
    for col, w in weights.items():
        if col in df.columns:
            df["omega"] += df[col] * w

    df["delta_p"] = df[TARGET_PARAM_COL] - df[BASE_PARAM_COL]

    df["omega_smooth"] = df["omega"].rolling(window=LOOKAHEAD_WINDOW, center=True).mean()
    df["delta_p_smooth"] = df["delta_p"].rolling(window=LOOKAHEAD_WINDOW, center=True).mean()
    df = df.bfill().ffill()

    sad_thresh = df["feat_sad"].quantile(0.97)
    scene_changes, _ = find_peaks(df["feat_sad"], height=sad_thresh, distance=60)

    # 画布设置：强制为单栏宽度 3.5 英寸
    fig, ax1 = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    t = df["frame_idx"]

    # --- 左轴: 综合特征 (背景) ---
    ax1.set_xlabel("Frame Index", fontsize=8, fontweight="bold")
    ax1.set_ylabel("Complexity ($\Omega_{agg}$)", color="gray", fontsize=8, fontweight="bold")

    ax1.fill_between(t, df["omega_smooth"], color=COLOR_BG, alpha=ALPHA_BG, edgecolor="none")
    ax1.plot(t, df["omega_smooth"], color=COLOR_BG, alpha=0.5, linewidth=0.8)

    ax1.tick_params(axis="both", which="major", labelsize=7, labelcolor="gray")
    
    # 左侧 Y 轴顶部留白 20%
    ax1.set_ylim(0, df["omega_smooth"].max() * 1.20)
    ax1.grid(True, which="major", linestyle=":", alpha=0.6, linewidth=0.5)

    # --- 右轴: 参数偏移 (前景) ---
    ax2 = ax1.twinx()
    param_label = f"Adj. ($\Delta$ {TARGET_PARAM.split('-')[0]})"
    ax2.set_ylabel(param_label, color=COLOR_FG, fontsize=8, fontweight="bold")

    ax2.plot(t, df["delta_p_smooth"], color=COLOR_FG, linewidth=1.5, linestyle="-")
    ax2.tick_params(axis="y", labelcolor=COLOR_FG, labelsize=7)

    # 为右侧 Y 轴动态增加 25% 的顶部空间，防止 Max Response 标注顶破边界
    y2_min, y2_max = df["delta_p_smooth"].min(), df["delta_p_smooth"].max()
    y2_range = y2_max - y2_min
    ax2.set_ylim(y2_min - y2_range * 0.05, y2_max + y2_range * 0.25)

    # --- 标注优化 ---
    if len(scene_changes) > 0:
        for i, idx in enumerate(scene_changes):
            ax1.axvline(x=idx, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
            if i == 0:
                # 移除换行符和 bbox，实现单行清爽显示
                ax1.text(idx + 4, ax1.get_ylim()[1] * 0.88, "Scene Change", 
                         ha="left", va="center", fontsize=6.5, color="black", alpha=0.85)

    max_idx = df["delta_p_smooth"].idxmax()
    max_val = df["delta_p_smooth"].max()

    # 细线箭头，相对位移锁定位置
    ax2.annotate(
        "Max Response",
        xy=(max_idx, max_val),
        xytext=(0, 15), 
        textcoords="offset points", 
        arrowprops=dict(arrowstyle="-|>", color=COLOR_FG, lw=1.2, mutation_scale=8),
        ha="center",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color=COLOR_FG,
    )

    # --- 图例 ---
    patch_bg = mpatches.Patch(color=COLOR_BG, alpha=ALPHA_BG, label="Video Complexity ($\Omega$)")
    line_fg = plt.Line2D([0], [0], color=COLOR_FG, linewidth=1.5, label=f"Adaptive Adj. ($\Delta$)")

    ax1.legend(
        handles=[patch_bg, line_fg],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,                     
        frameon=False,              
        fontsize=7,
        handletextpad=0.4,
        columnspacing=1.5
    )

    plt.tight_layout()
    output_img = "parameter_evolution_final.pdf"
    plt.savefig(output_img, bbox_inches="tight", pad_inches=0.02)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    main()