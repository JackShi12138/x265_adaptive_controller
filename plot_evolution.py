import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks

# === 绘图配置 ===
INPUT_CSV = "frame_level_data.csv"
MODEL_CONFIG_PATH = "config/model_config.json"

# [核心配置] 选择要展示的参数 (建议 psy-rd 或 aq-strength)
TARGET_PARAM = "aq-strength"  # 可选: "psy-rd", "aq-strength", "cutree-strength"
TARGET_PARAM_COL = "param_aq"  # 对应 CSV 中的列名
BASE_PARAM_COL = "base_aq"

# 平滑窗口 (模拟 Lookahead)
# 40帧 @ 30fps ≈ 1.3秒，符合 x265 默认 lookahead
LOOKAHEAD_WINDOW = 40

# === 样式配置 ===
COLOR_BG = "#A9A9A9"  # 背景特征颜色 (深灰)
COLOR_FG = "#8B0000"  # 前景参数颜色 (深红)
ALPHA_BG = 0.3  # 背景透明度

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def calculate_phi_weights(model_config):
    """
    根据 Phi Matrix 计算综合特征权重
    """
    phi = model_config["phi_matrix"]
    modules = model_config["modules"]  # ["VAQ", "CUTree", ...]
    feat_map = model_config["feature_mapping"]  # "VAQ" -> "w1_var"

    weights = {}

    # 对于每个输入特征 (Column j)
    for col_mod in modules:
        impact_sum = 0.0
        # 计算它对所有参数模块 (Row i) 的总绝对影响
        for row_mod in modules:
            val = phi.get(row_mod, {}).get(col_mod, 0.0)
            impact_sum += abs(val)

        # 映射到特征名
        feat_name = feat_map.get(col_mod)  # e.g., "w1_var"
        # 转换一下名字以匹配 CSV (w1_var -> feat_var)
        csv_name = (
            feat_name.replace("w1_", "feat_")
            .replace("w2_", "feat_")
            .replace("w3_", "feat_")
            .replace("w4_", "feat_")
            .replace("w5_", "feat_")
        )
        weights[csv_name] = impact_sum

    # 归一化
    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total

    return weights


def main():
    # 1. 加载数据
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    model_config = load_json(MODEL_CONFIG_PATH)

    # 2. 计算综合特征 (Omega)
    # 使用 Phi 矩阵权重
    weights = calculate_phi_weights(model_config)
    print("Feature Weights (from Phi):", weights)

    df["omega"] = 0.0
    for col, w in weights.items():
        if col in df.columns:
            df["omega"] += df[col] * w

    # 3. 计算参数偏移 (Delta P)
    df["delta_p"] = df[TARGET_PARAM_COL] - df[BASE_PARAM_COL]

    # 4. Lookahead 平滑 (均值滤波)
    # center=True 保证相位不滞后
    df["omega_smooth"] = (
        df["omega"].rolling(window=LOOKAHEAD_WINDOW, center=True).mean()
    )
    df["delta_p_smooth"] = (
        df["delta_p"].rolling(window=LOOKAHEAD_WINDOW, center=True).mean()
    )

    # 填充 NaN
    df = df.fillna(method="bfill").fillna(method="ffill")

    # 5. 检测关键事件 (用于标注)
    # 场景切换: SAD 突变点 (使用未平滑的 SAD)
    # height 阈值需要根据数据分布自适应，这里取 95% 分位点
    sad_thresh = df["feat_sad"].quantile(0.97)
    scene_changes, _ = find_peaks(df["feat_sad"], height=sad_thresh, distance=60)

    # 复杂纹理: Omega 高点
    omega_thresh = df["omega_smooth"].quantile(0.90)

    # 6. 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    t = df["frame_idx"]

    # --- 左轴: 综合特征 (背景) ---
    ax1.set_xlabel("Frame Index", fontsize=12, fontweight="bold")
    ax1.set_ylabel(
        "Objective Feature Complexity ($\Omega_{agg}$)",
        color="gray",
        fontsize=12,
        fontweight="bold",
    )

    # 面积图
    ax1.fill_between(t, df["omega_smooth"], color=COLOR_BG, alpha=ALPHA_BG)
    ax1.plot(t, df["omega_smooth"], color=COLOR_BG, alpha=0.6, linewidth=1)

    ax1.tick_params(axis="y", labelcolor="gray")
    ax1.set_ylim(0, df["omega_smooth"].max() * 1.3)  # 留出顶部空间给标注
    ax1.grid(True, which="major", linestyle="--", alpha=0.5)

    # --- 右轴: 参数偏移 (前景) ---
    ax2 = ax1.twinx()
    param_label = f"Parameter Adjustment ($\Delta {TARGET_PARAM}$)"
    ax2.set_ylabel(param_label, color=COLOR_FG, fontsize=12, fontweight="bold")

    # 粗实线
    ax2.plot(t, df["delta_p_smooth"], color=COLOR_FG, linewidth=2.5, linestyle="-")
    ax2.tick_params(axis="y", labelcolor=COLOR_FG)

    # --- 标注 ---
    # 1. 场景切换 (垂直虚线)
    if len(scene_changes) > 0:
        print(f"Detected {len(scene_changes)} scene changes.")
        for i, idx in enumerate(scene_changes):
            ax1.axvline(x=idx, color="black", linestyle="--", alpha=0.4)
            # 只在第一个切换点写文字，避免拥挤
            if i == 0:
                ax1.text(
                    idx,
                    ax1.get_ylim()[1] * 0.95,
                    "Scene\nChange",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="black",
                )

    # 2. 复杂纹理/高响应区 (箭头)
    # 找到 Delta P 的最高峰
    max_idx = df["delta_p_smooth"].idxmax()
    max_val = df["delta_p_smooth"].max()

    ax2.annotate(
        "Max Algorithm\nResponse",
        xy=(max_idx, max_val),
        xytext=(max_idx, max_val + 0.5),
        arrowprops=dict(facecolor=COLOR_FG, shrink=0.05, width=1.5, headwidth=8),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=COLOR_FG,
    )

    # --- 图例 ---
    # 自定义图例句柄以合并双轴
    patch_bg = mpatches.Patch(
        color=COLOR_BG, alpha=ALPHA_BG, label="Video Complexity ($\Omega$)"
    )
    line_fg = mpatches.Rectangle(
        (0, 0),
        1,
        1,
        color=COLOR_FG,
        label=f"Adaptive Response ($\Delta {TARGET_PARAM}$)",
    )

    plt.legend(
        handles=[patch_bg, line_fg],
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fontsize=10,
    )

    plt.title(
        f"Real-time Parameter Evolution: {TARGET_PARAM.upper()}\n(Smoothed with {LOOKAHEAD_WINDOW}-frame Lookahead)",
        fontsize=14,
        pad=15,
    )
    plt.tight_layout()

    output_img = "parameter_evolution.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")
    # plt.show() # 如果在无界面环境运行，请注释此行


if __name__ == "__main__":
    main()
