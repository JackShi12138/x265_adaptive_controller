import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks

# === 绘图配置 ===
INPUT_CSV = "frame_level_data.csv"
MODEL_CONFIG_PATH = "config/model_config.json"

# [核心修改] 定义要展示的所有5个参数 (Label, Target Col, Base Col, Color)
TARGET_PARAMS = [
    ("Psy-RD", "param_psy_rd", "base_psy_rd", "#D62728"),  # 红色
    ("Psy-RDOQ", "param_psy_rdoq", "base_psy_rdoq", "#FF7F0E"),  # 橙色
    ("AQ-Strength", "param_aq", "base_aq", "#2CA02C"),  # 绿色
    ("CUTree", "param_cutree", "base_cutree", "#1F77B4"),  # 蓝色
    ("QComp", "param_qcomp", "base_qcomp", "#9467BD"),  # 紫色
]

# 平滑窗口 (模拟 Lookahead)
# 40帧 @ 30fps ≈ 1.3秒，符合 x265 默认 lookahead
LOOKAHEAD_WINDOW = 40

# === 样式配置 ===
COLOR_BG = "#A9A9A9"  # 背景特征颜色 (深灰)
ALPHA_BG = 0.3  # 背景透明度

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def calculate_phi_weights(model_config):
    """
    根据 Phi Matrix 计算综合特征权重 (保持原逻辑不变)
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

    # 注意：如果本地没有 model_config.json，你需要注释掉下面这行并手动指定 weights
    try:
        model_config = load_json(MODEL_CONFIG_PATH)
        # 2. 计算综合特征 (Omega)
        weights = calculate_phi_weights(model_config)
        print("Feature Weights (from Phi):", weights)
    except FileNotFoundError:
        print("Warning: Config file not found. Using equal weights for features.")
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        weights = {c: 1.0 / len(feat_cols) for c in feat_cols}

    df["omega"] = 0.0
    for col, w in weights.items():
        if col in df.columns:
            df["omega"] += df[col] * w

    # 平滑背景特征
    df["omega_smooth"] = (
        df["omega"].rolling(window=LOOKAHEAD_WINDOW, center=True).mean()
    )

    # 3. [核心修改] 循环计算所有参数的偏移 (Delta P) 并平滑
    plot_data = []  # 用于存储绘图信息

    for label, target_col, base_col, color in TARGET_PARAMS:
        # 计算 Delta (Target - Base)
        delta_col = f"delta_{label}"

        # 确保列存在
        if target_col in df.columns and base_col in df.columns:
            df[delta_col] = df[target_col] - df[base_col]

            # 平滑
            smooth_col = f"smooth_{delta_col}"
            df[smooth_col] = (
                df[delta_col].rolling(window=LOOKAHEAD_WINDOW, center=True).mean()
            )

            # 存储绘图所需信息
            plot_data.append({"label": label, "color": color, "data": df[smooth_col]})
        else:
            print(f"Warning: Columns for {label} not found in CSV.")

    # 填充 NaN
    df = df.fillna(method="bfill").fillna(method="ffill")

    # 4. 检测关键事件 (用于标注)
    # 场景切换: SAD 突变点
    sad_thresh = df["feat_sad"].quantile(0.97)
    scene_changes, _ = find_peaks(df["feat_sad"], height=sad_thresh, distance=60)

    # 5. 绘图
    fig, ax1 = plt.subplots(figsize=(14, 7))  # 稍微加宽一点以容纳更多信息
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
    # 留出更多顶部空间给图例
    ax1.set_ylim(0, df["omega_smooth"].max() * 1.4)
    ax1.grid(True, which="major", linestyle="--", alpha=0.5)

    # --- 右轴: 参数偏移 (前景 - 多条曲线) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        "Parameter Adjustment ($\Delta P$)",
        color="#333333",
        fontsize=12,
        fontweight="bold",
    )

    # [核心修改] 循环绘制每一条曲线
    lines = []
    for item in plot_data:
        (l,) = ax2.plot(
            t,
            item["data"],
            color=item["color"],
            linewidth=2,
            linestyle="-",
            label=f"$\Delta$ {item['label']}",
        )
        lines.append(l)

    ax2.tick_params(axis="y", labelcolor="#333333")

    # --- 标注 (Scene Changes) ---
    if len(scene_changes) > 0:
        print(f"Detected {len(scene_changes)} scene changes.")
        for i, idx in enumerate(scene_changes):
            ax1.axvline(x=idx, color="black", linestyle="--", alpha=0.4)
            # 只在第一个点标注，避免拥挤
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

    # --- 图例 ---
    # 背景图例 Handle
    patch_bg = mpatches.Patch(
        color=COLOR_BG, alpha=ALPHA_BG, label="Video Complexity ($\Omega$)"
    )

    # 合并所有图例 Handles
    handles = [patch_bg] + lines

    plt.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, 1),  # 放在左上角外侧一点，或内部顶部
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        ncol=2,  # 分两列显示，节省垂直空间
    )

    plt.tight_layout()

    output_img = "parameter_evolution_multi.pdf"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")
    # plt.show()


if __name__ == "__main__":
    main()
