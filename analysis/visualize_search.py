import os
import sys
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 添加项目根目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

# === 配置 ===
DB_PATH = os.path.join(PROJECT_ROOT, "search_storage.db")
STORAGE_URL = f"sqlite:///{DB_PATH}"
STUDY_NAME = "x265_adaptive_optimization"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "plots")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_study():
    if not os.path.exists(DB_PATH):
        print(f"[Error] Database not found at {DB_PATH}")
        sys.exit(1)

    print(f"Loading study '{STUDY_NAME}' from {DB_PATH}...")
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
        print(f"Loaded {len(study.trials)} trials.")
        return study
    except Exception as e:
        print(f"[Error] Failed to load study: {e}")
        sys.exit(1)


def plot_optimization_history(study):
    print("Generating Optimization History Plot...")

    trials = study.trials
    # 过滤掉 Pruned (失败) 的 trial，只保留 Complete 的
    completed_trials = [
        t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("No completed trials to plot.")
        return

    numbers = [t.number for t in completed_trials]
    values = [t.value for t in completed_trials]

    # 计算“当前最佳”曲线
    best_values = []
    current_best = -float("inf")
    for v in values:
        if v > current_best:
            current_best = v
        best_values.append(current_best)

    plt.figure(figsize=(10, 6))

    # 绘制所有点
    plt.scatter(numbers, values, color="blue", alpha=0.5, label="Trial Score", s=20)

    # 绘制最佳曲线
    plt.plot(numbers, best_values, color="red", linewidth=2, label="Best So Far")

    plt.title("Optimization History (BD-VMAF)", fontsize=14)
    plt.xlabel("Trial Number", fontsize=12)
    plt.ylabel("BD-VMAF Score (Higher is Better)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    out_path = os.path.join(OUTPUT_DIR, "optimization_history.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved to {out_path}")


def plot_param_importances(study):
    print("Generating Parameter Importance Plot...")

    try:
        # 使用 Optuna 内置算法计算重要性 (默认是 fANOVA)
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"Skipping importance plot due to error (maybe not enough trials): {e}")
        return

    if not importances:
        return

    params = list(importances.keys())
    scores = list(importances.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=params, palette="viridis")

    plt.title("Hyperparameter Importance", fontsize=14)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Parameter", fontsize=12)
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)

    out_path = os.path.join(OUTPUT_DIR, "param_importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved to {out_path}")


def plot_slices(study):
    print("Generating Slice Plots (Sweet Spots)...")

    # 将数据转换为 DataFrame 方便绘图
    df = study.trials_dataframe()
    # 筛选成功的试验
    df = df[df.state == "COMPLETE"]

    # 提取参数列 (params_xxx) 和 目标值 (value)
    param_cols = [c for c in df.columns if c.startswith("params_")]

    # 计算子图布局
    n_params = len(param_cols)
    cols = 3
    rows = (n_params + cols - 1) // cols

    plt.figure(figsize=(15, 4 * rows))

    for i, col in enumerate(param_cols):
        param_name = col.replace("params_", "")

        plt.subplot(rows, cols, i + 1)

        # 绘制散点图
        plt.scatter(
            df[col],
            df["value"],
            alpha=0.6,
            c=df["value"],
            cmap="viridis",
            edgecolors="k",
        )

        # 拟合一个简单的趋势线 (Lowess) 帮助看趋势
        try:
            sns.regplot(
                x=df[col],
                y=df["value"],
                scatter=False,
                lowess=True,
                color="red",
                line_kws={"linewidth": 1.5},
            )
        except Exception:
            pass  # 数据太少可能无法拟合

        plt.title(f"{param_name}", fontsize=12)
        plt.xlabel(param_name)
        plt.ylabel("BD-VMAF")
        plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "param_slices.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved to {out_path}")


def main():
    # 设置绘图风格
    sns.set_style("whitegrid")

    # 1. 加载数据
    study = load_study()

    if len(study.trials) == 0:
        print("Study is empty.")
        return

    best_trial = study.best_trial
    print(f"\nBest Trial so far:")
    print(f"  Score: {best_trial.value:.4f}")
    print(f"  Params: {best_trial.params}")

    # 2. 绘制各种图表
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_slices(study)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
