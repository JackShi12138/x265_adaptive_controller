import os
import sys
import json
import optuna
import argparse
import time
from datetime import datetime

# 添加项目根目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from search.evaluator import ParallelEvaluator

# === 全局配置 ===
# 数据库路径 (用于断点续传)
DB_PATH = os.path.join(PROJECT_ROOT, "search_storage.db")
STORAGE_URL = f"sqlite:///{DB_PATH}"
STUDY_NAME = "x265_adaptive_optimization"

# 配置文件路径
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
# [Configuration] 使用离线最优结果作为 Anchor
# 这样计算出的 BD-VMAF 是相对于 "Offline Optimal" 的增益
ANCHOR_JSON = os.path.join(CONFIG_DIR, "offline_results.json")
META_JSON = os.path.join(CONFIG_DIR, "test_sequences.json")
INIT_JSON = os.path.join(CONFIG_DIR, "initial_params.json")
BEST_PARAMS_JSON = os.path.join(PROJECT_ROOT, "best_hyperparams.json")

# x265 路径 (请根据实际情况修改或通过命令行传入)
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
DEFAULT_DATASET_ROOT = "/home/shiyushen/x265_sequence/"


def objective(trial, evaluator):
    """
    Optuna 的目标函数 (Robust Mode)
    Goal: Maximize (Average Gain - Penalty * Negative Gain Sum)
    """

    # === 1. 定义搜索空间 (Search Space) ===
    # Sigmoid 形状参数
    param_a = trial.suggest_float("a", 0.5, 5.0)
    param_b = trial.suggest_float("b", 0.5, 5.0)

    # 模块权重 (Betas)
    # [修正] beta_CUTree 正式加入搜索，范围 0.0 ~ 10.0
    beta_vaq = trial.suggest_float("beta_VAQ", 0.0, 10.0)
    beta_cutree = trial.suggest_float("beta_CUTree", 0.0, 10.0)
    beta_psyrd = trial.suggest_float("beta_PsyRD", 0.0, 10.0)
    beta_psyrdoq = trial.suggest_float("beta_PsyRDOQ", 0.0, 10.0)
    beta_qcomp = trial.suggest_float("beta_QComp", 0.0, 10.0)

    # === 2. 打包参数 ===
    hyperparams = {
        "a": param_a,
        "b": param_b,
        "beta": {
            "VAQ": beta_vaq,
            "CUTree": beta_cutree,  # 现在它是动态变化的
            "PsyRD": beta_psyrd,
            "PsyRDOQ": beta_psyrdoq,
            "QComp": beta_qcomp,
        },
    }

    # === 3. 执行评估 ===
    print(
        f"\n[Trial {trial.number}] Evaluator starting with: {json.dumps(hyperparams)}"
    )

    # mean_score 是 evaluator 计算的原始平均分 (Avg BD-VMAF)
    # details 包含了每个序列的详细得分，用于计算惩罚
    mean_score, details, crash_report, stats = evaluator.evaluate_batch(hyperparams)

    # === 4. 处理异常 ===
    if mean_score == -9999.0:
        fail_reason = "Unknown Error"
        if crash_report["count"] > 0:
            fail_reason = (
                f"Crashes: {crash_report['count']} (e.g. {crash_report['errors'][:1]})"
            )
        elif "error" in stats:
            fail_reason = stats["error"]

        print(f"[Trial {trial.number}] FAILED. Reason: {fail_reason}")
        raise optuna.TrialPruned(fail_reason)

    # === 5. 计算惩罚项 (Robustness Penalty) ===
    # 逻辑：不仅要平均分高，还要没有“掉队”的视频
    penalty_sum = 0.0
    negative_count = 0

    # 遍历每个序列的详细结果
    for seq_name, info in details.items():
        # 获取该序列的 BD-VMAF
        val = info.get("bd_vmaf", 0.0)

        if val < 0:
            penalty_sum += abs(val)
            negative_count += 1

    # 定义惩罚系数 (Lambda)
    # 2.0 意味着: "消除 0.1 的负收益" 等价于 "获得 0.2 的正收益"
    PENALTY_LAMBDA = 2.0

    final_objective = mean_score - (PENALTY_LAMBDA * penalty_sum)

    print(f"[Trial {trial.number}] DONE.")
    print(f"  Raw Avg: {mean_score:.4f}")
    print(
        f"  Negatives: {negative_count} seqs (Penalty: -{PENALTY_LAMBDA * penalty_sum:.4f})"
    )
    print(f"  Final Objective: {final_objective:.4f}")

    # 记录辅助信息给 Optuna (用于后续分析/绘图)
    trial.set_user_attr("raw_avg_bd", mean_score)
    trial.set_user_attr("negative_count", negative_count)

    return final_objective


def run_optimization():
    parser = argparse.ArgumentParser(description="Optuna Search Runner (Robust Mode)")
    parser.add_argument(
        "--trials", type=int, default=200, help="Number of trials to run"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Max parallel workers (Strict limit: 16)",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET_ROOT, help="Path to YUV dataset root"
    )
    parser.add_argument("--lib", default=DEFAULT_LIB_PATH, help="Path to libx265.so")
    parser.add_argument(
        "--reset", action="store_true", help="Delete DB and start fresh"
    )
    args = parser.parse_args()

    # 1. 检查环境
    if not os.path.exists(args.lib):
        print(f"[Error] libx265 not found at {args.lib}")
        return
    if not os.path.exists(args.dataset):
        print(f"[Error] Dataset not found at {args.dataset}")
        return

    # 2. 初始化评估器
    print(f"Initializing Evaluator (Anchor: {ANCHOR_JSON})...")
    try:
        evaluator = ParallelEvaluator(
            anchor_json_path=ANCHOR_JSON,
            seq_meta_json_path=META_JSON,
            init_params_path=INIT_JSON,
            dataset_root=args.dataset,
            lib_path=args.lib,
            max_workers=args.workers,
        )
    except Exception as e:
        print(f"[Fatal] Evaluator init failed: {e}")
        return

    # 3. 初始化 Optuna Study
    if args.reset and os.path.exists(DB_PATH):
        print(f"Resetting database: {DB_PATH}")
        os.remove(DB_PATH)

    print(f"Loading/Creating Study '{STUDY_NAME}' at {STORAGE_URL}...")

    # 使用 TPE 采样器 (默认)，方向是 "maximize" (BD-VMAF 越大越好)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        load_if_exists=True,
    )

    print(f"Search started. Planning {args.trials} trials.")
    print("Press Ctrl+C to stop. Progress is saved automatically to DB.")

    try:
        study.optimize(lambda trial: objective(trial, evaluator), n_trials=args.trials)
    except KeyboardInterrupt:
        print("\n[Info] Search interrupted by user. Saving current best...")
    except Exception as e:
        print(f"\n[Error] Search crashed: {e}")
    finally:
        # 4. 结算与保存
        if len(study.trials) > 0:
            print("\n" + "=" * 40)
            print("       SEARCH FINISHED       ")
            print("=" * 40)

            try:
                best_trial = study.best_trial
                print(f"Best Penalized Score: {best_trial.value:.4f}")
                print(
                    f"Raw Avg Score: {best_trial.user_attrs.get('raw_avg_bd', 'N/A')}"
                )
                print("Best Params:")
                for k, v in best_trial.params.items():
                    print(f"  {k}: {v:.4f}")

                # [修正] 保存 logic 包含 beta_CUTree
                final_best = {
                    "a": best_trial.params["a"],
                    "b": best_trial.params["b"],
                    "beta": {
                        "VAQ": best_trial.params["beta_VAQ"],
                        "CUTree": best_trial.params["beta_CUTree"],  # 保存真实搜索值
                        "PsyRD": best_trial.params["beta_PsyRD"],
                        "PsyRDOQ": best_trial.params["beta_PsyRDOQ"],
                        "QComp": best_trial.params["beta_QComp"],
                    },
                    "score": best_trial.value,
                    "raw_score": best_trial.user_attrs.get("raw_avg_bd", 0.0),
                    "trial_number": best_trial.number,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                # 保存到文件
                with open(BEST_PARAMS_JSON, "w") as f:
                    json.dump(final_best, f, indent=4)
                print(f"Best hyperparameters saved to: {BEST_PARAMS_JSON}")

            except ValueError:
                print("No successful trials completed yet.")
        else:
            print("No trials executed.")


if __name__ == "__main__":
    run_optimization()

# nohup python3 search/runner.py --trials 400 --workers 16 --reset > run.log 2>&1 &
