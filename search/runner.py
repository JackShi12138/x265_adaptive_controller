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
ANCHOR_JSON = os.path.join(CONFIG_DIR, "anchor_results.json")
META_JSON = os.path.join(CONFIG_DIR, "test_sequences.json")
INIT_JSON = os.path.join(CONFIG_DIR, "initial_params.json")
BEST_PARAMS_JSON = os.path.join(PROJECT_ROOT, "best_hyperparams.json")

# x265 路径 (请根据实际情况修改或通过命令行传入)
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
DEFAULT_DATASET_ROOT = "/home/shiyushen/x265_sequence/"  # 你的真实 YUV 路径


def objective(trial, evaluator):
    """
    Optuna 的目标函数
    1. Suggest: 获取一组超参数
    2. Evaluate: 运行评估器
    3. Return: 返回 BD-VMAF 分数
    """

    # === 1. 定义搜索空间 (Search Space) ===
    # 使用 Uniform 分布，因为我们对参数的数量级没有先验偏好

    # Sigmoid 形状参数
    param_a = trial.suggest_float("a", 0.5, 5.0)
    param_b = trial.suggest_float("b", 0.5, 5.0)

    # 模块权重 (Betas)
    # 范围 0.0 ~ 10.0，允许模型选择激进(>5)或保守(<1)的策略
    beta_vaq = trial.suggest_float("beta_VAQ", 0.0, 10.0)
    beta_psyrd = trial.suggest_float("beta_PsyRD", 0.0, 10.0)
    beta_psyrdoq = trial.suggest_float("beta_PsyRDOQ", 0.0, 10.0)
    beta_qcomp = trial.suggest_float("beta_QComp", 0.0, 10.0)

    # beta_CUTree 锁定为 0.0 (不参与搜索，因为模型逻辑中不输出该参数)
    beta_cutree = 0.0

    # === 2. 打包参数 ===
    hyperparams = {
        "a": param_a,
        "b": param_b,
        "beta": {
            "VAQ": beta_vaq,
            "CUTree": beta_cutree,
            "PsyRD": beta_psyrd,
            "PsyRDOQ": beta_psyrdoq,
            "QComp": beta_qcomp,
        },
    }

    # === 3. 执行评估 ===
    # 注意：这就开始跑 88 个任务了，耗时较长
    print(
        f"\n[Trial {trial.number}] Evaluator starting with: {json.dumps(hyperparams)}"
    )

    score, details, crash_report, stats = evaluator.evaluate_batch(hyperparams)

    # === 4. 处理结果与异常 ===
    if score == -9999.0:
        # 记录失败原因
        fail_reason = "Unknown Error"
        if crash_report["count"] > 0:
            fail_reason = (
                f"Crashes: {crash_report['count']} (e.g. {crash_report['errors'][:1]})"
            )
        elif "error" in stats:  # 比如 metric missing
            fail_reason = stats["error"]

        print(f"[Trial {trial.number}] FAILED. Reason: {fail_reason}")

        # 告诉 Optuna 这个点无效 (Pruned)
        raise optuna.TrialPruned(fail_reason)

    print(f"[Trial {trial.number}] SUCCESS. Score: {score:.4f}")

    # 将一些额外的统计信息存入 trial.user_attrs，方便后续分析
    trial.set_user_attr("processed_seqs", stats.get("processed_seqs", 0))

    return score


def run_optimization():
    parser = argparse.ArgumentParser(description="Optuna Search Runner")
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

    # 2. 初始化评估器 (The Muscle)
    print(f"Initializing Evaluator (Workers: {args.workers})...")
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

    # 3. 初始化 Optuna Study (The Brain)
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
        # 使用 lambda 包装 objective 以传入 evaluator
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
                print(f"Best Score: {best_trial.value:.4f}")
                print("Best Params:")
                for k, v in best_trial.params.items():
                    print(f"  {k}: {v:.4f}")

                # 构造完整的最佳参数字典 (包含锁定的 beta_CUTree)
                final_best = {
                    "a": best_trial.params["a"],
                    "b": best_trial.params["b"],
                    "beta": {
                        "VAQ": best_trial.params["beta_VAQ"],
                        "CUTree": 0.0,
                        "PsyRD": best_trial.params["beta_PsyRD"],
                        "PsyRDOQ": best_trial.params["beta_PsyRDOQ"],
                        "QComp": best_trial.params["beta_QComp"],
                    },
                    "score": best_trial.value,
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

# nohup python3 search/runner.py --trials 200 --workers 16 > run.log 2>&1 &
