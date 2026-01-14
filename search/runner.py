import os
import sys
import json
import optuna
import argparse
import time
import numpy as np  # [新增] 需要 numpy 计算
from datetime import datetime

# 添加项目根目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from search.evaluator import ParallelEvaluator

# === 全局配置 ===
# 数据库路径
DB_PATH = os.path.join(PROJECT_ROOT, "search_storage.db")
STORAGE_URL = f"sqlite:///{DB_PATH}"
STUDY_NAME = "x265_adaptive_optimization"

# 配置文件路径
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# [关键修改] Anchor 指向 Offline Optimal (离线最优结果)
# 这样计算出的 BD-VMAF 就是 "相对于离线最优的增益"
ANCHOR_JSON = os.path.join(CONFIG_DIR, "offline_results.json")

META_JSON = os.path.join(CONFIG_DIR, "test_sequences.json")
INIT_JSON = os.path.join(CONFIG_DIR, "initial_params.json")
BEST_PARAMS_JSON = os.path.join(PROJECT_ROOT, "best_hyperparams.json")

# 默认路径
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
DEFAULT_DATASET_ROOT = "/home/shiyushen/x265_sequence/"


def objective(trial, evaluator):
    """
    带惩罚项的目标函数
    Goal: Maximize (Average Gain - Penalty * Negative Gain Sum)
    """

    # === 1. 定义搜索空间 ===
    param_a = trial.suggest_float("a", 0.5, 5.0)
    param_b = trial.suggest_float("b", 0.5, 5.0)

    # 允许更宽的 Beta 范围，让模型自己探索
    beta_vaq = trial.suggest_float("beta_VAQ", 0.0, 10.0)
    beta_psyrd = trial.suggest_float("beta_PsyRD", 0.0, 10.0)
    beta_psyrdoq = trial.suggest_float("beta_PsyRDOQ", 0.0, 10.0)
    beta_qcomp = trial.suggest_float("beta_QComp", 0.0, 10.0)

    beta_cutree = 0.0

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

    # === 2. 执行评估 ===
    print(f"\n[Trial {trial.number}] Running with: {json.dumps(hyperparams)}")

    # mean_score 是 evaluator 计算的平均分 (Avg BD-VMAF)
    mean_score, details, crash_report, stats = evaluator.evaluate_batch(hyperparams)

    # === 3. 处理异常 ===
    if mean_score == -9999.0:
        fail_reason = "Unknown Error"
        if crash_report["count"] > 0:
            fail_reason = f"Crashes: {crash_report['count']}"
        elif "error" in stats:
            fail_reason = stats["error"]
        print(f"[Trial {trial.number}] PRUNED. Reason: {fail_reason}")
        raise optuna.TrialPruned(fail_reason)

    # === 4. [关键修改] 计算惩罚项 ===
    # 从 details 中提取每个序列的 BD-VMAF
    seq_scores = []
    penalty_sum = 0.0
    negative_count = 0

    for seq_name, info in details.items():
        if "bd_vmaf" in info:
            val = info["bd_vmaf"]
            seq_scores.append(val)
            if val < 0:
                # 累加负值的绝对值作为惩罚
                penalty_sum += abs(val)
                negative_count += 1

    # 定义惩罚系数 (Lambda)
    # 2.0 意味着: "消除 0.1 的负收益" 等价于 "获得 0.2 的正收益"
    # 这会强烈驱动优化器去消除负值
    PENALTY_LAMBDA = 2.0

    final_objective = mean_score - (PENALTY_LAMBDA * penalty_sum)

    print(f"[Trial {trial.number}] DONE.")
    print(f"  Raw Avg: {mean_score:.4f}")
    print(
        f"  Negatives: {negative_count} seqs (Penalty: -{PENALTY_LAMBDA * penalty_sum:.4f})"
    )
    print(f"  Final Objective: {final_objective:.4f}")

    # 记录一些辅助信息给 Optuna 供后续分析
    trial.set_user_attr("raw_avg_bd", mean_score)
    trial.set_user_attr("negative_count", negative_count)

    return final_objective


def run_optimization():
    parser = argparse.ArgumentParser(description="Optuna Search Runner (Robust Mode)")
    parser.add_argument("--trials", type=int, default=200, help="Number of trials")
    parser.add_argument("--workers", type=int, default=16, help="Max workers")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ROOT, help="Dataset root")
    parser.add_argument("--lib", default=DEFAULT_LIB_PATH, help="libx265.so path")
    parser.add_argument("--reset", action="store_true", help="Start fresh")
    args = parser.parse_args()

    if not os.path.exists(args.lib) or not os.path.exists(args.dataset):
        print("[Error] Path invalid.")
        return

    # 1. 初始化 Evaluator
    print(f"Initializing Evaluator (Anchor: {ANCHOR_JSON})...")
    try:
        evaluator = ParallelEvaluator(
            anchor_json_path=ANCHOR_JSON,  # 指向 Offline Optimal
            seq_meta_json_path=META_JSON,
            init_params_path=INIT_JSON,
            dataset_root=args.dataset,
            lib_path=args.lib,
            max_workers=args.workers,
        )
    except Exception as e:
        print(f"[Fatal] {e}")
        return

    # 2. 初始化 Study
    if args.reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    storage = optuna.storages.RDBStorage(url=STORAGE_URL)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",  # 最大化 Penalized Score
        load_if_exists=True,
    )

    print(f"Search started. Trials: {args.trials}")

    try:
        study.optimize(lambda trial: objective(trial, evaluator), n_trials=args.trials)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Crash: {e}")
    finally:
        if len(study.trials) > 0:
            print("\n=== Search Finished ===")
            best_trial = study.best_trial
            print(f"Best Penalized Score: {best_trial.value:.4f}")
            print(f"Raw Avg Score: {best_trial.user_attrs.get('raw_avg_bd', 'N/A')}")

            # 保存结果
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
                "raw_score": best_trial.user_attrs.get("raw_avg_bd", 0.0),
            }
            with open(BEST_PARAMS_JSON, "w") as f:
                json.dump(final_best, f, indent=4)
            print(f"Saved to {BEST_PARAMS_JSON}")


if __name__ == "__main__":
    run_optimization()

# nohup python3 search/runner.py --trials 200 --workers 16 > run.log 2>&1 &
