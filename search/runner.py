import os
import sys
import json
import optuna
import argparse
from datetime import datetime
from optuna.samplers import CmaEsSampler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from search.evaluator import ParallelEvaluator

DB_PATH = os.path.join(PROJECT_ROOT, "search_storage.db")
STORAGE_URL = f"sqlite:///{DB_PATH}"
STUDY_NAME = "x265_adaptive_optimization"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
ANCHOR_JSON = os.path.join(CONFIG_DIR, "offline_results_91.json")
META_JSON = os.path.join(CONFIG_DIR, "test_sequences.json")
INIT_JSON = os.path.join(CONFIG_DIR, "initial_params.json")
BEST_PARAMS_JSON = os.path.join(PROJECT_ROOT, "best_hyperparams.json")
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
DEFAULT_DATASET_ROOT = "/home/shiyushen/x265_sequence/"

# [训练集] 典型视频
TRAINING_SET = [
    # Class A: 4K 基准，纹理细腻
    "PeopleOnStreet_2560x1600_30_crop",
    # Class B: 困难样本 Cactus (混合纹理) + 稳定样本 BQTerrace
    "Cactus_1920x1080_50",  # [新增] 替换 BasketballDrive
    "BQTerrace_1920x1080_60",
    # Class C: 困难样本集中营 (高纹理/高运动)
    "BasketballDrill_832x480_50",  # [保留] 重点攻克
    "PartyScene_832x480_50",  # [新增] 重点攻克噪声纹理
    # Class D: 低分辨率基准
    "RaceHorses_416x240_30",
    # Class E: 困难样本 KristenAndSara (静止背景)
    "KristenAndSara_1280x720_60",  # [新增] 替换 FourPeople
]


def objective(trial, evaluator):
    # 1. 定义搜索空间
    param_a = trial.suggest_float("a", 0.5, 5.0)
    param_b = trial.suggest_float("b", 0.5, 5.0)
    beta_vaq = trial.suggest_float("beta_VAQ", 0.0, 10.0)
    beta_cutree = trial.suggest_float("beta_CUTree", 0.0, 10.0)
    beta_psyrd = trial.suggest_float("beta_PsyRD", 0.0, 10.0)
    beta_psyrdoq = trial.suggest_float("beta_PsyRDOQ", 0.0, 10.0)
    beta_qcomp = trial.suggest_float("beta_QComp", 0.0, 10.0)

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

    print(
        f"\n[Trial {trial.number}] Evaluator starting with: {json.dumps(hyperparams)}"
    )

    # 2. 执行评估
    mean_score, details, crash_report, stats = evaluator.evaluate_batch(hyperparams)

    if mean_score == -9999.0:
        print(f"[Trial {trial.number}] Failed.")
        raise optuna.TrialPruned("Eval Failed")

    # === 3. 核心评分逻辑 (Min-Max 策略) ===
    scores = [info.get("bd_vmaf", 0.0) for info in details.values()]
    min_score = min(scores)
    mean_score = sum(scores) / len(scores)

    # 统计有多少个负收益，方便看日志
    negative_count = sum(1 for s in scores if s < 0)

    # 阈值 -0.09：允许轻微的负波动，但不能太离谱
    THRESHOLD = -0.09

    if min_score < THRESHOLD:
        # [生存模式] 有严重短板，全力消除短板
        # 放大梯度，强迫 CMA-ES 关注
        final_objective = min_score * 2.0
        mode = "SURVIVAL (Fix Min)"
    else:
        # [发展模式] 所有视频都及格了，追求高平均分
        # 加上 0.5 * min_score 是为了防守，防止为了平均分牺牲最低分
        final_objective = mean_score + 0.5 * min_score
        mode = "GROWTH (Max Mean)"

    # 4. 打印详细日志
    print(f"[Trial {trial.number}] DONE.")
    print(f"  Mode: {mode}")
    print(
        f"  Stats: Min={min_score:.4f} | Mean={mean_score:.4f} | Negatives={negative_count}"
    )
    print(f"  Final Objective: {final_objective:.4f}")

    # 记录辅助数据供 Optuna Dashboard 分析
    trial.set_user_attr("raw_avg_bd", mean_score)
    trial.set_user_attr("min_bd", min_score)
    trial.set_user_attr("negative_count", negative_count)

    return final_objective


def run_optimization():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--workers", type=int, default=10)  # 保持 CPU 高效利用
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lib", default=DEFAULT_LIB_PATH)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--full-set", action="store_true")
    args = parser.parse_args()

    target_seqs = None if args.full_set else TRAINING_SET
    print(
        f"Search Strategy: NO PRUNING, {'FULL SET' if args.full_set else 'TRAINING SUBSET'}, FULL FRAMES."
    )

    evaluator = ParallelEvaluator(
        anchor_json_path=ANCHOR_JSON,
        seq_meta_json_path=META_JSON,
        init_params_path=INIT_JSON,
        dataset_root=args.dataset,
        lib_path=args.lib,
        max_workers=args.workers,
        target_seqs=target_seqs,
        # [移除] max_frames
    )

    if args.reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    sampler = CmaEsSampler(restart_strategy="ipop", n_startup_trials=10)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,  # 传入 sampler
    )

    try:
        study.optimize(lambda trial: objective(trial, evaluator), n_trials=args.trials)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if len(study.trials) > 0:
            best_trial = study.best_trial
            print(f"\nBest Penalized Score: {best_trial.value:.4f}")
            print(f"Raw Avg Score: {best_trial.user_attrs.get('raw_avg_bd', 'N/A')}")

            final_best = {
                "a": best_trial.params["a"],
                "b": best_trial.params["b"],
                "beta": {
                    "VAQ": best_trial.params["beta_VAQ"],
                    "CUTree": best_trial.params["beta_CUTree"],
                    "PsyRD": best_trial.params["beta_PsyRD"],
                    "PsyRDOQ": best_trial.params["beta_PsyRDOQ"],
                    "QComp": best_trial.params["beta_QComp"],
                },
                "score": best_trial.value,
                "raw_score": best_trial.user_attrs.get("raw_avg_bd", 0.0),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(BEST_PARAMS_JSON, "w") as f:
                json.dump(final_best, f, indent=4)
            print(f"Saved to {BEST_PARAMS_JSON}")


if __name__ == "__main__":
    run_optimization()

# nohup python3 search/runner.py --trials 400 --workers 10 --reset > run.log 2>&1 &
