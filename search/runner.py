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
ANCHOR_JSON = os.path.join(CONFIG_DIR, "offline_results.json")
META_JSON = os.path.join(CONFIG_DIR, "test_sequences.json")
INIT_JSON = os.path.join(CONFIG_DIR, "initial_params.json")
BEST_PARAMS_JSON = os.path.join(PROJECT_ROOT, "best_hyperparams.json")
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
DEFAULT_DATASET_ROOT = "/home/shiyushen/x265_sequence/"

# [训练集] 典型视频
TRAINING_SET = [
    "BasketballDrive_1920x1080_50",
    "Cactus_1920x1080_50",
    "PartyScene_832x480_50",
    "RaceHorses_416x240_30",
    "FourPeople_1280x720_60",
    "SlideEditing_1280x720_30",
    "PartyScene_832x480_50",
    "SlideShow_1280x720_20",
    "BasketballDrill_832x480_50",
    "KristenAndSara_1280x720_60",
]


def objective(trial, evaluator):
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

    mean_score, details, crash_report, stats = evaluator.evaluate_batch(hyperparams)

    if mean_score == -9999.0:
        print(f"[Trial {trial.number}] Failed.")
        raise optuna.TrialPruned("Eval Failed")

    # === 惩罚项计算 ===
    penalty_sum = 0.0
    negative_count = 0
    for seq_name, info in details.items():
        val = info.get("bd_vmaf", 0.0)
        if val < 0:
            penalty_sum += abs(val)
            negative_count += 1

    # 系数 3.0: 强力驱逐负收益
    PENALTY_LAMBDA = 3.0
    final_objective = mean_score - (PENALTY_LAMBDA * penalty_sum)

    print(f"[Trial {trial.number}] DONE.")
    print(f"  Raw Avg: {mean_score:.4f}")
    print(
        f"  Negatives: {negative_count} seqs (Penalty: -{PENALTY_LAMBDA * penalty_sum:.4f})"
    )
    print(f"  Final Objective: {final_objective:.4f}")

    trial.set_user_attr("raw_avg_bd", mean_score)
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

# nohup python3 search/runner.py --trials 400 --workers 16 --reset > run.log 2>&1 &
