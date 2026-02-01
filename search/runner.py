import os
import sys
import json
import logging
import numpy as np
import optuna
import argparse
from datetime import datetime
from optuna.samplers import CmaEsSampler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from search.evaluator import ParallelEvaluator

# === 全局配置 ===
DB_PATH = os.path.join(PROJECT_ROOT, "search_storage.db")
STORAGE_URL = f"sqlite:///{DB_PATH}"
STUDY_NAME = "x265_adaptive_optimization"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
ANCHOR_JSON = os.path.join(CONFIG_DIR, "offline_results_91.json")
META_JSON = os.path.join(CONFIG_DIR, "test_sequences.json")
INIT_JSON = os.path.join(CONFIG_DIR, "initial_params.json")
BEST_PARAMS_JSON = os.path.join(PROJECT_ROOT, "best_hyperparams.json")
SEARCH_LOG_FILE = os.path.join(PROJECT_ROOT, "search_progress.log")

DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
DEFAULT_DATASET_ROOT = "/home/shiyushen/x265_sequence/"

# [训练集]
TRAINING_SET = [
    "PeopleOnStreet_2560x1600_30_crop",
    "Cactus_1920x1080_50",
    "BQTerrace_1920x1080_60",
    "BasketballDrill_832x480_50",
    "PartyScene_832x480_50",
    "RaceHorses_416x240_30",
    "KristenAndSara_1280x720_60",
]


# === 初始化独立 Logger ===
def setup_search_logger():
    logger = logging.getLogger("SearchMonitor")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(SEARCH_LOG_FILE, mode="w", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


search_logger = setup_search_logger()


# === 参数波动统计函数 ===
def analyze_parameter_variance(study, window=20):
    if len(study.trials) < window:
        return "Not enough data"

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    recent_trials = completed_trials[-window:]

    if not recent_trials:
        return "No complete trials"

    param_keys = [
        "a",
        "b",
        "beta_VAQ",
        "beta_CUTree",
        "beta_PsyRD",
        "beta_PsyRDOQ",
        "beta_QComp",
    ]
    variances = {}

    for key in param_keys:
        values = [t.params.get(key, 0) for t in recent_trials]
        std_dev = np.std(values)
        mean_val = np.mean(values)
        rel_std = std_dev / (abs(mean_val) + 1e-6)
        variances[key] = f"{std_dev:.2f}({rel_std:.1%})"

    return " | ".join([f"{k}:{v}" for k, v in variances.items()])


def objective(trial, evaluator, study):
    # === 1. 搜索空间限制 (物理约束) ===
    param_a = trial.suggest_float("a", 0.5, 5.0)
    param_b = trial.suggest_float("b", 0.5, 2.0)  # 限制斜率，防止开关效应

    # 限制 Beta 上限，防止过激调节
    beta_vaq = trial.suggest_float("beta_VAQ", 0.0, 4.0)
    beta_cutree = trial.suggest_float("beta_CUTree", 0.0, 4.0)
    beta_psyrd = trial.suggest_float("beta_PsyRD", 0.0, 4.0)
    beta_psyrdoq = trial.suggest_float("beta_PsyRDOQ", 0.0, 4.0)
    beta_qcomp = trial.suggest_float("beta_QComp", 0.0, 4.0)

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

    print(f"[Trial {trial.number}] Processing...")

    # 2. 执行评估
    mean_score, details, crash_report, stats = evaluator.evaluate_batch(hyperparams)

    if mean_score == -9999.0:
        search_logger.error(
            f"[Trial {trial.number}] FAILED. Params: {json.dumps(hyperparams)}"
        )
        raise optuna.TrialPruned("Eval Failed")

    # 3. 计算基础指标
    scores = [info.get("bd_vmaf", 0.0) for info in details.values()]
    min_score = min(scores)
    mean_val = sum(scores) / len(scores)
    negative_count = sum(1 for s in scores if s < 0)

    # === [核心修改] 软化边界策略 (Soft Barrier) ===
    # 目标：最大化 (平均分 + 0.5 * 最低分)
    # 但如果最低分低于安全线，施加平滑的平方惩罚

    base_objective = mean_val + 0.5 * min_score

    SAFE_LIMIT = -0.10  # 安全线，高于此线无惩罚
    barrier_penalty = 0.0

    for s in scores:
        if s < SAFE_LIMIT:
            # 平方惩罚：越过安全线越多，罚得越重（梯度连续）
            # 权重 5.0 是经验值，保证严重越界时罚分足够痛
            barrier_penalty += 5.0 * ((s - SAFE_LIMIT) ** 2)

    # === 4. L2 正则化 (抑制参数膨胀) ===
    REG_LAMBDA = 0.002
    param_magnitude = (param_b**2) + sum(v**2 for v in hyperparams["beta"].values())
    l2_penalty = REG_LAMBDA * param_magnitude

    # 最终得分 = 基础分 - 越界惩罚 - 参数大惩罚
    final_objective = base_objective - barrier_penalty - l2_penalty

    # 确定模式名称 (用于日志显示)
    if barrier_penalty > 0.1:
        mode = "UNSAFE"  # 越界严重
    elif barrier_penalty > 0:
        mode = "RISKY"  # 轻微越界
    else:
        mode = "SAFE"  # 完全在安全区

    # === 5. 日志记录 ===
    try:
        history_best = study.best_value
        global_best = max(history_best, final_objective)
    except ValueError:
        global_best = final_objective

    is_new_best = final_objective >= global_best
    marker = "★ NEW BEST ★" if is_new_best else ""

    log_msg = (
        f"[Trial {trial.number}] {marker}\n"
        f"  Mode: {mode} | Obj: {final_objective:.4f} (Base: {base_objective:.4f} - Barrier: {barrier_penalty:.4f} - L2: {l2_penalty:.4f})\n"
        f"  Metrics: Mean={mean_val:.4f} | Min={min_score:.4f} | Negs={negative_count}\n"
        f"  Params: a={param_a:.4f}, b={param_b:.4f} | "
        f"VAQ={beta_vaq:.4f}, CUTree={beta_cutree:.4f}, RD={beta_psyrd:.4f}, RDOQ={beta_psyrdoq:.4f}, QComp={beta_qcomp:.4f}\n"
    )

    if trial.number > 0 and trial.number % 10 == 0:
        variance_report = analyze_parameter_variance(study)
        log_msg += f"  [Convergence Stats (Last 20)]: {variance_report}\n"

    search_logger.info(log_msg)

    trial.set_user_attr("raw_avg_bd", mean_val)
    trial.set_user_attr("min_bd", min_score)
    trial.set_user_attr("barrier_penalty", barrier_penalty)

    return final_objective


def run_optimization():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lib", default=DEFAULT_LIB_PATH)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--full-set", action="store_true")
    args = parser.parse_args()

    target_seqs = None if args.full_set else TRAINING_SET

    search_logger.info("=" * 60)
    search_logger.info(
        f"SESSION START: trials={args.trials}, workers={args.workers}, reset={args.reset}"
    )
    search_logger.info(
        f"Policy: Soft Barrier (Limit -0.10) + Restricted Space + Seed + L2 Reg"
    )
    search_logger.info("=" * 60)

    evaluator = ParallelEvaluator(
        anchor_json_path=ANCHOR_JSON,
        seq_meta_json_path=META_JSON,
        init_params_path=INIT_JSON,
        dataset_root=args.dataset,
        lib_path=args.lib,
        max_workers=args.workers,
        target_seqs=target_seqs,
    )

    if args.reset and os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            search_logger.warning("Database reset success!")
        except OSError as e:
            search_logger.error(f"Error resetting DB: {e}")

    # 减小初始步长 sigma0，防止一开始就跳出好区域
    sampler = CmaEsSampler(restart_strategy="ipop", n_startup_trials=5, sigma0=0.5)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    # 注入修正后的优良种子
    if args.reset or len(study.trials) == 0:
        known_good_params = {
            "a": 2.5375,
            "b": 1.9242,
            "beta_VAQ": 2.7898,
            "beta_CUTree": 3.5,  # 手动削减到合规范围
            "beta_PsyRD": 4.0,  # 手动削减到合规范围 (原8.3)
            "beta_PsyRDOQ": 2.6921,
            "beta_QComp": 3.0856,
        }
        search_logger.info(f"Injecting seed params: {json.dumps(known_good_params)}")
        study.enqueue_trial(known_good_params)

    try:
        study.optimize(
            lambda trial: objective(trial, evaluator, study), n_trials=args.trials
        )
    except KeyboardInterrupt:
        search_logger.warning("Optimization interrupted by user.")
        print("\nInterrupted.")
    except Exception as e:
        search_logger.exception(f"Fatal error: {e}")
        raise
    finally:
        if len(study.trials) > 0:
            best_trial = study.best_trial
            search_logger.info(f"SESSION END. Best Objective: {best_trial.value:.4f}")
            search_logger.info(f"Best Params: {json.dumps(best_trial.params)}")

            final_best = {
                "a": best_trial.params["a"],
                "b": best_trial.params["b"],
                "beta": best_trial.params.copy(),  # 简单处理
                "score": best_trial.value,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            # 清理一下 beta 的结构以便保存
            del final_best["beta"]["a"]
            del final_best["beta"]["b"]
            final_best["beta"] = {
                k.replace("beta_", ""): v for k, v in final_best["beta"].items()
            }

            with open(BEST_PARAMS_JSON, "w") as f:
                json.dump(final_best, f, indent=4)
            print(f"Saved to {BEST_PARAMS_JSON}")


if __name__ == "__main__":
    run_optimization()

# === [运行示例] ===
# 重新搜索：nohup python3 search/runner.py --trials 400 --workers 10 --reset > run.log 2>&1 &
# 继续搜索：nohup python3 search/runner.py --trials 500 --workers 28 > run.log 2>&1 &
# 终止搜索：pkill -f "search/runner.py"
# 查看PID： ps -ef | grep search/runner.py
