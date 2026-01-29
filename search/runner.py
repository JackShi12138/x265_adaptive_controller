import os
import sys
import json
import logging
import numpy as np  # [新增] 用于计算标准差
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
SEARCH_LOG_FILE = os.path.join(
    PROJECT_ROOT, "search_progress.log"
)  # [新增] 独立日志路径

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


# === [新增] 初始化独立 Logger ===
def setup_search_logger():
    logger = logging.getLogger("SearchMonitor")
    logger.setLevel(logging.INFO)

    # 防止重复添加 Handler
    if not logger.handlers:
        # File Handler: 只写到 search_progress.log，不写到控制台(run.log)
        fh = logging.FileHandler(SEARCH_LOG_FILE, mode="w", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


search_logger = setup_search_logger()


# === [新增] 参数波动统计函数 ===
def analyze_parameter_variance(study, window=20):
    """计算最近 window 次试验的参数标准差，判断收敛趋势"""
    if len(study.trials) < window:
        return "Not enough data"

    # 获取最近完成的 trails
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    recent_trials = completed_trials[-window:]

    if not recent_trials:
        return "No complete trials"

    # 提取参数矩阵
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
        # 归一化波动率 (Std / Mean)，处理量纲不同
        rel_std = std_dev / (abs(mean_val) + 1e-6)
        variances[key] = f"{std_dev:.2f}({rel_std:.1%})"

    # 格式化输出
    # 重点关注几个波动最大的
    report = " | ".join([f"{k}:{v}" for k, v in variances.items()])
    return report


def objective(trial, evaluator, study):  # [修改] 传入 study 用于统计
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

    # 简略日志到控制台 (保持心跳)
    print(f"[Trial {trial.number}] Processing...")

    # 2. 执行评估
    mean_score, details, crash_report, stats = evaluator.evaluate_batch(hyperparams)

    if mean_score == -9999.0:
        search_logger.error(
            f"[Trial {trial.number}] FAILED. Params: {json.dumps(hyperparams)}"
        )
        raise optuna.TrialPruned("Eval Failed")

    # 3. 评分逻辑
    scores = [info.get("bd_vmaf", 0.0) for info in details.values()]
    min_score = min(scores)
    mean_val = sum(scores) / len(scores)
    negative_count = sum(1 for s in scores if s < 0)

    THRESHOLD = -0.1

    if min_score < THRESHOLD:
        final_objective = min_score * 2.0
        mode = "SURVIVAL"
    else:
        final_objective = mean_val + 0.5 * min_score
        mode = "GROWTH"

    # === [关键修改] 写入独立日志文件 ===
    # 计算当前是否打破了历史最佳记录
    try:
        # 注意：study.best_value 可能不包含当前正在跑的这一轮（取决于Optuna版本和并发更新时机）
        # 所以我们需要比较一下
        history_best = study.best_value
        global_best_score = max(history_best, final_objective)
    except ValueError:
        # 第一轮
        global_best_score = final_objective

    is_new_best = final_objective >= global_best_score

    marker = "★ NEW BEST ★" if is_new_best else ""

    # 构建详细日志条目
    log_msg = (
        f"[Trial {trial.number}] {marker} | Best So Far: {global_best_score:.4f}\n"
        f"  Mode: {mode} (Obj: {final_objective:.4f})\n"
        f"  Metrics: Mean={mean_val:.4f} | Min={min_score:.4f} | Negs={negative_count}\n"
        f"  Params: a={param_a:.6f}, b={param_b:.6f}, "
        f"VAQ={beta_vaq:.6f}, CUTree={beta_cutree:.6f}, PsyRD={beta_psyrd:.6f}, PsyRDOQ={beta_psyrdoq:.6f}, QComp={beta_qcomp:.6f}\n"
    )

    # 每 10 轮分析一次参数收敛情况
    if trial.number > 0 and trial.number % 10 == 0:
        variance_report = analyze_parameter_variance(study)
        log_msg += f"  [Convergence Stats (Last 20)]: {variance_report}\n"

    search_logger.info(log_msg)

    # 记录辅助数据
    trial.set_user_attr("raw_avg_bd", mean_val)
    trial.set_user_attr("min_bd", min_score)
    trial.set_user_attr("negative_count", negative_count)

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

    # 启动日志
    search_logger.info("=" * 60)
    search_logger.info(
        f"SESSION START: trials={args.trials}, workers={args.workers}, reset={args.reset}"
    )
    search_logger.info(f"Dataset: {args.dataset}")
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
        os.remove(DB_PATH)
        search_logger.warning("Database reset!")

    sampler = CmaEsSampler(restart_strategy="ipop", n_startup_trials=10)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    try:
        # 使用 lambda 将 study 传给 objective
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
            # 最终结果写入日志
            search_logger.info(f"SESSION END. Best Score: {best_trial.value:.4f}")
            search_logger.info(f"Best Params: {json.dumps(best_trial.params)}")

            # 保存到 JSON
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

# === [运行示例] ===
# 重新搜索：nohup python3 search/runner.py --trials 400 --workers 10 --reset > run.log 2>&1 &
# 继续搜索：nohup python3 search/runner.py --trials 500 --workers 10 > run.log 2>&1 &
