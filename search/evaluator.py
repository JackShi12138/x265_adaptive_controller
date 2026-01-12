import os
import csv
import json
import uuid
import shutil
import subprocess
import concurrent.futures
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd  # [新增] 引入 pandas 处理 CSV

# === 导入项目核心组件 ===
from core.adaptive_controller import AdaptiveController
from core.x265_wrapper import X265Wrapper
from utils.yuv_io import YUVReader

# === 导入评价指标模块 ===
try:
    from search.metric import calculate_bd_vmaf
except ImportError:
    print("[Error] Could not import search.metric. Please ensure metric.py is created.")
    calculate_bd_vmaf = None

# ==============================================================================
# [配置区域]
# ==============================================================================
TEMP_DIR = "/dev/shm/x265_search_temp"
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"

FFMPEG_EXEC = "ffmpeg"
VMAF_EXEC = "vmaf"

# ==============================================================================


def _parse_x265_bitrate(csv_file, fps):
    """
    [修改版] 计算真实码率
    优先策略：累加 Pandas 的 'Bits' 列 (最稳健)
    兜底策略：解析文本 Summary
    """
    real_bitrate = None

    # === 策略 1: 累加帧级 Bits (数学计算) ===
    try:
        # on_bad_lines='skip' 会丢弃底部的 summary，这正好方便我们统计帧数据
        df = pd.read_csv(csv_file, on_bad_lines="skip")

        # 寻找 Bits 列 (兼容带空格的情况)
        bits_col = None
        for col in df.columns:
            if col.strip() == "Bits" or col.strip() == "size(bits)":
                bits_col = col
                break

        if bits_col:
            total_bits = df[bits_col].sum()
            frame_count = len(df)

            if frame_count > 0 and fps > 0:
                duration = frame_count / fps
                real_bitrate = (total_bits / 1000.0) / duration
                return real_bitrate  # 成功返回
    except Exception:
        pass  # 失败则进入策略 2

    # === 策略 2: 纯文本解析 Summary (兜底) ===
    try:
        with open(csv_file, "r", errors="ignore") as f:
            lines = f.readlines()

        # 倒序读取最后 20 行
        for line in reversed(lines[-20:]):
            # x265 Summary 通常格式: "kb/s: 2045.23" 或 "Bitrate: 2045.23 kbps"
            if "kb/s:" in line:
                # 例子: "encoded 600 frames in ... 2045.23 kb/s"
                parts = line.split("kb/s")[0].strip().split()
                return float(parts[-1])
            elif "Bitrate" in line:
                # 例子: "Bitrate: 2045.23"
                parts = line.split(":")
                # 简单的提取数字逻辑
                import re

                match = re.search(r"(\d+\.?\d*)", parts[1])
                if match:
                    return float(match.group(1))
    except Exception:
        pass

    return None


def _worker_task(task_context):
    """
    Worker 任务函数
    """
    task_id = task_context["task_id"]
    seq_info = task_context["seq_info"]
    base_config = task_context["base_config"]
    hyperparams = task_context["hyperparams"]
    lib_path = task_context.get("lib_path", DEFAULT_LIB_PATH)

    unique_id = f"{task_id}_{uuid.uuid4().hex[:6]}"

    f_hevc = os.path.join(TEMP_DIR, f"{unique_id}.hevc")
    f_csv = os.path.join(TEMP_DIR, f"{unique_id}.csv")
    f_yuv_dec = os.path.join(TEMP_DIR, f"{unique_id}_dec.yuv")
    f_vmaf_json = os.path.join(TEMP_DIR, f"{unique_id}_vmaf.json")

    # === 1. Config ===
    run_config = deepcopy(base_config)
    run_config["output_file"] = f_hevc
    run_config["hyperparams"] = hyperparams

    # [修改] 设置 log-level 为 2 以配合 bitrate 读取逻辑
    run_config["base_params"]["csv"] = f_csv
    run_config["base_params"]["csv-log-level"] = 2

    # === 2. 编码 ===
    wrapper = X265Wrapper(lib_path)
    reader = None
    result = {"status": "failed", "error": ""}

    try:
        # 保持之前的修复：显式指定关键字参数
        reader = YUVReader(
            file_path=seq_info["path"],
            width=seq_info["width"],
            height=seq_info["height"],
            bit_depth=8,
            fps=seq_info["fps"],
        )

        controller = AdaptiveController(wrapper, reader, run_config)
        controller.run()

        # === 3. 解析结果 (使用 Pandas) ===
        if not os.path.exists(f_csv):
            if os.path.exists(f_hevc) and os.path.getsize(f_hevc) > 0:
                # 这种情况下无法读取精确码率，必须报错
                raise FileNotFoundError(f"Encoding finished but CSV missing: {f_csv}")
            raise FileNotFoundError(f"x265 CSV log not found: {f_csv}")

        # [修改] 调用新的解析逻辑，传入 FPS
        real_bitrate_kbps = _parse_x265_bitrate(f_csv, seq_info["fps"])

        if real_bitrate_kbps is None:
            # 如果两种策略都失败，抛出详细错误
            raise ValueError(f"Failed to calculate Bitrate from CSV: {f_csv}")

        # FFmpeg 解码
        cmd_decode = [
            FFMPEG_EXEC,
            "-y",
            "-v",
            "error",
            "-i",
            f_hevc,
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "0",
            f_yuv_dec,
        ]
        subprocess.run(cmd_decode, check=True)

        # VMAF 计算
        cmd_vmaf = [
            VMAF_EXEC,
            "-r",
            seq_info["path"],
            "-d",
            f_yuv_dec,
            "-w",
            str(seq_info["width"]),
            "-h",
            str(seq_info["height"]),
            "-p",
            "420",
            "-b",
            "8",
            "--json",
            "-o",
            f_vmaf_json,
        ]
        subprocess.run(
            cmd_vmaf, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # 解析 VMAF
        with open(f_vmaf_json, "r") as f:
            vmaf_data = json.load(f)
            if "pooled_metrics" in vmaf_data:
                vmaf_score = vmaf_data["pooled_metrics"]["vmaf"]["mean"]
            elif "VMAF score" in vmaf_data:
                vmaf_score = vmaf_data["VMAF score"]
            elif "vmaf" in vmaf_data:
                vmaf_score = vmaf_data["vmaf"]
            else:
                raise ValueError("Unknown VMAF JSON structure")

        result = {
            "status": "success",
            "task_id": task_id,
            "seq_name": seq_info["seq_name"],
            "target_bitrate": seq_info["target_bitrate"],
            "real_bitrate": real_bitrate_kbps,  # 这里的码率是 x265 Summary 报告的准确值
            "vmaf": vmaf_score,
            "anchor_real_bitrate": seq_info["anchor_real_bitrate"],
            "anchor_vmaf": seq_info["anchor_vmaf"],
        }

    except Exception as e:
        result["error"] = str(e)

    finally:
        if reader:
            reader.close()
        for f in [f_hevc, f_csv, f_yuv_dec, f_vmaf_json]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    return result


class ParallelEvaluator:
    def __init__(
        self,
        anchor_json_path,
        seq_meta_json_path,
        init_params_path,
        dataset_root,
        lib_path=DEFAULT_LIB_PATH,
        max_workers=16,
    ):
        self.max_workers = max_workers
        self.dataset_root = dataset_root
        self.lib_path = lib_path

        # 清理临时目录
        if os.path.exists(TEMP_DIR):
            try:
                for filename in os.listdir(TEMP_DIR):
                    file_path = os.path.join(TEMP_DIR, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            os.makedirs(TEMP_DIR, exist_ok=True)

        if not os.path.exists(init_params_path):
            raise FileNotFoundError(f"Initial params not found: {init_params_path}")
        with open(init_params_path, "r") as f:
            self.init_params = json.load(f)

        self.tasks_metadata = self._load_and_merge_metadata(
            anchor_json_path, seq_meta_json_path
        )
        print(
            f"[Evaluator] Initialized {len(self.tasks_metadata)} tasks. Workers: {max_workers}"
        )

    def _load_and_merge_metadata(self, anchor_path, meta_path):
        if not os.path.exists(anchor_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Config files missing")

        with open(anchor_path, "r") as f:
            anchors_data = json.load(f)
        with open(meta_path, "r") as f:
            seq_meta_data = json.load(f)

        tasks = []
        _id = 0

        for seq_key, anchors in anchors_data.items():
            if seq_key not in seq_meta_data:
                continue

            meta = seq_meta_data[seq_key]

            # 路径推导
            video_class = meta.get("class", "")
            class_folder = f"Class{video_class}" if video_class else ""
            full_path = os.path.join(self.dataset_root, class_folder, f"{seq_key}.yuv")

            if not os.path.exists(full_path):
                fallback_path = os.path.join(self.dataset_root, f"{seq_key}.yuv")
                if os.path.exists(fallback_path):
                    full_path = fallback_path

            defined_bitrates = meta.get("bitrates", {})
            for anchor in anchors:
                anchor_bit = anchor["bitrate"]
                profile_name = "Medium"
                if defined_bitrates:
                    profile_name = min(
                        defined_bitrates,
                        key=lambda k: abs(defined_bitrates[k] - anchor_bit),
                    )
                target_bitrate = defined_bitrates.get(profile_name, anchor_bit)

                tasks.append(
                    {
                        "task_id": _id,
                        "seq_name": seq_key,
                        "path": full_path,
                        "width": meta["width"],
                        "height": meta["height"],
                        "fps": meta["fps"],
                        "target_bitrate": target_bitrate,
                        "profile": profile_name,
                        "anchor_vmaf": anchor["vmaf"],
                        "anchor_real_bitrate": anchor.get(
                            "real_bitrate", anchor["bitrate"]
                        ),
                    }
                )
                _id += 1
        return tasks

    def evaluate_batch(self, hyperparams):
        work_items = []
        global_base_params = self.init_params.get("base_params", {})

        for meta in self.tasks_metadata:
            profile = meta.get("profile", "Medium")

            profiles_dict = self.init_params.get("profiles", {})
            profile_config = profiles_dict.get(profile, {})
            if not profile_config and "profiles" in self.init_params:
                profile_config = self.init_params["profiles"].get(profile, {})

            # Config 构造
            task_base_params = deepcopy(global_base_params)

            task_base_params["bitrate"] = int(meta["target_bitrate"])
            task_base_params["strict-cbr"] = 1

            # 显式注入元数据 (保持之前的修复)
            task_base_params["fps"] = int(meta["fps"])
            task_base_params["input-res"] = f"{meta['width']}x{meta['height']}"
            task_base_params["input-depth"] = 8
            task_base_params["internal-bit-depth"] = 8
            task_base_params["w"] = meta["width"]
            task_base_params["h"] = meta["height"]

            if "vbv-maxrate" not in task_base_params:
                task_base_params["vbv-maxrate"] = int(meta["target_bitrate"])
            if "vbv-bufsize" not in task_base_params:
                task_base_params["vbv-bufsize"] = int(meta["target_bitrate"])

            base_config = {
                "preset": "slow",
                "base_params": task_base_params,
                "mode_params": profile_config.get("mode_params", {}),
                "tune_params": profile_config.get("tune_params", {}),
            }

            context = {
                "task_id": meta["task_id"],
                "seq_info": meta,
                "base_config": base_config,
                "hyperparams": hyperparams,
                "lib_path": self.lib_path,
            }
            work_items.append(context)

        results = []
        fail_count = 0
        total_tasks = len(work_items)
        crash_report = {"count": 0, "errors": []}

        print(
            f"--- Starting Batch Evaluation ({total_tasks} tasks, {self.max_workers} workers) ---"
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_id = {
                executor.submit(_worker_task, item): item["task_id"]
                for item in work_items
            }

            for future in concurrent.futures.as_completed(future_to_id):
                try:
                    res = future.result()
                    if res["status"] == "success":
                        results.append(res)
                    else:
                        fail_count += 1
                        crash_report["errors"].append(
                            f"Task {future_to_id[future]} failed: {res.get('error')}"
                        )
                except Exception as exc:
                    fail_count += 1
                    crash_report["errors"].append(
                        f"Task {future_to_id[future]} exception: {str(exc)}"
                    )

                if fail_count > total_tasks * 0.15:
                    print(
                        f"[Circuit Breaker] Failure rate too high ({fail_count}/{total_tasks}). Aborting."
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    crash_report["count"] = fail_count
                    return -9999.0, {}, crash_report, {}

        crash_report["count"] = fail_count
        if not results:
            return -9999.0, {}, crash_report, {}

        seq_groups = defaultdict(list)
        for res in results:
            seq_groups[res["seq_name"]].append(res)

        bd_scores = []
        details = {}

        for seq_name, items in seq_groups.items():
            # 1. 无论点数多少，先提取数据 (方便调试)
            test_points = [(r["real_bitrate"], r["vmaf"]) for r in items]
            anchor_points = [
                (r["anchor_real_bitrate"], r["anchor_vmaf"]) for r in items
            ]

            # 2. 检查点数是否足够计算 BD-VMAF
            if len(items) < 3:
                # [关键修改] 即使计算失败，也返回 test_points 以便查看冒烟测试结果
                details[seq_name] = {
                    "error": "Insufficient points (<3) for BD-Metric",
                    "test_points": test_points,
                    "anchor_points": anchor_points,
                }
                continue

            # 3. 计算 BD-VMAF
            if calculate_bd_vmaf:
                score = calculate_bd_vmaf(anchor_points, test_points)
                if score != -9999.0:
                    bd_scores.append(score)
                    details[seq_name] = {"bd_vmaf": score, "test_points": test_points}
                else:
                    details[seq_name] = {
                        "error": "BD Calc Failed",
                        "test_points": test_points,
                    }
            else:
                return -9999.0, {}, {"error": "Metric missing"}, {}
            anchor_points = [
                (r["anchor_real_bitrate"], r["anchor_vmaf"]) for r in items
            ]

            if calculate_bd_vmaf:
                score = calculate_bd_vmaf(anchor_points, test_points)
                if score != -9999.0:
                    bd_scores.append(score)
                    details[seq_name] = {"bd_vmaf": score, "test_points": test_points}
                else:
                    details[seq_name] = {"error": "BD Calc Failed"}
            else:
                return -9999.0, {}, {"error": "Metric missing"}, {}

        if not bd_scores:
            return -9999.0, details, crash_report, {}
        final_score = np.mean(bd_scores)
        perf_stats = {"processed_seqs": len(bd_scores)}

        return final_score, details, crash_report, perf_stats
