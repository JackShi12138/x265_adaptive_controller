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
import pandas as pd

from core.adaptive_controller import AdaptiveController
from core.x265_wrapper import X265Wrapper
from utils.yuv_io import YUVReader

try:
    from search.metric import calculate_bd_vmaf
except ImportError:
    calculate_bd_vmaf = None

TEMP_DIR = "/dev/shm/x265_search_temp"
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
FFMPEG_EXEC = "ffmpeg"
VMAF_EXEC = "vmaf"


def _parse_x265_bitrate(csv_file, fps):
    try:
        df = pd.read_csv(csv_file, on_bad_lines="skip")
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
                return (total_bits / 1000.0) / duration
    except Exception:
        pass
    return None


def _worker_task(task_context):
    task_id = task_context["task_id"]
    seq_info = task_context["seq_info"]
    base_config = task_context["base_config"]
    hyperparams = task_context["hyperparams"]
    lib_path = task_context.get("lib_path", DEFAULT_LIB_PATH)
    max_frames = task_context.get("max_frames", 0)

    unique_id = f"{task_id}_{uuid.uuid4().hex[:6]}"
    f_hevc = os.path.join(TEMP_DIR, f"{unique_id}.hevc")
    f_csv = os.path.join(TEMP_DIR, f"{unique_id}.csv")
    f_yuv_dec = os.path.join(TEMP_DIR, f"{unique_id}_dec.yuv")
    f_vmaf_json = os.path.join(TEMP_DIR, f"{unique_id}_vmaf.json")

    run_config = deepcopy(base_config)
    run_config["output_file"] = f_hevc
    run_config["hyperparams"] = hyperparams
    run_config["base_params"]["csv"] = f_csv
    run_config["base_params"]["csv-log-level"] = 2
    if max_frames > 0:
        run_config["base_params"]["frames"] = max_frames

    wrapper = X265Wrapper(lib_path)
    reader = None
    result = {"status": "failed", "error": ""}

    try:
        reader = YUVReader(
            file_path=seq_info["path"],
            width=seq_info["width"],
            height=seq_info["height"],
            bit_depth=8,
            fps=seq_info["fps"],
        )
        controller = AdaptiveController(wrapper, reader, run_config)
        controller.run()

        if not os.path.exists(f_csv):
            if not (os.path.exists(f_hevc) and os.path.getsize(f_hevc) > 0):
                raise FileNotFoundError(f"x265 CSV log not found")

        real_bitrate_kbps = _parse_x265_bitrate(f_csv, seq_info["fps"])
        if real_bitrate_kbps is None:
            file_size = os.path.getsize(f_hevc)
            if file_size > 0:
                duration = max_frames / seq_info["fps"] if max_frames > 0 else 10.0
                real_bitrate_kbps = (file_size * 8 / 1000.0) / duration
            else:
                raise ValueError(f"Bitrate calc failed")

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
            "real_bitrate": real_bitrate_kbps,
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
        target_seqs=None,
        max_frames=0,
    ):
        self.max_workers = max_workers
        self.dataset_root = dataset_root
        self.lib_path = lib_path
        self.max_frames = max_frames
        self.target_seqs = target_seqs

        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR)
            except:
                pass
        os.makedirs(TEMP_DIR, exist_ok=True)

        if not os.path.exists(init_params_path):
            raise FileNotFoundError(f"Initial params not found")
        with open(init_params_path, "r") as f:
            self.init_params = json.load(f)
        self.tasks_metadata = self._load_and_merge_metadata(
            anchor_json_path, seq_meta_json_path
        )
        print(
            f"[Evaluator] Initialized {len(self.tasks_metadata)} tasks. Workers: {max_workers}. Frames Limit: {max_frames if max_frames > 0 else 'Full'}"
        )

    def _load_and_merge_metadata(self, anchor_path, meta_path):
        with open(anchor_path, "r") as f:
            anchors_data = json.load(f)
        with open(meta_path, "r") as f:
            seq_meta_data = json.load(f)
        tasks = []
        _id = 0
        for seq_key, anchors in anchors_data.items():
            if self.target_seqs is not None and seq_key not in self.target_seqs:
                continue
            if seq_key not in seq_meta_data:
                continue
            meta = seq_meta_data[seq_key]

            video_class = meta.get("class", "")
            class_folder = f"Class{video_class}" if video_class else ""
            full_path = os.path.join(self.dataset_root, class_folder, f"{seq_key}.yuv")
            if not os.path.exists(full_path):
                full_path = os.path.join(self.dataset_root, f"{seq_key}.yuv")

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

    def evaluate_batch(self, hyperparams):  # [改动] 不再接收 trial 参数
        work_items = []
        global_base_params = self.init_params.get("base_params", {})

        for meta in self.tasks_metadata:
            profile = meta.get("profile", "Medium")
            profiles_dict = self.init_params.get("profiles", {})
            profile_config = profiles_dict.get(profile, {})
            task_base_params = deepcopy(global_base_params)
            task_base_params.update(
                {
                    "bitrate": int(meta["target_bitrate"]),
                    "strict-cbr": 1,
                    "fps": int(meta["fps"]),
                    "input-res": f"{meta['width']}x{meta['height']}",
                    "input-depth": 8,
                    "internal-bit-depth": 8,
                    "w": meta["width"],
                    "h": meta["height"],
                    "vbv-maxrate": int(meta["target_bitrate"]),
                    "vbv-bufsize": int(meta["target_bitrate"]),
                }
            )
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
                "max_frames": self.max_frames,
            }
            work_items.append(context)

        results = []
        fail_count = 0
        total_tasks = len(work_items)
        crash_report = {"count": 0, "errors": []}

        # [改动] 移除所有 Pruning 相关的中间统计代码
        print(
            f"--- Batch Eval: {total_tasks} tasks (Subset), {self.max_frames} frames ---"
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
                        crash_report["errors"].append(f"Task fail: {res.get('error')}")
                except Exception as exc:
                    fail_count += 1
                    crash_report["errors"].append(f"Exception: {str(exc)}")

                if fail_count > total_tasks * 0.25:  # [保留] 错误熔断 (Circuit Breaker)
                    print(f"[Circuit Breaker] Too many failures. Aborting.")
                    executor.shutdown(wait=False, cancel_futures=True)
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
            test_points = [(r["real_bitrate"], r["vmaf"]) for r in items]
            anchor_points = [
                (r["anchor_real_bitrate"], r["anchor_vmaf"]) for r in items
            ]

            if len(items) < 3:
                details[seq_name] = {
                    "error": "Insufficient points",
                    "test_points": test_points,
                }
                continue

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
