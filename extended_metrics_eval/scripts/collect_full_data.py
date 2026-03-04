import os
import sys
import json
import uuid
import time
import subprocess
import pandas as pd
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 路径与环境配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
eval_root = os.path.dirname(current_dir)
project_root = os.path.dirname(eval_root)

# 将工程根目录加入系统路径，以便导入 core 和 utils
if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.x265_wrapper import X265Wrapper
from core.adaptive_controller import AdaptiveController
from utils.yuv_io import YUVReader
from calc_psnr_ssim import calculate_metrics

# --- 全局常量配置 (请根据服务器实际环境微调) ---
DATASET_ROOT = "/home/shiyushen/x265_sequence/"
LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
TEMP_DIR = (
    "/dev/shm/x265_eval_temp"  # 使用内存盘 /dev/shm 极大提升大量并发读写时的 IO 性能
)
MAX_WORKERS = 16

DEFAULT_HYPERPARAMS = {
    "a": 1.1033795668037922,
    "b": 1.069230192189941,
    "beta": {
        "VAQ": 0.8315122080542713,
        "CUTree": 0.9131054824257663,
        "PsyRD": 0.5567048950207648,
        "PsyRDOQ": 1.1188971214787788,
        "QComp": 0.20634856441171628,
    },
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_x265_bitrate(csv_file, fps):
    """从 x265 输出的 CSV 日志中精准提取实际码率"""
    try:
        df = pd.read_csv(csv_file, on_bad_lines="skip")
        bits_col = next(
            (col for col in df.columns if col.strip() in ["Bits", "size(bits)"]), None
        )
        if bits_col:
            total_bits = df[bits_col].sum()
            frame_count = len(df)
            if frame_count > 0 and fps > 0:
                duration = frame_count / fps
                return (total_bits / 1000.0) / duration
    except Exception:
        pass
    return None


def _decode_hevc_strictly(hevc_path, yuv_path):
    """极其严谨的解码函数，严格按照原始帧率和物理帧输出"""
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        hevc_path,
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "0",  # 强行关闭时间戳同步，防止丢帧/复帧
        "-f",
        "rawvideo",
        yuv_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def worker_task(task):
    """
    运行在独立子进程中的原子任务闭环 (无任何外部共享状态)。
    """
    seq_name = task["seq_name"]
    mode = task["mode"]
    profile = task["profile"]
    target_bitrate = task["target_bitrate"]

    # 构建沙盒级隔离的临时文件路径
    uid = uuid.uuid4().hex[:8]
    f_hevc = os.path.join(TEMP_DIR, f"{seq_name}_{mode}_{profile}_{uid}.hevc")
    f_csv = os.path.join(TEMP_DIR, f"{seq_name}_{mode}_{profile}_{uid}.csv")
    f_recon = os.path.join(TEMP_DIR, f"{seq_name}_{mode}_{profile}_{uid}_recon.yuv")

    wrapper = X265Wrapper(LIB_PATH)
    result = {
        "seq": seq_name,
        "mode": mode,
        "profile": profile,
        "status": "failed",
        "bitrate": -1.0,
        "psnr": -1.0,
        "ssim": -1.0,
        "error": "",
    }

    try:
        # === 步骤 1: 视频编码 ===
        if mode in ["baseline", "offline"]:
            param = wrapper.param_alloc()
            wrapper.param_default_preset(param, "slow", "None")
            wrapper.param_parse(param, "input-res", f"{task['width']}x{task['height']}")
            wrapper.param_parse(param, "fps", str(task["fps"]))
            wrapper.param_parse(param, "input-csp", "i420")
            wrapper.param_parse(param, "bitrate", str(int(target_bitrate)))
            wrapper.param_parse(param, "strict-cbr", "1")
            wrapper.param_parse(param, "vbv-maxrate", str(int(target_bitrate)))
            wrapper.param_parse(param, "vbv-bufsize", str(int(target_bitrate * 2)))
            wrapper.param_parse(param, "csv", f_csv)
            wrapper.param_parse(param, "csv-log-level", "2")
            wrapper.param_parse(param, "annexb", "1")
            wrapper.param_parse(param, "repeat-headers", "1")

            if mode == "offline" and task.get("offline_params"):
                for k, v in task["offline_params"].items():
                    wrapper.param_parse(param, k, str(v))

            encoder = wrapper.encoder_open(param)
            if not encoder:
                raise RuntimeError("Failed to open x265 encoder")

            pic_in = wrapper.picture_alloc()
            wrapper.picture_init(param, pic_in)
            pic_out = wrapper.picture_alloc()
            pic = pic_in.contents
            pic.bitDepth = 8
            pic.colorSpace = 1

            reader = YUVReader(
                task["ref_yuv"], task["width"], task["height"], 8, task["fps"]
            )
            encoded_frames = 0

            with open(f_hevc, "wb") as f_out:
                while reader.read_frame():
                    y, u, v = reader.get_pointers()
                    s_y, s_u, s_v = reader.get_strides()
                    pic.planes[0], pic.planes[1], pic.planes[2] = y, u, v
                    pic.stride[0], pic.stride[1], pic.stride[2] = s_y, s_u, s_v
                    pic.pts = encoded_frames

                    ret, nal_list = wrapper.encode(encoder, pic_in, pic_out)
                    for nal in nal_list:
                        f_out.write(nal)
                    if ret > 0:
                        encoded_frames += 1

                while True:
                    ret, nal_list = wrapper.encode(encoder, None, pic_out)
                    if ret <= 0:
                        break
                    for nal in nal_list:
                        f_out.write(nal)
                    encoded_frames += 1

            reader.close()
            wrapper.picture_free(pic_in)
            wrapper.picture_free(pic_out)
            wrapper.encoder_close(encoder)
            wrapper.param_free(param)

        elif mode == "online":
            ctrl_config = {
                "output_file": f_hevc,
                "preset": "slow",
                "base_params": {
                    "bitrate": int(target_bitrate),
                    "strict-cbr": 1,
                    "csv": f_csv,
                    "csv-log-level": 2,
                },
                "mode_params": task.get("mode_params", {}),
                "tune_params": task.get("tune_params", {}),
                "hyperparams": DEFAULT_HYPERPARAMS,
            }
            reader = YUVReader(
                task["ref_yuv"], task["width"], task["height"], 8, task["fps"]
            )
            controller = AdaptiveController(wrapper, reader, ctrl_config)
            controller.run()
            reader.close()

        # === 步骤 2: 解析码率 ===
        real_bitrate = _parse_x265_bitrate(f_csv, task["fps"])
        if not real_bitrate:
            raise RuntimeError("Failed to parse real bitrate from CSV")

        # === 步骤 3: 严谨解码 ===
        if not _decode_hevc_strictly(f_hevc, f_recon):
            raise RuntimeError("FFmpeg decoding failed")

        # === 步骤 4: 客观指标提取 ===
        metrics = calculate_metrics(
            dist_yuv=f_recon,
            ref_yuv=task["ref_yuv"],
            width=task["width"],
            height=task["height"],
            pixel_format="yuv420p",
        )

        if metrics["psnr"] <= 0 or metrics["ssim"] <= 0:
            raise RuntimeError("Metric calculation returned invalid values")

        # 组装成功载荷
        result.update(
            {
                "status": "success",
                "bitrate": float(real_bitrate),
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
            }
        )

    except Exception as e:
        result["error"] = str(e)
    finally:
        # === 步骤 5: 无痕清理沙盒 ===
        for f in [f_hevc, f_csv, f_recon]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

    return result


def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    out_json_dir = os.path.join(eval_root, "results_json")
    os.makedirs(out_json_dir, exist_ok=True)

    seq_config = load_json(os.path.join(project_root, "config", "test_sequences.json"))
    init_params = load_json(os.path.join(project_root, "config", "initial_params.json"))

    # 1. 任务扁平化 (Task Flattening)
    tasks = []
    modes = ["baseline", "offline", "online"]

    for seq_key, meta in seq_config.items():
        v_class = meta.get("class", "")
        yuv_path = (
            os.path.join(DATASET_ROOT, f"Class{v_class}", f"{seq_key}.yuv")
            if v_class
            else os.path.join(DATASET_ROOT, f"{seq_key}.yuv")
        )

        if not os.path.exists(yuv_path):
            print(f"[Warn] Missing YUV for {seq_key}, skipped.")
            continue

        for profile_name, target_bitrate in meta.get("bitrates", {}).items():
            profile_cfg = init_params.get("profiles", {}).get(profile_name, {})
            mode_params = profile_cfg.get("mode_params", {})
            tune_params = profile_cfg.get("tune_params", {})

            # 合并离线最佳参数
            offline_params = deepcopy(mode_params)
            offline_params.update(tune_params)

            for mode in modes:
                tasks.append(
                    {
                        "seq_name": seq_key,
                        "profile": profile_name,
                        "mode": mode,
                        "target_bitrate": target_bitrate,
                        "width": meta["width"],
                        "height": meta["height"],
                        "fps": meta["fps"],
                        "ref_yuv": yuv_path,
                        "offline_params": offline_params,
                        "mode_params": mode_params,
                        "tune_params": tune_params,
                    }
                )

    total_tasks = len(tasks)
    print(f"=== Starting High-Concurrency Data Collection ===")
    print(f"Total flattened tasks: {total_tasks}. Workers: {MAX_WORKERS}")

    # 2. 状态机字典初始化
    results_tree = {"baseline": {}, "offline": {}, "online": {}}

    start_time = time.time()
    completed = 0

    # 3. 高并发派发与回收
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(worker_task, t): t for t in tasks}

        for future in as_completed(future_to_task):
            res = future.result()
            completed += 1

            seq = res["seq"]
            mode = res["mode"]
            prof = res["profile"]

            if res["status"] == "success":
                if seq not in results_tree[mode]:
                    results_tree[mode][seq] = {}
                results_tree[mode][seq][prof] = {
                    "bitrate": res["bitrate"],
                    "psnr": res["psnr"],
                    "ssim": res["ssim"],
                }
                print(
                    f"[{completed}/{total_tasks}] SUCCESS: {seq} | {mode} | {prof} -> {res['psnr']:.2f} dB"
                )
            else:
                print(
                    f"[{completed}/{total_tasks}] FAILED: {seq} | {mode} | {prof} -> {res['error']}"
                )

            # 增量落盘保护 (每完成 20 个任务强制写一次盘，防止断电丢失)
            if completed % 20 == 0 or completed == total_tasks:
                for m_name in modes:
                    out_path = os.path.join(out_json_dir, f"{m_name}_metrics.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(results_tree[m_name], f, indent=4)

    print(f"\n=== All Tasks Completed in {time.time() - start_time:.2f}s ===")
    print(f"Metrics saved to: {out_json_dir}")


if __name__ == "__main__":
    main()

# nohup python3 -u extended_metrics_eval/scripts/collect_full_data.py > extended_metrics_eval/collect_data.log 2>&1 &
