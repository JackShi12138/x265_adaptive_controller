import os
import json
import time
import argparse
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# 假设核心模块在 core/ 和 utils/ 下，请确保 PYTHONPATH 包含这些路径
from core.x265_wrapper import X265Wrapper
from core.adaptive_controller import AdaptiveController
from utils.yuv_io import YUVReader

# === 配置区域 ===
DATASET_ROOT = "/home/shiyushen/x265_sequence/"
LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
OUTPUT_ROOT = "analysis_data"
VMAF_EXEC = "vmaf"  # 请确保 vmaf 在系统 PATH 中，或者写绝对路径

# 待分析的视频列表
TARGET_SEQS = [
    "RaceHorses_832x480_30",
    "PeopleOnStreet_2560x1600_30_crop",
    "ParkScene_1920x1080_24",
    "BasketballPass_416x240_50",
]

# 目标处理的清晰度档位
TARGET_PROFILES = ["Very Low", "Low", "Medium", "High"]

# Online 模式超参数配置 (支持按 Profile 区分，目前统一使用一套)
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

# 如果不同 Profile 有不同的最优参数，可以在这里定义覆盖
HYPERPARAMS_MAP = {
    "Very Low": DEFAULT_HYPERPARAMS,
    "Low": DEFAULT_HYPERPARAMS,
    "Medium": DEFAULT_HYPERPARAMS,
    "High": DEFAULT_HYPERPARAMS,
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def decode_hevc(hevc_path, yuv_path):
    """调用 ffmpeg 将 HEVC 解码为 YUV420P"""
    if not os.path.exists(hevc_path):
        print(f"[Warn] HEVC file missing: {hevc_path}")
        return False

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
        "0",
        "-f",
        "rawvideo",
        yuv_path,
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"[Error] FFmpeg decoding failed for {hevc_path}")
        return False


def run_vmaf_calculation(
    ref_yuv, dist_yuv, width, height, log_path, pixel_format="420", bit_depth=8
):
    """计算 VMAF 并保存为 JSON"""
    if os.path.exists(log_path):
        return  # 已存在则跳过

    if not os.path.exists(ref_yuv) or not os.path.exists(dist_yuv):
        print(f"[Error] Missing YUV for VMAF: {ref_yuv} or {dist_yuv}")
        return

    cmd_vmaf = [
        VMAF_EXEC,
        "-r",
        ref_yuv,
        "-d",
        dist_yuv,
        "-w",
        str(width),
        "-h",
        str(height),
        "-p",
        pixel_format,
        "-b",
        str(bit_depth),
        "--json",
        "-o",
        log_path,
    ]

    try:
        # 使用 subprocess 运行 vmaf
        subprocess.run(
            cmd_vmaf, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(f"[Error] VMAF calculation failed for {log_path}")
    except FileNotFoundError:
        print(f"[Error] VMAF executable '{VMAF_EXEC}' not found.")


def run_native_encode(wrapper, seq_info, output_hevc, csv_path, extra_params=None):
    """原生 x265 编码循环 (用于 Slow 和 Offline 模式)"""
    width = seq_info["width"]
    height = seq_info["height"]
    fps = seq_info["fps"]
    bitrate = seq_info["bitrate"]

    param = wrapper.param_alloc()
    wrapper.param_default_preset(param, "slow", "None")

    wrapper.param_parse(param, "input-res", f"{width}x{height}")
    wrapper.param_parse(param, "fps", str(fps))
    wrapper.param_parse(param, "input-csp", "i420")
    wrapper.param_parse(param, "bitrate", str(int(bitrate)))
    wrapper.param_parse(param, "strict-cbr", "1")
    wrapper.param_parse(param, "vbv-maxrate", str(int(bitrate)))
    wrapper.param_parse(param, "vbv-bufsize", str(int(bitrate * 2)))
    wrapper.param_parse(param, "csv", csv_path)
    wrapper.param_parse(param, "csv-log-level", "2")
    wrapper.param_parse(param, "annexb", "1")
    wrapper.param_parse(param, "repeat-headers", "1")

    if extra_params:
        for k, v in extra_params.items():
            if v is None:
                continue
            wrapper.param_parse(param, k, str(v))

    encoder = wrapper.encoder_open(param)
    if not encoder:
        print("[Error] Failed to open encoder")
        return

    pic_in = wrapper.picture_alloc()
    wrapper.picture_init(param, pic_in)
    pic_out = wrapper.picture_alloc()
    pic = pic_in.contents
    pic.bitDepth = 8
    pic.colorSpace = 1

    reader = YUVReader(seq_info["path"], width, height, 8, fps)
    encoded_frames = 0

    try:
        f_out = open(output_hevc, "wb")
        while True:
            if not reader.read_frame():
                break
            y, u, v = reader.get_pointers()
            s_y, s_u, s_v = reader.get_strides()
            pic.planes[0], pic.planes[1], pic.planes[2] = y, u, v
            pic.stride[0], pic.stride[1], pic.stride[2] = s_y, s_u, s_v
            pic.pts = encoded_frames

            ret, nal_list = wrapper.encode(encoder, pic_in, pic_out)
            if ret < 0:
                break
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
    finally:
        f_out.close()
        reader.close()
        wrapper.picture_free(pic_in)
        wrapper.picture_free(pic_out)
        wrapper.encoder_close(encoder)
        wrapper.param_free(param)


def process_sequence_profile(task_args):
    """
    处理单个任务：(序列, Profile)
    """
    seq_key, profile_name, base_dir, seq_meta, init_config_full = task_args

    # 初始化 Wrapper (进程内独立)
    wrapper = X265Wrapper(LIB_PATH)

    # 获取当前 Profile 的配置
    if profile_name not in init_config_full["profiles"]:
        print(f"[Error] Profile {profile_name} not found in init_config")
        return

    profile_data = init_config_full["profiles"][profile_name]
    target_mode_params = profile_data["mode_params"]
    target_tune_params = profile_data["tune_params"]

    meta = seq_meta[seq_key]
    if profile_name not in meta["bitrates"]:
        print(f"[Skip] Bitrate not defined for {seq_key} @ {profile_name}")
        return

    bitrate = meta["bitrates"][profile_name]

    # 寻找输入源 YUV
    video_class = meta.get("class", "")
    class_folder = f"Class{video_class}" if video_class else ""
    yuv_path = os.path.join(DATASET_ROOT, class_folder, f"{seq_key}.yuv")
    if not os.path.exists(yuv_path):
        yuv_path = os.path.join(DATASET_ROOT, f"{seq_key}.yuv")

    seq_info = {
        "name": seq_key,
        "path": yuv_path,
        "width": meta["width"],
        "height": meta["height"],
        "fps": meta["fps"],
        "bitrate": bitrate,
    }

    # === 目录结构: base_dir / seq_key / profile / method ===
    seq_profile_dir = os.path.join(base_dir, seq_key, profile_name)
    print(f"[{os.getpid()}] Processing {seq_key} [{profile_name}] @ {bitrate} kbps...")

    methods = ["slow", "offline", "online", "ssim"]

    for method in methods:
        method_dir = os.path.join(seq_profile_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        hevc_path = os.path.join(method_dir, "output.hevc")
        recon_path = os.path.join(method_dir, "recon.yuv")
        csv_path = os.path.join(method_dir, "x265_log.csv")
        vmaf_log_path = os.path.join(method_dir, "vmaf.json")

        # 1. Encoding
        if method == "slow":
            run_native_encode(wrapper, seq_info, hevc_path, csv_path, extra_params={})

        elif method == "ssim":
            ssim_params = {"tune": "ssim"}
            run_native_encode(
                wrapper, seq_info, hevc_path, csv_path, extra_params=ssim_params
            )

        elif method == "offline":
            offline_params = {}
            offline_params.update(target_mode_params)
            offline_params.update(target_tune_params)
            run_native_encode(
                wrapper, seq_info, hevc_path, csv_path, extra_params=offline_params
            )

        elif method == "online":
            ctrl_config = {
                "output_file": hevc_path,
                "controller_log": os.path.join(method_dir, "controller_log.json"),
                "preset": "slow",
                "base_params": {
                    "bitrate": int(bitrate),
                    "strict-cbr": 1,
                    "csv": csv_path,
                    "csv-log-level": 2,
                },
                "mode_params": target_mode_params,
                "tune_params": target_tune_params,
                # 根据 Profile 加载对应的超参数
                "hyperparams": HYPERPARAMS_MAP.get(profile_name, DEFAULT_HYPERPARAMS),
            }

            reader = YUVReader(
                seq_info["path"],
                seq_info["width"],
                seq_info["height"],
                8,
                seq_info["fps"],
            )
            controller = AdaptiveController(wrapper, reader, ctrl_config)
            controller.run()
            reader.close()

        # 2. Decoding
        success = decode_hevc(hevc_path, recon_path)

        # 3. VMAF Calculation
        if success:
            run_vmaf_calculation(
                ref_yuv=seq_info["path"],
                dist_yuv=recon_path,
                width=seq_info["width"],
                height=seq_info["height"],
                log_path=vmaf_log_path,
            )

    print(f"[{os.getpid()}] Finished {seq_key} [{profile_name}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of parallel workers"
    )
    args = parser.parse_args()

    # 加载配置
    try:
        seq_meta = load_json("config/test_sequences.json")
        init_config_full = load_json("config/initial_params.json")
    except FileNotFoundError as e:
        print(f"[Error] Config file missing: {e}")
        return

    # 创建输出根目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(OUTPUT_ROOT, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    print(f"=== Batch Encoding & Analysis Start ===")
    print(f"Output Dir: {base_dir}")
    print(f"Profiles: {TARGET_PROFILES}")

    # 构建任务列表：(Sequence, Profile) 的笛卡尔积
    tasks = []
    for seq_key in TARGET_SEQS:
        if seq_key not in seq_meta:
            print(f"[Warn] Sequence {seq_key} not in config, skipping.")
            continue

        for profile in TARGET_PROFILES:
            tasks.append((seq_key, profile, base_dir, seq_meta, init_config_full))

    print(f"Total Tasks: {len(tasks)}")

    # 并行执行
    start_time = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            executor.map(process_sequence_profile, tasks)
    else:
        for task in tasks:
            process_sequence_profile(task)

    print(f"\n=== All Done in {time.time() - start_time:.2f}s ===")


if __name__ == "__main__":
    main()
