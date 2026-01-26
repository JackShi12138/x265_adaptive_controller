import os
import json
import time
import argparse
import subprocess
from datetime import datetime
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from core.x265_wrapper import X265Wrapper
from core.adaptive_controller import AdaptiveController
from utils.yuv_io import YUVReader

# === 配置区域 ===
# 请替换为您的实际路径
DATASET_ROOT = "/home/shiyushen/x265_sequence/"
LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
OUTPUT_ROOT = "analysis_data"

# 待分析的视频列表 (请根据需要修改)
TARGET_SEQS = [
    "SlideEditing_1280x720_30",  # SCV
    "SlideShow_1280x720_20",  # SCV
    "ChinaSpeed_1024x768_30",  # Fast Motion
    "RaceHorses_832x480_30",  # High Texture
    "PeopleOnStreet_2560x1600_30_crop",  # 4K
    "BasketballDrive_1920x1080_50",  # Mixed Motion
    "ParkScene_1920x1080_24",
    "BlowingBubbles_416x240_50",
]

# Online 模式使用的超参数 (请替换为您搜索出的最优参数)
BEST_HYPERPARAMS = {
    "a": 3.044929438592169,
    "b": 2.36521356355403,
    "beta": {
        "VAQ": 6.317753857221986,
        "CUTree": 3.0987080653981685,
        "PsyRD": 3.1385016893339084,
        "PsyRDOQ": 6.070245329186924,
        "QComp": 7.167432624475883,
    },
}

TARGET_PROFILE = "Medium"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def decode_hevc(hevc_path, yuv_path):
    """
    [Updated] 调用 ffmpeg 将 HEVC 解码为 YUV420P
    使用 -vsync 0 确保帧数绝对对齐
    """
    if not os.path.exists(hevc_path):
        print(f"[Warn] HEVC file missing: {hevc_path}")
        return

    # 构造命令
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出
        "-v",
        "error",  # 只打印错误日志
        "-i",
        hevc_path,  # 输入
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "0",  # [关键] 严禁丢帧/补帧，保持原始帧数
        "-f",
        "rawvideo",  # [建议] 显式指定格式，防止后缀名识别错误
        yuv_path,  # 输出
    ]

    try:
        # 移除 stdout/stderr=DEVNULL，让 -v error 生效
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[Error] FFmpeg decoding failed for {hevc_path}")
    except FileNotFoundError:
        print(f"[Error] FFmpeg not found. Please install ffmpeg.")


def run_native_encode(wrapper, seq_info, output_hevc, csv_path, extra_params=None):
    """
    原生 x265 编码循环 (用于 Slow 和 Offline 模式)
    """
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
            if wrapper.param_parse(param, k, str(v)) < 0:
                print(f"[Warn] Failed to set {k}={v}")

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
            pic.planes[0] = y
            pic.planes[1] = u
            pic.planes[2] = v
            pic.stride[0] = s_y
            pic.stride[1] = s_u
            pic.stride[2] = s_v
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


def process_single_sequence(task_args):
    """
    单个序列的处理任务 (封装以便并行调用)
    """
    seq_key, base_dir, seq_meta, init_config_full = task_args

    # [关键] 在进程内部初始化 Wrapper，避免跨进程共享 ctypes 对象
    wrapper = X265Wrapper(LIB_PATH)

    # 提取配置
    profile_data = init_config_full["profiles"][TARGET_PROFILE]
    target_mode_params = profile_data["mode_params"]
    target_tune_params = profile_data["tune_params"]

    meta = seq_meta[seq_key]
    bitrate = meta["bitrates"][TARGET_PROFILE]

    # 路径处理
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

    seq_out_dir = os.path.join(base_dir, seq_key)
    print(f"[{os.getpid()}] Processing {seq_key} @ {bitrate} kbps...")

    # === Mode 1: Slow ===
    mode_dir = os.path.join(seq_out_dir, "slow")
    os.makedirs(mode_dir, exist_ok=True)
    hevc_path = os.path.join(mode_dir, "output.hevc")

    run_native_encode(
        wrapper,
        seq_info,
        output_hevc=hevc_path,
        csv_path=os.path.join(mode_dir, "x265_log.csv"),
        extra_params={},
    )
    decode_hevc(hevc_path, os.path.join(mode_dir, "recon.yuv"))

    # === Mode 2: Offline ===
    mode_dir = os.path.join(seq_out_dir, "offline")
    os.makedirs(mode_dir, exist_ok=True)
    hevc_path = os.path.join(mode_dir, "output.hevc")

    offline_params = {}
    offline_params.update(target_mode_params)
    offline_params.update(target_tune_params)

    run_native_encode(
        wrapper,
        seq_info,
        output_hevc=hevc_path,
        csv_path=os.path.join(mode_dir, "x265_log.csv"),
        extra_params=offline_params,
    )
    decode_hevc(hevc_path, os.path.join(mode_dir, "recon.yuv"))

    # === Mode 3: Online ===
    mode_dir = os.path.join(seq_out_dir, "online")
    os.makedirs(mode_dir, exist_ok=True)
    hevc_path = os.path.join(mode_dir, "output.hevc")

    reader = YUVReader(
        seq_info["path"], seq_info["width"], seq_info["height"], 8, seq_info["fps"]
    )
    ctrl_config = {
        "output_file": hevc_path,
        "controller_log": os.path.join(mode_dir, "controller_log.json"),
        "preset": "slow",
        "base_params": {
            "bitrate": int(bitrate),
            "strict-cbr": 1,
            "csv": os.path.join(mode_dir, "x265_log.csv"),
            "csv-log-level": 2,
        },
        "mode_params": target_mode_params,
        "tune_params": target_tune_params,
        "hyperparams": BEST_HYPERPARAMS,
    }

    controller = AdaptiveController(wrapper, reader, ctrl_config)
    controller.run()
    reader.close()

    decode_hevc(hevc_path, os.path.join(mode_dir, "recon.yuv"))

    print(f"[{os.getpid()}] Finished {seq_key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers"
    )
    args = parser.parse_args()

    seq_meta = load_json("config/test_sequences.json")
    init_config_full = load_json("config/initial_params.json")

    # 验证 Profile
    if TARGET_PROFILE not in init_config_full["profiles"]:
        raise ValueError(f"Profile {TARGET_PROFILE} not found")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(OUTPUT_ROOT, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    print(f"=== Data Collection Start ===")
    print(f"Output Dir: {base_dir}")
    print(f"Workers: {args.workers}")
    print(f"Profile: {TARGET_PROFILE}")

    # 准备任务列表
    tasks = []
    for seq_key in TARGET_SEQS:
        if seq_key not in seq_meta:
            print(f"[Skip] {seq_key} config not found")
            continue
        tasks.append((seq_key, base_dir, seq_meta, init_config_full))

    # 并行或串行执行
    start_time = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            executor.map(process_single_sequence, tasks)
    else:
        for task in tasks:
            process_single_sequence(task)

    print(f"\n=== All Done in {time.time() - start_time:.2f}s ===")


if __name__ == "__main__":
    main()

# nohup python3 collect_analysis_data.py --workers 8 > run.log 2>&1 &
