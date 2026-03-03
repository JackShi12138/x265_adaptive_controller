import os
import json
import time
import argparse
import sys
import csv
from datetime import datetime

# 确保能导入核心模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.x265_wrapper import X265Wrapper
from core.adaptive_controller import AdaptiveController
from utils.yuv_io import YUVReader

# === 配置区域 ===
DATASET_ROOT = "/home/shiyushen/x265_sequence/"
LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
OUTPUT_ROOT = "benchmark_results_timing_only"  # 结果保存目录

# 最优超参数
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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def run_native_benchmark(wrapper, seq_info, profile_name, output_path, extra_params=None):
    """
    运行原生 x265 (Slow) 并测量耗时 (输出指向空设备)
    """
    param = wrapper.param_alloc()
    wrapper.param_default_preset(param, "slow", "None")
    
    wrapper.param_parse(param, "input-res", f"{seq_info['width']}x{seq_info['height']}")
    wrapper.param_parse(param, "fps", str(seq_info['fps']))
    wrapper.param_parse(param, "input-csp", "i420")
    wrapper.param_parse(param, "bitrate", str(int(seq_info['bitrate'])))
    wrapper.param_parse(param, "strict-cbr", "1")
    wrapper.param_parse(param, "vbv-maxrate", str(int(seq_info['bitrate'])))
    wrapper.param_parse(param, "vbv-bufsize", str(int(seq_info['bitrate'] * 2)))
    wrapper.param_parse(param, "annexb", "1")
    wrapper.param_parse(param, "csv-log-level", "0") # 禁用 CSV
    
    if extra_params:
        for k, v in extra_params.items():
            if v is not None:
                wrapper.param_parse(param, k, str(v))

    encoder = wrapper.encoder_open(param)
    if not encoder:
        print("    [Native] Error: Failed to open encoder")
        return 0

    pic_in = wrapper.picture_alloc()
    wrapper.picture_init(param, pic_in)
    pic_out = wrapper.picture_alloc()
    
    reader = YUVReader(seq_info["path"], seq_info["width"], seq_info["height"], 8, seq_info["fps"])
    
    start_time = time.time()
    
    try:
        # [修改] 这里 output_path 将被传入 os.devnull
        with open(output_path, "wb") as f_out:
            while True:
                if not reader.read_frame():
                    break
                
                y, u, v = reader.get_pointers()
                ys, us, vs = reader.get_strides()
                pic = pic_in.contents
                pic.planes[0] = y; pic.planes[1] = u; pic.planes[2] = v
                pic.stride[0] = ys; pic.stride[1] = us; pic.stride[2] = vs
                
                ret, nal_list = wrapper.encode(encoder, pic_in, pic_out)
                if ret < 0: break
                for nal in nal_list: f_out.write(nal) # 写入空设备

            while True:
                ret, nal_list = wrapper.encode(encoder, None, pic_out)
                if ret <= 0: break
                for nal in nal_list: f_out.write(nal)
            
    finally:
        reader.close()
        wrapper.picture_free(pic_in)
        wrapper.picture_free(pic_out)
        wrapper.encoder_close(encoder)
        wrapper.param_free(param)
        
    return time.time() - start_time


def run_adaptive_benchmark(wrapper, seq_info, profile_name, output_path, config_profiles):
    """
    运行 Adaptive Controller 并测量耗时 (输出指向空设备)
    """
    profile_data = config_profiles[profile_name]
    
    ctrl_config = {
        "output_file": output_path, # [修改] 传入 os.devnull
        "preset": "slow",
        "base_params": {
            "bitrate": int(seq_info['bitrate']),
            "strict-cbr": 1,
            "csv-log-level": 0,
        },
        "mode_params": profile_data["mode_params"],
        "tune_params": profile_data["tune_params"],
        "hyperparams": BEST_HYPERPARAMS,
    }

    reader = YUVReader(seq_info["path"], seq_info["width"], seq_info["height"], 8, seq_info["fps"])
    
    start_time = time.time()
    
    # AdaptiveController 内部会打开 output_file，这里是空设备，因此不会占用磁盘
    controller = AdaptiveController(wrapper, reader, ctrl_config)
    controller.run()
    
    duration = time.time() - start_time
    reader.close()
    
    return duration


def main():
    # 1. 加载配置
    if not os.path.exists("config/test_sequences.json"):
        print("[Error] config/test_sequences.json not found.")
        return
    
    seq_meta = load_json("config/test_sequences.json")
    init_config_full = load_json("config/initial_params.json")
    
    all_seq_keys = list(seq_meta.keys())
    all_profiles = list(init_config_full["profiles"].keys())

    # 创建保存结果报告的目录，而不是视频文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(OUTPUT_ROOT, timestamp)
    os.makedirs(report_dir, exist_ok=True)
    
    results = []
    
    print(f"=== Benchmark (Timing Only) Start ===")
    print(f"Output Report Dir: {report_dir}")
    print(f"Video Output: {os.devnull} (Discarded)")
    print("-" * 60)

    total_tasks = len(all_seq_keys) * len(all_profiles)
    current_task = 0

    for seq_key in all_seq_keys:
        meta = seq_meta[seq_key]
        
        # 路径处理
        video_class = meta.get("class", "")
        class_folder = f"Class{video_class}" if video_class else ""
        yuv_path = os.path.join(DATASET_ROOT, class_folder, f"{seq_key}.yuv")
        if not os.path.exists(yuv_path):
            yuv_path = os.path.join(DATASET_ROOT, f"{seq_key}.yuv")
            
        if not os.path.exists(yuv_path):
            print(f"[Skip] YUV missing: {seq_key}")
            current_task += len(all_profiles)
            continue

        print(f">>> Processing Sequence: {seq_key}")
        wrapper = X265Wrapper(LIB_PATH)

        for profile in all_profiles:
            current_task += 1
            bitrate = meta["bitrates"].get(profile)
            if bitrate is None:
                continue

            seq_info = {
                "name": seq_key,
                "path": yuv_path,
                "width": meta["width"],
                "height": meta["height"],
                "fps": meta["fps"],
                "bitrate": bitrate
            }

            print(f"  [{current_task}/{total_tasks}] Profile: {profile:<10} Bitrate: {bitrate} kbps")

            # [关键修改] 将输出路径设为系统空设备
            null_output = os.devnull

            # Run Native
            t_native = run_native_benchmark(wrapper, seq_info, profile, null_output)

            # Run Adaptive
            t_adapt = run_adaptive_benchmark(wrapper, seq_info, profile, null_output, init_config_full["profiles"])

            # 统计
            overhead = (t_adapt - t_native) / t_native * 100 if t_native > 0 else 0
            
            results.append({
                "Seq": seq_key,
                "Profile": profile,
                "Native_Time": round(t_native, 4),
                "Adapt_Time": round(t_adapt, 4),
                "Overhead_Pct": round(overhead, 2)
            })
            
            print(f"    Result: Native={t_native:.2f}s, Adapt={t_adapt:.2f}s, Overhead=+{overhead:.2f}%")

    # === 保存和显示结果 ===
    
    # 1. 打印表格
    print("\n" + "="*85)
    print(f"{'Sequence':<25} | {'Profile':<10} | {'Native(s)':<10} | {'Adapt(s)':<10} | {'Overhead':<10}")
    print("-" * 85)
    
    total_overhead = 0
    for res in results:
        print(f"{res['Seq']:<25} | {res['Profile']:<10} | {res['Native_Time']:<10.2f} | {res['Adapt_Time']:<10.2f} | +{res['Overhead_Pct']:.2f}%")
        total_overhead += res['Overhead_Pct']
        
    print("-" * 85)
    if results:
        print(f"Average Overhead: +{total_overhead/len(results):.2f}%")
    print("="*85)

    # 2. 保存为 JSON
    json_path = os.path.join(report_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"JSON report saved: {json_path}")

    # 3. 保存为 CSV
    csv_path = os.path.join(report_dir, "benchmark_results.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"CSV report saved:  {csv_path}")

if __name__ == "__main__":
    main()

# nohup python3 time_test.py > benchmark_run.log 2>&1 &