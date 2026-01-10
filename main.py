import argparse
import os
import sys
import json
import time

# 导入核心模块
# 假设脚本在项目根目录运行，添加当前目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.x265_wrapper import X265Wrapper
from utils.yuv_io import YUVReader
from core.adaptive_controller import AdaptiveController

# 默认配置常量
DEFAULT_YUV_DIR = "/home/shiyushen/x265_sequence/"
DEFAULT_LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


def load_json(filename):
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def run_adaptive_coding(
    sequence_name: str,
    quality_level: str,
    output_hevc_path: str,
    output_csv_path: str,
    yuv_base_dir: str = DEFAULT_YUV_DIR,
    lib_path: str = DEFAULT_LIB_PATH,
    hyperparams_dict: dict = None,
) -> dict:
    """
    自适应编码执行函数 (API 接口)
    供 main 函数或外部超参数搜索脚本调用。
    """

    print(f"\n>>> Starting Job: {sequence_name} [{quality_level}]")

    # 1. 加载配置文件
    seq_config = load_json("test_sequences.json")
    init_params_json = load_json("initial_params.json")

    # [关键修复] 获取 profiles 字典
    if "profiles" not in init_params_json:
        raise ValueError("Invalid initial_params.json format: Missing 'profiles' key")
    init_params_profiles = init_params_json["profiles"]

    # 2. 获取序列元数据与自动推导路径
    if sequence_name not in seq_config:
        raise ValueError(f"Sequence '{sequence_name}' not found in test_sequences.json")

    seq_info = seq_config[sequence_name]
    video_class = seq_info["class"]
    width = seq_info["width"]
    height = seq_info["height"]
    fps = seq_info["fps"]

    # 自动拼接 YUV 路径: /base/ClassA/Name.yuv
    class_folder = f"Class{video_class}" if video_class else ""
    input_yuv_path = os.path.join(yuv_base_dir, class_folder, f"{sequence_name}.yuv")

    if not os.path.exists(input_yuv_path):
        # 尝试不带 Class 目录直接查找
        input_yuv_path_fallback = os.path.join(yuv_base_dir, f"{sequence_name}.yuv")
        if os.path.exists(input_yuv_path_fallback):
            input_yuv_path = input_yuv_path_fallback
        else:
            raise FileNotFoundError(f"YUV file not found at: {input_yuv_path}")

    print(f"    Input: {input_yuv_path}")
    print(f"    Res: {width}x{height} @ {fps}fps")

    # 3. 获取初始参数与码率
    # [关键修复] 从 profiles 中查找画质档位
    if quality_level not in init_params_profiles:
        valid_keys = list(init_params_profiles.keys())
        raise ValueError(
            f"Quality '{quality_level}' not defined in initial_params.json profiles. Available: {valid_keys}"
        )

    # 获取 bitrates 字典中对应档位的码率
    target_bitrate = seq_info["bitrates"].get(quality_level)
    if target_bitrate is None:
        raise ValueError(
            f"Bitrate for quality '{quality_level}' not defined for sequence '{sequence_name}'"
        )

    # 提取 mode 和 tune 参数
    preset_config = init_params_profiles[quality_level]
    mode_params = preset_config.get("mode_params", {}).copy()
    tune_params = preset_config.get("tune_params", {}).copy()

    # 4. 准备超参数 (默认值 vs 覆盖值)
    # 默认超参数
    final_hyperparams = {
        "a": 1.0,
        "b": 1.0,
        "beta": {"VAQ": 0.5, "CUTree": 0.0, "PsyRD": 1.0, "PsyRDOQ": 1.0, "QComp": 0.1},
    }
    # 如果传入了 override，则更新
    if hyperparams_dict:
        if "a" in hyperparams_dict:
            final_hyperparams["a"] = hyperparams_dict["a"]
        if "b" in hyperparams_dict:
            final_hyperparams["b"] = hyperparams_dict["b"]
        if "beta" in hyperparams_dict:
            # 深度更新 beta 字典，而不是直接替换
            if "beta" in final_hyperparams and isinstance(
                final_hyperparams["beta"], dict
            ):
                final_hyperparams["beta"].update(hyperparams_dict["beta"])
            else:
                final_hyperparams["beta"] = hyperparams_dict["beta"]

    # 5. 组装最终 Config
    controller_config = {
        "output_file": output_hevc_path,
        "preset": "slow",
        "base_params": {
            "bitrate": target_bitrate,
            "strict-cbr": 1,
            # VBV 由 Controller 自动推导
            "csv": output_csv_path,
            "csv-log-level": 2,
        },
        "mode_params": mode_params,
        "tune_params": tune_params,
        "hyperparams": final_hyperparams,
    }

    # 6. 执行编码流程
    start_time = time.time()

    wrapper = X265Wrapper(lib_path)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_hevc_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    with YUVReader(input_yuv_path, width, height, fps=fps) as reader:
        controller = AdaptiveController(wrapper, reader, controller_config)
        controller.run()

    total_time = time.time() - start_time

    # 7. 返回结果
    result = {
        "sequence": sequence_name,
        "quality": quality_level,
        "hevc_path": output_hevc_path,
        "csv_path": output_csv_path,
        "duration": total_time,
        "status": "Success",
    }
    print(f"<<< Job Finished in {total_time:.2f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="x265 Adaptive Controller CLI")

    # 必需参数
    parser.add_argument(
        "--sequence-name",
        required=True,
        help="Video sequence name (key in test_sequences.json)",
    )
    parser.add_argument(
        "--quality",
        required=True,
        choices=["Very Low", "Low", "Medium", "High"],
        help="Quality level",
    )
    parser.add_argument("--output", required=True, help="Output HEVC file path")
    parser.add_argument("--csv", required=True, help="Output CSV log file path")

    # 环境配置
    parser.add_argument(
        "--yuv-dir", default=DEFAULT_YUV_DIR, help="Base directory for YUV sequences"
    )
    parser.add_argument(
        "--lib-path", default=DEFAULT_LIB_PATH, help="Path to libx265.so"
    )

    # 超参数覆盖 (7个参数)
    parser.add_argument("--param-a", type=float, default=1.0, help="Model parameter a")
    parser.add_argument("--param-b", type=float, default=1.0, help="Model parameter b")

    parser.add_argument("--beta-vaq", type=float, help="Beta for VAQ")
    parser.add_argument("--beta-cutree", type=float, help="Beta for CUTree")
    parser.add_argument("--beta-psyrd", type=float, help="Beta for PsyRD")
    parser.add_argument("--beta-psyrdoq", type=float, help="Beta for PsyRDOQ")
    parser.add_argument("--beta-qcomp", type=float, help="Beta for QComp")

    args = parser.parse_args()

    # 构建超参数字典 (仅包含用户指定的值，以便与默认值合并)
    hyperparams_override = {"a": args.param_a, "b": args.param_b, "beta": {}}

    # 仅当用户在命令行显式传入时才覆盖
    if args.beta_vaq is not None:
        hyperparams_override["beta"]["VAQ"] = args.beta_vaq
    if args.beta_cutree is not None:
        hyperparams_override["beta"]["CUTree"] = args.beta_cutree
    if args.beta_psyrd is not None:
        hyperparams_override["beta"]["PsyRD"] = args.beta_psyrd
    if args.beta_psyrdoq is not None:
        hyperparams_override["beta"]["PsyRDOQ"] = args.beta_psyrdoq
    if args.beta_qcomp is not None:
        hyperparams_override["beta"]["QComp"] = args.beta_qcomp

    try:
        run_adaptive_coding(
            sequence_name=args.sequence_name,
            quality_level=args.quality,
            output_hevc_path=args.output,
            output_csv_path=args.csv,
            yuv_base_dir=args.yuv_dir,
            lib_path=args.lib_path,
            hyperparams_dict=hyperparams_override,
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# 执行示例
# python main.py   --sequence-name BasketballDrive_1920x1080_50   --quality Medium   --output ./results/basketball_out.hevc   --csv ./results/basketball_log.csv   --beta-vaq 0.6   --beta-psyrd 1.2
