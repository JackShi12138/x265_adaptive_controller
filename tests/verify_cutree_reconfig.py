import sys
import os
import time
import argparse
import subprocess

# 确保能导入 core 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from core.x265_wrapper import X265Wrapper
from utils.yuv_io import YUVReader


def run_encode_task(args, task_name, do_reconfig=False):
    """
    执行单次编码任务
    :param do_reconfig: 是否在第 100 帧触发参数变更
    :return: 输出文件路径
    """
    output_path = os.path.join(os.path.dirname(args.output), f"{task_name}.hevc")
    print(f"\n=== 任务: {task_name} (Reconfig={do_reconfig}) ===")

    # 1. 初始化
    wrapper = X265Wrapper(args.lib)
    param = wrapper.param_alloc()

    # 2. 参数配置
    wrapper.param_default_preset(param, args.preset, "None")
    wrapper.param_parse(param, "input-res", f"{args.width}x{args.height}")
    wrapper.param_parse(param, "fps", str(args.fps))
    wrapper.param_parse(param, "input-csp", "i420")

    # 固定码率控制模式 (CRF 方便观察画质变化，或者 Bitrate 固定观察 VMAF)
    # 这里用 bitrate 模式，qcomp 的变化会改变码率分配策略
    wrapper.param_parse(param, "strict-cbr", "1")
    wrapper.param_parse(param, "bitrate", "2000")
    wrapper.param_parse(param, "vbv-maxrate", "2000")
    wrapper.param_parse(param, "vbv-bufsize", "4000")

    # 初始 qcomp (默认是 0.6)
    wrapper.param_parse(param, "cutree-strength", "2.0")

    wrapper.param_parse(param, "annexb", "1")
    wrapper.param_parse(param, "repeat-headers", "1")

    encoder = wrapper.encoder_open(param)
    if not encoder:
        print("[FAIL] Encoder open failed")
        return None

    # 3. 图像准备
    pic_in_ptr = wrapper.picture_alloc()
    wrapper.picture_init(param, pic_in_ptr)
    pic_out_ptr = wrapper.picture_alloc()

    # 4. 打开输出文件
    f_out = open(output_path, "wb")

    try:
        with YUVReader(args.input, args.width, args.height, fps=args.fps) as yuv_reader:
            # 配置 Picture 指针
            pic = pic_in_ptr.contents
            pic.bitDepth = 8
            pic.colorSpace = 1

            y_ptr, u_ptr, v_ptr = yuv_reader.get_pointers()
            pic.planes[0], pic.planes[1], pic.planes[2] = y_ptr, u_ptr, v_ptr

            y_stride, u_stride, v_stride = yuv_reader.get_strides()
            pic.stride[0], pic.stride[1], pic.stride[2] = y_stride, u_stride, v_stride

            for i in range(args.frames):
                if not yuv_reader.read_frame():
                    break

                pic.pts = i

                # ==============================================================
                # [关键点] 在第 100 帧触发 Reconfig
                # ==============================================================
                if do_reconfig and i == 100:
                    print(
                        f"\n[INFO] Frame {i}: Triggering Reconfig (cutree-strength 2.0 -> 1.5)..."
                    )

                    # 1. 修改参数结构体
                    # 注意：x265_param_parse 会修改 param 指向的内存
                    wrapper.param_parse(param, "cutree-strength", "1.5")

                    # 2. 调用 reconfig 应用更改
                    # encoder_reconfig(encoder, param_ptr)
                    ret_reconfig = wrapper.encoder_reconfig(encoder, param)

                    if (
                        ret_reconfig == 1
                    ):  # 通常 1 表示有变动且成功 (视版本而定，有时是0)
                        print("[SUCCESS] x265_encoder_reconfig returned 1 (Success).")
                    else:
                        print(
                            f"[WARNING] x265_encoder_reconfig returned {ret_reconfig}."
                        )
                # ==============================================================

                ret, nal_list = wrapper.encode(encoder, pic_in_ptr, pic_out_ptr)
                if ret < 0:
                    break
                for nal in nal_list:
                    f_out.write(nal)

                if i % 50 == 0:
                    print(f"Proc Frame {i}/{args.frames}...", end="\r")

            # Flush
            while True:
                ret, nal_list = wrapper.encode(encoder, None, pic_out_ptr)
                if ret <= 0:
                    break
                for nal in nal_list:
                    f_out.write(nal)

    finally:
        f_out.close()
        wrapper.picture_free(pic_in_ptr)
        wrapper.picture_free(pic_out_ptr)
        wrapper.encoder_close(encoder)
        wrapper.param_free(param)
        print(f"\nTask {task_name} finished.")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Verify x265 Reconfig Capability")
    # 默认使用 ClassD 小视频快速测试
    default_input = "/home/shiyushen/x265_sequence/ClassD/RaceHorses_416x240_30.yuv"
    parser.add_argument("--input", default=default_input)
    parser.add_argument("--output", default="./tests/temp_file/verify.hevc")
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--frames", type=int, default=200)  # 跑 200 帧足够覆盖 100 帧
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", default="medium")  # 用 medium 稍微快点
    parser.add_argument("--lib", default="/home/shiyushen/program/x265_4.0/libx265.so")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. 运行对照组 (Baseline)
    base_file = run_encode_task(args, "baseline", do_reconfig=False)

    # 2. 运行实验组 (Experiment)
    exp_file = run_encode_task(args, "experiment", do_reconfig=True)

    # 3. 结果验证
    print("\n" + "=" * 40)
    print("       VERIFICATION RESULTS       ")
    print("=" * 40)

    if not base_file or not exp_file:
        print("[ERROR] Encoding failed.")
        return

    size_base = os.path.getsize(base_file)
    size_exp = os.path.getsize(exp_file)

    print(f"Baseline File Size   : {size_base} bytes")
    print(f"Experiment File Size : {size_exp} bytes")
    print(f"Size Diff            : {size_exp - size_base} bytes")

    if size_base == size_exp:
        print("\n[CONCLUSION] FAILED. Files are identical.")
        print("Reason: x265 ignored the reconfig command for 'cutree-strength'.")
        print("Action: Lock 'beta_CUTree' to 0 in your optimization model.")
    else:
        print("\n[CONCLUSION] SUCCESS! Files are different.")
        print("Reason: Reconfig successfully altered encoding behavior.")
        print(
            "Action: You can safely include 'beta_CUTree' (via cutree-strength) in search."
        )


if __name__ == "__main__":
    main()
