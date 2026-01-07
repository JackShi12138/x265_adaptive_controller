import sys
import os
import ctypes
import time
import argparse

# 确保能导入 core 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from core.x265_wrapper import X265Wrapper
from utils.yuv_io import YUVReader  # [新增] 导入 YUV 读取工具类


def check_file_size(filepath, width, height, frames):
    """验证文件大小是否符合分辨率计算出的理论值"""
    if not os.path.exists(filepath):
        return False, "File not found"

    file_size = os.path.getsize(filepath)
    # YUV420P: 1 pixel = 1 byte (Y) + 0.25 byte (U) + 0.25 byte (V) = 1.5 bytes
    frame_size = width * height * 3 // 2
    expected_size = frame_size * frames

    print(f"[Check] 单帧大小: {frame_size} 字节")
    print(f"[Check] 理论总大小: {expected_size} 字节 ({frames} 帧)")
    print(f"[Check] 实际文件大小: {file_size} 字节")

    if file_size < expected_size:
        return (
            False,
            f"文件过小！实际只有理论值的 {file_size/expected_size*100:.1f}%。请检查分辨率(Width/Height)是否设置正确。",
        )

    if file_size > expected_size + frame_size:
        return (
            True,
            f"注意：文件比理论值大 {(file_size - expected_size)/1024/1024:.2f} MB，可能是帧数多于指定值或包含头部信息。",
        )

    return True, "文件大小校验通过。"


def run_test():
    parser = argparse.ArgumentParser(
        description="x265 Wrapper Basic Encode Test (With YUVReader)"
    )

    # 针对 RaceHorses_416x240_30.yuv 设置默认参数
    default_input = "/home/shiyushen/x265_sequence/ClassD/RaceHorses_416x240_30.yuv"
    parser.add_argument(
        "--input", type=str, default=default_input, help="Input YUV file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./tests/temp_file/test_output.hevc",
        help="Output HEVC file path",
    )

    parser.add_argument("--width", type=int, default=416, help="Video width")
    parser.add_argument("--height", type=int, default=240, help="Video height")
    parser.add_argument(
        "--frames", type=int, default=300, help="Number of frames to encode"
    )
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")

    parser.add_argument(
        "--lib",
        type=str,
        default="/home/shiyushen/program/x265_4.0/libx265.so",
        help="Path to libx265.so",
    )

    # 码率控制参数
    parser.add_argument(
        "--bitrate",
        type=int,
        default=300,
        help="Target bitrate in kbps (Default 300 to match CLI)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=0,
        help="Constant Rate Factor (0 to disable if bitrate is set)",
    )
    parser.add_argument(
        "--preset", type=str, default="slow", help="Preset (medium, slow, etc.)"
    )

    args = parser.parse_args()

    # 0. 预检查
    is_valid, msg = check_file_size(args.input, args.width, args.height, args.frames)
    print(f"[{'OK' if is_valid else 'WARNING'}] {msg}")

    print(f"=== 开始全流程编码测试 (YUVReader 集成版) ===")
    print(f"输入: {args.input} ({args.width}x{args.height} @ {args.fps}fps)")
    print(f"输出: {args.output}")

    # 1. 初始化
    wrapper = X265Wrapper(args.lib)
    param = wrapper.param_alloc()

    # 2. 参数配置
    print(f"配置模式: Preset={args.preset}")
    wrapper.param_default_preset(param, args.preset, "None")

    wrapper.param_parse(param, "input-res", f"{args.width}x{args.height}")
    wrapper.param_parse(param, "fps", str(args.fps))
    wrapper.param_parse(param, "input-csp", "i420")

    if args.bitrate > 0:
        print(f"配置模式: ABR (Bitrate: {args.bitrate} kbps)")
        wrapper.param_parse(param, "bitrate", str(args.bitrate))
        wrapper.param_parse(param, "vbv-maxrate", str(args.bitrate))
        wrapper.param_parse(param, "vbv-bufsize", str(args.bitrate * 2))
    elif args.crf > 0:
        print(f"配置模式: CRF (Quality: {args.crf})")
        wrapper.param_parse(param, "crf", str(args.crf))

    wrapper.param_parse(param, "annexb", "1")
    wrapper.param_parse(param, "repeat-headers", "1")

    encoder = wrapper.encoder_open(param)
    if not encoder:
        print("[FAIL] Encoder open failed")
        return

    # 3. 图像准备
    pic_in_ptr = wrapper.picture_alloc()
    wrapper.picture_init(param, pic_in_ptr)
    pic_out_ptr = wrapper.picture_alloc()

    # 4. YUV IO 与 编码循环
    # [修改] 使用 YUVReader 上下文管理器，自动处理文件打开、内存缓冲和指针计算
    f_out = open(args.output, "wb")

    try:
        with YUVReader(args.input, args.width, args.height, fps=args.fps) as yuv_reader:

            # 配置 Picture 结构体属性
            pic = pic_in_ptr.contents
            pic.bitDepth = 8
            pic.colorSpace = 1  # X265_CSP_I420

            # [关键] 从 YUVReader 获取安全的内存指针
            y_ptr, u_ptr, v_ptr = yuv_reader.get_pointers()
            pic.planes[0] = y_ptr
            pic.planes[1] = u_ptr
            pic.planes[2] = v_ptr

            # [关键] 从 YUVReader 获取正确的 Stride
            y_stride, u_stride, v_stride = yuv_reader.get_strides()
            pic.stride[0] = y_stride
            pic.stride[1] = u_stride
            pic.stride[2] = v_stride

            print(
                f"Pic Config -> BitDepth: {pic.bitDepth}, ColorSpace: {pic.colorSpace}"
            )
            print(f"Stride Y: {pic.stride[0]}, U: {pic.stride[1]}, V: {pic.stride[2]}")

            start_time = time.time()
            input_frames = 0
            encoded_frames = 0

            for i in range(args.frames):
                # [关键] 读取下一帧数据到内部 Buffer
                if not yuv_reader.read_frame():
                    print(f"[INFO] End of input file at frame {i}")
                    break

                pic.pts = i

                # 编码
                ret, nal_list = wrapper.encode(encoder, pic_in_ptr, pic_out_ptr)

                if ret < 0:
                    print(f"[FAIL] Encode error at frame {i}")
                    break

                for nal in nal_list:
                    f_out.write(nal)

                if ret > 0:
                    encoded_frames += 1

                input_frames += 1
                if i % 50 == 0:
                    print(f"Proc Frame {i}/{args.frames}...")

            # Flush 编码器
            print(f"--- 输入完成，Flush 编码器 ---")
            while True:
                ret, nal_list = wrapper.encode(encoder, None, pic_out_ptr)
                if ret <= 0:
                    break
                for nal in nal_list:
                    f_out.write(nal)
                encoded_frames += 1

            end_time = time.time()
            duration = end_time - start_time
            fps = encoded_frames / duration if duration > 0 else 0
            print(
                f"Performance: {encoded_frames} frames in {duration:.2f}s ({fps:.2f} fps)"
            )

    finally:
        f_out.close()
        if pic_in_ptr:
            wrapper.picture_free(pic_in_ptr)
        if pic_out_ptr:
            wrapper.picture_free(pic_out_ptr)
        if encoder:
            wrapper.encoder_close(encoder)
        if param:
            wrapper.param_free(param)

    print(f"=== 完成: 已写入 {args.output} ===")


if __name__ == "__main__":
    run_test()
