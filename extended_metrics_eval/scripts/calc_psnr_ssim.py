import subprocess
import re
import os


def parse_ffmpeg_output(output_text):
    """
    使用正则表达式从 FFmpeg 的 stderr 输出中提取全局平均 PSNR(Y) 和 SSIM(All)。
    """
    result = {"psnr": -1.0, "ssim": -1.0}

    # 提取 PSNR Y 分量
    # 典型输出格式: [Parsed_psnr_0 @ 0x...] PSNR y:37.550123 u:40.12 v:41.02 average:38.22 min:36.12 max:40.11
    psnr_match = re.search(r"PSNR\s+y:([0-9\.]+)", output_text)
    if psnr_match:
        result["psnr"] = float(psnr_match.group(1))

    # 提取 SSIM All 分量
    # 典型输出格式: [Parsed_ssim_1 @ 0x...] SSIM Y:0.955000 (13.5) U:0.96... V:0.97... All:0.961000 (14.2)
    ssim_match = re.search(r"All:([0-9\.]+)", output_text)
    if ssim_match:
        result["ssim"] = float(ssim_match.group(1))

    return result


def calculate_metrics(dist_yuv, ref_yuv, width, height, pixel_format="yuv420p"):
    """
    调用 FFmpeg 计算重建 YUV 与原始 YUV 的 PSNR 和 SSIM。
    要求输入必须是已经解码好且格式对齐的 Raw YUV 文件。

    Args:
        dist_yuv (str): 解码后的重建 YUV 文件路径。
        ref_yuv (str): 原始参考 YUV 文件路径。
        width (int): 视频宽度。
        height (int): 视频高度。
        pixel_format (str): YUV 的像素格式，默认为 yuv420p。

    Returns:
        dict: 包含 {"psnr": float, "ssim": float} 的字典。
    """
    if not os.path.exists(dist_yuv) or not os.path.exists(ref_yuv):
        print(
            f"[Warn] YUV files missing. dist:{os.path.exists(dist_yuv)}, ref:{os.path.exists(ref_yuv)}"
        )
        return {"psnr": -1.0, "ssim": -1.0}

    # 构造 FFmpeg 命令
    # 严谨做法：对两个输入流都强制指定分辨率和像素格式，杜绝任何隐式的格式转换或帧率猜测
    cmd = [
        "ffmpeg",
        "-y",
        "-video_size",
        f"{width}x{height}",
        "-pixel_format",
        pixel_format,
        "-i",
        dist_yuv,  # 输入0: 重建 YUV
        "-video_size",
        f"{width}x{height}",
        "-pixel_format",
        pixel_format,
        "-i",
        ref_yuv,  # 输入1: 原始 YUV
        "-lavfi",
        "[0:v][1:v]psnr;[0:v][1:v]ssim",  # 挂载评价滤镜
        "-f",
        "null",
        "-",  # 仅计算指标，无实体输出
    ]

    try:
        # 运行子进程，捕获 stderr (FFmpeg 的滤镜日志默认输出到 stderr)
        process_result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        output = process_result.stderr
        return parse_ffmpeg_output(output)

    except subprocess.CalledProcessError as e:
        print(f"[Error] FFmpeg metrics calculation failed: {e.stderr}")
        return {"psnr": -1.0, "ssim": -1.0}
