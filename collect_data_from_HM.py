import os
import shutil
import subprocess
import sys

# ==================== 1. 全局配置区域 ====================

# HM 解码器路径 (确保这是你修改过源码并重新编译后的版本)
DECODER_EXE = "/home/shiyushen/program/HM/TAppDecoderStatic"

# 实验配置：标签 -> (码流输入路径, Trace输出路径, Coeffs输出路径)
# 路径基于你昨天的 plot_cu_boxmap.py 进行了对齐
EXPERIMENTS = {
    "Baseline (Slow)": {
        "bitstream": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/slow/output.hevc",
        "trace_out": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/slow/trace_baseline.txt",
        "coeff_out": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/slow/coeffs_baseline.txt",
    },
    "Offline Opt.": {
        "bitstream": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/offline/output.hevc",
        "trace_out": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/offline/trace_offline.txt",
        "coeff_out": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/offline/coeffs_offline.txt",
    },
    "Online (Proposed)": {
        "bitstream": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/online/output.hevc",
        "trace_out": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/online/trace_online.txt",
        "coeff_out": "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/online/coeffs_online.txt",
    },
}

# ==================== 2. 核心逻辑 ====================


def run_extraction():
    print("=" * 60)
    print(f"🚀 开始全自动数据生成")
    print(f"🔧 解码器: {DECODER_EXE}")
    print("=" * 60)

    # 0. 检查解码器是否存在
    if not os.path.exists(DECODER_EXE):
        print(f"❌ 错误: 找不到解码器程序: {DECODER_EXE}")
        sys.exit(1)

    for label, config in EXPERIMENTS.items():
        bitstream = config["bitstream"]
        trace_target = config["trace_out"]
        coeff_target = config["coeff_out"]

        print(f"\n>>> 正在处理: {label}")

        # 1. 检查输入码流
        if not os.path.exists(bitstream):
            print(f"  ⚠️ 跳过: 找不到码流文件 {bitstream}")
            continue

        # 2. 清理环境 (删除当前目录下可能残留的 TraceDec.txt)
        if os.path.exists("TraceDec.txt"):
            try:
                os.remove("TraceDec.txt")
            except Exception as e:
                print(f"  ⚠️ 警告: 无法删除旧的 TraceDec.txt: {e}")

        # 3. 构造命令
        # 注意：系数是通过 stdout 打印的，所以我们需要在 Python 里捕获它
        cmd = [
            DECODER_EXE,
            "-b",
            bitstream,
            "-o",
            os.devnull,  # 不输出 YUV，节省空间
        ]

        print(f"  1. 运行解码器并捕获数据...")

        try:
            # 打开用于保存 Coefficients 的文件
            with open(coeff_target, "w") as f_coeff:
                # 执行命令，将 stdout (屏幕输出) 重定向到文件
                # stderr 还是打印到屏幕，方便看进度
                process = subprocess.run(
                    cmd, stdout=f_coeff, stderr=subprocess.PIPE, text=True
                )

            if process.returncode != 0:
                print(f"  ❌ 解码器返回错误码: {process.returncode}")
                print(f"  错误信息: {process.stderr}")
                continue

            # 4. 处理 Trace 文件 (TraceDec.txt -> trace_xxx.txt)
            if os.path.exists("TraceDec.txt"):
                if os.path.getsize("TraceDec.txt") > 0:
                    # 如果目标文件已存在，先删除
                    if os.path.exists(trace_target):
                        os.remove(trace_target)
                    shutil.move("TraceDec.txt", trace_target)
                    print(
                        f"  ✅ Trace 文件生成并重命名为: {os.path.basename(trace_target)}"
                    )
                else:
                    print(f"  ❌ 警告: 生成的 TraceDec.txt 为空")
            else:
                print(f"  ❌ 警告: 未找到 TraceDec.txt (请确认宏 ENC_DEC_TRACE 已开启)")

            # 5. 检查 Coeffs 文件
            if os.path.exists(coeff_target) and os.path.getsize(coeff_target) > 0:
                print(
                    f"  ✅ Coefficients 文件捕获成功: {os.path.basename(coeff_target)}"
                )

                # 简单验证一下是否包含 COEFF_DUMP
                with open(coeff_target, "r") as f:
                    head = f.read(1024)
                    if "COEFF_DUMP" in head:
                        print("     (验证: 文件头包含 COEFF_DUMP 标记)")
                    else:
                        print(
                            "     (⚠️ 警告: 文件头未发现 COEFF_DUMP，请检查 C++ 代码是否生效)"
                        )
            else:
                print(f"  ❌ 警告: Coefficients 文件为空")

        except Exception as e:
            print(f"  ❌ 执行异常: {e}")

    print("\n" + "=" * 60)
    print("🏁 所有任务完成！")
    print("现在可以使用绘图脚本 (plot_cu_boxmap.py 或 plot_coeffs.py) 进行分析了。")
    print("=" * 60)


if __name__ == "__main__":
    run_extraction()
