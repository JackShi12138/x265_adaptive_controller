import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 1. 设置视频路径 (建议使用解码后的 YUV 或高质量 MP4)
# 如果是 .hevc 文件，建议先用 ffmpeg 转成 .mp4 或者 .yuv，OpenCV 对 hevc 支持可能依赖系统编码器
BASELINE_VIDEO = "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/slow/output.hevc"  # Baseline 视频
ONLINE_VIDEO = "/home/shiyushen/x265_adaptive_controller/analysis_data/20260125_154912/RaceHorses_832x480_30/online/output.hevc"  # Online 算法视频

# 2. 选择要可视化的帧 (比如第 100 帧，找一个纹理复杂的地方)
FRAME_INDEX = 40

# 3. 增益系数 (关键！因为残差通常很小，需要放大才能看清热力分布)
# 如果生成的图全黑或全蓝，就把这个数改大；如果全红，就改小。
GAIN = 5.0
# ===========================================


def generate_heatmap():
    # 打开视频
    cap_base = cv2.VideoCapture(BASELINE_VIDEO)
    cap_online = cv2.VideoCapture(ONLINE_VIDEO)

    if not cap_base.isOpened() or not cap_online.isOpened():
        print("错误：无法打开视频文件，请检查路径。")
        return

    # 跳转到指定帧
    cap_base.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    cap_online.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)

    # 读取帧
    ret1, frame_base = cap_base.read()
    ret2, frame_online = cap_online.read()

    if not ret1 or not ret2:
        print("错误：无法读取指定帧。")
        return

    # 转为灰度图 (Luma channel)，因为人眼对亮度误差最敏感
    gray_base = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)
    gray_online = cv2.cvtColor(frame_online, cv2.COLOR_BGR2GRAY)

    # 计算绝对差值 (Residual / Error)
    # convert to int16 to avoid overflow/underflow during subtraction
    diff = cv2.absdiff(gray_base, gray_online)

    # === 可视化部分 ===
    plt.figure(figsize=(10, 6))

    # 绘制热力图
    # cmap='jet' 是经典的蓝-红热力图 (蓝=低误差, 红=高误差)
    # vmin=0, vmax=20 这里的 vmax 决定了红色的阈值。
    # 一般残差在 0-10 之间，偶尔有 20+。根据你的视频内容调整 vmax 效果最好。
    plt.imshow(diff, cmap="jet", vmin=0, vmax=20)

    plt.colorbar(label="Absolute Reconstruction Error")
    plt.title(f"Spatial Distribution of Distortion (Frame {FRAME_INDEX})")
    plt.axis("off")  # 论文插图通常不需要像素坐标轴

    # 保存图片
    output_filename = "Fig_A_Residual_Heatmap.png"
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"成功！热力图已保存为: {output_filename}")
    print("请查看图片，如果颜色太淡，请减小代码中的 vmax 值；如果太深，请增大 vmax。")

    # 释放资源
    cap_base.release()
    cap_online.release()


if __name__ == "__main__":
    generate_heatmap()
