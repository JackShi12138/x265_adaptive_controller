import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ================= 配置区域 =================
# 1. 视频路径
BASELINE_VIDEO = "/home/shiyushen/x265_adaptive_controller/analysis_data/20260213_230133_208/RaceHorses_832x480_30/slow/output.hevc"
ONLINE_VIDEO = "/home/shiyushen/x265_adaptive_controller/analysis_data/20260213_230133_208/RaceHorses_832x480_30/online/output.hevc"

# 2. 帧索引
FRAME_INDEX = 40

# 3. 热力图显示范围 (根据您的残差大小调整)
VMAX_VAL = 20

# 4. 感兴趣区域 (ROI) 定义 [x, y, width, height]
# *** 请根据 RaceHorses 实际画面调整以下坐标 ***
# 提示：用播放器打开视频，截图并在画图软件里看像素坐标
ROI_TEXTURE = [360, 100, 100, 100]  # 示例：马身或骑手 (红色框) - 期望看到纹理更清晰
ROI_FLAT = [30, 30, 100, 100]  # 示例：背景草地或天空 (绿色框) - 期望看到无明显伪影

# ===========================================


def generate_composite_figure():
    # 1. 读取视频帧
    cap_base = cv2.VideoCapture(BASELINE_VIDEO)
    cap_online = cv2.VideoCapture(ONLINE_VIDEO)

    if not cap_base.isOpened() or not cap_online.isOpened():
        print("错误：无法打开视频文件。")
        return

    cap_base.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    cap_online.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)

    ret1, frame_base_bgr = cap_base.read()
    ret2, frame_online_bgr = cap_online.read()

    cap_base.release()
    cap_online.release()

    if not ret1 or not ret2:
        print("错误：无法读取指定帧。")
        return

    # 2. 预处理
    # 转换为灰度图计算残差
    gray_base = cv2.cvtColor(frame_base_bgr, cv2.COLOR_BGR2GRAY)
    gray_online = cv2.cvtColor(frame_online_bgr, cv2.COLOR_BGR2GRAY)

    # 计算差异图 (Diff Map)
    # 注意：这里计算的是 |Baseline - Online|，展示的是两者不一样的地方
    diff_map = cv2.absdiff(gray_base, gray_online)

    # 转换为 RGB 用于显示细节图 (Matplotlib 默认 RGB，OpenCV 默认 BGR)
    frame_base_rgb = cv2.cvtColor(frame_base_bgr, cv2.COLOR_BGR2RGB)
    frame_online_rgb = cv2.cvtColor(frame_online_bgr, cv2.COLOR_BGR2RGB)

    # 3. 创建画布布局
    fig = plt.figure(figsize=(14, 6))  # 宽长一点以容纳右侧子图
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 0.8, 0.8])
    # 左边两列合并给大热力图，右边两列给细节图

    # === 左侧：大热力图 ===
    ax_main = fig.add_subplot(gs[:, :2])
    im = ax_main.imshow(diff_map, cmap="jet", vmin=0, vmax=VMAX_VAL)
    ax_main.axis("off")
    ax_main.set_title(f"Difference Heatmap (Frame {FRAME_INDEX})", y=-0.1, fontsize=12)

    # 添加 Colorbar
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Difference Level")

    # 在热力图上画框
    def add_box_to_map(ax, roi, color, label):
        rect = patches.Rectangle(
            (roi[0], roi[1]),
            roi[2],
            roi[3],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        # 可选：在框旁边加标签
        ax.text(roi[0], roi[1] - 10, label, color=color, fontweight="bold", fontsize=10)

    add_box_to_map(ax_main, ROI_TEXTURE, "red", "Texture")
    add_box_to_map(ax_main, ROI_FLAT, "lime", "Flat")

    # === 右侧：局部放大对比 ===

    # 辅助函数：绘制局部 Crop
    def plot_patch(ax_idx, img, roi, border_color, title):
        ax = fig.add_subplot(ax_idx)
        # 裁剪图像: img[y:y+h, x:x+w]
        patch = img[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        ax.imshow(patch)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        # 给子图加边框，颜色与热力图上的框对应
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    # --- 第一行：纹理区域 (红色框) ---
    # 显示 Baseline (Slow)
    plot_patch(
        gs[0, 2], frame_base_rgb, ROI_TEXTURE, "red", "Baseline (Slow)\nBlurry Texture"
    )
    # 显示 Online (Proposed)
    plot_patch(
        gs[0, 3],
        frame_online_rgb,
        ROI_TEXTURE,
        "red",
        "Proposed (Online)\nSharper Detail",
    )

    # --- 第二行：平坦区域 (绿色框) ---
    # 显示 Baseline (Slow)
    plot_patch(
        gs[1, 2], frame_base_rgb, ROI_FLAT, "lime", "Baseline (Slow)\nFlat Region"
    )
    # 显示 Online (Proposed)
    plot_patch(
        gs[1, 3], frame_online_rgb, ROI_FLAT, "lime", "Proposed (Online)\nNo Artifacts"
    )

    # 4. 保存
    plt.tight_layout()
    output_filename = "Fig_Modified_Comparison.pdf"
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"成功生成组合对比图: {output_filename}")


if __name__ == "__main__":
    generate_composite_figure()
