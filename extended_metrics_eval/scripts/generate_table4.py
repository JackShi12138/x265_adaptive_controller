import os
import json
import csv
import sys

# 将当前目录加入路径，以便导入同目录下的 compute_bd_stats
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from compute_bd_stats import calculate_bd_score

# --- 配置路径 ---
# 假设该脚本在 extended_metrics_eval/scripts/ 下运行
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(current_dir)
)  # 指向 x265_adaptive_controller 根目录
EVAL_ROOT = os.path.dirname(current_dir)  # 指向 extended_metrics_eval 目录

SEQ_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "test_sequences.json")
BASELINE_JSON = os.path.join(EVAL_ROOT, "results_json", "baseline_metrics.json")
OFFLINE_JSON = os.path.join(EVAL_ROOT, "results_json", "offline_metrics.json")
ONLINE_JSON = os.path.join(EVAL_ROOT, "results_json", "online_metrics.json")
OUTPUT_CSV = os.path.join(EVAL_ROOT, "Table_IV_extended.csv")


def load_json(path):
    if not os.path.exists(path):
        print(f"[Warn] JSON file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_rd_points(seq_data):
    """
    从单个序列的字典中提取 R-D 坐标点数组。
    输入示例: {"Very Low": {"bitrate": 1500, "psnr": 35.2, "ssim": 0.92}, ...}
    返回: psnr_pts [(rate, psnr), ...], ssim_pts [(rate, ssim), ...]
    """
    psnr_pts = []
    ssim_pts = []
    for profile, metrics in seq_data.items():
        if "bitrate" in metrics and "psnr" in metrics and "ssim" in metrics:
            psnr_pts.append((metrics["bitrate"], metrics["psnr"]))
            ssim_pts.append((metrics["bitrate"], metrics["ssim"]))
    return psnr_pts, ssim_pts


def main():
    print("=== Generating Extended Table IV (BD-PSNR & BD-SSIM) ===")

    # 1. 加载配置与数据
    seq_config = load_json(SEQ_CONFIG_PATH)
    baseline_data = load_json(BASELINE_JSON)
    offline_data = load_json(OFFLINE_JSON)
    online_data = load_json(ONLINE_JSON)

    if not seq_config or not baseline_data:
        print("[Error] Missing core JSON files. Please run data collection first.")
        return

    # 2. 准备 CSV 写入
    # 按照指标进行聚类，方便论文表格中纵向与横向的对比
    headers = [
        "Sequence",
        "Offline_BD-PSNR",
        "Online_BD-PSNR",
        "Offline_BD-SSIM",
        "Online_BD-SSIM",
    ]

    results = []

    # 3. 按配置文件中的物理顺序遍历序列 (保证输出顺序与论文一致)
    for seq_name in seq_config.keys():
        # 如果某个序列在 JSON 中缺失，填入 N/A 保留行位置
        if seq_name not in baseline_data:
            results.append([seq_name, "N/A", "N/A", "N/A", "N/A"])
            continue

        # 提取 Anchor (Baseline) 散点
        base_psnr, base_ssim = extract_rd_points(baseline_data[seq_name])

        # 提取 Offline 散点并计算
        off_psnr_pts, off_ssim_pts = extract_rd_points(offline_data.get(seq_name, {}))
        off_bd_psnr = calculate_bd_score(base_psnr, off_psnr_pts, drop_threshold=1.0)
        off_bd_ssim = calculate_bd_score(base_ssim, off_ssim_pts, drop_threshold=0.05)

        # 提取 Online 散点并计算
        on_psnr_pts, on_ssim_pts = extract_rd_points(online_data.get(seq_name, {}))
        on_bd_psnr = calculate_bd_score(base_psnr, on_psnr_pts, drop_threshold=1.0)
        on_bd_ssim = calculate_bd_score(base_ssim, on_ssim_pts, drop_threshold=0.05)

        # 4. 格式化输出，严格对齐 headers 设定的顺序
        row = [
            seq_name,
            f"{off_bd_psnr:.3f}" if off_bd_psnr != -9999.0 else "Error",
            f"{on_bd_psnr:.3f}" if on_bd_psnr != -9999.0 else "Error",
            f"{off_bd_ssim:.4f}" if off_bd_ssim != -9999.0 else "Error",
            f"{on_bd_ssim:.4f}" if on_bd_ssim != -9999.0 else "Error",
        ]
        results.append(row)
        print(
            f"Processed: {seq_name:<35} | PSNR(Off/On): {row[1]:>7}/{row[2]:>7} | SSIM(Off/On): {row[3]:>7}/{row[4]:>7}"
        )

    # 5. 落盘为 CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

    print(f"\n[Success] Table generated successfully at: {OUTPUT_CSV}")
    print(
        "Tip: You can now paste these columns directly into your LaTeX tabular environment."
    )


if __name__ == "__main__":
    main()
