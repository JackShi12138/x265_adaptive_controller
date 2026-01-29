import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# 确保能导入核心模块
from core.feature_extractor import FeatureExtractor
from core.optimization_model import OptimizationModel
from utils.yuv_io import YUVReader

# === 配置区域 ===
# [Updated] 使用您刚才报错日志中的视频路径
VIDEO_PATH = "/home/shiyushen/x265_sequence/ClassB/BQTerrace_1920x1080_60.yuv"
WIDTH = 1920
HEIGHT = 1080
FPS = 60

# 配置文件路径
MODEL_CONFIG_PATH = "config/model_config.json"
INITIAL_PARAMS_PATH = "config/initial_params.json"
PROFILE = "High"  # 使用 Medium 档位作为基准

# [Updated] 使用您之前搜索出的最优超参数
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

OUTPUT_CSV = "frame_level_data.csv"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    print(f"=== Starting Frame-Level Data Collection ===")
    print(f"Target Video: {VIDEO_PATH}")

    if not os.path.exists(VIDEO_PATH):
        print(f"[Error] Video file not found: {VIDEO_PATH}")
        return

    # 1. 初始化组件
    init_config = load_json(INITIAL_PARAMS_PATH)
    if PROFILE not in init_config["profiles"]:
        raise ValueError(f"Profile {PROFILE} not found in initial_params.json")

    profile_data = init_config["profiles"][PROFILE]
    # 合并 mode 和 tune 参数作为基准
    base_params = {}
    base_params.update(profile_data["mode_params"])
    base_params.update(profile_data["tune_params"])

    # 初始化模型
    model = OptimizationModel(
        profile_data["tune_params"], BEST_HYPERPARAMS, config_path=MODEL_CONFIG_PATH
    )

    # 初始化 Reader 和 Extractor
    # [关键] gop_size=1 实现逐帧精细采集
    reader = YUVReader(VIDEO_PATH, WIDTH, HEIGHT, 8, FPS)
    extractor = FeatureExtractor(reader, gop_size=1, processing_width=256)

    data_list = []
    frame_idx = 0

    # 2. 逐帧扫描
    pbar = tqdm(desc="Scanning Frames", unit="frm")

    while True:
        # [关键修复] 记录当前位置
        current_pos = reader.frame_idx

        # 提取特征 (读取后会自动重置回 current_pos)
        is_valid, features = extractor.get_next_gop_features()
        if not is_valid:
            break

        # 模型推理 (计算该帧的理论参数)
        target_params = model.compute_adjustments(features)

        # 3. 记录数据
        record = {
            "frame_idx": frame_idx,
            # 原始特征 (Normalized by FeatureExtractor)
            "feat_var": features.get("w1_var", 0),
            "feat_sad": features.get("w2_sad", 0),
            "feat_grad": features.get("w3_grad", 0),
            "feat_tex": features.get("w4_tex", 0),
            "feat_cplx": features.get("w5_cplx", 0),
            # 模型输出参数
            "param_psy_rd": target_params.get("psy-rd", 0),
            "param_psy_rdoq": target_params.get("psy-rdoq", 0),
            "param_aq": target_params.get("aq-strength", 0),
            "param_cutree": target_params.get("cutree-strength", 0),
            "param_qcomp": target_params.get("qcomp", 0),
            # 基准值 (用于计算 Delta)
            "base_psy_rd": base_params.get("psy-rd", 0),
            "base_psy_rdoq": base_params.get("psy-rdoq", 0),
            "base_aq": base_params.get("aq-strength", 0),
            "base_cutree": base_params.get("cutree-strength", 0),
            "base_qcomp": base_params.get("qcomp", 0),
        }
        data_list.append(record)

        # [关键修复] 手动推进文件指针！
        # 因为我们设了 gop_size=1，所以每处理完一次就前进 1 帧
        reader.seek(current_pos + 1)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    reader.close()

    # 4. 保存
    df = pd.DataFrame(data_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Data saved to {OUTPUT_CSV}. Total frames: {frame_idx}")


if __name__ == "__main__":
    main()
