import time
import os
import ctypes
import json
from .feature_extractor import FeatureExtractor
from .optimization_model import OptimizationModel


class AdaptiveController:
    """
    x265 自适应编码控制器 (Two-Pass 模式)
    [Enhanced] 支持保存决策日志
    """

    def __init__(self, wrapper, yuv_reader, config):
        self.wrapper = wrapper
        self.reader = yuv_reader
        self.config = config
        self.output_file = config["output_file"]

        # 提取微调参数作为模型的初始基准
        self.initial_tune_params = config.get("tune_params", {}).copy()

        # 初始化优化模型
        self.model = OptimizationModel(self.initial_tune_params, config["hyperparams"])

        # 初始化特征提取器 (GOP=40)
        self.gop_size = 40
        self.feature_extractor = FeatureExtractor(
            self.reader, gop_size=self.gop_size, processing_width=256
        )

        # 内部状态
        self.encoder = None
        self.pic_in = None
        self.pic_out = None
        self.param = None
        self.encoded_frames = 0

        # 存储分析结果
        self.gop_plan_list = []

    def _apply_params_from_dict(self, param_dict, stage_name):
        """辅助函数：批量应用参数字典"""
        if not param_dict:
            return

        # [增强] 自动推导 VBV 参数 (仅针对 base_params 阶段)
        if stage_name == "Stage 3" and "bitrate" in param_dict:
            bitrate = int(param_dict["bitrate"])
            if "vbv-maxrate" not in param_dict:
                if (
                    self.wrapper.param_parse(self.param, "vbv-maxrate", str(bitrate))
                    < 0
                ):
                    pass
            if "vbv-bufsize" not in param_dict:
                bufsize = bitrate * 2
                if (
                    self.wrapper.param_parse(self.param, "vbv-bufsize", str(bufsize))
                    < 0
                ):
                    pass

        for k, v in param_dict.items():
            val_str = str(v)
            if self.wrapper.param_parse(self.param, k, val_str) < 0:
                print(f"[Warning] [{stage_name}] Failed to set {k}={v}")

    def _setup_encoder(self):
        """初始化 x265 编码器 (仅在 Pass 2 使用)"""
        self.param = self.wrapper.param_alloc()

        # === Stage 1: Preset ===
        preset = self.config.get("preset", "slow")
        self.wrapper.param_default_preset(self.param, preset, "None")

        # === Stage 2: Basic Input ===
        self.wrapper.param_parse(
            self.param, "input-res", f"{self.reader.width}x{self.reader.height}"
        )
        self.wrapper.param_parse(self.param, "fps", str(self.reader.fps))
        self.wrapper.param_parse(self.param, "input-csp", "i420")
        self.wrapper.param_parse(self.param, "annexb", "1")
        self.wrapper.param_parse(self.param, "repeat-headers", "1")

        # === Stage 3: Base Params (Bitrate, VBV, Logging) ===
        self._apply_params_from_dict(self.config.get("base_params"), "Stage 3")

        # === Stage 4: Mode Params (Switchs, Modes) ===
        self._apply_params_from_dict(self.config.get("mode_params"), "Stage 4")

        # === Stage 5: Tune Params (Initial Strengths) ===
        self._apply_params_from_dict(self.config.get("tune_params"), "Stage 5")

        # 打开编码器
        self.encoder = self.wrapper.encoder_open(self.param)
        if not self.encoder:
            raise RuntimeError("Failed to open x265 encoder")

        # 分配 Picture
        self.pic_in = self.wrapper.picture_alloc()
        self.wrapper.picture_init(self.param, self.pic_in)
        self.pic_out = self.wrapper.picture_alloc()

        pic = self.pic_in.contents
        pic.bitDepth = 8
        pic.colorSpace = 1  # I420

    def _run_analysis_pass(self):
        """Pass 1: 全局分析与参数规划"""
        print("\n=== Pass 1: Global Analysis ===")
        start_time = time.time()
        self.reader.seek(0)

        gop_idx = 0
        while True:
            current_pos = self.reader.frame_idx
            is_valid, features = self.feature_extractor.get_next_gop_features()

            if not is_valid:
                break

            frames_in_this_gop = features["frames_in_gop"]
            target_params = self.model.compute_adjustments(features)

            plan = {
                "gop_idx": gop_idx,
                "params": target_params,
                "frames": frames_in_this_gop,
                "features": features,
            }
            self.gop_plan_list.append(plan)

            debug_info = []
            for k in ["aq-strength", "cutree-strength", "psy-rd"]:
                val = target_params.get(k)
                if val is not None:
                    debug_info.append(f"{k}={val:.2f}")

            print(
                f"[Analysis GOP {gop_idx}] Frames: {frames_in_this_gop} | "
                + " | ".join(debug_info)
            )

            new_pos = current_pos + frames_in_this_gop
            self.reader.seek(new_pos)
            gop_idx += 1

        duration = time.time() - start_time
        print(f"Pass 1 Complete. Analyzed {gop_idx} GOPs in {duration:.2f}s.")

    def _run_encoding_pass(self):
        """Pass 2: 执行编码"""
        print("\n=== Pass 2: Encoding ===")
        self.reader.seek(0)
        self._setup_encoder()

        f_out = open(self.output_file, "wb")
        try:
            total_start = time.time()
            for plan in self.gop_plan_list:
                gop_idx = plan["gop_idx"]
                params = plan["params"]
                frames_to_encode = plan["frames"]

                print(f"--- Encoding GOP {gop_idx} ({frames_to_encode} frames) ---")
                self._apply_reconfig(params)

                # 配置 Picture 指针
                pic = self.pic_in.contents
                y, u, v = self.reader.get_pointers()
                ys, us, vs = self.reader.get_strides()
                pic.planes[0] = y
                pic.planes[1] = u
                pic.planes[2] = v
                pic.stride[0] = ys
                pic.stride[1] = us
                pic.stride[2] = vs

                for _ in range(frames_to_encode):
                    if not self.reader.read_frame():
                        break

                    pic.pts = self.encoded_frames
                    ret, nal_list = self.wrapper.encode(
                        self.encoder, self.pic_in, self.pic_out
                    )

                    if ret < 0:
                        return

                    for nal in nal_list:
                        f_out.write(nal)

                    if ret > 0:
                        self.encoded_frames += 1

            # Flush
            print("--- Flushing Encoder ---")
            while True:
                ret, nal_list = self.wrapper.encode(self.encoder, None, self.pic_out)
                if ret <= 0:
                    break
                for nal in nal_list:
                    f_out.write(nal)
                self.encoded_frames += 1

            total_time = time.time() - total_start
            print(
                f"Pass 2 Complete. Total Frames: {self.encoded_frames}, Time: {total_time:.2f}s"
            )

        finally:
            f_out.close()
            if self.pic_in:
                self.wrapper.picture_free(self.pic_in)
            if self.pic_out:
                self.wrapper.picture_free(self.pic_out)
            if self.encoder:
                self.wrapper.encoder_close(self.encoder)
            if self.param:
                self.wrapper.param_free(self.param)

    def _apply_reconfig(self, new_params):
        """调用 x265_encoder_reconfig"""
        if not new_params:
            return
        valid_params = {k: v for k, v in new_params.items() if v is not None}
        if not valid_params:
            return
        for k, v in valid_params.items():
            if self.wrapper.param_parse(self.param, k, str(v)) < 0:
                print(f"[Warning] Reconfig parse failed: {k}={v}")
        self.wrapper.encoder_reconfig(self.encoder, self.param)

    def save_decision_log(self, json_path):
        """[新增] 保存分析阶段的决策日志"""
        try:
            # 转换 numpy 类型为 float 以便 JSON 序列化
            def convert(o):
                if hasattr(o, "item"):
                    return o.item()
                return o

            with open(json_path, "w") as f:
                json.dump(self.gop_plan_list, f, indent=2, default=convert)
            print(f"[Log] Decision log saved to {json_path}")
        except Exception as e:
            print(f"[Error] Failed to save decision log: {e}")

    def run(self):
        """主入口"""
        self._run_analysis_pass()

        # [新增] 如果配置了日志路径，自动保存
        if "controller_log" in self.config:
            self.save_decision_log(self.config["controller_log"])

        self._run_encoding_pass()
