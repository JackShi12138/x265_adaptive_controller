import time
import os
import ctypes
from .feature_extractor import FeatureExtractor
from .optimization_model import OptimizationModel

class AdaptiveController:
    """
    x265 自适应编码控制器 (Two-Pass 模式)
    
    工作流程:
    1. Pass 1 (Analysis): 扫描全视频，计算每个 GOP 的最优参数。
    2. Pass 2 (Encoding): 重置视频流，应用参数并执行编码。
    """

    def __init__(self, wrapper, yuv_reader, config):
        """
        :param wrapper: X265Wrapper 实例
        :param yuv_reader: YUVReader 实例
        :param config: 配置字典，结构如下 (参考命令行示例):
        
        Example CLI:
        x265 --input input.yuv --input-res 416x240 --fps 30 -o output.mp4 \\
             --preset slow \\                                      <-- preset
             --bitrate 300 --strict-cbr --vbv-bufsize 36000 \\     <-- base_params
             --csv-log-level 2 --csv log.csv --vbv-maxrate 300 \\  <-- base_params
             --aq-mode 2 --cutree --rd 3 --rdoq-level 2 \\         <-- mode_params
             --aq-strength 1.0 --cutree-strength 2.0 \\            <-- tune_params (initial)
             --psy-rd 2 --psy-rdoq 1 --qcomp 0.6                   <-- tune_params (initial)
        
        Config Dict Structure:
        {
            "output_file": "/path/to/output.hevc",
            
            # Stage 1: Preset
            "preset": "slow",
            
            # Stage 3: Base Params (码率控制、VBV、日志等基础参数)
            "base_params": {
                "bitrate": 300,
                "strict-cbr": 1,
                # 可选: vbv-maxrate, vbv-bufsize (若缺省则自动根据 bitrate 推导)
                "csv": "/path/to/log.csv",
                "csv-log-level": 2
            },
            
            # Stage 4: Mode Params (算法开关、模式选择)
            "mode_params": {
                "aq-mode": 2,
                "cutree": 1,      # 1=On (对应命令行 --cutree), 0=Off (对应命令行 --no-cutree)
                "rd": 3,
                "rdoq-level": 2
            },
            
            # Stage 5: Tune Params (需要被模型动态微调的参数初始值)
            "tune_params": {
                "aq-strength": 1.0,
                "cutree-strength": 2.0,
                "psy-rd": 2.0,
                "psy-rdoq": 1.0,
                "qcomp": 0.6
            },
            
            # Optimization Model Hyperparameters
            "hyperparams": {
                "a": 1.0, "b": 1.0,
                "beta": {"VAQ": 0.5, ...}
            }
        }
        """
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
        # 如果这是 Stage 3 (Base Params) 且包含 bitrate
        if stage_name == "Stage 3" and "bitrate" in param_dict:
            bitrate = int(param_dict["bitrate"])
            
            # 检查并自动补全 vbv-maxrate
            if "vbv-maxrate" not in param_dict:
                # 默认 MaxRate = Bitrate (CBR)
                if self.wrapper.param_parse(self.param, "vbv-maxrate", str(bitrate)) < 0:
                    print(f"[Warning] Auto-set vbv-maxrate failed")
                else:
                    # 仅在调试时打印
                    # print(f"[Info] Auto-set vbv-maxrate={bitrate}")
                    pass

            # 检查并自动补全 vbv-bufsize
            if "vbv-bufsize" not in param_dict:
                # 默认 Bufsize = 2 * Bitrate (宽松缓存) (单位通常是 kb)
                # x265 --bitrate 单位是 kbps, --vbv-bufsize 单位是 kbits
                bufsize = bitrate * 2 
                # 注意 x265 默认 bufsize 很大，这里设为 2秒 比较合理
                if self.wrapper.param_parse(self.param, "vbv-bufsize", str(bufsize)) < 0:
                    print(f"[Warning] Auto-set vbv-bufsize failed")
                else:
                    # print(f"[Info] Auto-set vbv-bufsize={bufsize}")
                    pass

        for k, v in param_dict.items():
            # 修正：不跳过 cutree-strength
            
            # 尝试设置参数
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
        self.wrapper.param_parse(self.param, "input-res", f"{self.reader.width}x{self.reader.height}")
        self.wrapper.param_parse(self.param, "fps", str(self.reader.fps))
        self.wrapper.param_parse(self.param, "input-csp", "i420")
        self.wrapper.param_parse(self.param, "annexb", "1")
        self.wrapper.param_parse(self.param, "repeat-headers", "1")
        
        # === Stage 3: Base Params (Bitrate, VBV, Logging) ===
        # 这一步会自动触发 VBV 推导
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

        # 配置 Picture 输入格式
        pic = self.pic_in.contents
        pic.bitDepth = 8
        pic.colorSpace = 1 # I420

    def _run_analysis_pass(self):
        """Pass 1: 全局分析与参数规划"""
        print("\n=== Pass 1: Global Analysis ===")
        start_time = time.time()
        
        self.reader.seek(0)
        
        gop_idx = 0
        while True:
            current_pos = self.reader.frame_idx
            
            # 提取特征
            is_valid, features = self.feature_extractor.get_next_gop_features()
            
            if not is_valid:
                break
                
            frames_in_this_gop = features['frames_in_gop']
            
            # 计算参数
            target_params = self.model.compute_adjustments(features)
            
            # 保存计划
            plan = {
                'gop_idx': gop_idx,
                'params': target_params,
                'frames': frames_in_this_gop,
                'features': features
            }
            self.gop_plan_list.append(plan)
            
            # 打印调试信息
            debug_info = []
            for k in ['aq-strength', 'cutree-strength', 'psy-rd']:
                val = target_params.get(k)
                if val is not None:
                    debug_info.append(f"{k}={val:.2f}")
            
            print(f"[Analysis GOP {gop_idx}] Frames: {frames_in_this_gop} | " + " | ".join(debug_info))
            
            # 手动推进文件指针
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
                gop_idx = plan['gop_idx']
                params = plan['params']
                frames_to_encode = plan['frames']
                
                print(f"--- Encoding GOP {gop_idx} ({frames_to_encode} frames) ---")
                
                # 应用参数 (Reconfig)
                self._apply_reconfig(params)
                
                # 配置 Picture 指针
                pic = self.pic_in.contents
                y, u, v = self.reader.get_pointers()
                ys, us, vs = self.reader.get_strides()
                pic.planes[0] = y; pic.planes[1] = u; pic.planes[2] = v
                pic.stride[0] = ys; pic.stride[1] = us; pic.stride[2] = vs

                for _ in range(frames_to_encode):
                    if not self.reader.read_frame():
                        print("[Warning] Unexpected EOF in Pass 2")
                        break
                    
                    pic.pts = self.encoded_frames
                    
                    ret, nal_list = self.wrapper.encode(self.encoder, self.pic_in, self.pic_out)
                    
                    if ret < 0:
                        print(f"[Error] Encode failed at frame {self.encoded_frames}")
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
            print(f"Pass 2 Complete. Total Frames: {self.encoded_frames}, Time: {total_time:.2f}s, FPS: {self.encoded_frames/total_time:.2f}")

        finally:
            f_out.close()
            # 资源释放
            if self.pic_in: self.wrapper.picture_free(self.pic_in)
            if self.pic_out: self.wrapper.picture_free(self.pic_out)
            if self.encoder: self.wrapper.encoder_close(self.encoder)
            if self.param: self.wrapper.param_free(self.param)

    def _apply_reconfig(self, new_params):
        """调用 x265_encoder_reconfig"""
        if not new_params:
            return

        valid_params = {k: v for k, v in new_params.items() if v is not None}
        if not valid_params:
            return

        # [修正] 移除对 cutree-strength 的跳过逻辑
        for k, v in valid_params.items():
            if self.wrapper.param_parse(self.param, k, str(v)) < 0:
                print(f"[Warning] Reconfig parse failed: {k}={v}")
                
        # 执行 Reconfig
        self.wrapper.encoder_reconfig(self.encoder, self.param)

    def run(self):
        """主入口"""
        self._run_analysis_pass()
        self._run_encoding_pass()