import numpy as np
import ctypes


class FeatureExtractor:
    """
    视频特征提取器 (分辨率自适应修正版)
    """

    def __init__(self, yuv_reader, gop_size=40, processing_width=256):
        """
        :param yuv_reader: YUVReader 实例
        :param gop_size: GOP 长度
        :param processing_width: 下采样处理宽度 (推荐 256)
        """
        self.reader = yuv_reader
        self.gop_size = gop_size

        # 计算下采样步长
        self.scale_step = max(1, self.reader.width // processing_width)

        # 内部状态：存储上一帧的下采样数据 (用于时域计算)
        self.prev_frame_y = None

        # === [关键修改] 分辨率自适应归一化常数 ===
        # 逻辑：高分辨率细节多，上限设高；低分辨率画面软，上限设低。
        width = self.reader.width

        if width >= 1920:  # 1080p, 2K, 4K
            self.NORM_REF = {
                "var": 4000.0,  # 高纹理上限
                "sad": 60.0,  # 高像素量下的运动差异可能更大
                "grad": 100.0,  # 锐利边缘
                "luma": 255.0,
            }
        elif width >= 1280:  # 720p
            self.NORM_REF = {"var": 3000.0, "sad": 50.0, "grad": 80.0, "luma": 255.0}
        else:  # 480p, 360p, 240p (Class C, D)
            self.NORM_REF = {
                "var": 2000.0,  # 画面较软，降低门槛
                "sad": 40.0,
                "grad": 50.0,  # 梯度通常较小
                "luma": 255.0,
            }

        print(
            f"[FeatureExtractor] Adaptive Norm Refs for {width}x{self.reader.height}: {self.NORM_REF}"
        )

    def _ptr_to_numpy(self, y_ptr):
        # ... (以下代码保持不变) ...
        buffer_interface = np.ctypeslib.as_array(
            ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_ubyte)),
            shape=(self.reader.height, self.reader.width),
        )
        return buffer_interface

    def _downsample(self, img_full):
        """快速下采样"""
        return img_full[:: self.scale_step, :: self.scale_step].astype(np.float32)

    def _compute_frame_features(self, curr_y_ds):
        """计算单帧特征"""
        h, w = curr_y_ds.shape
        pixel_count = h * w

        # 1. 空间方差 (w1)
        spatial_var = np.var(curr_y_ds)

        # 2. 梯度能量 (w3)
        gx = np.abs(curr_y_ds[:, 1:] - curr_y_ds[:, :-1])
        gy = np.abs(curr_y_ds[1:, :] - curr_y_ds[:-1, :])
        # [微调] 防止除以0，虽然 pixel_count 不可能为0
        grad_mean = (np.sum(gx) + np.sum(gy)) / max(1, pixel_count)

        # 3. 亮度
        luma_mean = np.mean(curr_y_ds)

        # 4. 时域帧差 (w2)
        if self.prev_frame_y is None:
            # 如果是全视频第一帧，没有历史参考，SAD 设为 0
            temporal_sad = 0.0
        else:
            diff = np.abs(curr_y_ds - self.prev_frame_y)
            temporal_sad = np.mean(diff)

        return {
            "var": spatial_var,
            "grad": grad_mean,
            "sad": temporal_sad,
            "luma": luma_mean,
        }

    def _normalize_features(self, raw_feats):
        """标准化映射到 [0, 1]"""
        norm = {}
        # 增加极小值保护，防止除零（虽然 NORM_REF 是常数）
        norm["w1_var"] = min(raw_feats["var"] / self.NORM_REF["var"], 1.0)
        norm["w2_sad"] = min(raw_feats["sad"] / self.NORM_REF["sad"], 1.0)
        norm["w3_grad"] = min(raw_feats["grad"] / self.NORM_REF["grad"], 1.0)
        norm["w4_tex"] = norm["w3_grad"]

        # w5: 0.5 Spatial + 0.5 Temporal
        norm["w5_cplx"] = 0.5 * norm["w1_var"] + 0.5 * norm["w2_sad"]
        norm["avg_luma"] = raw_feats["luma"] / 255.0

        return norm

    def get_next_gop_features(self):
        """
        核心方法：预读并分析下一个 GOP
        """
        accumulated = {"var": 0.0, "grad": 0.0, "sad": 0.0, "luma": 0.0}
        frame_count = 0

        # 记录当前文件指针位置
        start_frame_idx = self.reader.frame_idx
        last_processed_frame = None

        try:
            for _ in range(self.gop_size):
                if not self.reader.read_frame():
                    break

                y_ptr, _, _ = self.reader.get_pointers()
                y_full = self._ptr_to_numpy(y_ptr)
                y_ds = self._downsample(y_full)

                current_prev = (
                    self.prev_frame_y if frame_count == 0 else last_processed_frame
                )

                if current_prev is None:
                    temporal_sad = 0.0
                else:
                    temporal_sad = np.mean(np.abs(y_ds - current_prev))

                frame_feats = self._compute_frame_features(y_ds)
                frame_feats["sad"] = temporal_sad  # 修正 SAD

                last_processed_frame = y_ds.copy()

                for k in frame_feats:
                    accumulated[k] += frame_feats[k]

                frame_count += 1

        finally:
            # 回溯指针
            self.reader.seek(start_frame_idx)

        if frame_count == 0:
            return False, {}

        # 更新全局 prev_frame_y
        if last_processed_frame is not None:
            self.prev_frame_y = last_processed_frame

        # 计算平均值并标准化
        avg_raw = {k: v / frame_count for k, v in accumulated.items()}
        final_feats = self._normalize_features(avg_raw)
        final_feats["frames_in_gop"] = frame_count

        return True, final_feats
