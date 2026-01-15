import numpy as np
import ctypes


class FeatureExtractor:
    """
    视频特征提取器 (Step 4 修正版)

    功能：
    1. 从 YUVReader 预读 (Peek) GOP 长度的帧。
    2. 计算特征并聚合。
    3. [关键修复] 分析完成后自动回溯 (Rewind) 文件指针，确保不影响后续编码器的读取。

    论文对应特征:
    - w1 (Var): 空间方差 -> VAQ
    - w2 (TempSAD): 时域帧差 -> CUTree
    - w3 (Grad): 梯度能量 -> Psy-RDO
    - w5 (Complex): 综合复杂度 -> QComp
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

        # 标准化参考常数
        self.NORM_REF = {"var": 4000.0, "sad": 60.0, "grad": 100.0, "luma": 255.0}

    def _ptr_to_numpy(self, y_ptr):
        """将 ctypes 指针转换为 numpy 数组视图"""
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
        grad_mean = (np.sum(gx) + np.sum(gy)) / pixel_count

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
        注意：此方法会短暂移动文件指针，但在返回前会恢复指针位置，
        以保证编码器能正确读取这一段数据。
        """
        accumulated = {"var": 0.0, "grad": 0.0, "sad": 0.0, "luma": 0.0}
        frame_count = 0

        # [关键修正 1] 记录当前文件指针位置 (帧索引)
        start_frame_idx = self.reader.frame_idx

        # 临时存储本 GOP 最后一帧数据，用于更新 self.prev_frame_y
        last_processed_frame = None

        try:
            for _ in range(self.gop_size):
                if not self.reader.read_frame():
                    break

                y_ptr, _, _ = self.reader.get_pointers()
                y_full = self._ptr_to_numpy(y_ptr)
                y_ds = self._downsample(y_full)

                # 计算特征 (注意：这里使用的是 self.prev_frame_y，即上一个 GOP 的最后一帧)
                # 在 GOP 内部循环时，我们需要实时更新“上一帧”的概念吗？
                # 是的，计算 Frame N 的 SAD 需要 Frame N-1。
                # 所以我们在这里使用一个临时的 prev_frame 变量

                current_prev = (
                    self.prev_frame_y if frame_count == 0 else last_processed_frame
                )

                # 临时计算逻辑复刻
                if current_prev is None:
                    temporal_sad = 0.0
                else:
                    temporal_sad = np.mean(np.abs(y_ds - current_prev))

                # 重新计算其他特征...
                # 为了复用 _compute_frame_features，我们需要临时 hack 一下 self.prev_frame_y
                # 或者更干净的做法是把 prev_frame 传进去。
                # 为了保持代码简单，这里手动展开计算 SAD，其他调用函数

                frame_feats = self._compute_frame_features(y_ds)
                # 修正 SAD (因为 _compute_frame_features 用的是 self.prev_frame_y)
                frame_feats["sad"] = temporal_sad

                last_processed_frame = y_ds.copy()  # 更新循环内的“上一帧”

                for k in frame_feats:
                    accumulated[k] += frame_feats[k]

                frame_count += 1

        finally:
            # [关键修正 2] 无论发生什么，必须将文件指针 seek 回起始位置
            # 这样编码器才能读到刚才分析的帧
            self.reader.seek(start_frame_idx)

        if frame_count == 0:
            return False, {}

        # [关键修正 3] 更新全局的 prev_frame_y 为本 GOP 的最后一帧
        # 这样下一个 GOP 分析时，能正确计算与本 GOP 最后一帧的差值
        if last_processed_frame is not None:
            self.prev_frame_y = last_processed_frame

        # 计算平均值并标准化
        avg_raw = {k: v / frame_count for k, v in accumulated.items()}
        final_feats = self._normalize_features(avg_raw)
        final_feats["frames_in_gop"] = frame_count

        return True, final_feats
