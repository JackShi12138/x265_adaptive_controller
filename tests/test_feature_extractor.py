import unittest
import os
import sys
import numpy as np
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from utils.yuv_io import YUVReader
from core.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # 使用之前的 RaceHorses 素材 (如果存在)，否则生成假数据
        self.real_input = (
            "/home/shiyushen/x265_sequence/ClassD/BasketballPass_416x240_50.yuv"
        )
        self.width = 416
        self.height = 240
        self.fps = 50
        self.gop = 40

        self.use_real_file = os.path.exists(self.real_input)
        if not self.use_real_file:
            print("[Info] Real YUV not found, generating synthetic pattern...")
            self.gen_file = "temp_test_pattern.yuv"
            self._generate_pattern_yuv(
                self.gen_file, self.width, self.height, 60
            )  # 2 GOPs
            self.input_file = self.gen_file
        else:
            self.input_file = self.real_input

    def _generate_pattern_yuv(self, filename, w, h, frames):
        """生成一个有简单运动和纹理的测试视频"""
        y_size = w * h
        uv_size = y_size // 2
        with open(filename, "wb") as f:
            for i in range(frames):
                x = np.linspace(0, 10 * np.pi, w)
                y_row = (np.sin(x + i * 0.2) * 127 + 128).astype(np.uint8)
                y_plane = np.tile(y_row, (h, 1))

                f.write(y_plane.tobytes())
                f.write(bytes([128] * uv_size))  # U/V gray

    def tearDown(self):
        if not self.use_real_file and os.path.exists(self.input_file):
            os.remove(self.input_file)

    def test_gop_features(self):
        print(
            f"\n=== Testing GOP Feature Extraction (Input: {os.path.basename(self.input_file)}) ==="
        )

        with YUVReader(self.input_file, self.width, self.height) as reader:
            # 初始化提取器，下采样到宽 128 以提升速度
            extractor = FeatureExtractor(
                reader, gop_size=self.gop, processing_width=128
            )

            gop_idx = 0
            total_time = 0

            while True:
                # 1. 记录当前位置 (验证 rewind 功能)
                start_idx = reader.frame_idx

                # 2. 提取特征 (此时 reader 会预读 gop_size 帧，然后 seek 回 start_idx)
                start_t = time.time()
                is_valid, feats = extractor.get_next_gop_features()
                cost_t = time.time() - start_t

                if not is_valid:
                    break

                # [验证] 确保提取器真的把指针还回来了
                self.assertEqual(
                    reader.frame_idx,
                    start_idx,
                    "Error: FeatureExtractor did not rewind file pointer!",
                )

                total_time += cost_t
                frame_count = feats["frames_in_gop"]

                print(
                    f"\n[GOP {gop_idx}] Processed {frame_count} frames in {cost_t*1000:.2f} ms"
                )
                print(f"  -> Speed: {frame_count/cost_t:.1f} FPS (Analysis Only)")
                print("  -> Normalized Features:")
                print(f"     w1 (Var) : {feats['w1_var']:.4f}")
                print(f"     w2 (SAD) : {feats['w2_sad']:.4f}")
                print(f"     w3 (Grad): {feats['w3_grad']:.4f}")
                print(f"     w5 (Cplx): {feats['w5_cplx']:.4f}")

                # 验证标准化约束
                self.assertTrue(0.0 <= feats["w1_var"] <= 1.0, "w1 out of range")
                self.assertTrue(0.0 <= feats["w2_sad"] <= 1.0, "w2 out of range")
                self.assertTrue(0.0 <= feats["w3_grad"] <= 1.0, "w3 out of range")

                # 3. [关键更新] 手动模拟编码器消耗数据，推进文件指针
                # 在真实场景中，这里会调用 encoder.encode()
                # 在测试中，我们直接 seek 到下一段
                new_idx = start_idx + frame_count
                reader.seek(new_idx)

                gop_idx += 1

            print(f"\nTotal Analysis FPS: {(gop_idx * self.gop) / total_time:.1f}")


if __name__ == "__main__":
    unittest.main()
