import unittest
import sys
import os

# 将上一级目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scripts.calc_psnr_ssim import calculate_metrics


class TestCalcPsnrSsimReal(unittest.TestCase):

    def setUp(self):
        """
        使用服务器上的真实解码重建 YUV 和原始 YUV 文件进行集成测试。
        """
        self.dist_yuv = "/home/shiyushen/x265_adaptive_controller/analysis_data/20260214_230748/BasketballDrive_1920x1080_50/online/recon.yuv"
        self.ref_yuv = (
            "/home/shiyushen/x265_sequence/ClassB/BasketballDrive_1920x1080_50.yuv"
        )
        # 根据文件名 BasketballDrive_1920x1080_50 设定分辨率
        self.width = 1920
        self.height = 1080

    def test_real_ffmpeg_execution(self):
        """
        测试调用真实的 FFmpeg 进程，严格对比两个 Raw YUV 文件的输出。
        """
        # 安全机制：如果测试文件被清理或移动，优雅地跳过测试而不是抛出异常
        if not os.path.exists(self.dist_yuv) or not os.path.exists(self.ref_yuv):
            self.skipTest(
                f"真实测试文件缺失，跳过集成测试。\nRecon YUV: {self.dist_yuv}\nOrigin YUV: {self.ref_yuv}"
            )

        # 调用我们刚刚更新过的、强制双向限定分辨率和格式的严谨函数
        result = calculate_metrics(
            dist_yuv=self.dist_yuv,
            ref_yuv=self.ref_yuv,
            width=self.width,
            height=self.height,
            pixel_format="yuv420p",
        )

        self.assertIn("psnr", result)
        self.assertIn("ssim", result)

        # 真实视频编码的 PSNR 通常落在 30 dB 到 50 dB 的合理区间内
        self.assertGreater(result["psnr"], 30.0)
        self.assertLess(result["psnr"], 50.0)

        # 真实视频编码的 SSIM 通常在 0.85 到 1.0 之间
        self.assertGreater(result["ssim"], 0.85)
        self.assertLessEqual(result["ssim"], 1.0)

        print(
            f"集成测试结果 - PSNR: {result['psnr']:.2f} dB, SSIM: {result['ssim']:.4f}"
        )

    def test_missing_files_handling(self):
        """
        测试当 YUV 文件不存在时，函数是否能优雅地拦截并返回 -1.0。
        """
        result = calculate_metrics(
            dist_yuv="non_existent_dist.yuv",
            ref_yuv="non_existent_ref.yuv",
            width=self.width,
            height=self.height,
        )

        self.assertEqual(result["psnr"], -1.0)
        self.assertEqual(result["ssim"], -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
