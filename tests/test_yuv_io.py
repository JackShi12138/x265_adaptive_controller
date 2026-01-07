import unittest
import os
import sys
import ctypes

# 添加项目根目录到路径，以便导入模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

# 假设 utils 文件夹已创建，如果没有请手动创建 (mkdir utils)
# 在 python 中导入时确保 utils/__init__.py 存在或作为命名空间包导入
from utils.yuv_io import YUVReader


class TestYUVReader(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_gen.yuv"
        self.width = 64
        self.height = 64
        self.frames = 5
        self.bit_depth = 8

        # 计算尺寸
        self.y_size = self.width * self.height
        self.uv_size = (self.width // 2) * (self.height // 2)

        # 生成一个测试 YUV 文件
        # 模式：Y平面填充帧序号，U平面填充 0xAA，V平面填充 0x55
        with open(self.test_file, "wb") as f:
            for i in range(self.frames):
                # Y: 全是 i
                f.write(bytes([i % 256] * self.y_size))
                # U: 全是 0xAA (170)
                f.write(bytes([0xAA] * self.uv_size))
                # V: 全是 0x55 (85)
                f.write(bytes([0x55] * self.uv_size))

    def tearDown(self):
        # 清理测试文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_read_sequential(self):
        """测试顺序读取帧"""
        print("\n=== Testing Sequential Read ===")
        with YUVReader(
            self.test_file, self.width, self.height, self.bit_depth
        ) as reader:
            count = 0
            while reader.read_frame():
                y_ptr, u_ptr, v_ptr = reader.get_pointers()

                # 验证 Y 平面第一个像素
                y_val = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_ubyte))[0]
                print(
                    f"Frame {count}: First Pixel Y Value = {y_val} (Expected {count})"
                )
                self.assertEqual(y_val, count)

                # 验证 U 平面第一个像素
                u_val = ctypes.cast(u_ptr, ctypes.POINTER(ctypes.c_ubyte))[0]
                self.assertEqual(u_val, 0xAA)

                count += 1

            self.assertEqual(count, self.frames)
            print(f"Successfully read {count} frames.")

    def test_strides(self):
        """测试 Stride 计算"""
        print("\n=== Testing Strides ===")
        reader = YUVReader(self.test_file, self.width, self.height)
        strides = reader.get_strides()
        self.assertEqual(strides, (64, 32, 32))
        reader.close()

    def test_seek(self):
        """测试 Seek 功能"""
        print("\n=== Testing Seek ===")
        with YUVReader(self.test_file, self.width, self.height) as reader:
            target_frame = 3
            reader.seek(target_frame)
            reader.read_frame()

            y_ptr, _, _ = reader.get_pointers()
            y_val = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_ubyte))[0]

            print(f"Seek to frame {target_frame}, read value: {y_val}")
            self.assertEqual(y_val, target_frame)


if __name__ == "__main__":
    unittest.main()
