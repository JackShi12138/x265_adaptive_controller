import os
import ctypes


class YUVReader:
    """
    YUV420P (I420) 格式视频读取器。
    负责文件读取、内存缓冲管理以及向编码器提供正确的内存指针。
    """

    def __init__(self, file_path, width, height, bit_depth=8, fps=30):
        self.file_path = file_path
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.fps = fps

        self.file = None
        self.frame_idx = 0

        # 校验位深
        if bit_depth == 8:
            self.pixel_size = 1  # byte (uint8)
            self.ctype = ctypes.c_ubyte
        elif bit_depth in (10, 12):
            self.pixel_size = 2  # bytes (uint16)
            self.ctype = ctypes.c_uint16
        else:
            raise ValueError(
                f"Unsupported bit depth: {bit_depth}. Only 8, 10, 12 are supported."
            )

        # 计算各平面大小 (I420: Y + U/2 + V/2)
        self.y_size = width * height * self.pixel_size
        self.uv_width = width // 2
        self.uv_height = height // 2
        self.uv_size = self.uv_width * self.uv_height * self.pixel_size
        self.frame_size = self.y_size + 2 * self.uv_size

        # 分配 ctypes 缓冲区 (用于存放一帧数据)
        # 注意：buffer_size 是字节数，创建数组时需要除以 pixel_size
        self.buffer_len = self.frame_size // self.pixel_size
        self.buffer = (self.ctype * self.buffer_len)()

        # 计算各平面的内存地址偏移
        self.addr_base = ctypes.addressof(self.buffer)
        self.addr_y = self.addr_base
        self.addr_u = self.addr_y + self.y_size
        self.addr_v = self.addr_u + self.uv_size

        self._open_file()

    def _open_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"YUV file not found: {self.file_path}")
        self.file = open(self.file_path, "rb")

        # 可选：检查文件大小是否足够至少一帧
        file_size = os.path.getsize(self.file_path)
        if file_size < self.frame_size:
            print(
                f"[Warning] File size ({file_size}) is smaller than one frame ({self.frame_size})"
            )

    def read_frame(self):
        """
        读取下一帧数据到内部缓冲区。
        :return: True 如果读取成功, False 如果已到文件末尾 (EOF)
        """
        if not self.file:
            return False

        # readinto 直接读取数据到 ctypes 数组，避免 Python 字节对象复制，高效
        read_bytes = self.file.readinto(self.buffer)

        if read_bytes < self.frame_size:
            # 读不满一帧，视为结束
            return False

        self.frame_idx += 1
        return True

    def get_pointers(self):
        """
        获取当前帧各平面的指针，用于传递给 x265 编码器。
        :return: (y_ptr, u_ptr, v_ptr) as ctypes.c_void_p
        """
        return (
            ctypes.cast(self.addr_y, ctypes.c_void_p),
            ctypes.cast(self.addr_u, ctypes.c_void_p),
            ctypes.cast(self.addr_v, ctypes.c_void_p),
        )

    def get_strides(self):
        """
        获取各平面的行宽 (Stride)。
        :return: (y_stride, u_stride, v_stride)
        """
        # 对于标准 I420，Stride 通常等于宽度（乘以每像素字节数）
        return (
            self.width * self.pixel_size,
            self.uv_width * self.pixel_size,
            self.uv_width * self.pixel_size,
        )

    def seek(self, frame_index):
        """
        跳转到指定帧。
        """
        if self.file:
            offset = frame_index * self.frame_size
            self.file.seek(offset, 0)
            self.frame_idx = frame_index

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    # 支持上下文管理器 (with YUVReader(...) as yuv:)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
