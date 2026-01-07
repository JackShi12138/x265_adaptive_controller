import sys
import os
import ctypes
import time


# === 1. 严格定义的 x265 结构体 (适配您提供的源码定义) ===
class X265Param(ctypes.Structure):
    _fields_ = [("cpuid", ctypes.c_int), ("frameNumThreads", ctypes.c_int)]  # 仅占位


class X265Picture(ctypes.Structure):
    _fields_ = [
        ("pts", ctypes.c_int64),  # 0
        ("dts", ctypes.c_int64),  # 8
        ("vbvEndFlag", ctypes.c_int),  # 16 (新增字段)
        # ctypes 会自动在此处填充 4 字节，以保证下一字段 userData (void*) 的 8 字节对齐
        ("userData", ctypes.c_void_p),  # 24
        ("planes", ctypes.c_void_p * 4),  # 32 (大小为4)
        ("stride", ctypes.c_int * 4),  # 64 (大小为4，类型为int)
        ("bitDepth", ctypes.c_int),  # 80
        ("sliceType", ctypes.c_int),  # 84
        ("poc", ctypes.c_int),  # 88
        ("colorSpace", ctypes.c_int),  # 92
        ("forceqp", ctypes.c_int),  # 96
        # 后续字段如 analysisData 为结构体传值，定义复杂且在此处未被使用，故省略
        # 由于我们只操作指针，部分定义是安全的
    ]


class X265Nal(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("sizeBytes", ctypes.c_int),
        ("payload", ctypes.POINTER(ctypes.c_ubyte)),
    ]


# === 2. 加载库与函数 ===
LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"

try:
    lib = ctypes.CDLL(LIB_PATH)
except OSError as e:
    print(f"[Error] 无法加载库: {e}")
    sys.exit(1)

# [关键修正] 自动寻找正确的 encoder_open 函数名
encoder_open_func = None
if hasattr(lib, "x265_encoder_open_215"):
    encoder_open_func = lib.x265_encoder_open_215
    print("[Info] Detected API: x265_encoder_open_215")
elif hasattr(lib, "x265_encoder_open"):
    encoder_open_func = lib.x265_encoder_open
    print("[Info] Detected API: x265_encoder_open")
else:
    # 暴力搜索
    import re

    # 这里无法直接列出符号，只能尝试常见版本
    for v in range(200, 230):
        name = f"x265_encoder_open_{v}"
        if hasattr(lib, name):
            encoder_open_func = getattr(lib, name)
            print(f"[Info] Detected API: {name}")
            break

if not encoder_open_func:
    print("[Error] 找不到 x265_encoder_open 函数")
    sys.exit(1)

# 定义函数签名
lib.x265_param_alloc.restype = ctypes.POINTER(X265Param)
lib.x265_param_free.argtypes = [ctypes.POINTER(X265Param)]
lib.x265_param_default_preset.argtypes = [
    ctypes.POINTER(X265Param),
    ctypes.c_char_p,
    ctypes.c_char_p,
]
lib.x265_param_parse.argtypes = [
    ctypes.POINTER(X265Param),
    ctypes.c_char_p,
    ctypes.c_char_p,
]

encoder_open_func.argtypes = [ctypes.POINTER(X265Param)]
encoder_open_func.restype = ctypes.c_void_p

lib.x265_picture_alloc.restype = ctypes.POINTER(X265Picture)
lib.x265_picture_init.argtypes = [
    ctypes.POINTER(X265Param),
    ctypes.POINTER(X265Picture),
]
lib.x265_picture_free.argtypes = [ctypes.POINTER(X265Picture)]

lib.x265_encoder_encode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(X265Nal)),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(X265Picture),
    ctypes.POINTER(X265Picture),
]
lib.x265_encoder_encode.restype = ctypes.c_int
lib.x265_encoder_close.argtypes = [ctypes.c_void_p]


# === 3. 主测试逻辑 ===
def run_diagnostic():
    print("=== x265 结构体对齐修正验证 (Based on Source) ===")

    width = 416
    height = 240
    fps = 30

    # 1. 准备参数
    param = lib.x265_param_alloc()
    lib.x265_param_default_preset(param, b"medium", None)

    config = {
        "input-res": f"{width}x{height}",
        "fps": str(fps),
        "input-csp": "i420",
        "bitrate": "300",
        "vbv-maxrate": "300",
        "vbv-bufsize": "36000",
        "annexb": "1",
        "repeat-headers": "1",
    }

    for k, v in config.items():
        lib.x265_param_parse(param, k.encode("utf-8"), v.encode("utf-8"))

    encoder = encoder_open_func(param)
    if not encoder:
        print("[Fail] 编码器无法打开")
        return

    pic_in = lib.x265_picture_alloc()
    lib.x265_picture_init(param, pic_in)

    # 2. 准备纯色数据
    y_size = width * height
    uv_size = y_size // 4
    total_size = y_size + uv_size * 2

    buffer = (ctypes.c_ubyte * total_size)()
    ctypes.memset(buffer, 128, total_size)  # 灰色

    # 3. 设置指针
    addr_base = ctypes.addressof(buffer)
    addr_y = addr_base
    addr_u = addr_base + y_size
    addr_v = addr_u + uv_size

    pic = pic_in.contents

    # 手动初始化字段
    pic.poc = 0
    pic.userData = None
    # pic.userSEI, pic.numUserSEI 已从定义中移除，无需赋值

    pic.bitDepth = 8
    pic.colorSpace = 1  # I420

    # 正确写入 planes
    pic.planes[0] = ctypes.cast(addr_y, ctypes.c_void_p)
    pic.planes[1] = ctypes.cast(addr_u, ctypes.c_void_p)
    pic.planes[2] = ctypes.cast(addr_v, ctypes.c_void_p)

    pic.stride[0] = width
    pic.stride[1] = width // 2
    pic.stride[2] = width // 2

    print(f"Debug: Planes Address Written -> Y:{hex(addr_y)}")

    # 4. 编码循环
    out_file = "debug_gray_test_fixed.hevc"
    f_out = open(out_file, "wb")

    nal_ptr = ctypes.POINTER(X265Nal)()
    nal_count = ctypes.c_uint32()

    print("开始编码 50 帧...")
    try:
        for i in range(50):
            pic.pts = i
            ret = lib.x265_encoder_encode(
                encoder, ctypes.byref(nal_ptr), ctypes.byref(nal_count), pic_in, None
            )

            if ret > 0:
                for j in range(nal_count.value):
                    nal = nal_ptr[j]
                    data = ctypes.string_at(nal.payload, nal.sizeBytes)
                    f_out.write(data)
    except Exception as e:
        print(f"\n[CRASH] 依然崩溃: {e}")
        return

    # Flush
    while True:
        ret = lib.x265_encoder_encode(
            encoder, ctypes.byref(nal_ptr), ctypes.byref(nal_count), None, None
        )
        if ret <= 0:
            break
        for j in range(nal_count.value):
            nal = nal_ptr[j]
            data = ctypes.string_at(nal.payload, nal.sizeBytes)
            f_out.write(data)

    f_out.close()
    lib.x265_encoder_close(encoder)
    lib.x265_param_free(param)
    lib.x265_picture_free(pic_in)

    print(f"诊断完成。输出: {out_file}")
    print("如果这次不崩溃且画面是灰色，请把这个 X265Picture 类定义更新到主代码中。")


if __name__ == "__main__":
    run_diagnostic()
