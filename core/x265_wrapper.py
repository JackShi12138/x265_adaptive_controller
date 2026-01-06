import ctypes
import os
import sys


# --- 定义 x265_api 结构体 ---
# 这是 x265 官方提供的函数指针表，用于安全获取所有接口
class X265_API(ctypes.Structure):
    _fields_ = [
        ("api_major_version", ctypes.c_int),
        ("api_build_number", ctypes.c_int),
        ("sizeof_param", ctypes.c_int),
        ("sizeof_picture", ctypes.c_int),
        ("sizeof_analysis_data", ctypes.c_int),
        ("sizeof_zone", ctypes.c_int),
        ("sizeof_stats", ctypes.c_int),
        ("bit_depth", ctypes.c_int),
        ("version_str", ctypes.c_char_p),
        ("build_info_str", ctypes.c_char_p),
        # 函数指针 (按 source/x265.h 顺序定义)
        ("param_alloc", ctypes.CFUNCTYPE(ctypes.c_void_p)),
        ("param_free", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("param_default", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        (
            "param_parse",
            ctypes.CFUNCTYPE(
                ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p
            ),
        ),
        ("scenecut_aware_qp_param_parse", ctypes.c_void_p),  # 占位
        ("param_apply_profile", ctypes.c_void_p),  # 占位
        (
            "param_default_preset",
            ctypes.CFUNCTYPE(
                ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p
            ),
        ),
        ("picture_alloc", ctypes.CFUNCTYPE(ctypes.c_void_p)),
        ("picture_free", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("picture_init", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)),
        ("encoder_open", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)),
        ("encoder_parameters", ctypes.c_void_p),  # 占位
        (
            "encoder_reconfig",
            ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p),
        ),
        ("encoder_reconfig_zone", ctypes.c_void_p),
        ("encoder_headers", ctypes.c_void_p),
        ("configure_vbv_end", ctypes.c_void_p),
        (
            "encoder_encode",
            ctypes.CFUNCTYPE(
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ),
        ),
        ("encoder_get_stats", ctypes.c_void_p),
        ("encoder_log", ctypes.c_void_p),
        ("encoder_close", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("cleanup", ctypes.CFUNCTYPE(None)),
    ]


class X265Wrapper:
    def __init__(self, lib_path):
        self.lib = self._load_library(lib_path)
        self.api = self._load_api_table()

    def _load_library(self, lib_path):
        try:
            lib = ctypes.CDLL(lib_path)
            print(f"[INFO] Library loaded: {lib_path}")
            return lib
        except OSError as e:
            print(f"[ERROR] Failed to load library: {e}")
            sys.exit(1)

    def _load_api_table(self):
        """
        使用 x265_api_get 获取函数指针表，这是最安全的调用方式
        """
        # 1. 尝试查找 x265_api_get (及其后缀变体)
        suffixes = ["_215", "_216", "_212", "_210", "_200", "", "_get"]
        api_get_func = None

        # 优先找 x265_api_get_215 这种带版本的，最后找通用的
        candidates = [f"x265_api_get{s}" for s in suffixes]

        for name in candidates:
            if hasattr(self.lib, name):
                api_get_func = getattr(self.lib, name)
                print(f"[INFO] Found API entry point: {name}")
                break

        if not api_get_func:
            print(
                "[CRITICAL] Could not find 'x265_api_get' function. Library seems incompatible."
            )
            sys.exit(1)

        # 2. 调用 API 获取指针表
        # 参数: bitDepth (0 表示默认)
        api_get_func.argtypes = [ctypes.c_int]
        api_get_func.restype = ctypes.POINTER(X265_API)

        api_ptr = api_get_func(0)

        if not api_ptr:
            print("[CRITICAL] x265_api_get returned NULL.")
            sys.exit(1)

        api = api_ptr.contents
        print(
            f"[INFO] Loaded x265 API: Version {api.api_major_version}.{api.api_build_number} | BitDepth: {api.bit_depth}"
        )
        print(f"[DEBUG] sizeof(x265_param) = {api.sizeof_param}")

        return api

    # --- 封装调用接口 (直接使用 api 表中的函数指针) ---

    def param_alloc(self):
        return self.api.param_alloc()

    def param_free(self, param_ptr):
        self.api.param_free(param_ptr)

    def param_parse(self, param_ptr, name, value):
        # 必须传入 bytes
        return self.api.param_parse(
            param_ptr, name.encode("utf-8"), value.encode("utf-8")
        )

    def encoder_open(self, param_ptr):
        return self.api.encoder_open(param_ptr)

    def encoder_reconfig(self, encoder_ptr, param_ptr):
        return self.api.encoder_reconfig(encoder_ptr, param_ptr)

    def encoder_close(self, encoder_ptr):
        self.api.encoder_close(encoder_ptr)

    def picture_alloc(self):
        return self.api.picture_alloc()

    def picture_free(self, pic_ptr):
        self.api.picture_free(pic_ptr)

    def picture_init(self, param_ptr, pic_ptr):
        self.api.picture_init(param_ptr, pic_ptr)


# --- 测试入口 ---
if __name__ == "__main__":
    # 替换为您的实际路径
    LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"

    wrapper = X265Wrapper(LIB_PATH)

    print("\n[TEST] 尝试初始化 x265...")
    param = wrapper.param_alloc()
    if param:
        print(f"[SUCCESS] x265_param_alloc returned pointer: {hex(param)}")

        # 尝试设置一个参数，看是否还会返回 -1
        res = wrapper.param_parse(param, "bframes", "3")
        print(f"[TEST] param_parse result: {res} (0 means success)")

        if res == 0:
            encoder = wrapper.encoder_open(param)
            if encoder:
                print(f"[SUCCESS] x265_encoder_open returned pointer: {hex(encoder)}")
                wrapper.encoder_close(encoder)
            else:
                print("[FAIL] x265_encoder_open failed.")
        else:
            print("[FAIL] param_parse failed (Structure mismatch likely).")

        wrapper.param_free(param)
    else:
        print("[FAIL] x265_param_alloc failed.")
