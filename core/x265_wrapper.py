import ctypes
import os
import sys

# --- 1. 核心结构体定义 ---


class X265_NAL(ctypes.Structure):
    """编码后的 NAL 单元结构 (核心输出)"""

    _fields_ = [
        ("type", ctypes.c_uint32),  # [修正] 使用 uint32 匹配 x265.h 定义
        ("sizeBytes", ctypes.c_uint32),  # [修正] 使用 uint32
        ("payload", ctypes.POINTER(ctypes.c_ubyte)),  # uint8_t*
    ]


class X265_PICTURE(ctypes.Structure):
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


class X265_API(ctypes.Structure):
    """API 函数表"""

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
        # --- 函数指针 ---
        ("param_alloc", ctypes.CFUNCTYPE(ctypes.c_void_p)),
        ("param_free", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("param_default", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        (
            "param_parse",
            ctypes.CFUNCTYPE(
                ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p
            ),
        ),
        ("scenecut_aware_qp_param_parse", ctypes.c_void_p),
        ("param_apply_profile", ctypes.c_void_p),
        (
            "param_default_preset",
            ctypes.CFUNCTYPE(
                ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p
            ),
        ),
        ("picture_alloc", ctypes.CFUNCTYPE(ctypes.POINTER(X265_PICTURE))),
        ("picture_free", ctypes.CFUNCTYPE(None, ctypes.POINTER(X265_PICTURE))),
        (
            "picture_init",
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(X265_PICTURE)),
        ),
        ("encoder_open", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)),
        ("encoder_parameters", ctypes.c_void_p),
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
                ctypes.POINTER(ctypes.POINTER(X265_NAL)),  # pp_nal
                ctypes.POINTER(ctypes.c_uint32),  # pi_nal
                ctypes.POINTER(X265_PICTURE),
                ctypes.POINTER(X265_PICTURE),
            ),
        ),
        ("encoder_get_stats", ctypes.c_void_p),
        ("encoder_log", ctypes.c_void_p),
        ("encoder_close", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("cleanup", ctypes.CFUNCTYPE(None)),
    ]


# --- 2. 包装类实现 ---


class X265Wrapper:
    def __init__(self, lib_path):
        self.lib = self._load_library(lib_path)
        self.api = self._load_api_table()

    def _load_library(self, lib_path):
        try:
            lib = ctypes.CDLL(lib_path)
            # print(f"[INFO] Library loaded: {lib_path}")
            return lib
        except OSError as e:
            print(f"[ERROR] Failed to load library: {e}")
            sys.exit(1)

    def _load_api_table(self):
        suffixes = ["_215", "_216", "_212", "_210", "_200", "", "_get"]
        api_get_func = None
        candidates = [f"x265_api_get{s}" for s in suffixes]

        for name in candidates:
            if hasattr(self.lib, name):
                api_get_func = getattr(self.lib, name)
                # print(f"[INFO] Found API entry point: {name}")
                break

        if not api_get_func:
            print("[CRITICAL] Could not find 'x265_api_get' function.")
            sys.exit(1)

        api_get_func.argtypes = [ctypes.c_int]
        api_get_func.restype = ctypes.POINTER(X265_API)

        api_ptr = api_get_func(0)
        if not api_ptr:
            print("[CRITICAL] x265_api_get returned NULL.")
            sys.exit(1)

        return api_ptr.contents

    # --- 接口封装 ---

    def param_alloc(self):
        return self.api.param_alloc()

    def param_free(self, p):
        self.api.param_free(p)

    def param_default_preset(self, p, preset, tune):
        tune_b = tune.encode("utf-8") if tune else None
        return self.api.param_default_preset(p, preset.encode("utf-8"), tune_b)

    def param_parse(self, p, name, val):
        return self.api.param_parse(p, name.encode("utf-8"), str(val).encode("utf-8"))

    def encoder_open(self, p):
        return self.api.encoder_open(p)

    def encoder_close(self, e):
        self.api.encoder_close(e)

    def encoder_reconfig(self, e, p):
        return self.api.encoder_reconfig(e, p)

    def picture_alloc(self):
        return self.api.picture_alloc()

    def picture_free(self, p):
        self.api.picture_free(p)

    def picture_init(self, param, pic):
        self.api.picture_init(param, pic)

    def encode(self, encoder, pic_in, pic_out):
        """
        核心编码函数 (最终修正版：Array Cast + Binary Safe Copy)
        :return: (ret, nal_bytes_list)
        """
        pp_nal = ctypes.POINTER(X265_NAL)()  # X265_NAL*
        pi_nal = ctypes.c_uint32(0)  # uint32_t

        ret = self.api.encoder_encode(
            encoder, ctypes.byref(pp_nal), ctypes.byref(pi_nal), pic_in, pic_out
        )

        encoded_bytes = []
        if ret > 0 and pp_nal:
            num_nal = pi_nal.value

            # [关键] 强制转换为数组指针，确保指针算术正确
            nal_array = ctypes.cast(pp_nal, ctypes.POINTER(X265_NAL * num_nal)).contents

            for i in range(num_nal):
                nal = nal_array[i]

                # [关键] 使用 string_at 读取指定长度的二进制数据
                if nal.sizeBytes > 0 and nal.payload:
                    data = ctypes.string_at(nal.payload, nal.sizeBytes)
                    encoded_bytes.append(data)

        return ret, encoded_bytes
