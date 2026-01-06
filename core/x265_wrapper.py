import ctypes
import os
import sys


class X265Wrapper:
    def __init__(self, lib_path):
        self.lib = self._load_library(lib_path)
        self._resolve_functions()

    def _load_library(self, lib_path):
        try:
            # 加载动态库
            lib = ctypes.CDLL(lib_path)
            print(f"[INFO] Library loaded: {lib_path}")
            return lib
        except OSError as e:
            print(f"[ERROR] Failed to load library: {e}")
            sys.exit(1)

    def _resolve_functions(self):
        """
        自动探测函数名（处理 _215 等版本后缀）并建立映射
        """
        # 定义我们需要用到的所有核心函数名
        required_funcs = [
            "x265_param_alloc",
            "x265_param_free",
            "x265_param_parse",
            "x265_encoder_open",
            "x265_encoder_close",
            "x265_encoder_encode",
            "x265_encoder_reconfig",
            "x265_picture_alloc",
            "x265_picture_free",
        ]

        # 可能的后缀列表：空字符串（标准），_215（当前版本），以及未来可能的版本
        suffixes = ["", "_215", "_216", "_217", "_210", "_200", "_199"]

        self.api = {}  # 存储最终可调用的函数对象

        for func_name in required_funcs:
            found = False
            for suffix in suffixes:
                real_name = func_name + suffix
                if hasattr(self.lib, real_name):
                    # 获取函数对象
                    func_ptr = getattr(self.lib, real_name)
                    # 将其保存到 api 字典中，键名为标准名称（无后缀）
                    self.api[func_name] = func_ptr
                    print(f"[DEBUG] Mapped {func_name} -> {real_name}")
                    found = True
                    break

            if not found:
                print(
                    f"[CRITICAL] Could not find symbol for {func_name}. Tried suffixes: {suffixes}"
                )
                sys.exit(1)

    # --- 封装调用接口 ---

    def param_alloc(self):
        # 此时 self.api['x265_param_alloc'] 已经指向了正确的 x265_param_alloc_215
        func = self.api["x265_param_alloc"]
        func.restype = ctypes.c_void_p  # 显式声明返回类型为指针
        return func()

    def param_free(self, param_ptr):
        func = self.api["x265_param_free"]
        func.argtypes = [ctypes.c_void_p]
        func(param_ptr)

    def param_parse(self, param_ptr, name, value):
        func = self.api["x265_param_parse"]
        func.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        # 注意：ctypes 需要传入 bytes 类型，而非 str
        return func(param_ptr, name.encode("utf-8"), value.encode("utf-8"))

    def encoder_open(self, param_ptr):
        func = self.api["x265_encoder_open"]
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        return func(param_ptr)

    # ... 其他函数的封装类似 ...


# --- 简单的测试入口 ---
if __name__ == "__main__":
    # 替换为您的实际路径
    LIB_PATH = "/home/shiyushen/program/x265_4.0/libx265.so"

    wrapper = X265Wrapper(LIB_PATH)

    print("\n[TEST] 尝试初始化 x265...")
    param = wrapper.param_alloc()
    if param:
        print("[SUCCESS] x265_param_alloc returned a pointer.")

        res = wrapper.param_parse(param, "preset", "ultrafast")
        print(f"[TEST] param_parse result: {res} (0 means success)")

        encoder = wrapper.encoder_open(param)
        if encoder:
            print("[SUCCESS] x265_encoder_open returned a pointer.")
        else:
            print("[FAIL] x265_encoder_open failed.")
    else:
        print("[FAIL] x265_param_alloc failed.")
