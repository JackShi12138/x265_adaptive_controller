import unittest
import os
import sys
import shutil

# 添加项目根目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from core.x265_wrapper import X265Wrapper
from utils.yuv_io import YUVReader
from core.adaptive_controller import AdaptiveController

class TestAdaptiveController(unittest.TestCase):
    
    def setUp(self):
        # 1. 准备文件路径
        # 请根据您的实际环境修改 input_file 路径
        self.input_file = "/home/shiyushen/x265_sequence/ClassD/RaceHorses_416x240_30.yuv"
        
        # 如果输入文件不存在，跳过测试
        if not os.path.exists(self.input_file):
            print(f"[Info] Input file not found: {self.input_file}")
            print("[Info] Skipping AdaptiveController integration test.")
            self.skipTest("Input file not found")
            
        self.output_dir = os.path.join(CURRENT_DIR, "temp_output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.output_file = os.path.join(self.output_dir, "adaptive_out.hevc")
        self.csv_file = os.path.join(self.output_dir, "encode_log.csv")
        self.lib_path = "/home/shiyushen/program/x265_4.0/libx265.so"
        
        self.width = 416
        self.height = 240
        self.fps = 30
        
        # 2. 构建分层配置字典 (模拟 x265 命令行)
        # x265 --preset slow --bitrate 300 --strict-cbr --vbv-bufsize 36000 
        #      --csv-log-level 2 --csv log.csv --vbv-maxrate 300
        #      --aq-mode 2 --cutree --rd 3 --rdoq-level 2
        #      --aq-strength 1.0 --cutree-strength 2.0 --psy-rd 2 --psy-rdoq 1 --qcomp 0.6
        
        self.config = {
            "output_file": self.output_file,
            
            # Stage 1: Preset
            "preset": "slow",
            
            # Stage 3: Base Params
            "base_params": {
                "bitrate": 300,
                "strict-cbr": 1,        # 开启 strict-cbr
                "vbv-maxrate": 300,
                "vbv-bufsize": 36000,
                "csv": self.csv_file,
                "csv-log-level": 2
            },
            
            # Stage 4: Mode Params
            "mode_params": {
                "aq-mode": 2,
                "cutree": 1,            # 开启 cutree
                "rd": 3,
                "rdoq-level": 2
            },
            
            # Stage 5: Tune Params (初始值，模型将基于此进行调整)
            "tune_params": {
                "aq-strength": 1.0,
                "cutree-strength": 2.0, # 将被模型计算，但可能不被 x265 应用(如果wrapper不支持)
                "psy-rd": 2.0,
                "psy-rdoq": 1.0,
                "qcomp": 0.6
            },
            
            # Optimization Model Hyperparameters
            "hyperparams": {
                "a": 1.0, 
                "b": 1.0,
                "beta": {
                    "VAQ": 0.5,      # aq-strength 调整力度
                    "CUTree": 0.0,   # 不调 cutree
                    "PsyRD": 1.0,    # psy-rd 调整力度
                    "PsyRDOQ": 1.0,  # psy-rdoq 调整力度
                    "QComp": 0.1     # qcomp 调整力度
                }
            }
        }
        
    def tearDown(self):
        # 清理输出目录
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_full_pipeline(self):
        print("\n=== Testing Adaptive Controller Pipeline (Step 6) ===")
        
        # 初始化 Wrapper
        if not os.path.exists(self.lib_path):
             self.skipTest(f"Library not found: {self.lib_path}")
        wrapper = X265Wrapper(self.lib_path)
        
        # 初始化 Reader
        with YUVReader(self.input_file, self.width, self.height, fps=self.fps) as reader:
            
            # 初始化控制器
            controller = AdaptiveController(
                wrapper=wrapper,
                yuv_reader=reader,
                config=self.config
            )
            
            # 运行全流程 (Pass 1 -> Pass 2)
            print("Starting Controller run...")
            controller.run()
            
            # === 验证结果 ===
            
            # 1. 验证 HEVC 输出文件
            self.assertTrue(os.path.exists(self.output_file), "Output HEVC file should exist")
            file_size = os.path.getsize(self.output_file)
            print(f"Output File Size: {file_size} bytes")
            self.assertGreater(file_size, 0, "Output HEVC file should not be empty")
            
            # 2. 验证 CSV 日志文件 (由 x265 生成)
            self.assertTrue(os.path.exists(self.csv_file), "CSV log file should exist")
            self.assertGreater(os.path.getsize(self.csv_file), 0, "CSV log file should not be empty")
            
            # 3. 验证参数自适应性 (Pass 1 的结果)
            num_plans = len(controller.gop_plan_list)
            print(f"Generated Plans for {num_plans} GOPs.")
            self.assertGreater(num_plans, 0, "Should generate at least one GOP plan")
            
            # 检查前几个 GOP 的参数是否有变化
            print("\nParameter Adaptation Sample (First 5 GOPs):")
            aq_values = []
            for i, plan in enumerate(controller.gop_plan_list[:5]):
                params = plan['params']
                aq = params.get('aq-strength')
                psy = params.get('psy-rd')
                qcomp = params.get('qcomp')
                
                print(f"  GOP {i}: AQ={aq:.4f}, PsyRD={psy:.4f}, QComp={qcomp:.4f}")
                
                if aq is not None:
                    aq_values.append(aq)
            
            # 简单验证参数不是死值 (如果视频内容有变化，参数应该波动)
            if len(aq_values) > 1:
                # 计算方差或直接比较是否全等
                is_constant = all(x == aq_values[0] for x in aq_values)
                if not is_constant:
                    print("=> Verified: Parameters are adapting dynamically.")
                else:
                    print("=> Note: Parameters are constant (Input video might be static).")

if __name__ == "__main__":
    unittest.main()