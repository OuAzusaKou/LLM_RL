
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGE-M3 嵌入模型测试脚本
用于测试 FlagEmbedding 库中的 BGEM3FlagModel 的相似度计算功能
"""

import torch
import numpy as np


from FlagEmbedding import BGEM3FlagModel
import os
import sys

class BGEM3Tester:
    """BGE-M3 模型测试类"""
    
    def __init__(self, model_path="/9950backfile/liguoqi/wangzihang/bge-m3"):
        """
        初始化 BGE-M3 模型
        
        Args:
            model_path (str): 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """加载 BGE-M3 模型"""
        try:
            print(f"正在加载 BGE-M3 模型从: {self.model_path}")
            print(f"使用设备: {self.device}")
            
            self.model = BGEM3FlagModel(
                model_name_or_path=self.model_path,
                use_fp16=True,
                device=self.device
            )
            
            print("✅ 模型加载成功!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def get_embedding(self, text_or_texts):
        """
        获取文本的嵌入向量，处理不同的返回格式
        
        Args:
            text_or_texts: 单个文本或文本列表
            
        Returns:
            嵌入向量（numpy数组或torch张量）
        """
        result = self.model.encode(text_or_texts)
        
        # 检查返回结果的类型
        if isinstance(result, dict):
            # 如果是字典，尝试获取嵌入向量
            if 'embedding' in result:
                embedding = result['embedding']
            elif 'embeddings' in result:
                embedding = result['embeddings']
            else:
                # 打印字典的键来了解结构
                print(f"返回的字典键: {list(result.keys())}")
                # 尝试获取第一个值
                embedding = list(result.values())[0]
        else:
            embedding = result
            
        return embedding
    
    def test_basic_encoding(self):
        """测试基本的文本编码功能"""
        if self.model is None:
            print("❌ 模型未加载，请先调用 load_model()")
            return False
            
        try:
            print("\n=== 测试基本编码功能 ===")
            
            # 测试单个文本编码
            text1 = "1 \
                    aaa"
            result1 = self.model.encode(text1)
            print(f"文本1: {text1}")
            print(f"编码结果类型: {type(result1)}")
            
            if isinstance(result1, dict):
                print(f"编码结果键: {list(result1.keys())}")
                embedding1 = self.get_embedding(text1)
            else:
                embedding1 = result1
                
            print(f"嵌入向量形状: {embedding1.shape}")
            print(f"嵌入向量类型: {type(embedding1)}")
            
            # 测试批量文本编码
            texts = ["第一个文本", "第二个文本", "第三个文本"]
            result_batch = self.model.encode(texts)
            print(f"\n批量文本: {texts}")
            print(f"批量编码结果类型: {type(result_batch)}")
            
            if isinstance(result_batch, dict):
                print(f"批量编码结果键: {list(result_batch.keys())}")
                embeddings = self.get_embedding(texts)
            else:
                embeddings = result_batch
                
            print(f"批量嵌入向量形状: {embeddings.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ 基本编码测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_similarity_calculation(self):
        """测试相似度计算功能"""
        if self.model is None:
            print("❌ 模型未加载，请先调用 load_model()")
            return False
            
        try:
            print("\n=== 测试相似度计算功能 ===")
            
            # 测试用例1: 相似的解释
            model_explanation = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            ground_truth_explanation = "事件1. 行号: 3824, 函数调用 gst_object_ref(splitmux->provided_sink) 直接或者间接地返回了输入参数 splitmux->provided_sink 的信息。\n事件2. 行号: 3824, 赋值 provided_sink = gst_object_ref(splitmux->provided_sink) 。\n事件3. 行号: 3850, 在 gst_bin_add 中释放 provided_sink 。\n事件4. 行号: 3852, 在 gst_object_unref 中再次释放 provided_sink 。\n"
            
            # 测试用例2: 不相似的解释
            model_explanation2 = "1"
            ground_truth_explanation2 = "事件1. 行号: 3824, 函数调用 gst_object_ref(splitmux->provided_sink) 直接或者间接地返回了输入参数 splitmux->provided_sink 的信息。\n事件2. 行号: 3824, 赋值 provided_sink = gst_object_ref(splitmux->provided_sink) 。\n事件3. 行号: 3850, 在 gst_bin_add 中释放 provided_sink 。\n事件4. 行号: 3852, 在 gst_object_unref 中再次释放 provided_sink 。\n"
            
            # 测试用例3: 完全相同的解释
            model_explanation3 = "用户点击了登录按钮"
            ground_truth_explanation3 = "用户点击了登录按钮"
            
            test_cases = [
                ("相似解释", model_explanation, ground_truth_explanation),
                ("不相似解释", model_explanation2, ground_truth_explanation2),
                ("相同解释", model_explanation3, ground_truth_explanation3)
            ]
            
            for case_name, text1, text2 in test_cases:
                print(f"\n--- {case_name} ---")
                print(f"文本1: {text1}")
                print(f"文本2: {text2}")
                
                # 计算嵌入向量
                embedding1 = self.get_embedding(text1)
                embedding2 = self.get_embedding(text2)
                
                # 确保嵌入向量是numpy数组或torch张量
                if hasattr(embedding1, 'numpy'):
                    embedding1 = embedding1.numpy()
                if hasattr(embedding2, 'numpy'):
                    embedding2 = embedding2.numpy()
                
                # 计算余弦相似度
                similarity = embedding1 @ embedding2.T
                
                print(f"余弦相似度: {similarity.item():.4f}")
                
                # 验证相似度值是否在合理范围内
                if -1 <= similarity.item() <= 1:
                    print("✅ 相似度值在合理范围内")
                else:
                    print("❌ 相似度值超出合理范围")
            
            return True
            
        except Exception as e:
            print(f"❌ 相似度计算测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_edge_cases(self):
        """测试边界情况"""
        if self.model is None:
            print("❌ 模型未加载，请先调用 load_model()")
            return False
            
        try:
            print("\n=== 测试边界情况 ===")
            
            # 测试空字符串
            print("测试空字符串...")
            empty_embedding = self.get_embedding("")
            print(f"空字符串嵌入向量形状: {empty_embedding.shape}")
            
            # 测试很长的文本
            print("测试长文本...")
            long_text = "这是一个很长的文本。" * 100
            long_embedding = self.get_embedding(long_text)
            print(f"长文本嵌入向量形状: {long_embedding.shape}")
            
            # 测试特殊字符
            print("测试特殊字符...")
            special_text = "特殊字符: !@#$%^&*()_+-=[]{}|;':\",./<>?"
            special_embedding = self.get_embedding(special_text)
            print(f"特殊字符嵌入向量形状: {special_embedding.shape}")
            
            # 测试中英文混合
            print("测试中英文混合...")
            mixed_text = "This is a mixed text with 中文 and English"
            mixed_embedding = self.get_embedding(mixed_text)
            print(f"中英文混合嵌入向量形状: {mixed_embedding.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ 边界情况测试失败: {e}")
            return False
    
    def test_batch_processing(self):
        """测试批量处理功能"""
        if self.model is None:
            print("❌ 模型未加载，请先调用 load_model()")
            return False
            
        try:
            print("\n=== 测试批量处理功能 ===")
            
            # 准备测试数据
            texts = [
                "用户登录系统",
                "用户注册账户", 
                "用户购买商品",
                "用户查看订单",
                "用户修改密码"
            ]
            
            print(f"批量处理 {len(texts)} 个文本...")
            
            # 批量编码
            embeddings = self.get_embedding(texts)
            print(f"批量嵌入向量形状: {embeddings.shape}")
            
            # 计算所有文本之间的相似度矩阵
            similarity_matrix = embeddings @ embeddings.T
            print(f"相似度矩阵形状: {similarity_matrix.shape}")
            
            # 显示相似度矩阵
            print("\n相似度矩阵:")
            for i, text1 in enumerate(texts):
                for j, text2 in enumerate(texts):
                    sim = similarity_matrix[i, j].item()
                    print(f"{sim:.3f}", end="\t")
                print()
            
            return True
            
        except Exception as e:
            print(f"❌ 批量处理测试失败: {e}")
            return False
    
    def test_model_info(self):
        """测试模型信息"""
        if self.model is None:
            print("❌ 模型未加载，请先调用 load_model()")
            return False
            
        try:
            print("\n=== 模型信息 ===")
            
            # 获取模型信息
            if hasattr(self.model, 'model'):
                model = self.model.model
                if hasattr(model, 'config'):
                    config = model.config
                    print(f"模型名称: {getattr(config, 'model_type', 'Unknown')}")
                    print(f"词汇表大小: {getattr(config, 'vocab_size', 'Unknown')}")
                    print(f"隐藏层大小: {getattr(config, 'hidden_size', 'Unknown')}")
                    print(f"最大序列长度: {getattr(config, 'max_position_embeddings', 'Unknown')}")
            
            # 测试编码维度
            test_embedding = self.get_embedding("测试")
            print(f"嵌入向量维度: {test_embedding.shape[1]}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型信息获取失败: {e}")
            return False
    
    def test_grpo_similarity_logic(self):
        """测试与GRPO训练器中相同的相似度计算逻辑"""
        if self.model is None:
            print("❌ 模型未加载，请先调用 load_model()")
            return False
            
        try:
            print("\n=== 测试GRPO相似度计算逻辑 ===")
            
            # 模拟GRPO训练器中的代码逻辑
            model_explanation = "用户点击了登录按钮，系统验证了用户名和密码"
            ground_truth_explanation = "用户进行了登录操作，系统执行了身份验证"
            
            print(f"模型解释: {model_explanation}")
            print(f"标准解释: {ground_truth_explanation}")
            
            # 使用与GRPO训练器相同的逻辑
            response_embedding = self.get_embedding(model_explanation)
            explanation_embedding = self.get_embedding(ground_truth_explanation)
            
            print(f"响应嵌入向量形状: {response_embedding.shape}")
            print(f"解释嵌入向量形状: {explanation_embedding.shape}")
            
            # 计算余弦相似度（与GRPO训练器相同的计算方式）
            similarity = response_embedding @ explanation_embedding.T
            print(f"余弦相似度: {similarity.item():.4f}")
            
            # 验证相似度值
            if -1 <= similarity.item() <= 1:
                print("✅ GRPO相似度计算逻辑测试通过")
                return True
            else:
                print("❌ 相似度值超出合理范围")
                return False
                
        except Exception as e:
            print(f"❌ GRPO相似度计算逻辑测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始 BGE-M3 模型测试")
        print("=" * 50)
        
        # 检查模型路径
        if not os.path.exists(self.model_path):
            print(f"❌ 模型路径不存在: {self.model_path}")
            return False
        
        # 加载模型
        if not self.load_model():
            return False
        
        # 运行各项测试
        tests = [
            ("基本编码功能", self.test_basic_encoding),
            ("相似度计算功能", self.test_similarity_calculation),
            ("GRPO相似度计算逻辑", self.test_grpo_similarity_logic),
            ("边界情况", self.test_edge_cases),
            ("批量处理功能", self.test_batch_processing),
            ("模型信息", self.test_model_info)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        
        print(f"\n{'='*50}")
        print(f"测试结果: {passed_tests}/{total_tests} 项测试通过")
        
        if passed_tests == total_tests:
            print("✅ 所有测试通过！BGE-M3 模型工作正常")
        else:
            print("⚠️  部分测试失败，请检查模型配置")
        
        return passed_tests == total_tests


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "/9950backfile/liguoqi/wangzihang/bge-m3"
    
    # 创建测试器并运行测试
    tester = BGEM3Tester(model_path)
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ 测试完成，模型可以正常使用")
        sys.exit(0)
    else:
        print("\n❌ 测试失败，请检查模型配置")
        sys.exit(1)


if __name__ == "__main__":
    main()
