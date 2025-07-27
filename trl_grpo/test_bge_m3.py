import torch
import os
from ollama_bge import OllamaBGE
import time
import psutil
import GPUtil
import requests
import json

def check_ollama_service():
    """检查Ollama服务状态"""
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        print("Ollama服务正常运行")
        return True
    else:
        print(f"Ollama服务异常，状态码: {response.status_code}")
        return False

def get_gpu_memory_usage(gpu_id):
    """获取指定GPU的显存使用情况"""
    gpu = GPUtil.getGPUs()[gpu_id]
    return {
        'total': gpu.memoryTotal,
        'used': gpu.memoryUsed,
        'free': gpu.memoryFree,
        'utilization': gpu.memoryUtil * 100
    }

def test_bge_m3(gpu_id=0, batch_size=32, num_texts=100):
    """测试BGE-M3模型的显存占用
    
    Args:
        gpu_id: 使用的GPU ID
        batch_size: 批处理大小
        num_texts: 测试文本数量
    """
    # 检查Ollama服务状态
    if not check_ollama_service():
        print("请确保Ollama服务正常运行后再试")
        return

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"使用GPU: {gpu_id}")
    print(f"初始显存使用情况:")
    print(get_gpu_memory_usage(gpu_id))
    
    # 创建测试文本
    test_texts = [f"这是一个测试文本 {i}" for i in range(num_texts)]
    
    # 初始化模型
    print("\n开始加载模型...")
    start_time = time.time()
    model = OllamaBGE(model="bge-m3", batch_size=batch_size)
    load_time = time.time() - start_time
    print(f"模型加载完成，耗时: {load_time:.2f}秒")
    print("加载后显存使用情况:")
    print(get_gpu_memory_usage(gpu_id))
    
    # 测试单个文本编码
    print("\n测试单个文本编码...")
    single_embedding = model.encode([test_texts[0]])
    print(single_embedding)
    print("单个文本编码成功")
    
    # 测试批量编码
    print("\n开始批量编码测试...")
    start_time = time.time()
    embeddings = model.encode(test_texts)
    encode_time = time.time() - start_time
    print(f"批量编码完成，耗时: {encode_time:.2f}秒")
    print("编码后显存使用情况:")
    print(get_gpu_memory_usage(gpu_id))
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    print("\n清理后显存使用情况:")
    print(get_gpu_memory_usage(gpu_id))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="要使用的GPU ID")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--num_texts", type=int, default=100, help="测试文本数量")
    args = parser.parse_args()
    
    test_bge_m3(
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        num_texts=args.num_texts
    ) 