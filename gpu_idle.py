import torch
import time
import os

def occupy_gpu(gpu_id):
    # 设置当前设备
    torch.cuda.set_device(gpu_id)
    # 创建一个大的张量来占用显存
    tensor = torch.randn(100000, 100000, device=f'cuda:{gpu_id}')
    print(f"GPU {gpu_id} 已占用，显存使用量: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.2f} MB")
    return tensor

def main():
    # 检查可用的GPU数量
    gpu_count = torch.cuda.device_count()
    if gpu_count < 8:
        print(f"警告：系统只有 {gpu_count} 个GPU，少于请求的8个")
        return

    print(f"开始占用8个GPU...")
    tensors = []
    
    # 占用所有8个GPU
    for i in range(8):
        tensors.append(occupy_gpu(i))
    
    print("所有GPU已占用，程序将保持运行状态...")
    
    try:
        while True:
            # 保持程序运行
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        # 清理GPU内存
        for tensor in tensors:
            del tensor
        torch.cuda.empty_cache()
        print("已释放所有GPU资源")

if __name__ == "__main__":
    main() 