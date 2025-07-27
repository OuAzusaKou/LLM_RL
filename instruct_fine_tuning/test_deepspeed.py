import deepspeed
import torch
import torch.distributed as dist

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    
    # 定义模型和配置
    model = torch.nn.Linear(10, 10).cuda()
    config = {
        "train_batch_size": 8,          # 全局批次大小
        "train_micro_batch_size_per_gpu": 1,  # 每个GPU的微批次大小
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-5}},
        "fp16": {"enabled": True}
    }
    
    # 初始化DeepSpeed引擎
    engine, *_ = deepspeed.initialize(model=model, config=config)
    print("[Success] DeepSpeed engine initialized!")

if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()  # 安全销毁进程组
