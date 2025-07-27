import torch
import torch.distributed as dist
import time
import datetime
import subprocess
import json

def get_gpu_temp():
    try:
        # 使用nvidia-smi命令获取GPU温度
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            encoding='utf-8'
        )
        temps = [int(x) for x in result.strip().split('\n')]
        return max(temps)  # 返回最高温度
    except Exception as e:
        print(f"Failed to get GPU temperature: {str(e)}")
        return 0

def init_process_group():
    dist.init_process_group(backend='nccl')

def run_allreduce_test(tensor_size, duration_minutes, max_temp=80):
    device = torch.device("cuda")
    tensor = torch.ones(tensor_size, device=device)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    iteration = 0
    
    while time.time() < end_time:
        try:
            # 检查GPU温度
            current_temp = get_gpu_temp()
            if current_temp > max_temp:
                print(f"Warning: GPU temperature too high ({current_temp}°C) at {datetime.datetime.now()}")
                time.sleep(60)  # 等待冷却
                continue
                
            dist.all_reduce(tensor)
            iteration += 1
            if iteration % 100 == 0:
                print(f"Completed {iteration} iterations at {datetime.datetime.now()}")
                print(f"Current GPU temperature: {current_temp}°C")
                
        except Exception as e:
            print(f"Error occurred at {datetime.datetime.now()}")
            print(f"Error details: {str(e)}")
            print(f"GPU temperature at error: {get_gpu_temp()}°C")
            return False
    
    return True

if __name__ == "__main__":
    init_process_group()
    # 测试不同大小的tensor
    sizes = [(1024*1024), (10*1024*1024), (100*1024*1024)]
    
    # 记录开始时的GPU状态
    print(f"Initial GPU temperature: {get_gpu_temp()}°C")
    
    for size in sizes:
        print(f"Testing with tensor size: {size}")
        success = run_allreduce_test(size, duration_minutes=60, max_temp=80)
        if not success:
            break 