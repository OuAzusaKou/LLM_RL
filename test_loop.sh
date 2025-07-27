#!/bin/bash

# 添加环境变量设置
export LD_LIBRARY_PATH=/usr/local/nccl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

check_gpu_temp() {
    # 获取所有GPU的温度
    temps=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    max_temp=0
    
    # 找出最高温度
    for temp in $temps; do
        if [ $temp -gt $max_temp ]; then
            max_temp=$temp
        fi
    done
    
    echo "Maximum GPU temperature: ${max_temp}°C"
    
    # 如果温度超过阈值（例如80度），返回1
    if [ $max_temp -gt 80 ]; then
        return 1
    fi
    return 0
}

while true
do
    date
    
    # 检查GPU温度
    check_gpu_temp
    if [ $? -ne 0 ]; then
        echo "Warning: GPU temperature too high at $(date)"
        sleep 60  # 等待冷却
        continue
    fi
    
    ./build/all_reduce_perf -b 10G -e 40G -f 2 -g 4
    
    # 检查返回值，如果测试失败则退出
    if [ $? -ne 0 ]; then
        echo "Test failed at $(date)"
        nvidia-smi  # 打印GPU状态
        exit 1
    fi
    
    echo "Test completed successfully at $(date)"
    echo "-----------------------------------"
done 