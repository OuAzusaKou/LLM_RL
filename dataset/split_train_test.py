import json
import random

def split_dataset(input_file, train_file, test_file, train_ratio=0.8):
    """
    将数据集分割为训练集和测试集
    
    Args:
        input_file (str): 输入的jsonl文件路径
        train_file (str): 训练集输出文件路径
        test_file (str): 测试集输出文件路径
        train_ratio (float): 训练集占总数据的比例，默认0.8
    """
    # 读取所有数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f if line.strip()]
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算分割点
    split_idx = int(len(data) * train_ratio)
    
    # 分割数据
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(item + '\n')
    
    # 写入测试集
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(item + '\n')
    
    print(f"总数据量: {len(data)}")
    print(f"训练集数量: {len(train_data)}")
    print(f"测试集数量: {len(test_data)}")

def main():
    input_file = "./dataset/converted_data.jsonl"
    train_file = "./dataset/train.jsonl"
    test_file = "./dataset/test.jsonl"
    train_ratio = 0.8
    
    # 设置随机种子以确保结果可复现
    random.seed(42)
    
    split_dataset(input_file, train_file, test_file, train_ratio)

if __name__ == "__main__":
    main()
