import json
import os
import random
from collections import defaultdict

def convert_format(input_file, output_file):
    """转换单个文件的格式"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    input_data = data['input']
    output_data = data['output']
    if len(data['instruction']) > 0:
        subtype = data['instruction'][0]['subtype']
    else:
        return None
    
    new_format = {
        "subtype": subtype,
        "text": [
            {
                "role": "user",
                "content": str(f"任务：{input_data['task']}\n\n代码：\n{input_data['code']}\n\n事件：\n{input_data['event']}\n\n输出格式：\n{input_data['output_format']}")
            },
            {
                "role": "assistant",
                "content": str(f"判断:{output_data['judgement']}\n\n原因：{output_data['reason']}")
            }
        ]
    }
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(new_format, ensure_ascii=False) + '\n')
    
    return subtype

def convert_dataset(input_dir, output_file):
    """转换整个数据集并返回subtype统计信息"""
    if os.path.exists(output_file):
        os.remove(output_file)
    
    subtype_count = defaultdict(int)
    
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.json'):
                input_file = os.path.join(root, filename)
                relative_path = os.path.relpath(input_file, input_dir)
                print(f"处理文件: {relative_path}")
                subtype = convert_format(input_file, output_file)
                if subtype is not None:
                    subtype_count[subtype] += 1
    
    return subtype_count

def split_dataset(input_file, train_file, test_file, test_ratio=0.2):
    """按照subtype分层抽样分割数据集"""
    # 读取所有数据并按subtype分组
    subtype_data = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                subtype = data['subtype']
                subtype_data[subtype].append(line)
    
    train_samples = []
    test_samples = []
    
    # 对每个subtype进行分层抽样
    for subtype, samples in subtype_data.items():
        n_test = max(1, int(len(samples) * test_ratio))  # 确保每个subtype至少有一个测试样本
        
        # 随机打乱当前subtype的样本
        random.shuffle(samples)
        
        # 分割数据
        test_samples.extend(samples[:n_test])
        train_samples.extend(samples[n_test:])
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_samples:
            f.write(item)
    
    # 写入测试集
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_samples:
            f.write(item)
    
    return len(train_samples), len(test_samples)

def main():
    input_dir = "./dataset/output_unbalanced"  # 输入目录
    converted_file = "./dataset/converted_data.jsonl"
    train_file = "./dataset/train.jsonl"
    test_file = "./dataset/test.jsonl"
    test_ratio = 0.2
    
    # 设置随机种子以确保结果可复现
    random.seed(42)
    
    # 转换数据集并获取subtype统计信息
    print("开始转换数据集...")
    subtype_count = convert_dataset(input_dir, converted_file)
    
    # 打印subtype统计信息
    print("\nsubtype统计信息:")
    for subtype, count in subtype_count.items():
        print(f"{subtype}: {count}条")
    
    # 分割数据集
    print("\n开始分割数据集...")
    train_count, test_count = split_dataset(converted_file, train_file, test_file, test_ratio)
    
    print(f"\n数据集分割完成:")
    print(f"训练集数量: {train_count}")
    print(f"测试集数量: {test_count}")
    print(f"总数据量: {train_count + test_count}")

if __name__ == "__main__":
    main()
