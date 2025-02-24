import json
import random
import sys
import os
import glob
import argparse

# 添加父目录到导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step_grpo_trainer import THINK_STR_START,THINK_STR_END,ANSWER_STR_START,ANSWER_STR_END

def convert_format(data, is_training=True,answer_tags="answer"):
    """Convert math problems to conversation format"""
    conversations = []
    for item in data:
        conversation = {
            "text": [
                {
                    "role": "system",
                    "content": f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within {THINK_STR_START} {THINK_STR_END} and {ANSWER_STR_START} {ANSWER_STR_END} tags, respectively, i.e., {THINK_STR_START} reasoning process 1 here {THINK_STR_END} {THINK_STR_START} reasoning process n here {THINK_STR_END} {ANSWER_STR_START} answer here {ANSWER_STR_END}. Do not need to add any other text."""
                },
                {
                    "role": "user", 
                    "content": item["problem"]
                },
                {
                    "role": "assistant",
                    "content": item[answer_tags]
                }
            ]
        }
        conversations.append(conversation)
    return conversations

def save_jsonl(data, filename):
    """Save data in JSON Lines format"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='转换JSON文件为训练数据格式')
    parser.add_argument('--input_dir',default='instruct_finetuning_dataset/math_dataset')
    parser.add_argument('--output', '-o', default='instruct_finetuning_dataset/math_training_data.json', 
                      help='输出文件名 (默认: math_training_data.json)')
    args = parser.parse_args()
    
    # 确保输入目录存在
    if not os.path.exists(args.input_dir):
        print(f"错误：输入目录 '{args.input_dir}' 不存在")
        return
    
    # 获取所有json文件
    all_data = []
    json_files = glob.glob(os.path.join(args.input_dir, '*.json'))
    
    if not json_files:
        print(f"在目录 '{args.input_dir}' 中没有找到任何JSON文件")
        return
        
    for json_file in json_files:
        # 跳过输出文件
        if os.path.basename(json_file) == args.output:
            continue
            
        print(f"处理文件: {os.path.basename(json_file)}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        except json.JSONDecodeError:
            print(f"警告：文件 '{json_file}' 不是有效的JSON格式，已跳过")
        except Exception as e:
            print(f"警告：处理文件 '{json_file}' 时出错：{str(e)}")
    
    if not all_data:
        print("没有找到任何有效的数据进行处理")
        return
        
    # 随机打乱数据
    random.seed(42)  # 设置随机种子以保证结果可复现
    random.shuffle(all_data)
    
    # 分割训练集和测试集
    split_index = len(all_data) // 10  # 10%作为测试集
    test_data = all_data[:split_index]
    train_data = all_data[split_index:]
    
    # 转换格式
    train_conversations = convert_format(train_data, is_training=True, answer_tags="multi_steps_response")
    
    # 保存到指定输出路径
    output_path = args.output
    save_jsonl(train_conversations, output_path)
    
    print(f"\n转换完成。总数据量: {len(all_data)}")
    print(f"训练集大小: {len(train_data)}，测试集大小: {len(test_data)}")
    print(f"已保存到: {output_path}")

if __name__ == "__main__":
    main() 