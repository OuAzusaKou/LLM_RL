import json
import random
from step_grpo_trainer import THINK_STR,ANSWER_STR
def convert_format(data, is_training=True):
    """Convert math problems to conversation format"""
    conversations = []
    for item in data:
        conversation = {
            "text": [
                {
                    "role": "system",
                    "content": f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant. \\
                    需要根据用户指令来回答问题，需要有一个思考过程，最后输出答案，思考过程用{THINK_STR}来分割，答案用{ANSWER_STR}来分割。注意{ANSWER_STR}后只有答案，不需要其他字符，请使用中文回复。\\
                    示例：
                        user: x+y = 5 , x =3 , y = ?
                        assistant: 首先,我们知道x+y = 5 {THINK_STR} 然后，我们知道x = 3{THINK_STR} 所以 y = 5 - x = 5 - 3 = 2 {ANSWER_STR}2
                    """
                },
                {
                    "role": "user", 
                    "content": item["problem"]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
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
    # 读取原始数据
    with open('math.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机打乱数据
    random.seed(42)  # 设置随机种子以保证结果可复现
    random.shuffle(data)
    
    # 分割训练集和测试集
    split_index = len(data) // 10  # 10%作为测试集
    test_data = data[:split_index]
    train_data = data[split_index:]
    
    # 转换格式
    train_conversations = convert_format(train_data, is_training=True)
    test_conversations = convert_format(test_data, is_training=False)
    
    # 保存训练集和测试集为JSONL格式
    save_jsonl(train_conversations, 'math_training_data.json')
    save_jsonl(test_conversations, 'math_test_data.json')
    
    print(f"转换完成。训练集大小: {len(train_data)}，测试集大小: {len(test_data)}")

if __name__ == "__main__":
    main() 