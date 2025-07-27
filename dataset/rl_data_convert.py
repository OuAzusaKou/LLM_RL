import json

def convert_qa_format(input_file, output_file):
    """
    将原始QA数据转换为新格式
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 转换格式
    converted_data = []
    for line in lines:
        try:
            data = json.loads(line)
            # 获取问题和答案
            qa_pair = data['text']
            question = qa_pair[0]['content']
            answer = qa_pair[1]['content']
            
            # 构建新格式
            new_format = {
                'prompt': [
                    {
                        'role': 'user',
                        'content': question+'请一步一步思考，并用 "####" 来提示最终答案.'
                    }
                ],
                'reward_model': {
                    'ground_truth': answer,
                    'style': 'rule'
                }
            }
            
            # 添加到结果列表
            converted_data.append(new_format)
            
        except json.JSONDecodeError as e:
            print(f"解析JSON时出错: {e}")
            continue
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_file = "/home/ubuntu/LLM_RL/dataset/train.jsonl"
    output_file = "/home/ubuntu/LLM_RL/dataset/train_gsmk_format.jsonl"
    convert_qa_format(input_file, output_file) 