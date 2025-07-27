import json
import os

def convert_format(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取需要的内容
    input_data = data['input']
    output_data = data['output']
    
    # 构建新的格式
    # new_format = {"text": [{"role": "system","content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and \\boxed{ } tags, respectively, i.e., <think> reasoning process 1 here </think> <think> reasoning process n here  \\boxed{ answer here } </think>. Do not need to add any other text."},{"role": "user","content": f"任务：{input_data['task']}\n\n事件：\n{input_data['event']}\n\n输出格式：\n{input_data['output_format']}"},{"role": "assistant","content": f"<think>{output_data['think']}</think>\n\\boxed{{{output_data['judgement']}}}\n\n原因：{output_data['reason']}"}]}
    
    new_format = {"text": [{"role": "user","content": str(f"任务：{input_data['task']}\n\n事件：\n{input_data['event']}\n\n输出格式：\n{input_data['output_format']}")},{"role": "assistant","content": str(f"判断:{output_data['judgement']}\n\n原因：{output_data['reason']}") }]}
    
    # 写入输出文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(new_format, ensure_ascii=False) + '\n')

def main():
    input_dir = "/home/ubuntu/LLM_RL/dataset/output"  # 输入目录
    output_file = "converted_data.jsonl"  # 使用.jsonl扩展名表示每行一个JSON
    
    # 确保输出文件是空的，以防止追加到已有内容
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 使用os.walk递归遍历所有子目录
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.json'):
                input_file = os.path.join(root, filename)
                relative_path = os.path.relpath(input_file, input_dir)
                print(f"处理文件: {relative_path}")
                try:
                    convert_format(input_file, output_file)
                except Exception as e:
                    print(f"处理文件 {relative_path} 时出错: {str(e)}")


if __name__ == "__main__":
    main()
