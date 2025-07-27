import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict


import sys
import os


# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trl_grpo.reward_function.reward import code_compute_score

def calculate_weight_norms(model):
    """计算并打印模型权重的模长"""
    print("\n=== 模型权重模长统计 ===")
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2).item()  # L2范数
            total_norm += param_norm ** 2
            param_count += 1
            print(f"{name}: {param_norm:.6f}")
    
    total_norm = total_norm ** 0.5
    print(f"\n总权重模长: {total_norm:.6f}")
    print(f"参数数量: {param_count}")
    print("=" * 30)
def load_model_and_tokenizer(model_path,tokenizer_path=None):
    """加载本地模型和分词器"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='cuda:0',
        torch_dtype=torch.float32
    )
    if tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=2048):
    """生成模型回答"""
    inputs = tokenizer.apply_chat_template(
            [eval(prompt)],
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
            truncation=True,
            max_length=1024,
            # padding='max_length',
            # padding_side='left'
        ).to('cuda:0')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_generate_tokens=500,
            max_new_tokens=1024,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id,151643]
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def plot_subtype_accuracy(subtype_stats):
    """Plot accuracy bar chart for each subtype"""
    subtypes = []
    accuracies = []
    total_samples = []
    
    for subtype, stats in subtype_stats.items():
        subtypes.append(subtype)
        accuracies.append(stats['accuracy'])
        total_samples.append(stats['total'])
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot accuracy bars
    bars = ax1.bar(subtypes, accuracies, color='skyblue')
    ax1.set_ylabel('Accuracy', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    
    # Add accuracy labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    # Add sample count line plot
    ax2 = ax1.twinx()
    ax2.plot(subtypes, total_samples, color='red', marker='o', linestyle='-', linewidth=2)
    ax2.set_ylabel('Sample Count', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add sample count labels on line
    for i, v in enumerate(total_samples):
        ax2.text(i, v, str(v), ha='center', va='bottom', color='red')
    
    plt.title('Defect Detection Accuracy and Sample Count by Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('subtype_accuracy.png')
    plt.close()

def evaluate_model(model_path, test_file_path, tokenizer_path=None):
    """评估模型性能"""
    # 加载模型和分词器
    print("正在加载模型...")
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    
    # 计算并打印权重模长
    calculate_weight_norms(model)
    
    # 加载测试数据集
    print("正在加载测试数据集...")
    test_data = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                test_data.append(json.loads(line))
    
    # 初始化统计数据
    subtype_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total = len(test_data)
    
    # 创建结果文件，写入文件头
    results_file = "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('{\n')
        f.write('  "detailed_results": [\n')
    
    print("开始评估...")
    for i, item in enumerate(tqdm(test_data)):
        subtype = item['subtype']
        prompt = f"{item['prompt'][0]}"
        
        # 获取模型回答
        model_response = generate_response(model, tokenizer, prompt)
        
        ground_truth = item['reward_model']['ground_truth']

        # 计算得分
        score = code_compute_score(model_response, ground_truth)

        # 提取预期答案
        is_correct = (score == 1.0)
        if is_correct:
            total_correct += 1
            subtype_stats[subtype]['correct'] += 1
        
        subtype_stats[subtype]['total'] += 1
        
        # 立即写入结果到文件
        result_item = {
            "input": item['prompt'][0],
            "expected": ground_truth,
            "predicted": model_response,
            "is_correct": is_correct,
            "subtype": subtype
        }
        
        # 写入单个结果
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write('    ' + json.dumps(result_item, ensure_ascii=False, indent=2))
            if i < total - 1:  # 不是最后一个项目
                f.write(',\n')
            else:  # 最后一个项目
                f.write('\n')
    
    # 计算最终统计信息
    total_accuracy = total_correct / total
    
    # 计算每个subtype的准确率
    for subtype in subtype_stats:
        stats = subtype_stats[subtype]
        stats['accuracy'] = stats['correct'] / stats['total']
    
    # 完成文件写入，添加统计信息
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write('  ],\n')
        f.write('  "overall_accuracy": ' + str(total_accuracy) + ',\n')
        f.write('  "total_samples": ' + str(total) + ',\n')
        f.write('  "correct_predictions": ' + str(total_correct) + ',\n')
        f.write('  "subtype_statistics": ' + json.dumps(dict(subtype_stats), ensure_ascii=False, indent=2) + '\n')
        f.write('}\n')
    
    # 绘制subtype准确率图表
    plot_subtype_accuracy(subtype_stats)
    
    print(f"\n评估完成！")
    print(f"总体准确率: {total_accuracy:.2%}")
    print(f"总体正确预测数: {total_correct}/{total}")
    print("\n各类型准确率:")
    for subtype, stats in subtype_stats.items():
        print(f"{subtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print(f"\n详细结果已保存到 {results_file}")
    print(f"准确率图表已保存到 subtype_accuracy.png")

if __name__ == "__main__":
    # 设置模型路径和测试数据集路径
    MODEL_PATH = "/9950backfile/liguoqi/wangzihang/LLM_RL/trl_grpo/Qwen2.5_7b_rl_api/checkpoint-2620"
    # MODEL_PATH = "/data/Qwen2.5-Coder-7B-Instruct"
    # TEST_DATASET_PATH = "/9950backfile/liguoqi/wangzihang/LLM_RL/dataset/rl_test.jsonl"
    TEST_DATASET_PATH = "/9950backfile/liguoqi/wangzihang/LLM_RL/dataset/rl_train.jsonl"
    tokenizer_path = "/9950backfile/liguoqi/wangzihang/LLM_RL/trl_grpo/Qwen2.5_7b_rl_api/checkpoint-2620"
    
    evaluate_model(MODEL_PATH, TEST_DATASET_PATH, tokenizer_path) 