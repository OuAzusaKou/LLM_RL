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


def load_model_and_tokenizer(model_path,tokenizer_path=None):
    """加载本地模型和分词器"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='cuda:1',
        torch_dtype=torch.float16
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
            max_length=8196,
            # padding=True
        ).to('cuda:1')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_generate_tokens=500,
            max_new_tokens=50,
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
    results = []
    
    print("开始评估...")
    for item in tqdm(test_data):
        subtype = item['subtype']
        prompt = f"{item['text'][0]}"
        
        # 获取模型回答
        model_response = generate_response(model, tokenizer, prompt)
        
        # 提取预期答案
        expected_answer = "缺陷真实存在" if "缺陷真实存在" in item['text'][1]['content'] else "疑似误报"
        
        # 判断是否正确
        is_correct = ("缺陷真实存在" in model_response) == ("缺陷真实存在" in expected_answer)
        if is_correct:
            total_correct += 1
            subtype_stats[subtype]['correct'] += 1
        
        subtype_stats[subtype]['total'] += 1
        
        # 保存结果
        results.append({
            "input": item['text'][0],
            "expected": expected_answer,
            "predicted": model_response,
            "is_correct": is_correct,
            "subtype": subtype
        })
    
    # 计算总体准确率和每个subtype的准确率
    total_accuracy = total_correct / total
    
    # 计算每个subtype的准确率
    for subtype in subtype_stats:
        stats = subtype_stats[subtype]
        stats['accuracy'] = stats['correct'] / stats['total']
    
    # 保存评估结果
    output = {
        "overall_accuracy": total_accuracy,
        "total_samples": total,
        "correct_predictions": total_correct,
        "subtype_statistics": dict(subtype_stats),
        "detailed_results": results
    }
    
    with open("evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # 绘制subtype准确率图表
    plot_subtype_accuracy(subtype_stats)
    
    print(f"\n评估完成！")
    print(f"总体准确率: {total_accuracy:.2%}")
    print(f"总体正确预测数: {total_correct}/{total}")
    print("\n各类型准确率:")
    for subtype, stats in subtype_stats.items():
        print(f"{subtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print(f"\n详细结果已保存到 evaluation_results.json")
    print(f"准确率图表已保存到 subtype_accuracy.png")

if __name__ == "__main__":
    # 设置模型路径和测试数据集路径
    # MODEL_PATH = "/data/qwen_instruct_lora_finetuned_0418_1rd/checkpoint-180"
    MODEL_PATH = "/public/liguoqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    TEST_DATASET_PATH = "./dataset/test.jsonl"
    tokenizer_path = "/public/liguoqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    
    evaluate_model(MODEL_PATH, TEST_DATASET_PATH, tokenizer_path) 
