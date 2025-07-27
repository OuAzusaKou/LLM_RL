# train_grpo.py
import os
import sys
from datasets import load_dataset
import re
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from openai import OpenAI
# from verl.utils.reward_score.math import compute_score
# from verl.utils.reward_score.gsm8k import compute_score
# 获取当前文件所在的目录
os.environ["WANDB_MODE"] = "offline"
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径
sys.path.append(current_dir)
print(sys.path)
# 加载数据集
# dataset = load_dataset("parquet", data_files="/data/math/train.parquet", split="train")
# dataset = load_dataset("parquet", data_files="/data/gsm8k/train.parquet", split="train")
train_data = '/9950backfile/liguoqi/wangzihang/LLM_RL/dataset/rl_train.jsonl'
test_data = '/9950backfile/liguoqi/wangzihang/LLM_RL/dataset/rl_test.jsonl'
dataset = load_dataset("json", data_files={
        "train": train_data,
        "test": test_data
    })

# from FlagEmbedding import BGEM3FlagModel
# from ollama_bge import OllamaBGE
import torch.nn.functional as F

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-LjjwKIT8YaKGAY4uk2aYmXT98x1n2JI9fMhyB3EqAIZcMd7F',
)
            
            # 加载BGE-M3模型
# embedding_encoder = BGEM3FlagModel(model_name_or_path="/9950backfile/liguoqi/wangzihang/bge-m3", batch_size=32)

    # 编码单个文本

# 如果有多个parquet文件，可以这样加载
# dataset = load_dataset("parquet", data_files={
#     "train": "path/to/train.parquet",
#     "validation": "path/to/validation.parquet"
# })

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]
def fantasy_compute_score(solution_str: str, ground_truth: str) -> int:
    # 提取 '####' 后的字母
    match = re.search(r'##\s*([a-zA-Z]+)', solution_str)
    if not match:
        return 0  # 未找到匹配项，返回 0
    extracted_text = match.group(1).lower()
    return 1.0 if extracted_text == ground_truth.lower() else 0

def get_semantic_similarity_openai(text1, text2, client):
    """
    使用OpenAI API计算两个文本的语义相似度
    返回0-1之间的分数
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的文本相似度评估专家。请分析两个文本的语义相似度，并给出0到1之间的分数，其中0表示完全不相似，1表示完全相同。请只返回数字，不要其他解释。"
                },
                {
                    "role": "user",
                    "content": f"请评估以下两个文本的语义相似度，给出0-1之间的分数：\n\n文本1：{text1}\n\n文本2：{text2}"
                }
            ],
            temperature=0.1,
            max_completion_tokens=10
        )
        
        # 提取分数
        score_text = response.choices[0].message.content
        try:
            score = float(score_text)
            # 确保分数在0-1范围内
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            print(f"Warning: Could not parse similarity score '{score_text}'. Using fallback score.")
            return 0.5
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return 0.5  # 出错时返回中等相似度

def code_compute_score(solution_str, ground_truth, use_openai=True, explanation=None, similarity_weight=0.8):
    # 修改正则表达式以匹配第一个##到第二个##之间的所有内容
    match = re.search(r'^.*?##\s*(.*?)\s*##', solution_str, re.DOTALL)

    if not match:
        return 0  # 未找到匹配项，返回 0

    model_response = match.group(1)

    # 匹配'相关事件:'之后的所有内容
    explanation_match = re.search(r'相关事件:\s*(.*)', solution_str, re.DOTALL)
    if explanation_match:   
        model_explanation = explanation_match.group(1)
    else:
        model_explanation = "1"
    
    expected_answer = "缺陷真实存在" if "缺陷真实存在" in ground_truth else "缺陷不存在"
        
    # 判断是否正确
    is_correct = ("缺陷真实存在" in model_response) == ("缺陷真实存在" in expected_answer)
    
    base_score = 1.0 if is_correct else 0.0
    
    # 如果启用OpenAI语义相似度计算
    if use_openai and explanation is not None:
        # 使用OpenAI API计算语义相似度
        similarity = get_semantic_similarity_openai(model_explanation, explanation, client)
        print('OpenAI similarity', similarity)
        
        # 结合基础分数和相似度分数
        final_score = (1 - similarity_weight) * base_score + similarity_weight * similarity
        
        return final_score
    else:
        return base_score

def reward_score(completions, **kwargs):
    scores = []
    for i, completion in enumerate(completions):
        solution_str = completion[0]['content']
        ground_truth = kwargs['reward_model'][i]['ground_truth']
        
        # 检查是否提供了explanation
        explanation = kwargs['reward_model'][i].get('explanation', None)
        print('explanation', explanation)
        print('solution_str', solution_str)
        print('ground_truth', ground_truth)
        
        # 调用修改后的函数
        score = code_compute_score(
            solution_str, 
            ground_truth, 
            explanation=explanation
        )
        
        scores.append(score)
    return scores

# 更新训练配置
training_args = GRPOConfig(
    output_dir="./Qwen2.5_7b_rl_api",
    logging_steps=10,
    use_vllm=True,
    per_device_train_batch_size=2,
    num_generations=14,
    max_prompt_length=1024,
    max_completion_length=1024,
    gradient_accumulation_steps=5,
    ds3_gather_for_generation = True,
    vllm_gpu_memory_utilization = 0.6,
    # 添加DeepSpeed相关配置
    learning_rate=3e-7,
    save_strategy="steps",
    ddp_backend="nccl",
    save_only_model = True,
    # save_safetensors = True,
    save_steps=20,
    save_total_limit=2,
    beta=0.1,
    # deepspeed="ds_config.json",  # 指向DeepSpeed配置文件
    bf16=True,  # 启用混合精度训练
    # fp16=True,
    num_train_epochs=2,
)

# processing_class = AutoTokenizer.from_pretrained("/public/liguoqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")


trainer = GRPOTrainer(
    # processing_class=processing_class,
    # model="/data/Qwen2-7B-Instruct",
    # model="/data/Qwen2.5-1.5B-Instruct",
    model = "/public/liguoqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    # model = "/9950backfile/liguoqi/wangzihang/qwen_instruct_lora_finetuned_0418_1rd/checkpoint-180",
    # model = "/9950backfile/liguoqi/wangzihang/LLM_RL/trl_grpo/Qwen1.5_rl_0630/bin1040",
    # model = "/public/liguoqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    # model="/home/ubuntu/Qwen2.5-7B-Insctruct",
    reward_funcs=reward_score,
    args=training_args,
    train_dataset=dataset['train'],
    
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./Qwen2.5_model_1epoch")