# train_grpo.py
import os
import sys
from datasets import load_dataset
import re
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
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

from FlagEmbedding import BGEM3FlagModel
from ollama_bge import OllamaBGE
import torch.nn.functional as F
            
            # 加载BGE-M3模型
embedding_encoder = BGEM3FlagModel(model_name_or_path="/9950backfile/liguoqi/wangzihang/bge-m3", batch_size=32)

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

def code_compute_score(solution_str, ground_truth, use_embedding=True, explanation=None, embedding_weight=0.8):
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
    
    # 如果启用嵌入向量相似度计算
    if use_embedding and explanation is not None:

        # 计算嵌入向量
        response_embedding = embedding_encoder.encode(model_explanation)['dense_vecs']
        explanation_embedding = embedding_encoder.encode(explanation)['dense_vecs']
        
        # 计算余弦相似度
        similarity = response_embedding @ explanation_embedding.T
        print('similarity', similarity)
        # 结合基础分数和相似度分数
        final_score = (1 - embedding_weight) * base_score + embedding_weight * similarity
        
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
    output_dir="./Qwen2.5_1.5b_rl_test",
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
    # bf16=True,  # 启用混合精度训练
    fp16=True,
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