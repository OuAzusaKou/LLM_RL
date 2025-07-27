# train_grpo.py
import os
import sys
from datasets import load_dataset
import re
sys.path.append('/9950backfile/liguoqi/wangzihang/LLM_RL/trl_grpo')
from trl import GRPOConfig
from grpo_code_trainer import GRPOTrainer
from FlagEmbedding import BGEM3FlagModel
import torch
import torch.nn.functional as F

os.environ["WANDB_MODE"] = "offline"
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径
sys.path.append(current_dir)
print(sys.path)

# 加载数据集
train_data = '/9950backfile/liguoqi/wangzihang/LLM_RL/dataset/rl_train.jsonl'
test_data = '/9950backfile/liguoqi/wangzihang/LLM_RL/dataset/rl_test.jsonl'
dataset = load_dataset("json", data_files={
        "train": train_data,
        "test": test_data
    })

# 全局变量存储模型实例
embedding_model = None

def get_embedding_model(trainer):
    model = BGEM3FlagModel(
        model_name_or_path="/path/to/your/local/bge-m3",  # 替换为你的本地模型路径
        use_fp16=True,
        device=trainer.accelerator.device
    )
    return trainer.accelerator.prepare_model(model, evaluation_mode=True)

def get_embedding(text, trainer):
    global embedding_model
    if embedding_model is None:
        embedding_model = get_embedding_model(trainer)
    with torch.no_grad():
        embedding = embedding_model.encode(text)
    return embedding

def code_compute_score(solution_str, ground_truth):
    # 修改正则表达式以匹配第一个##到第二个##之间的所有内容
    match = re.search(r'^.*?##\s*(.*?)\s*##', solution_str, re.DOTALL)

    if not match:
        return 0  # 未找到匹配项，返回 0

    model_response = match.group(1)
    
    expected_answer = "缺陷真实存在" if "缺陷真实存在" in ground_truth else "缺陷不存在"
        
    # 判断是否正确
    is_correct = ("缺陷真实存在" in model_response) == ("缺陷真实存在" in expected_answer)
    
    return 1.0 if is_correct else 0.0

def reward_score(completions, **kwargs):
    scores = []
    for i, completion in enumerate(completions):
        solution_str = completion[0]['content']
        ground_truth = kwargs['reward_model'][i]['ground_truth']
        
        # 调用简化后的函数
        score = code_compute_score(solution_str, ground_truth)
        
        scores.append(score)
    return scores

# 更新训练配置
training_args = GRPOConfig(
    output_dir="./Qwen7_rl_0610",
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
    save_strategy="steps",
    ddp_backend="nccl",
    save_only_model = True,
    save_safetensors = True,
    save_steps=20,
    save_total_limit=2,
    bf16=True,  # 启用混合精度训练
    num_train_epochs=50
)

# 设置 wandb 离线模式的环境变量


trainer = GRPOTrainer(
    model = "/public/liguoqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    reward_funcs=reward_score,
    args=training_args,
    train_dataset=dataset['train'],
    embedding_model_path="/9950backfile/liguoqi/wangzihang/bge-m3"  # 替换为你的本地embedding模型路径
)

if __name__ == "__main__":
    trainer.train()