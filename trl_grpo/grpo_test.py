# train_grpo.py
from datasets import load_dataset
import re
from trl import GRPOConfig, GRPOTrainer
from verl.utils.reward_score.math import compute_score
# from verl.utils.reward_score.gsm8k import compute_score

# 加载数据集
dataset = load_dataset("parquet", data_files="/home/ubuntu/deepscaler/data/train.parquet", split="train")
# dataset = load_dataset("parquet", data_files="/data/gsm8k/train.parquet", split="train")
# train_data = '/home/ubuntu/LLM_RL/trl_grpo/fantasy_data_test.json'
# dataset = load_dataset("json", data_files={
#         "train": train_data,
#     })
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
    match = re.search(r'####\s*([a-zA-Z]+)', solution_str)
    if not match:
        return 0  # 未找到匹配项，返回 0
    extracted_text = match.group(1).lower()
    return 1.0 if extracted_text == ground_truth.lower() else 0

def reward_score(completions, **kwargs):
    scores = []
    for i,completion in enumerate(completions):
        solution_str = completion[0]['content']
        ground_truth = kwargs['reward_model'][i]['ground_truth']
        # print('solution',solution_str)
        # print('ground_truth',ground_truth)
        score = compute_score(solution_str, ground_truth)
        # score = fantasy_compute_score(solution_str,ground_truth)
        # print(score)
        scores.append(score)
    return scores

# 更新训练配置
training_args = GRPOConfig(
    output_dir="/data/deepseek-r1_rl",
    logging_steps=10,
    use_vllm=True,
    per_device_train_batch_size=8,
    num_generations=14,
    max_prompt_length=1024,
    max_completion_length=1024,
    gradient_accumulation_steps=5,
    temperature=1.3,
    # ds3_gather_for_generation = False,
    vllm_gpu_memory_utilization = 0.6,
    # 添加DeepSpeed相关配置
    save_strategy="steps",
    ddp_backend="nccl",
    save_only_model = True,
    save_safetensors = True,
    save_steps=20,
    save_total_limit=2,
    # deepspeed="ds_config.json",  # 指向DeepSpeed配置文件
    bf16=True,  # 启用混合精度训练
    num_train_epochs=50
)

trainer = GRPOTrainer(
    # model="/data/Qwen2-7B-Instruct",
    # model="/data/Qwen2.5-1.5B-Instruct",
    model = "/home/ubuntu/.cache/huggingface/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # model="/home/ubuntu/Qwen2.5-7B-Insctruct",
    reward_funcs=reward_score,
    args=training_args,
    train_dataset=dataset,
)

if __name__ == "__main__":
    trainer.train()