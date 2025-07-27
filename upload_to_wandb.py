import wandb
import json

# 初始化 wandb
wandb.init(
    project="qwen-instruct-lora",  # 项目名称
    name="finetuning-run-0317",    # 实验名称
    config={
        "model": "Qwen",
        "dataset": "instruct",
        "learning_rate": "3e-5",
    }
)

# 读取训练状态文件
with open('/data/qwen_instruct_lora_finetuned_0317/checkpoint-420/trainer_state.json') as f:
    data = json.load(f)

# 上传训练日志
for log in data['log_history']:
    wandb.log(log)

wandb.finish() 