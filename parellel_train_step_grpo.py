from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import load_dataset
from step_grpo_trainer import StepGRPOTrainer,THINK_STR_START,THINK_STR_END,ANSWER_STR_START,ANSWER_STR_END
import torch
from typing import List
import datetime
import argparse
import torch.distributed as dist
from deepspeed.comm import init_distributed
import subprocess
import accelerate

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from datasets import Features, Value
from peft import PeftModel, PeftConfig

def setup_model_and_tokenizer(model_name: str,load_lora:bool=False,checkpoint_path:str='checkpoint-10500'):
    """设置模型和分词器，并应用LoRA
    
    Args:
        model_name: 模型名称或路径
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None
    )
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # 配置LoRA
    if not load_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # LoRA秩
            lora_alpha=32,  # LoRA alpha参数
            lora_dropout=0.1,  # Dropout概率
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj"],  # 需要根据具体模型架构调整
        )
        
        # 准备模型进行训练
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    else:
        # 加载LoRA配置
        config = PeftConfig.from_pretrained(checkpoint_path)
        # 加载LoRA权重
        model = PeftModel.from_pretrained(model, checkpoint_path,config=config)
        # model = PeftModel.from_pretrained(model, checkpoint_path)
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = True
        model.train()
    # 应用LoRA

    model.print_trainable_parameters()  # 打印可训练参数比例
    
    return model, tokenizer



def reward_function(steps: List[str], answer: List[str], others: List[str], labels: str) -> float:
    """计算奖励值的函数
    
    Args:
        steps: 思考步骤列表
        answer: 最终答案
        
    Returns:
        reward: 奖励值
    """
    # 这里实现您的奖励计算逻辑
    # 示例：根据步骤数和答案长度计算简单奖励
    base_reward = 5
    step_reward = len(steps) * 0.1  # 每个步骤给0.1的奖励
    # answer_reward = min(len(answer) / 100, 0.5)  # 答案长度奖励，最大0.5
    if len(steps) > 0 and len(steps) < 4:
        step_reward = len(steps) * 1
    other_reward = 0
    for i in range(0,len(others)):
        other_reward_buf =  - (len(others[i]) * 0.05)
        if abs(other_reward_buf) > 0.5:
            other_reward_buf =  - 0.5
        other_reward += other_reward_buf
    if other_reward <= (-4.9):
        other_reward = -4.9

    answer_reward = 0
    if len(answer) == 1:
        answer_reward = 2
        if labels[0] in answer[-1]:
            answer_reward += 10
        

    return base_reward + step_reward + answer_reward + other_reward

# custom_features = Features({
#     'input_ids': Sequence(Value('int32')),  # 输入ID序列
#     'attention_mask': Sequence(Value('int32')),  # 注意力掩码序列
#     'la': Sequence(Value('string')),  # 原始答案文本（如果是文本生成任务）
# })



def tokenize_function(example, tokenizer, max_length=2048):
    """自定义数据处理函数 - 单样本处理版本"""
    # 获取对话
    conversation = example['text']

    # 分离输入消息
    input_messages = conversation[:2]  # system和user消息
    label_message = conversation[2]    # assistant消息
    
    # 处理输入部分
    input_tokens = tokenizer.apply_chat_template(
        [input_messages],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    
    # 确保label_message['content']不为None
    content = label_message['content'] if label_message['content'] is not None else ""
    # 构建返回数据
    return {
        "input_ids": input_tokens['input_ids'][0],
        "attention_mask": input_tokens['attention_mask'][0],
        # "label": content  # 原始答案文本
        "labels":content
    }

def custom_data_collator(features):
    # print(f"Collator received features length: {len(features)}")  # 调试信息
    """自定义数据整理函数
    
    Args:
        features: 一个batch的数据样本列表
    """
    # 强制确保只处理一个样本

        
    # 收集batch中的样本
    input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
    attention_mask = torch.stack([torch.tensor(f["attention_mask"]) for f in features])
    labels = [f["labels"] for f in features]
    
    # 构建batch
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    return batch

def main():
    # 配置参数
    #  "../../DeepSeek-R1-Distill-Qwen-7B/"
    model_name =  "../Qwen2.5-Math-1.5B-Instruct/" #"/home/ps/.cache/huggingface/hub/models--agentica-org--DeepScaleR-1.5B-Preview/snapshots/24a92eff29154a702a928249812162644208ac5b/"
    output_dir = "output"
    num_train_epochs = 3
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 1
    learning_rate = 2e-5  # 对于LoRA可以使用稍大的学习率
    max_steps = 500
    num_samples = 3
    max_generation_steps = 4
    load_lora = True
    max_generate_tokens = 300
    save_steps = 10
    checkpoint_path = '../checkpoint-10500'
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    # 定义一个函数来获取和打印NCCL设置
    def print_nccl_settings():
        print("\n=== NCCL Settings ===")
        nccl_vars = ["NCCL_TIMEOUT", "NCCL_DEBUG", "NCCL_SOCKET_NTHREADS", 
                     "NCCL_NSOCKS_PERTHREAD", "NCCL_P2P_DISABLE"]
        for var in nccl_vars:
            value = os.getenv(var, "Not set")
            print(f"{var}: {value}")
        
        # 如果是分布式训练，打印PyTorch分布式设置
        if dist.is_initialized():

            print("\n=== PyTorch Distributed Settings ===")
            print(f"World Size: {dist.get_world_size()}")
            print(f"Rank: {dist.get_rank()}")
            print(f"Backend: {dist.get_backend()}")
            try:
                timeout = dist.get_timeout()
                print(f"Communication Timeout: {timeout}")
            except:
                print("Unable to get communication timeout")
        print("========================\n")
    
    # 初始化分布式环境
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    if local_rank != -1:
        # 使用 DeepSpeed 的初始化方法
        init_distributed(timeout=datetime.timedelta(seconds=7200))
        
        # 设置 NCCL 超时
        os.environ["NCCL_TIMEOUT"] = "7200"
        os.environ["NCCL_DEBUG"] = "INFO"
        
        # 打印分布式设置信息
        print_nccl_settings()

    # 加载模型和分词器
    model, tokenizer = setup_model_and_tokenizer(model_name, load_lora=load_lora, checkpoint_path=checkpoint_path)
    
    # 将模型移动到对应的设备
    if local_rank != -1:
        model = model.to(f'cuda:{local_rank}')

    # 加载数据集
    train_data = 'math_training_data.json'#'/home/ps/Documents/deepscaler/LLM_RL/math_training_data.json'
    test_data = 'math_test_data.json'#'/home/ps/Documents/deepscaler/LLM_RL/math_test_data.json'

    # 加载数据集时禁用缓存
    dataset = load_dataset(
        "json", 
        data_files={
            "train": train_data,
            "test": test_data
        },
        cache_dir=None,  # 禁用缓存
        # features=custom_features  # 使用自定义特征
    )

    # 修改数据集处理方式
    train_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False  # 强制重新处理
    )
    # print(train_dataset.features)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=10,
        weight_decay=0.01,
        save_steps=save_steps,
        save_total_limit=3,
        save_strategy='steps',
        fp16=True,
        bf16=False,
        dataloader_num_workers=4,
        ddp_timeout=7200,
        deepspeed="ds_config.json",
        local_rank=local_rank,
        ddp_backend="nccl",
        # gradient_checkpointing=True,  # 添加梯度检查点以节省显存
    )
    
    # 创建一个自定义的save_model方法
    def save_full_model(self, output_dir):
        """保存完整的模型，包括基础模型和LoRA权重"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取完整模型
        model = self.model
        # 如果是PeftModel，先合并权重
        if isinstance(model, PeftModel):
            merged_model = model.merge_and_unload()
        else:
            merged_model = model
            
        # 先保存配置
        merged_model.config.save_pretrained(output_dir)
        
        # 然后保存模型权重
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"  # 将大模型分片保存
        )
    
    # 修改StepGRPOTrainer的_save_checkpoint方法来保存完整模型
    original_save = StepGRPOTrainer._save_checkpoint
    def new_save_checkpoint(self, model, trial):
        # 调用原始的保存方法
        original_save(self, model, trial)
        # 创建checkpoint目录路径
        checkpoint_dir = os.path.join(output_dir, f"base_model-{self.state.global_step}")
        
        # 保存完整模型
        save_full_model(self, checkpoint_dir)

    # save full model
    StepGRPOTrainer._save_checkpoint = new_save_checkpoint
    
    # 初始化训练器
    trainer = StepGRPOTrainer(
        reward_function=reward_function,
        num_samples=num_samples,
        max_steps=max_generation_steps,
        max_generate_tokens=max_generate_tokens,
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        kl_coef=0,
        data_collator=custom_data_collator,  # 使用自定义的data_collator
    )
    
    # 测试生成过程
    # print("测试生成过程...")
    # test_input = {
    #     "input_ids": torch.tensor(train_dataset['train'][0]["input_ids"]).unsqueeze(0).to(device),
    #     "attention_mask": torch.tensor(train_dataset['train'][0]["attention_mask"]).unsqueeze(0).to(device),
    # }
    # # trainer.test_generation(model, test_input)
    # test_output = trainer.model(test_input['input_ids'])
    # print(test_output)
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

if __name__ == "__main__":
    main()
