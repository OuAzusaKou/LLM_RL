from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset
from step_grpo_trainer import StepGRPOTrainer,THINK_STR_START,THINK_STR_END,ANSWER_STR_START,ANSWER_STR_END
import torch
from typing import List
import os
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from datasets import Features, Value, Sequence

def setup_model_and_tokenizer(model_name: str):
    """设置模型和分词器，并应用LoRA
    
    Args:
        model_name: 模型名称或路径
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # 配置LoRA
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
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
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
    step_reward = len(steps) * 0.1  # 每个步骤给0.1的奖励
    # answer_reward = min(len(answer) / 100, 0.5)  # 答案长度奖励，最大0.5
    if len(steps) >= 4:
        step_reward = 1
    else:
        step_reward = 0
    other_reward = 0
    for i in range(4,len(others)):
        other_reward_buf = len(others[i]) * 0.05
        if abs(other_reward_buf) > 0.5:
            other_reward_buf =  - 0.5
        other_reward += other_reward_buf

    answer_reward = 0
    if len(answer) > 2:
        answer_reward = 2
    if answer[-1] == labels:
        answer_reward += 10
    

    return step_reward + answer_reward + other_reward

# custom_features = Features({
#     'input_ids': Sequence(Value('int32')),  # 输入ID序列
#     'attention_mask': Sequence(Value('int32')),  # 注意力掩码序列
#     'la': Sequence(Value('string')),  # 原始答案文本（如果是文本生成任务）
# })



def tokenize_function(example, tokenizer, max_length=512):
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
    """自定义数据整理函数
    
    Args:
        features: 一个batch的数据样本列表
    """
    if not isinstance(features, list):
        features = [features]
        
    # 收集batch中的所有样本
    input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
    attention_mask = torch.stack([torch.tensor(f["attention_mask"]) for f in features])
    labels = [f["labels"] for f in features]  # 保持原始文本格式
    
    # 构建batch
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    return batch

def main():
    # 配置参数
    model_name =  "../../DeepSeek-R1-Distill-Qwen-7B/" #"/home/ps/.cache/huggingface/hub/models--agentica-org--DeepScaleR-1.5B-Preview/snapshots/24a92eff29154a702a928249812162644208ac5b/"
    output_dir = "output"
    num_train_epochs = 3
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 1
    learning_rate = 2e-4  # 对于LoRA可以使用稍大的学习率
    max_steps = 500
    num_samples = 2
    max_generation_steps = 4
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和分词器（现在包含LoRA）
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        # remove_unused_columns=False,
        fp16=True,  # 使用混合精度训练
        optim="paged_adamw_32bit",  # 使用8位优化器
    )
    
    # 初始化训练器
    trainer = StepGRPOTrainer(
        reward_function=reward_function,
        num_samples=num_samples,
        max_steps=max_generation_steps,
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
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
