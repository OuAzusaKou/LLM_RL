import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import deepspeed


def parse_args():
    parser = argparse.ArgumentParser(description='LoRA 训练脚本')
    # 添加自定义参数组
    training_args = parser.add_argument_group('training arguments')
    # 基础参数
    training_args.add_argument('--model_path', type=str, default="../Qwen2.5-7B",
                      help='预训练模型的路径')
    training_args.add_argument('--train_data', type=str, default="training_data.json",
                      help='训练数据的JSON文件路径')
    training_args.add_argument('--test_data', type=str, default="test_data.json",
                      help='测试数据的JSON文件路径')
    training_args.add_argument('--output_dir', type=str, default="./qwen_instruct_lora_finetuned_0115",
                      help='输出目录')
    
    # 新增参数
    training_args.add_argument('--max_length', type=int, default=2048,
                      help='最大序列长度')
    training_args.add_argument('--lr_scheduler_type', type=str, default="linear",
                      choices=["linear", "cosine", "constant"],
                      help='学习率调度器类型')
    training_args.add_argument('--lora_rank', type=int, default=8,
                      help='LoRA 秩数')
    training_args.add_argument('--lora_alpha', type=int, default=32,
                      help='LoRA Alpha值')
    
    # 添加训练相关参数
    training_args.add_argument('--num_train_epochs', type=int, default=100,
                      help='最大训练轮数')
    training_args.add_argument('--per_device_train_batch_size', type=int, default=2,
                      help='每个设备的训练批次大小')
    training_args.add_argument('--gradient_accumulation_steps', type=int, default=8,
                      help='梯度累积步数')
    training_args.add_argument('--learning_rate', type=float, default=2e-4,
                      help='学习率')
    training_args.add_argument('--weight_decay', type=float, default=0.01,
                      help='权重衰减')
    training_args.add_argument('--warmup_steps', type=int, default=0,
                      help='预热步数')
    training_args.add_argument('--logging_steps', type=int, default=10,
                      help='日志记录步数')
    training_args.add_argument('--save_steps', type=int, default=100,
                      help='保存模型步数')
    training_args.add_argument('--eval_steps', type=int, default=100,
                      help='评估步数')
    
    # 添加local_rank参数，这是deepspeed需要的
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='local rank passed from distributed launcher')
    
    # 添加deepspeed参数
    parser = deepspeed.add_config_arguments(parser)
    
    # 在parse_args函数中添加新的参数
    training_args.add_argument('--focus_tokens', type=str, nargs='+', default=[],
                      help='需要重点关注的token列表')
    training_args.add_argument('--focus_weight', type=float, default=2.0,
                      help='重点token附近的loss权重')
    training_args.add_argument('--context_window', type=int, default=5,
                      help='重点token影响的上下文窗口大小')
    
    args = parser.parse_args()
    return args

# 在main函数之前添加自定义训练器
class WeightedLossTrainer(Trainer):
    def __init__(self, focus_tokens, focus_weight, context_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 将focus_tokens转换为token_ids
        self.focus_token_ids = []
        for token in focus_tokens:
            ids = self.tokenizer(token, add_special_tokens=False)['input_ids']
            self.focus_token_ids.extend(ids)
        self.focus_weight = focus_weight
        self.context_window = context_window
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        
        # 创建权重矩阵
        weights = torch.ones_like(labels, dtype=torch.float)
        
        # 在batch中查找focus_tokens
        for b in range(labels.shape[0]):  # 遍历batch
            for pos in range(labels.shape[1]):  # 遍历序列
                if labels[b, pos] in self.focus_token_ids:
                    # 为目标token周围的上下文应用权重
                    start = max(0, pos - self.context_window)
                    end = min(labels.shape[1], pos + self.context_window + 1)
                    weights[b, start:end] = self.focus_weight
        
        # 计算加权损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(labels.shape) * weights
        loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss

def main():
    args = parse_args()
    
    # 使用args替换硬编码的参数
    model_path = args.model_path
    train_data = args.train_data
    test_data = args.test_data
    
    # 使用命令行参数初始化模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None  # 使用DeepSpeed时不要设置device_map
    )

    # 配置LoRA，使用命令行参数
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,                     # LoRA 秩
        lora_alpha=args.lora_alpha,          # LoRA alpha参数
        lora_dropout=0.1,       # Dropout概率
        bias="none",
        # use_gradient_checkpointing=True,    # 添加梯度检查点以节省显存
        target_modules=['q_proj', 'v_proj', 'k_proj'],  # 需要根据模型结构调整
        # target_modules=['gate_up_proj','down_proj'],
        # target_modules=['dense_h_to_4h','dense_4h_to_h']
        # target_modules=['query_key_value']
    )

    # 准备模型进行LoRA训练
    model = prepare_model_for_kbit_training(model)
    print(model)
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()

    # 使用命令行参数加载数据集
    dataset = load_dataset("json", data_files={
        "train": train_data,
        "test": test_data
    })
    
    def tokenize_function(examples):
        # 确保输入是列表格式
        if isinstance(examples['messages'], str):
            examples['messages'] = [examples['messages']]
            
        return tokenizer.apply_chat_template(
            examples['messages'],
            return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            # tokenize = False
        )
        # return tokenizer(
        #     examples["text"],
        #     padding="max_length",
        #     truncation=True,
        #     max_length=1024  # 与数据处理保持一致
        # )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        
    )

    # 更新training_args使用命令行参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_pin_memory=False,  # 避免潜在的内存问题
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        deepspeed="ds_config_lora.json",
        fp16=True,
        bf16=False,
        local_rank=int(os.getenv("LOCAL_RANK", -1)),
        ddp_backend="nccl",
    )

    # 使用自定义训练器替换原有的Trainer
    trainer = WeightedLossTrainer(
        focus_tokens=args.focus_tokens,
        focus_weight=args.focus_weight,
        context_window=args.context_window,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,  # 需要添加tokenizer
    )

    # 开始训练
    trainer.train()
    
    # 保存LoRA权重
    model.save_pretrained("./qwen_lora_weights")

if __name__ == "__main__":
    main() 