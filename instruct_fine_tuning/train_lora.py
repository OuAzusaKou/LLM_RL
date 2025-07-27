import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
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
    training_args.add_argument('--model_path', type=str, default="/data/Qwen2.5-Coder-7B-Instruct/",
                      help='预训练模型的路径')
    training_args.add_argument('--train_data', type=str, default="/home/ubuntu/LLM_RL/dataset/train.jsonl",
                      help='训练数据的JSON文件路径')
    training_args.add_argument('--test_data', type=str, default="/home/ubuntu/LLM_RL/dataset/test.jsonl",
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
    training_args.add_argument('--learning_rate', type=float, default=3e-05,
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
        # 在parse_args函数中添加新的参数
    training_args.add_argument('--focus_tokens', type=str, nargs='+', default=[],
                      help='需要重点关注的token列表')
    training_args.add_argument('--focus_weight', type=float, default=2.0,
                      help='重点token附近的loss权重')
    training_args.add_argument('--context_window', type=int, default=5,
                      help='重点token影响的上下文窗口大小')
    
    # 添加deepspeed参数
    parser = deepspeed.add_config_arguments(parser)
    
    # 添加是否使用LoRA的参数
    training_args.add_argument('--use_lora', action='store_true',
                      help='是否使用LoRA进行微调')
    training_args.add_argument('--target_modules', type=str, nargs='+', 
                      default=['q_proj', 'v_proj', 'k_proj'],
                      help='LoRA目标模块')
    
    args = parser.parse_args()
    return args


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
    labels = torch.stack([torch.tensor(f["labels"]) for f in features])
    
    # 构建batch
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    return batch

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
        
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=0):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # 创建权重矩阵
        weights = torch.ones_like(shift_labels, dtype=torch.float)
        
        # 在batch中查找focus_tokens
        for b in range(labels.shape[0]):  # 遍历batch
            for pos in range(labels.shape[1]):  # 遍历序列
                if labels[b, pos] in self.focus_token_ids:
                    # 为目标token周围的上下文应用权重
                    start = max(0, pos - self.context_window)
                    end = min(labels.shape[1], pos + self.context_window + 1)
                    weights[b, start:end] = self.focus_weight
        
        # 计算加权损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape) * weights
        
        # 只对非零元素计算平均值
        non_zero_mask = (loss != 0)
        if non_zero_mask.sum() > 0:  # 确保有非零元素
            loss = loss.sum() / non_zero_mask.sum()
        else:
            loss = loss.mean()  # 如果全为零，返回普通平均值
        
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
        model_path,      # 预训练模型的路径
        torch_dtype=torch.float16,    # 使用float16精度加载模型以节省显存
        trust_remote_code=True,       # 信任远程代码，允许加载自定义模型代码
        device_map=None  # 使用DeepSpeed时不要设置device_map
    )

    # 修改模型加载和训练准备部分
    if args.use_lora:
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
            r=args.lora_rank,              # LoRA的秩，决定了低秩矩阵的维度
            lora_alpha=args.lora_alpha,    # LoRA的缩放参数，用于控制适应强度
            lora_dropout=0.1,              # LoRA层的dropout率，用于防止过拟合
            bias="none",                   # 是否训练偏置项，none表示不训练
            target_modules=args.target_modules,  # 需要应用LoRA的目标模块名称列表
        )

        # 准备模型进行LoRA训练
        model = prepare_model_for_kbit_training(model)
        print(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # 全参数微调模式
        print("使用全参数微调模式")
        # 遍历模型的所有参数
        for param in model.parameters():
            # 将每个参数设置为可训练状态
            param.requires_grad = True
        # 计算所有可训练参数的总数
        # numel()返回张量中元素的总数
        # 只统计requires_grad=True的参数
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数总数: {total_params}")

    # 使用命令行参数加载数据集
    dataset = load_dataset("json", data_files={
        "train": train_data,
        "test": test_data
    })
    
    def tokenize_function(examples):
        # 确保输入是列表格式
        if isinstance(examples['text'], str):
            examples['text'] = [examples['text']]

        try:
            input_tokens = tokenizer.apply_chat_template(
                examples['text'],
                return_tensors="pt",
                add_generation_prompt=False,
                return_dict=True,
                truncation=False,  # 修改为False以便检查长度
                max_length=args.max_length,
            )
            
            # 获取input_ids
            input_ids = input_tokens.data['input_ids'][0]
            
            # 检查长度是否超过最大长度
            if len(input_ids) > args.max_length:
                print(f"样本长度 {len(input_ids)} 超过最大长度 {args.max_length}，跳过该样本")
                return None
            
            # print('input_ids:', examples['text'])
            # 创建labels，初始化为-100
            labels = torch.full_like(input_ids, -100)
            
            # 查找assistant标记的位置
            assistant_start = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
            assistant_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            
            # 找到所有assistant开始和结束的位置
            current_pos = 0
            while True:
                # 查找下一个assistant开始位置
                start_pos = -1
                for i in range(current_pos, len(input_ids) - len(assistant_start) + 1):
                    if all(input_ids[i + j] == assistant_start[j] for j in range(len(assistant_start))):
                        start_pos = i + len(assistant_start)
                        break
                
                if start_pos == -1:
                    # print("没有找到assistant开始位置")
                    break
                    
                # 查找对应的结束位置
                end_pos = -1
                for i in range(start_pos, len(input_ids) - len(assistant_end) + 1):
                    if all(input_ids[i + j] == assistant_end[j] for j in range(len(assistant_end))):
                        end_pos = i
                        break
                
                if end_pos == -1:
                    # print("没有找到assistant结束位置")
                    end_pos = len(input_ids)
                
                # 设置labels
                labels[start_pos:end_pos] = input_ids[start_pos:end_pos]
                current_pos = end_pos + len(assistant_end)
        
            return {
                "input_ids": input_ids,
                "attention_mask": input_tokens.data['attention_mask'][0],
                "labels": torch.tensor(labels)
            }
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            return None

    # 对数据集进行tokenize处理,将文本转换为模型可以理解的token序列
    tokenized_dataset = dataset.map(
        # 使用自定义的tokenize函数处理每个样本
        tokenize_function,
        # batched=False表示一次处理一个样本,而不是批量处理
        batched=False,
        # 移除原始数据集中的所有列,只保留tokenize后的结果
        remove_columns=dataset["train"].column_names,
        # 禁用缓存,每次都重新处理数据
        load_from_cache_file=False,  # 强制重新处理
        # 可选的过滤函数,用于过滤掉None值的样本
        # filter_fn=lambda x: x is not None  # 添加过滤函数
    )

    # 更新training_args使用命令行参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,                    # 模型和训练日志的输出目录
        num_train_epochs=args.num_train_epochs,        # 训练的总轮数
        per_device_train_batch_size=args.per_device_train_batch_size,    # 每个设备的训练批次大小
        per_device_eval_batch_size=args.per_device_train_batch_size,     # 每个设备的评估批次大小
        gradient_accumulation_steps=args.gradient_accumulation_steps,     # 梯度累积步数，用于增大实际批次大小
        learning_rate=args.learning_rate,              # 学习率
        weight_decay=args.weight_decay,                # 权重衰减，用于L2正则化
        lr_scheduler_type="cosine",                    # 学习率调度器类型，使用余弦退火
        logging_steps=args.logging_steps,              # 每多少步记录一次日志
        save_steps=args.save_steps,                    # 每多少步保存一次模型
        eval_steps=args.eval_steps,                    # 每多少步进行一次评估
        evaluation_strategy="steps",                   # 评估策略，按步数进行评估
        load_best_model_at_end=True,                  # 训练结束时加载最佳模型
        metric_for_best_model="loss",                 # 用于选择最佳模型的指标
        deepspeed="ds_config_lora.json",              # DeepSpeed配置文件路径
        fp16=True,                                    # 是否启用16位浮点数训练
        local_rank=int(os.getenv("LOCAL_RANK", -1)),  # 分布式训练的本地进程序号
        ddp_backend="nccl",                           # 分布式训练后端
        save_total_limit=3,                           # 最多保存的检查点数量
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],  # 将test_dataset改为eval_dataset
        # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        # data_collator=custom_data_collator,
    )

    # trainer = WeightedLossTrainer(
    #     focus_tokens=args.focus_tokens,
    #     focus_weight=args.focus_weight,
    #     context_window=args.context_window,
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["test"],
    #     data_collator=custom_data_collator,
    #     tokenizer=tokenizer,  # 需要添加tokenizer
    # )



    # 开始训练
    # 定义一个自定义的回调函数来记录训练过程中的指标
    # class CustomCallback(TrainerCallback):
    #     def on_step_end(self, args, state, control, model, logs=None, **kwargs):
    #         if logs is not None and "loss" in logs:
    #             # 这里可以添加你想要记录的自定义指标
    #             wandb.log({
    #                 "custom_error_category_2": logs["loss"] * 1.5,  # 示例:记录类别2的error
    #                 "current_lr": trainer.optimizer.param_groups[0]["lr"]
    #             })

    # # 初始化wandb用于记录指标
    # wandb.init(project="qwen_training_metrics")
    
    # # 添加自定义回调
    # trainer.add_callback(CustomCallback())
    
    # 定义一个字典来存储每个类别的loss历史记录
    # subtype_loss_history = {i: [] for i in range(20)}  # 20个类别的loss历史
    # iteration_history = []  # 记录iteration次数

    # # 定义测试集评估的回调函数
    # class SubtypeLossCallback(TrainerCallback):
    #     def __init__(self, eval_dataset, trainer):
    #         self.eval_dataset = eval_dataset
    #         self.trainer = trainer
    #         self.current_iteration = 0
        
    #     def on_step_end(self, args, state, control, **kwargs):
    #         self.current_iteration += 1
            
    #         # 计算每个类别的loss
    #         subtype_losses = {i: 0.0 for i in range(20)}
    #         subtype_counts = {i: 0 for i in range(20)}
            
    #         # 遍历测试集
    #         for item in self.eval_dataset:
    #             # 获取样本的subtype
    #             subtype = item['subtype'] if 'subtype' in item else 0
                
    #             # 使用trainer的compute_loss计算单个样本的loss
    #             inputs = self.trainer.data_collator([item])
    #             inputs = {k: v.to(self.trainer.model.device) for k, v in inputs.items()}
    #             loss = self.trainer.compute_loss(self.trainer.model, inputs)
                
    #             # 累加该类别的loss
    #             subtype_losses[subtype] += loss.item()
    #             subtype_counts[subtype] += 1
            
    #         # 计算每个类别的平均loss并记录
    #         for subtype in range(20):
    #             if subtype_counts[subtype] > 0:
    #                 avg_loss = subtype_losses[subtype] / subtype_counts[subtype]
    #                 subtype_loss_history[subtype].append(avg_loss)
            
    #         # 记录当前iteration
    #         iteration_history.append(self.current_iteration)
            
    #         # 可以选择将数据记录到wandb
    #         # if wandb.run is not None:
    #         #     for subtype in range(20):
    #         #         if subtype_counts[subtype] > 0:
    #         #             wandb.log({f'subtype_{subtype}_loss': subtype_losses[subtype] / subtype_counts[subtype]},
    #         #                     step=self.current_iteration)

    # # 添加回调函数到trainer
    # trainer.add_callback(SubtypeLossCallback(tokenized_dataset["test"], trainer))
    
    # 开始训练
    trainer.train()
    
    # 训练结束后，可以使用matplotlib绘制loss曲线
    # import matplotlib.pyplot as plt
    
    # plt.figure(figsize=(15, 10))
    # for subtype in range(20):
    #     if len(subtype_loss_history[subtype]) > 0:
    #         plt.plot(iteration_history, subtype_loss_history[subtype], label=f'Subtype {subtype}')
    
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss by Subtype over Training Iterations')
    # plt.legend()
    # plt.savefig('subtype_loss_curves.png')
    # plt.close()
    
    # 保存LoRA权重
    model.save_pretrained("./qwen_lora_weights")

if __name__ == "__main__":
    main() 