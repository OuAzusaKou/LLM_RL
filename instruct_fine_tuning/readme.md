# 指令微调训练和评估工具

本项目提供了模型指令微调训练、评估和可视化的完整工具链，支持LoRA高效微调和全参数微调两种模式。

## 目录结构

```
dataset/  # 数据处理相关代码
  ├── convert_and_split.py  # 数据转换和分割程序
  └── output/              # 原始数据目录
instruct_fine_tuning/  # 训练相关代码
  ├── train_lora.py    # LoRA训练主程序
  ├── run_training_lora.sh  # 训练启动脚本
  └── readme.md        # 使用说明
eval/  # 评估相关代码
  ├── model_evaluation.py  # 模型评估程序
  └── plot_subtype_acc.py  # 结果可视化程序
```

## 数据处理说明

### 数据转换和分割

`convert_and_split.py`程序用于将原始数据转换为训练所需的格式，并进行训练集和测试集的分割。

#### 主要功能

1. 数据格式转换
- 将原始JSON格式转换为模型训练所需的对话格式
- 保留缺陷类型(subtype)信息
- 结构化用户输入和助手回答

2. 数据集分割
- 使用分层抽样方法进行训练集和测试集划分
- 确保每种缺陷类型在测试集中至少有一个样本
- 支持自定义测试集比例(默认20%)

#### 使用方法

```bash
python dataset/convert_and_split.py
```

#### 参数配置

在`convert_and_split.py`中可以修改以下参数：

```python
input_dir = "./dataset/output"  # 原始数据目录
converted_file = "./dataset/converted_data.jsonl"  # 转换后的数据文件
train_file = "./dataset/train.jsonl"  # 训练集输出路径
test_file = "./dataset/test.jsonl"  # 测试集输出路径
test_ratio = 0.2  # 测试集比例
```

#### 输入数据格式要求

原始数据应为JSON格式，包含以下字段：
```json
{
    "instruction": [{"subtype": "缺陷类型"}],
    "input": {
        "task": "任务描述",
        "event": "事件描述",
        "output_format": "输出格式要求"
    },
    "output": {
        "judgement": "判断结果",
        "reason": "判断原因"
    }
}
```

#### 输出数据格式

转换后的数据为JSONL格式，每行一个样本：
```json
{
    "subtype": "缺陷类型",
    "text": [
        {
            "role": "user",
            "content": "任务：{task}\n\n事件：\n{event}\n\n输出格式：\n{output_format}"
        },
        {
            "role": "assistant",
            "content": "判断:{judgement}\n\n原因：{reason}"
        }
    ]
}
```

## 训练程序使用说明

### 1. LoRA微调训练

训练程序支持使用LoRA方法进行高效微调，可以通过修改`run_training_lora.sh`脚本中的默认参数来配置训练。

#### 主要参数说明

在`run_training_lora.sh`中可以修改以下默认参数：

```bash
# 默认参数配置
MODEL_PATH="/data/Qwen2.5-Coder-7B-Instruct/"      # 基础模型路径
TRAIN_DATA="../dataset/train.jsonl"                # 训练数据路径
TEST_DATA="../dataset/test.jsonl"                  # 测试数据路径
OUTPUT_DIR="/data/qwen_instruct_lora_finetuned_0401"  # 输出目录
LEARNING_RATE=3e-5                                 # 学习率
MAX_LENGTH=4096                                    # 最大序列长度
SCHEDULER="linear"                                 # 学习率调度器
LORA_RANK=32                                      # LoRA秩数
LORA_ALPHA=32                                     # LoRA Alpha值
MAX_EPOCHS=300                                    # 最大训练轮数
SAVE_STEPS=20                                     # 保存间隔步数
PER_DEVICE_TRAIN_BATCH_SIZE=1                     # 每设备训练批次大小
GRADIENT_ACCUMULATION_STEPS=8                     # 梯度累积步数
```

#### 使用方法

直接运行训练脚本：
```bash
./run_training_lora.sh
```

如需修改训练参数，请直接编辑`run_training_lora.sh`文件中的默认值。


### 2. 训练程序核心功能

`train_lora.py`是训练的核心程序，主要功能包括：

- 支持LoRA和全参数两种微调模式
- 自定义数据处理和标签生成
- 支持DeepSpeed分布式训练
- 支持重点token加权训练
- 自动评估和保存最佳模型
- 集成Weights & Biases (wandb)训练监控

### 3. 训练监控与模型选择

#### Weights & Biases 可视化

训练过程会自动使用wandb进行监控和可视化，可以通过以下步骤查看：

1. 训练开始时会显示wandb链接：
```
wandb: 🚀 View run at https://wandb.ai/<username>/huggingface/runs/<run_id>
```

2. 点击链接可以实时查看：
- 训练损失(train_loss)
- 评估损失(eval_loss)
- 学习率变化
- GPU利用率等指标

#### 最佳模型选择

训练过程中会自动保存检查点，建议选择eval_loss最低的模型进行测试：

1. 在wandb界面找到eval_loss最低的检查点编号
2. 对应的模型保存在`OUTPUT_DIR/checkpoint-{step}`目录下
3. 在评估时使用该检查点路径：
```python
# 使用eval_loss最低的检查点进行评估
MODEL_PATH = "/data/qwen_instruct_lora_finetuned_0401/checkpoint-140"  # 示例路径
```


## 模型评估程序

### 使用方法

```bash
python eval/model_evaluation.py
```

评估程序会加载指定的模型和测试数据集，生成模型回答并与标准答案比较，计算准确率。

#### 主要参数

在`model_evaluation.py`文件中修改以下参数：

```python
# 模型路径
MODEL_PATH = "/data/qwen_instruct_lora_finetuned_0401/checkpoint-140"
# 测试数据集路径
TEST_DATASET_PATH = "./dataset/test.jsonl"
# 分词器路径
tokenizer_path = "/data/Qwen2.5-Coder-7B-Instruct"
```

### 评估结果

评估程序会生成以下输出：

1. `evaluation_results.json` - 包含详细评估结果的JSON文件
2. `subtype_accuracy.png` - 各缺陷类型准确率可视化图表

## 结果可视化程序

### 使用方法

```bash
python eval/plot_subtype_acc.py
```

可视化程序可以比较两个模型的评估结果，生成对比图表。

#### 主要参数

在`plot_subtype_acc.py`文件中修改以下参数：

```python
result1_path = "/home/ubuntu/LLM_RL/evaluation_results_ori_0403.json"  # 基线模型结果
result2_path = "/home/ubuntu/LLM_RL/evaluation_results.json"  # 改进模型结果
```

### 可视化输出

可视化程序会生成以下图表：

1. `model_comparison.png` - 两个模型在各缺陷类型上的准确率对比
2. `overall_accuracy_comparison.png` - 两个模型的总体准确率对比
3. `type_accuracy_comparison.png` - 两个模型在不同类型样本上的准确率对比

## 数据格式要求

训练和测试数据应为JSONL格式，每行包含一个JSON对象，格式如下：

```json
{"text": [{"role": "user", "content": "用户输入..."}, {"role": "assistant", "content": "助手回答..."}], "subtype": "缺陷类型"}
```

## 注意事项

1. 确保已安装所有依赖库：transformers, torch, deepspeed, peft等
2. 使用DeepSpeed训练需要正确配置`ds_config_lora.json`文件
