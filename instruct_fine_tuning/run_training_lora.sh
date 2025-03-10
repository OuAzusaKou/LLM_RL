#!/bin/bash

# 默认值设置
MODEL_PATH="../Qwen2.5-7B"
TRAIN_DATA="train.json"
TEST_DATA="test.json"
OUTPUT_DIR="./qwen_instruct_lora_finetuned_0115"
LEARNING_RATE=3e-4  # 默认学习率
MAX_LENGTH=2048    # 默认最大序列长度
SCHEDULER="linear" # 默认学习率调度器
LORA_RANK=8       # 默认LoRA秩数
LORA_ALPHA=32     # 默认LoRA Alpha值
MAX_EPOCHS=300    # 默认最大训练轮数
SAVE_STEPS=100    # 默认保存间隔步数

# 处理命令行参数
while getopts ":l:m:t:v:o:x:s:r:a:e:p:" opt; do
    case $opt in
        l) LEARNING_RATE="$OPTARG";;  # 学习率
        m) MODEL_PATH="$OPTARG";;     # 模型路径
        t) TRAIN_DATA="$OPTARG";;     # 训练数据
        v) TEST_DATA="$OPTARG";;      # 验证(测试)数据
        o) OUTPUT_DIR="$OPTARG";;     # 输出目录
        x) MAX_LENGTH="$OPTARG";;    # 最大序列长度
        s) SCHEDULER="$OPTARG";;      # 学习率调度器
        r) LORA_RANK="$OPTARG";;      # LoRA秩数
        a) LORA_ALPHA="$OPTARG";;     # LoRA Alpha值
        e) MAX_EPOCHS="$OPTARG";;     # 最大训练轮数
        p) SAVE_STEPS="$OPTARG";;     # 保存间隔步数
        \?) echo "无效的选项: -$OPTARG" >&2; exit 1;;
        :) echo "选项 -$OPTARG 需要参数" >&2; exit 1;;
    esac
done

# 显示参数信息
echo "使用以下参数:"
echo "模型路径: $MODEL_PATH"
echo "训练数据: $TRAIN_DATA"
echo "测试数据: $TEST_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "学习率: $LEARNING_RATE"
echo "最大序列长度: $MAX_LENGTH"
echo "学习率调度器: $SCHEDULER"
echo "LoRA秩数: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
echo "最大训练轮数: $MAX_EPOCHS"
echo "保存间隔步数: $SAVE_STEPS"

# 使用jq更新ds_config_lora.json中的学习率
jq --arg lr "$LEARNING_RATE" '.optimizer.params.lr = ($lr|tonumber)' ds_config_lora.json > ds_config_lora.json.tmp && mv ds_config_lora.json.tmp ds_config_lora.json
jq --arg lr "$LEARNING_RATE" '.scheduler.params.warmup_max_lr = ($lr|tonumber)' ds_config_lora.json > ds_config_lora.json.tmp && mv ds_config_lora.json.tmp ds_config_lora.json

# 检查必要文件是否存在
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误：训练数据文件 $TRAIN_DATA 不存在" >&2
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "错误：测试数据文件 $TEST_DATA 不存在" >&2
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误：模型目录 $MODEL_PATH 不存在" >&2
    exit 1
fi

# 启动训练
deepspeed --include="localhost:2,3" train_lora.py \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$MAX_EPOCHS" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0.01 \
    --warmup_steps 0 \
    --logging_steps 10 \
    --save_steps "$SAVE_STEPS" \
    --eval_steps 100 \
    --max_length "$MAX_LENGTH" \
    --lr_scheduler_type "$SCHEDULER" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA"
    # --per_device_train_batch_size=2 \
    # --gradient_accumulation_steps=8 \
    # --num_train_epochs=300 \
    # --learning_rate=2e-4 \
    # --fp16 
    # --model_path $MODEL_PATH \
    # --train_data $TRAIN_DATA \
    # --test_data $TEST_DATA \