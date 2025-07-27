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
model_path = 'Qwen/Qwen2.5-1.5B-Instruct'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype='auto',

    device_map='auto'  # 使用DeepSpeed时不要设置device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

# 加载基础模型
