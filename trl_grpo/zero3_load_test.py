from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

model_path = "/9950backfile/liguoqi/wangzihang/LLM_RL/trl_grpo/Qwen1.5_rl_0630/bin1040"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 初始化空模型
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_path)

# 自动设备映射（必要）
device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"])

# 加载分片权重
model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map)

# 推理
inputs = tokenizer("讲个关于猫的童话故事", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))