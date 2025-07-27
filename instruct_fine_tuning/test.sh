import deepspeed
import torch

# 确认DeepSpeed是否能识别GPU
print("CUDA available:", torch.cuda.is_available())
print("DeepSpeed version:", deepspeed.__version__)

# 初始化DeepSpeed引擎
model = torch.nn.Linear(10, 10).cuda()
engine, *_ = deepspeed.initialize(model=model, config={"train_batch_size": 1})
print("Engine initialized successfully")
