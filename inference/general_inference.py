import torch
from transformers import AutoTokenizer
import os
from peft import PeftModel, PeftConfig
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import sys
# 获取当前文件所在目录的父目录（项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append('../')
# from custom_generate.custom_generate import custom_generate
from step_grpo_trainer import ANSWER_STR_END, ANSWER_STR_START, THINK_STR_START, SequenceStoppingCriteria
from transformers import StoppingCriteria, StoppingCriteriaList

def load_model(base_model,tokenizer_address,checkpoint_path, device="cuda:0", use_lora=False,sink_flag=False):
    """
    加载模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 使用的设备，默认为"cuda:3"
        use_lora: 是否使用LoRA模型
    """
    # 加载分词器
    model_name = base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_address, trust_remote_code=True)
    if sink_flag:
    # 加载基础模型
        from attention_sinks import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attention_sink_size=4,
            attention_sink_window_size=2048,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
    if use_lora:
        # 加载LoRA配置
        config = PeftConfig.from_pretrained(checkpoint_path)
        # 加载LoRA权重
        model = PeftModel.from_pretrained(model, checkpoint_path,config=config)
    # else:
    #     # 加载完整的模型权重
    #     model = load_state_dict_from_zero_checkpoint(model, checkpoint_path)
    
    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, history=None, max_length=2048, temperature=1.0, top_p=0.95, repetition_penalty=1.1, no_repeat_ngram_size=20,custom_generate_flag=False):
    """
    生成回答
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入的问题
        history: 聊天历史记录，格式为[{"role": "user/assistant", "content": "消息内容"}, ...]
        max_length: 最大生成长度
        temperature: 温度参数
        top_p: top-p采样参数
    """
    if history is None:
        history = []
    system_prompt = [{"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and \\boxed{ } tags, respectively, i.e., <think> reasoning process 1 here </think> <think> reasoning process n here  \\boxed{ answer here } </think>. Do not need to add any other text."}]
    message = [{"role": "user", "content": prompt}]
    # message = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}, {"role": "user", "content": " "}, {"role": "assistant", "content": "”她不服气地开口。 仲瑾颐脸上的笑容，彻彻底底消失了。他冷着脸看她"}]
    inputs = tokenizer.apply_chat_template(
        message,
        # return_tensors="pt",
        add_generation_prompt=True,
        return_dict=False,
        tokenize = False
    )
    # inputs = tokenizer.apply_chat_template(
    #     [inputs],
    #     return_tensors="pt",
    #     add_generation_prompt=False,
    #     return_dict=True,
    #     tokenize = True
    # ).to(model.device)
    inputs=tokenizer.encode(inputs,return_tensors="pt",add_special_tokens=False).to(model.device)

    # inputs = tokenizer.apply_chat_template(
    #     message,
    #     return_tensors="pt",
    #     add_generation_prompt=True,
    #     # return_dict=True,
    #     tokenize = False
    # )


    # inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    answer_start_sequence = [tokenizer.encode(ANSWER_STR_START,add_special_tokens=False)[0],tokenizer.encode(ANSWER_STR_START,add_special_tokens=False)[1],
                             tokenizer.encode(ANSWER_STR_START,add_special_tokens=False)[2],
                            ]


    generate_kwargs = {
        "input_ids": inputs,
        # "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_length,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "eos_token_id": tokenizer.eos_token_id,
        # "stopping_criteria": StoppingCriteriaList([
        #                     SequenceStoppingCriteria(answer_start_sequence)
        #                 ])
    }
    if custom_generate_flag:
        # out = custom_generate(model,generate_kwargs,tokenizer)
        response = tokenizer.decode(out,skip_special_tokens=True)
        return response
    else:


        out = model.generate(**generate_kwargs)


    response = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return response

def print_chat_header():
    """打印聊天界面的标题"""
    print("\n" + "="*50)
    print("欢迎使用AI助手！")
    print("- 输入 'quit' 退出对话")
    print("- 输入 'clear' 清除历史记录")
    print("="*50 + "\n")

def main():
    # 设置最大保留的对话轮数
    MAX_HISTORY_TURNS = 5
    sink_flag = False
    use_lora = False
    custom_generate_flag = False
    base_model = '/home/ubuntu/LLM_RL/instruct_fine_tuning/qwen_instruct_lora_finetuned_0311/checkpoint-200'
    tokenizer_address = '../Qwen2.5-Math-1.5B-Instruct'
    if use_lora:
        checkpoint_path = "./instruct_fine_tuning/qwen_instruct_lora_finetuned_0311/checkpoint-200"
    else:
        checkpoint_path = "./glm_finetuned/checkpoint-300"
    
    try:
        model, tokenizer = load_model(base_model, tokenizer_address, checkpoint_path, use_lora=use_lora,sink_flag=sink_flag)
        print_chat_header()
        
        # 用于存储对话历史
        history = []
        
        while True:
            user_input = input("\n用户: ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                history = []
                print("\n已清除历史记录！")
                continue
            
            try:
                
                response = generate_response(model, tokenizer, user_input, history,custom_generate_flag=custom_generate_flag)
                print("\n助手:", response)
                
                # 更新历史记录
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                
                # 如果历史记录过长，只保留最近的几轮对话
                if len(history) > MAX_HISTORY_TURNS * 2:
                    history = history[-MAX_HISTORY_TURNS * 2:]
                    
            except Exception as e:
                print(f"\n生成回答时发生错误: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n对话结束！")

if __name__ == "__main__":
    main() 