"""
math_multi_steps_pp2.py
Author: Hongfeng Ai
Function: Gnerate Math Dataset into multi-steps thinking processing.
History:
20250220    Hongfeng Ai v1
"""
import pandas as pd
import json
import os
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from typing import List
from tqdm import tqdm
import concurrent.futures
import math

import sys
sys.path.append("/home/gptdemo/aihongfeng/cot/deepscaler")
from verl.utils import hf_tokenizer
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd


def load_deepscaler_model(model_path:str):
    """load pretrained deepscaler model"""
    tokenizer = hf_tokenizer(model_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype="auto",
                                                 device_map="cuda")
    return tokenizer, model


def deepscaler_answer(query:str,
                      tokenizer,
                      model):
    """Use DeepScaleR generate answer for the given query"""
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    question = f"{query} {instruction}"
    messages = [{"role":"user", "content":question}]
    text = tokenizer.apply_chat_template(messages,
                                            add_generation_prompt=True,
                                            padding=True,
                                            truncation=True,
                                            max_length=1536,
                                            return_tensors='pt',
                                            tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=32768)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
    ]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def deepscaler_answer_batch(queries: List[str], 
                            tokenizer, 
                            model, 
                            batch_size:int=8):
    """Use DeepScaleR batch generate answers for the given querys"""
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    batched_texts = []
    for query in queries:
        question = f"{query} {instruction}"
        messages = [{"role":"user", "content":question}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, padding=True, 
            truncation=True, max_length=1536, return_tensors='pt', 
            tokenize=False
        )
        batched_texts.append(text)
    
    inputs = tokenizer(batched_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=32768)
    
    responses = []
    for i in range(len(queries)):
        generated_ids = outputs[i][len(inputs.input_ids[i]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(response)
    return responses

def check_response(response:str, answer:str):
    """check deepscaler response format
        1. <think> format is right or not
        2. answer is correct or not
    """
    think_process = ""
    # get think process and answer
    # if format error and answer wrong, resample
    pattern = r"<think>(.*?)</think>"
    if '<think>' in response and '</think>' in response:
        think_process = re.findall(pattern, response, re.DOTALL)
        model_solution = response.split('</think>')[1]
    else:
        print('no <think> or </think>')
        return False

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        print('no model_answer:', model_solution)
        return False

    # Process the ground truth(s)
    ground_truths = answer
    if ground_truths is None:
        print("ground_truths is None")
        return False

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]
        
    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        print("processed_ground_truths:", processed_ground_truths)
        return False

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return True
        
    if len(think_process) == 0 or is_correct == False:
        print("len(think_process) == 0 or is_correct == False")
        return False
    
    return True


def llm_generate(query:str):
    """Use LLM generate answer for the given query"""
    client = OpenAI(
        api_key = "94f5cbab-0ef8-4712-a923-3838356ee67f",
        base_url = "https://ark.cn-beijing.volces.com/api/v3",
    )

    # Non-streaming:
    # print("----- standard request -----")
    completion = client.chat.completions.create(
        model = "ep-20250214154816-2qlp9",  # your model endpoint ID
        messages = [
            # {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content

    

def llm_split_think_process(think_process:str,
                            split_step_upper_limit:int=5):
    """split thinking process to multi-step thinking content"""
    if think_process == "":
        return ""
    
    prompt = f"""Please decompose the following total thinking process into multiple smaller thinking steps which has no more than {split_step_upper_limit} steps, and output them in the form of a Python list. Each thinking step you generate should not be modified in the content of the given total thinking process.
    For example:
    - Total thinking process: "total_think"
    - Output: ["think 1", "think 2", "think 3"]

    Now, we have total thinking process as below:

    '
    {think_process}
    '

    please split to small think steps as list format as ["think 1", "think 2", "think 3", ...] with no-more-than {split_step_upper_limit} steps.
    """
    return llm_generate(prompt)


def llm_split_think_process_batch(think_processes: List[str], 
                                  split_step_upper_limit: int = 5):
    """Concurrently process multiple thinking processes using multiple processes."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(llm_split_think_process, process, split_step_upper_limit)
            for process in think_processes
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results


def check_multi_steps_response(multi_think_steps:str,
                               len_of_gt_think:int,
                               split_step_upper_limit:int,
                               token_diff_limit:int):
    """check the generated result of the multi-steps thinking process is right or not"""
    try:
        multi_think_steps_list = eval(multi_think_steps)
        
        # If list has too many steps, break out
        if len(multi_think_steps_list) > split_step_upper_limit:
            print("len(multi_think_steps_list) >= split_step_upper_limit:", len(multi_think_steps_list), split_step_upper_limit)
            return [], False
        
        # if token numbers are so different, break out
        token_diff_num = abs(len(multi_think_steps) - len_of_gt_think)
        if  token_diff_num > token_diff_limit:
            print(f"token numbers are so different with ({len(multi_think_steps)}, {len_of_gt_think}): {token_diff_num}")
            return [], False
        
    except:
        return [], False
        
    return multi_think_steps_list, True


def batch_split_multi_steps(samples:List,
                            tokenizer,
                            model,
                            call_upper_limit:int=3,
                            split_step_upper_limit:int=5,
                            token_diff_limit:int=500,
                            batch_size:int=3
                            ):
    """batch splitting
    
    when multi_steps_status:
        -1: deepscaler response fails with the format check.
        -2: fail to generate right multi-steps thinking process.
        1: success with generating right multi-steps thinking process.
    """
    # 每次取出batch_size个样本
    num_batches = math.ceil(len(samples) / batch_size)

    total_results = []
    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = start + batch_size
        batch = samples[start:min(end, len(samples))]

        queries = [sample['problem'] for sample in batch]

        # get deepscaler thinking process and answer
        responses = deepscaler_answer_batch(queries=queries, 
                                     tokenizer=tokenizer, model=model, 
                                     batch_size=batch_size)
        # check responses' format
        checked_responses = []
        for response, sample in zip(responses, batch):
            # check response's format
            response_is_correct = check_response(response, sample['answer'])
            if response_is_correct:
                sample.update({'response':response,
                               'response_status':1})
            else:
                sample.update({'response':response,
                               'response_status':-1})
            checked_responses.append(sample)

        # extract total thinking process
        think_processes = []
        pattern = r"<think>(.*?)</think>"
        for response in checked_responses:
            if response['response_status'] == -1:
                think_process = ""
            else:
                think_process = re.findall(pattern, response['response'], re.DOTALL)[0]
            think_processes.append(think_process)

        multi_think_steps_responses = llm_split_think_process_batch(think_processes, split_step_upper_limit)

        # check multi_think_steps' format
        checked_multi_responses = []
        for multi_think_steps, think_process, checked_response in zip(multi_think_steps_responses, think_processes, checked_responses):
            multi_think_steps_list, split_status = check_multi_steps_response(multi_think_steps=multi_think_steps,
                                                                        len_of_gt_think=len(think_process),
                                                                        split_step_upper_limit=split_step_upper_limit,
                                                                        token_diff_limit=token_diff_limit)
            if split_status:
                checked_response.update({
                    'multi_steps_response':"".join([f"<think>{step_think}</think>"for step_think in multi_think_steps_list]),
                    'multi_steps_response_status':1,
                                         })
            else:
                checked_response.update({'multi_steps_response':"",
                                         'multi_steps_response_status':-1})
            checked_multi_responses.append(checked_response)   
            
        total_results.append(checked_multi_responses) 

    return total_results


def main():
    # load deepscaler model
    tokenizer, model = load_deepscaler_model(model_path="agentica-org/DeepScaleR-1.5B-Preview")

    with open('/home/gptdemo/aihongfeng/cot/deepscaler/deepscaler/data/train/math.json', 'rb') as f:
        train = json.load(f)

    with open('/home/gptdemo/aihongfeng/cot/deepscaler/deepscaler/data/test/math.json', 'rb') as f:
        test = json.load(f)

    print("train number:", len(train))
    print("test number:", len(test))
    call_upper_limit=2
    split_step_upper_limit=5
    token_diff_limit=500
    batch_size = 2 # parallel numbers

    # TODO if formal run pls change train[:4] to train
    train_steps_results = batch_split_multi_steps(train[:4],
                                                tokenizer,
                                                model,
                                                call_upper_limit,
                                                split_step_upper_limit,
                                                token_diff_limit,
                                                batch_size)
    
    # test_steps_results = batch_split_multi_steps(test,
    #                                         tokenizer,
    #                                         model,
    #                                         call_upper_limit,
    #                                         split_step_upper_limit,
    #                                         token_diff_limit,
    #                                         batch_size)
    
    with open('/home/gptdemo/aihongfeng/cot/deepscaler/deepscaler/data/train/math_multi_steps.json', 'w') as f:
        json.dump(train_steps_results, f, ensure_ascii=False, indent=4)
        
    # with open('/home/gptdemo/aihongfeng/cot/deepscaler/deepscaler/data/test/math_multi_steps.json', 'w') as f:
    #     json.dump(test_steps_results, f, ensure_ascii=False, indent=4)


if __name__=="__main__":
    main()


