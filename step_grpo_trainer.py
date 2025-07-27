import numpy as np
from transformers import Trainer
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import re
# from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
THINK_STR_START = '<think>'
THINK_STR_END = '</think>'
ANSWER_STR_START = '\\boxed{'
ANSWER_STR_END = '}'

if is_apex_available():
    from apex import amp
import datetime
from transformers import AutoModelForCausalLM
import copy


from transformers import StoppingCriteria, StoppingCriteriaList




class ThoughtNode:
    """思考树的节点类"""
    def __init__(self, text, output, score, input_length, parent=None):
        self.text = text  # 节点的文本内容
        self.output = output  # 生成这个节点的模型输出
        self.score = score  # 生成这个节点的模型输出
        self.input_length = input_length
        self.sample_weight = 0
        self.labels = None
        # self.grad_logits = grad_logits
        self.value = 0
        self.advantage = 0  # 添加advantage属性
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.is_answer = False  # 是否是答案节点
    
    def add_child(self, child):
        """添加子节点"""
        self.children.append(child)
        child.parent = self

# Get the per-token log probabilities for the completions for the model and the reference model
def get_per_token_logps(model, input_ids, num_logits_to_keep):
    # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


class StepGRPOTrainer(Trainer):
    def __init__(self, reward_function, num_samples=5, max_steps=5, max_generate_tokens=100, kl_coef=0,*args, **kwargs):
        """初始化训练器
        
        Args:
            reward_function: 奖励计算函数
            num_samples: 每步采样次数
            max_steps: 最大步骤数
            kl_coef: KL散度的权重系数
        """
        super().__init__(*args, **kwargs)
        self.reward_function = reward_function
        self.num_samples = num_samples
        self.max_steps = max_steps
        self.max_generate_tokens = max_generate_tokens
        self.kl_coef = kl_coef
        
        # 创建日志文件
        self.log_file = open('training_loss.log', 'a')
        
        # 加载reference模型
        if kl_coef != 0:
            print("Creating reference model...")
            # 深度复制当前模型
            self.ref_model = copy.deepcopy(self.model).to(self.model.device)
            
            # 冻结reference模型
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()

        self.answer_start_sequence = [self.tokenizer.encode(ANSWER_STR_START,add_special_tokens=False)[0],
                             self.tokenizer.encode(ANSWER_STR_START,add_special_tokens=False)[1],
                             self.tokenizer.encode(ANSWER_STR_START,add_special_tokens=False)[2]]

    def __del__(self):
        """析构函数，确保日志文件被正确关闭"""
        if hasattr(self, 'log_file'):
            self.log_file.close()

    def _parse_generation(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """解析生成的文本，分离思考步骤、最终答案和其他文本
        
        Args:
            text: 生成的文本
            
        Returns:
            steps: 思考步骤列表
            answers: 最终答案列表
            others: 非匹配部分和嵌套文本列表
        """
        # 初始化结果列表
        steps = []
        answers = []
        others = []
        
        # 找到第一个"assistant\n"后的文本
        assistant_start = text.find("assistant\n")
        if assistant_start != -1:
            # 从assistant后开始处理文本
            text = text[assistant_start + len("assistant\n"):]
        
        # 按顺序找出所有标记
        all_matches = []
        
        # 查找所有思考步骤
        think_pattern = f'{THINK_STR_START}.*?{THINK_STR_END}'
        think_matches = list(re.finditer(think_pattern, text, re.DOTALL))
        for match in think_matches:
            all_matches.append(('think', match.start(), match.end(), match.group()))
        
        # 查找答案部分
        answer_pattern = r'\\boxed'+f'.*?{ANSWER_STR_END}'
        answer_matches = list(re.finditer(answer_pattern, text, re.DOTALL))
        for match in answer_matches:
            all_matches.append(('answer', match.start(), match.end(), match.group()))
        
        # 按开始位置排序
        all_matches.sort(key=lambda x: x[1])
        
        # 按顺序处理所有匹配项
        current_pos = 0
        for match_type, start, end, content in all_matches:
            # 检查非匹配部分并加入到others
            if current_pos < start:
                non_match = text[current_pos:start].strip()
                if non_match:
                    others.append(non_match)
            
            # 根据类型将内容添加到对应列表
            if match_type == 'think':
                steps.append(content)
            else:
                answers.append(content)
            
            current_pos = end
        
        # 处理最后的非匹配部分
        if current_pos < len(text):
            final_non_match = text[current_pos:].strip()
            if final_non_match:
                others.append(final_non_match)
        
        return steps, answers, others

    def sample_generate_and_adv_cal(self, model, inputs, return_outputs=False, num_items_in_batch=1,exploration_sample_flag=False):
        """计算训练损失，每一步都进行多次采样
        
        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
        """
        # 获取输入数据
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        batch_size = input_ids.shape[0]
        if exploration_sample_flag:
            exploration_sample = []
            for i in range(batch_size):
                exploration_sample.append(inputs.get("exploration_sample")[i])
        
        # input_str = self.tokenizer.decode(input_ids[0])

        # split_positions = (input_ids[0] == 151644).nonzero(as_tuple=True)[0]
        # if len(split_positions) >= 2:
        #     # 获取第二个分隔符的位置
        #     second_split_pos = split_positions[2]
        #     # 截取前两组内容
        #     input_ids = input_ids[:, :second_split_pos+3]
        #     attention_mask = attention_mask[:, :second_split_pos+3]


        
        # labels = input_str.split('<|im_start|>')[3].split('assistant\n')[1].split('<|im_end|>\n')[0]

        total_loss = 0
        
        # 存储所有思考树的根节点
        all_trees = []
        
        for i in range(batch_size):
            # 获取单个样本
            sample_input_ids = input_ids[i,:].view(1,-1)
            sample_attention_mask = attention_mask[i,:].view(1,-1)
            
            query_texts = self.tokenizer.decode(sample_input_ids[0], skip_special_tokens=False)
            
            r_node = ThoughtNode(query_texts, sample_input_ids[0], 0,len(sample_input_ids[0]))


            # 生成第一步的多个样本
            # attention_mask在generate中得效果是什么？后续并未使用。
            first_step_output = model.generate(
                input_ids=sample_input_ids,
                attention_mask=sample_attention_mask,
                max_new_tokens=self.max_generate_tokens,
                num_return_sequences=self.num_samples,
                do_sample=True,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([
                            SequenceStoppingCriteria(self.answer_start_sequence)
                        ]),
                eos_token_id=[self.tokenizer.encode(THINK_STR_START,add_special_tokens=False)[-1],
                              self.tokenizer.encode(self.tokenizer.eos_token,add_special_tokens=False)[-1],151643]
            )
            if exploration_sample_flag:
                # to list append(exploration_sample_information)
                first_step_output = exploration_sample[i]
            # logits = model(first_step_output.sequences).logits
            
            # grad_logits = logits[:, :-1, :]
            
            first_step_texts = self.tokenizer.batch_decode(first_step_output.sequences, skip_special_tokens=False)
            
            first_step_texts = [self._clean_special_tokens_at_end(text) for text in first_step_texts]

            first_scores = first_step_output.logits

            first_input_length = sample_input_ids.shape[1]
            # 对第一步的每个样本继续生成
            # tolist 第一次为根节点，现在第一次思考为根节点了。
            # modified in range(self.num_samples)
            for first_step_idx,first_step_text in enumerate(first_step_texts):
                # 创建根节点
                score = []
                for i in range(len(first_scores)):
                    score.append(first_scores[i][first_step_idx])

                root = ThoughtNode(first_step_texts[first_step_idx], first_step_output[0][first_step_idx], score,first_input_length)
                all_trees.append(root)
                r_node.add_child(root)
                # 初始化当前分支的队列，现在包含节点对象
                branch_queue = [root]
                
                while branch_queue:
                    current_node = branch_queue.pop(0)
                    current_text = current_node.text
                    
                    # 如果当前文本已包含answer或达到最大步数，则生成答案
                    if (ANSWER_STR_START in current_text[len(query_texts):]) or (len(current_text[len(query_texts):].split(THINK_STR_END)) >= self.max_steps) \
                        or (len(current_text[len(query_texts):]) > 1200) or (len(current_text[len(query_texts):].split(THINK_STR_START)) >= self.max_steps):
                        # 生成最终答案
                        if current_text.endswith(ANSWER_STR_START):
                            final_input_ids = self.tokenizer.encode(
                                current_text,
                                return_tensors="pt",
                                add_special_tokens=False
                                ).to(input_ids.device)
                        else:
                            final_input_ids = self.tokenizer.encode(
                                current_text,
                                return_tensors="pt",
                                add_special_tokens=False
                            ).to(input_ids.device)

                        final_output = model.generate(
                            input_ids=final_input_ids,
                            max_new_tokens=self.max_generate_tokens,
                            num_return_sequences=self.num_samples,
                            do_sample=True,
                            output_scores=True,
                            output_logits=True,
                            return_dict_in_generate=True,
                            pad_token_id=self.tokenizer.pad_token_id,

                            eos_token_id=[
                                          self.tokenizer.encode(self.tokenizer.eos_token,add_special_tokens=False)[-1],151643]
                        )
                        final_scores = final_output.logits

                        # logits = model(final_output.sequences).logits
            
                        # grad_logits = logits[:, :-1, :]

                        final_texts = self.tokenizer.batch_decode(final_output.sequences, skip_special_tokens=False)
                        
                        final_texts = [self._clean_special_tokens_at_end(text) for text in final_texts]

                        final_input_length = final_input_ids.shape[1]
                        # 为每个答案创建叶子节点
                        for sample_num, final_text in enumerate(final_texts):
                            final_score = []
                            for i in range(len(final_scores)):
                                final_score.append(final_scores[i][sample_num])

                            
                            answer_node = ThoughtNode(final_text, final_output[0][sample_num], final_score,final_input_length)
                            
                            answer_node.is_answer = True
                            
                            answer_node.labels = labels

                            current_node.add_child(answer_node)

                        continue
                        
                    
                    # 准备输入
                    if current_text.endswith(THINK_STR_START):
                        current_input_ids = self.tokenizer.encode(
                            current_text,
                            return_tensors="pt",
                            add_special_tokens=False
                        ).to(input_ids.device)
                    else:
                        current_input_ids = self.tokenizer.encode(
                            current_text,
                            return_tensors="pt",
                            add_special_tokens=False
                        ).to(input_ids.device)
                        
                    # 为当前步骤生成多个样本
                    next_outputs = model.generate(
                        input_ids=current_input_ids,
                        max_new_tokens=self.max_generate_tokens,
                        num_return_sequences=self.num_samples,
                        do_sample=True,
                        output_scores=True,
                        output_logits=True,
                        return_dict_in_generate=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        stopping_criteria=StoppingCriteriaList([
                            SequenceStoppingCriteria(self.answer_start_sequence)
                        ]),
                        eos_token_id=[self.tokenizer.encode(THINK_STR_START,add_special_tokens=False)[-1],
                              self.tokenizer.encode(self.tokenizer.eos_token,add_special_tokens=False)[-1],151643]
                    )
                    
                    if exploration_sample_flag:
                        # to do list append(exploration_sample_information)
                        next_outputs.append(exploration_sample[i])
                    # logits = model(next_outputs.sequences).logits
            
                    # grad_logits = logits[:, :-1, :]


                    next_texts = self.tokenizer.batch_decode(next_outputs.sequences, skip_special_tokens=False)
                    # 清理每个生成文本的末尾特殊token
                    next_texts = [self._clean_special_tokens_at_end(text) for text in next_texts]
                    
                    current_scores = next_outputs.logits
                    current_input_length = current_input_ids.shape[1]
                    # 为每个生成的样本创建新节点并加入队列
                    for sample_num, next_text in enumerate(next_texts):
                        score = []
                        for i in range(len(current_scores)):
                            score.append(current_scores[i][sample_num])


                        new_node = ThoughtNode(next_text, next_outputs[0][sample_num], score,current_input_length)
                        current_node.add_child(new_node)
                        branch_queue.append(new_node)
                
                # 计算损失时遍历树结构
                def compute_node_value(node, accumulated_outputs=[]):
                    node.value = 0
                    node.sample_weight = 0
                    if node.is_answer:
                        # 到达叶子节点，计算整条路径的损失
                        path_text = node.text
                        # path_outputs = accumulated_outputs + [node.output]
                        steps, answer,others = self._parse_generation(path_text)
                        reward = self.reward_function(steps, answer, others,node.labels) 
                        node.value = reward

                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # 记录loss到日志文件
                        self.log_file.write(f"{current_time} - Reward: {reward}\n")
                        self.log_file.flush()  # 确保立即写入文件


                        return reward
                    
                    else:
                        prob_list = []
                        for child in node.children:
                            # to do 用score 去和logits mask
                            child_logits = child.score
                            child_logits = torch.stack(child_logits, dim=0)
                            child_probs = torch.nn.functional.softmax(child_logits, dim=-1)

                            # 在log空间计算序列概率
                            sequence_log_prob = torch.tensor(0.0).to(input_ids.device)
                            for i,pos in enumerate(range(child.input_length,child.output.shape[0])):
                                if child.output[pos]==self.tokenizer.eos_token_id or child.output[pos]==151643:
                                    break
                                token_id = child.output[pos]
                                token_prob = child_probs[i, token_id]
                                prob_list.append(token_prob)
                                sequence_log_prob += torch.log(token_prob)
                            
                            # 计算child_value并保持在log空间
                            child_value = compute_node_value(child, [])
                            log_value = sequence_log_prob + torch.log(torch.tensor(child_value + 1e-10))  # 添加小值防止log(0)
                            
                            # if torch.isnan(log_value).any() or torch.isinf(log_value).any():
                            #     print(log_value)
                            #     raise ValueError("NaN or Inf detected in log_value")
                            # 收集所有子节点的log空间值
                            if not hasattr(node, 'log_values'):
                                node.log_values = []
                            node.log_values.append(log_value)
                            
                            # 累积sample_weight (在log空间)
                            if not hasattr(node, 'log_weights'):
                                node.log_weights = []
                            node.log_weights.append(sequence_log_prob)
                        
                        # 使用log-sum-exp技巧计算最终的value
                        if hasattr(node, 'log_values') and node.log_values:
                            # 找到最大log值
                            max_log_value = max(node.log_values)
                            
                            # 计算总和（使用log-sum-exp技巧）
                            log_sum = max_log_value + torch.log(
                                sum(torch.exp(log_value - max_log_value) 
                                    for log_value in node.log_values)
                            )
                            
                            # 计算权重和（同样使用log-sum-exp）
                            max_log_weight = max(node.log_weights)
                            log_weight_sum = max_log_weight + torch.log(
                                sum(torch.exp(log_weight - max_log_weight) 
                                    for log_weight in node.log_weights)
                            )
                            
                            # 计算最终的value
                            node.value = torch.exp(log_sum - log_weight_sum)
                            node.sample_weight = torch.exp(log_weight_sum)
                            
                            # 清理临时属性
                            del node.log_values
                            del node.log_weights
                        else:
                            node.value = 0
                            node.sample_weight = 0

                        if node.value>50 or torch.isnan(node.value).any() or torch.isinf(node.value).any():
                            print(node.value)
                            raise ValueError("NaN or Inf detected in value")
                        # 计算当前节点的总损失
                        # total_subtree_loss = sum(child.value for child in node.children)

                        return node.value
                
                # 计算每棵树的损失
                for tree in all_trees:
                    compute_node_value(tree)
                    # 计算完value后，计算advantage
                    compute_advantage(tree)
                    



        # 计算平均损失
        # avg_loss = total_loss / (len(all_trees) * batch_size)
        
        # return (avg_loss, all_trees) if return_outputs else avg_loss
        return all_trees
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        torch.cuda.empty_cache()
        print('training_step',torch.cuda.memory_summary())
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        self.optimizer.zero_grad()
        inputs = self._prepare_inputs(inputs)
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)
        
        trees = self.sample_generate_and_adv_cal(model, inputs, num_items_in_batch=num_items_in_batch)
        node_num = 0
        
        # rd = np.random.randn(0,1)

        average_loss = 0
        del inputs
        for tree in trees:
            node = tree 
            node_deque = []
            node_deque.append(node)
            while node_deque:
                node = node_deque.pop(0)
                if node.children:
                    node_deque.extend(node.children)
                
                with self.compute_loss_context_manager():
                    # loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                    node_num += 1
                    # rd = np.random.randn(0,1)
                    # if rd > 0.5 and rd < 0.7:
                    #     continue
                    # else:
                    loss = self.compute_node_loss(model,node)
                    average_loss += loss
                    # 或者方式3：同时检查

                # 
                if (
                    self.args.torch_empty_cache_steps is not None
                    and self.state.global_step % self.args.torch_empty_cache_steps == 0
                ):
                    if is_torch_xpu_available():
                        torch.xpu.empty_cache()
                    elif is_torch_mlu_available():
                        torch.mlu.empty_cache()
                    elif is_torch_musa_available():
                        torch.musa.empty_cache()
                    elif is_torch_npu_available():
                        torch.npu.empty_cache()
                    elif is_torch_mps_available(min_version="2.0"):
                        torch.mps.empty_cache()
                    else:
                        torch.cuda.empty_cache()

                kwargs = {}
                if not loss.requires_grad:
                    continue
                # For LOMO optimizers you need to explicitly use the learnign rate
                if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                    kwargs["learning_rate"] = self._get_learning_rate()

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    # Finally we need to normalize the loss for reporting
                    if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                        loss = loss / self.args.gradient_accumulation_steps

                    self.accelerator.backward(loss, **kwargs)

                    for name, param in model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    print(f"Warning: NaN or Inf detected in gradients for {name}")
                                    print(f"grad stats: min={param.grad.min()}, max={param.grad.max()}")
                        
                        # 清除梯度以准备下一次计算
                        # model.zero_grad()
                    
            # 在计算完loss后记录到日志文件
                # 获取当前时间
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 记录loss到日志文件
                self.log_file.write(f"{current_time} - Loss: {loss.detach().cpu().item()}\n")
                self.log_file.flush()  # 确保立即写入文件

            
        

        # print(loss)
        # 在训练步骤结束时销毁所有树
        for tree in trees:
            destroy_tree(tree)
        del trees
        average_loss = average_loss / node_num
        return average_loss.detach()
    
    def compute_node_loss(self, model, node):
        """计算单个节点的损失，包括策略梯度损失和KL散度
        
        Args:
            model: 模型
            node: 节点
        """
        # 计算当前模型的logits
        logits_new = model(node.output.view(1,-1)).logits[:,:-1,:]
        probs = torch.nn.functional.softmax(logits_new, dim=-1)
        if self.kl_coef != 0:
            if self.ref_model.device=='cpu':
                self.ref_model = self.ref_model.to(logits_new.device)
                self.ref_model.eval()
        # 计算reference模型的logits
        if self.kl_coef != 0:   
            with torch.no_grad():
                ref_logits = self.ref_model(node.output.to(logits_new.device).view(1,-1)).logits[:,:-1,:]
                ref_probs = torch.nn.functional.softmax(ref_logits, dim=-1)
        
        token_prob_list = []
        token_prob_olden_list = []
        token_prob_ref_list = []
        # sequence_log_new = torch.tensor(0.0).to(logits_new.device)
        # kl_div_sum = torch.tensor(0.0).to(logits_new.device)
        loss_num = 0
        
        # sequence_log_ref = torch.tensor(0.0).to(ref_probs.device)
        # 计算序列概率和KL散度
        for i,pos in enumerate(range(node.input_length-1,node.output.shape[0]-1)):
            if node.output[pos+1]==self.tokenizer.eos_token_id or node.output[pos+1]==151643:
                break
            token_id = node.output[pos+1]
            if self.kl_coef != 0:
                token_prob_ref = ref_probs[0, pos, token_id]
                token_prob_ref_list.append(torch.log(token_prob_ref))
                # sequence_log_ref += torch.log(token_prob_ref)
            # 计算当前token的概率
            token_prob = probs[0, pos, token_id]
            token_prob_list.append(torch.log(token_prob))
            # token_prob = child_probs[pos, token_id]
            # sequence_log_new += torch.log(token_prob)
        # sequence_log_old = torch.tensor(0.0).to(logits_new.device)
        logits_old = node.score
        logits_old = torch.stack(logits_old, dim=0)
        old_probs = torch.nn.functional.softmax(logits_old, dim=-1)
        # sequence_log_old = torch.tensor(0.0).to(old_probs.device)
        for i,pos in enumerate(range(node.input_length,node.output.shape[0])):
            if node.output[pos]==self.tokenizer.eos_token_id or node.output[pos]==151643:
                break
            token_id = node.output[pos]
            token_prob = old_probs[i, token_id]
            token_prob_olden_list.append(torch.log(token_prob))
            # sequence_log_old += torch.log(token_prob)
            loss_num += 1
        if loss_num != 0:
            ratio_loss = torch.exp(torch.stack(token_prob_list) - torch.stack(token_prob_olden_list)) * node.advantage
            if self.kl_coef != 0:
                kl_div_loss = torch.exp(torch.stack(token_prob_ref_list) - torch.stack(token_prob_list)) - \
                (torch.stack(token_prob_ref_list) - torch.stack(token_prob_list)) - 1
                loss = - (ratio_loss - self.kl_coef * kl_div_loss)
            else:
                loss = - ratio_loss
            
            loss = loss.sum(dim=-1) / loss_num
            # 检查loss是否有nan/inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Warning: NaN or Inf detected in loss")
                # print(f"sequence_log_new: {sequence_log_new}")
                # print(f"sequence_log_old: {sequence_log_old}")
                print(f"advantage: {node.advantage}")
                print(f"loss: {loss}")
                raise ValueError("NaN or Inf detected in loss")
        else:
            loss = torch.tensor(0.0).to(logits_new.device)

        # print(torch.cuda.memory_summary())
        del token_prob_list
        del token_prob_olden_list
        del token_prob_ref_list

        return loss
    def _compute_step_loss(self, outputs, step_idx: int, reward: float):
        """计算单个步骤的损失
        
        这个方法需要根据具体的损失计算方法来实现
        
        Args:
            outputs: 模型输出
            step_idx: 步骤索引
            reward: 奖励值
            
        Returns:
            loss: 该步骤的损失值
        """
        raise NotImplementedError("请实现具体的步骤损失计算方法")

    def test_generation(self, model, inputs):
        """测试生成过程，打印每个步骤的采样结果
        
        Args:
            model: 模型
            inputs: 输入数据
        """
        # 获取输入数据
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        print("输入文本:")
        print(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
        print("\n" + "="*50 + "\n")
        
        # 获取单个样本
        sample_input_ids = input_ids[0:1]
        sample_attention_mask = attention_mask[0:1]
        
        query_texts = self.tokenizer.decode(sample_input_ids[0], skip_special_tokens=True)
        # 存储所有采样路径
        all_paths = []
        all_path_outputs = []
        
        # 生成第一步的多个样本
        print(f"第1步 ({self.num_samples}个采样):")
        first_step_output = model.generate(
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask,
            max_new_tokens=500,
            num_return_sequences=self.num_samples,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.encode(THINK_STR)[0],self.tokenizer.encode(ANSWER_STR)[0]]
        )
        
        first_step_texts = self.tokenizer.batch_decode(first_step_output.sequences, skip_special_tokens=True)
        for idx, text in enumerate(first_step_texts):
            print(f"采样 {idx + 1}:")
            print(text)
            print("-" * 30)
        
        # 对第一步的一个样本继续生成（为了演示，我们只跟踪第一个样本）
        current_text = first_step_texts[0]
        current_outputs = [first_step_output]
        
        step_count = 1
        while "answer" not in current_text[len(query_texts):] and step_count < self.max_steps:
            step_count += 1
            print(f"\n第{step_count}步 ({self.num_samples}个采样):")
            
            # 准备输入
            current_input_ids = self.tokenizer.encode(
                current_text + "think",
                return_tensors="pt",
                add_special_tokens=False
            ).to(input_ids.device)
            
            # 为当前步骤生成多个样本
            next_outputs = model.generate(
                input_ids=current_input_ids,
                max_new_tokens=500,
                num_return_sequences=self.num_samples,
                do_sample=True,
                output_scores=True,
                # output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.tokenizer.encode("think")[0],self.tokenizer.encode("answer")[0]]
            )
            
            next_texts = self.tokenizer.batch_decode(next_outputs.sequences, skip_special_tokens=True)
            
            # 显示所有采样结果
            for idx, next_text in enumerate(next_texts):
                print(f"采样 {idx + 1}:")
                print(next_text)
                print("-" * 30)
            
            # 继续跟踪第一个采样
            current_text = next_texts[0]
            current_outputs.append(next_outputs)
        
        # 生成最终答案
        print("\n生成最终答案:")
        final_input_ids = self.tokenizer.encode(
            current_text + "answer",
            return_tensors="pt",
            add_special_tokens=False
        ).to(input_ids.device)
        
        final_output = model.generate(
            input_ids=final_input_ids,
            max_new_tokens=500,
            num_return_sequences=1,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        final_text = self.tokenizer.decode(final_output.sequences[0], skip_special_tokens=True)
        print(final_text)
        
        # 解析最终结果
        print("\n" + "="*50)
        print("最终结果解析:")
        steps, answer = self._parse_generation(final_text)
        print("\n思考步骤:")
        for idx, step in enumerate(steps):
            print(f"步骤 {idx + 1}: {step}")
        print("\n最终答案:", answer)

    def _clean_special_tokens_at_end(self, text: str) -> str:
        """清理文本末尾的特殊token
        
        Args:
            text: 需要清理的文本
            
        Returns:
            清理后的文本
        """
        # 定义需要清理的特殊token
        tail_special_tokens = [
            self.tokenizer.eos_token,
            '<|im_end|>',
            '<|im_start|>',
            '<|endoftext|>',
        ]
        
        # 从最长的特殊token开始检查，避免部分匹配问题
        tail_special_tokens.sort(key=len, reverse=True)
        
        cleaned_text = text
        while True:
            original_length = len(cleaned_text)
            for token in tail_special_tokens:
                if cleaned_text.strip().endswith(token):
                    cleaned_text = cleaned_text[:cleaned_text.rstrip().rfind(token)].rstrip()
            
            # 如果没有发生变化，说明已经清理完成
            if len(cleaned_text) == original_length:
                break
                
        return cleaned_text.strip()

def compute_advantage(node):
    """计算树中每个节点的advantage值
    
    Args:
        node: 当前节点
    """
    # 如果节点有父节点，计算其兄弟节点的平均value
    if node.parent:
        siblings = node.parent.children
        sibling_values = [child.value for child in siblings if child != node]
        if sibling_values:  # 如果有兄弟节点
            baseline = sum(sibling_values) / len(sibling_values)
            node.advantage = node.value - baseline
        else:  # 如果没有兄弟节点
            node.advantage = 0.0
    else:  # 根节点的advantage为0
        node.advantage = 0.0
    
    # 递归计算所有子节点的advantage
    for child in node.children:
        compute_advantage(child)

def destroy_tree(tree):
    """销毁树中的所有节点
    
    Args:
        tree: 树的根节点
    """
    if not tree:
        return
    if tree.parent:
        del tree.parent
    node_queue = [tree]
    while node_queue:
        current_node = node_queue.pop(0)
        # 将子节点加入队列
        node_queue.extend(current_node.children)
        
        # 清除节点的引用
        current_node.children = []
        current_node.parent = None
        current_node.output = None
        current_node.score = None
        current_node.text = None
        current_node.labels = None
        current_node.value = 0
        current_node.advantage = 0  # 添加advantage属性
        current_node.sample_weight = 0
        current_node.input_length = 0


class SequenceStoppingCriteria(StoppingCriteria):
    """自定义停止条件：检查最后N个token是否匹配指定序列"""
    
    def __init__(self, stop_token_sequence):
        """
        Args:
            stop_token_sequence: 停止序列的token列表
        """
        self.stop_token_sequence = torch.tensor(stop_token_sequence)
        self.sequence_length = len(stop_token_sequence)
    
    def __call__(self, input_ids, scores, **kwargs):
        # 获取每个序列最后N个token，N是停止序列的长度
        last_tokens = input_ids[:, -self.sequence_length:]
        
        # 如果生成的token数量不够，返回False
        if last_tokens.shape[1] < self.sequence_length:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool)
        
        # 将停止序列移到正确的设备上
        stop_sequence = self.stop_token_sequence.to(input_ids.device)
        
        # 检查是否匹配停止序列
        matches = (last_tokens == stop_sequence).all(dim=1)
        return matches