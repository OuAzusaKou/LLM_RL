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
THINK_STR = 'Tk'
ANSWER_STR = 'Ans'

if is_apex_available():
    from apex import amp

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
    def __init__(self, reward_function, num_samples=5, max_steps=5, max_generate_tokens=100,*args, **kwargs):
        """初始化训练器
        
        Args:
            reward_function: 奖励计算函数
            num_samples: 每步采样次数
            max_steps: 最大步骤数
        """
        super().__init__(*args, **kwargs)
        self.reward_function = reward_function
        self.num_samples = num_samples
        self.max_steps = max_steps
        self.max_generate_tokens = max_generate_tokens

    def _parse_generation(self, text: str) -> Tuple[List[str], str]:
        """解析生成的文本，分离思考步骤和最终答案
        
        Args:
            text: 生成的文本
            
        Returns:
            steps: 思考步骤列表
            answer: 最终答案
        """
        # 分割思考步骤和答案
        parts = text.split(ANSWER_STR)
        answer = parts[-1]
        steps = ''
        return steps, answer.strip()

    def sample_generate_and_adv_cal(self, model, inputs, return_outputs=False, num_items_in_batch=1):
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
        
        input_str = self.tokenizer.decode(input_ids[0])

        split_positions = (input_ids[0] == 151644).nonzero(as_tuple=True)[0]
        if len(split_positions) >= 2:
            # 获取第二个分隔符的位置
            second_split_pos = split_positions[2]
            # 截取前两组内容
            input_ids = input_ids[:, :second_split_pos+3]
            attention_mask = attention_mask[:, :second_split_pos+3]


        
        labels = input_str.split('<|im_start|>')[3].split('assistant\n')[1].split('<|im_end|>\n')[0]

        total_loss = 0
        
        # 存储所有思考树的根节点
        all_trees = []
        
        for i in range(batch_size):
            # 获取单个样本
            sample_input_ids = input_ids[i,:].view(1,-1)
            sample_attention_mask = attention_mask[i,:].view(1,-1)
            
            query_texts = self.tokenizer.decode(sample_input_ids[0], skip_special_tokens=True)
            
            # 生成第一步的多个样本
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
                eos_token_id=[self.tokenizer.encode(THINK_STR)[0],self.tokenizer.encode(ANSWER_STR)[0]]
            )

            # logits = model(first_step_output.sequences).logits
            
            # grad_logits = logits[:, :-1, :]
            
            first_step_texts = self.tokenizer.batch_decode(first_step_output.sequences, skip_special_tokens=True)
            
            first_scores = first_step_output.logits

            first_input_length = sample_input_ids.shape[1]
            # 对第一步的每个样本继续生成
            # tolist 第一次为根节点，现在第一次思考为根节点了。
            for first_step_idx in range(self.num_samples):
                # 创建根节点
                score = []
                for i in range(len(first_scores)):
                    score.append(first_scores[i][first_step_idx])

                root = ThoughtNode(first_step_texts[first_step_idx], first_step_output[0][first_step_idx], score,first_input_length)
                all_trees.append(root)
                
                # 初始化当前分支的队列，现在包含节点对象
                branch_queue = [root]
                
                while branch_queue:
                    current_node = branch_queue.pop(0)
                    current_text = current_node.text
                    
                    # 如果当前文本已包含answer或达到最大步数，则生成答案
                    if ANSWER_STR in current_text[len(query_texts):] or len(current_text[len(query_texts):].split(THINK_STR)) >= self.max_steps:
                        # 生成最终答案
                        final_input_ids = self.tokenizer.encode(
                            current_text + ANSWER_STR,
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
                            eos_token_id=[self.tokenizer.encode(THINK_STR)[0],self.tokenizer.encode(ANSWER_STR)[0]]
                        )
                        final_scores = final_output.logits

                        # logits = model(final_output.sequences).logits
            
                        # grad_logits = logits[:, :-1, :]

                        final_texts = self.tokenizer.batch_decode(final_output.sequences, skip_special_tokens=True)
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
                    if current_text.endswith(THINK_STR):
                        current_input_ids = self.tokenizer.encode(
                            current_text,
                            return_tensors="pt",
                            add_special_tokens=False
                        ).to(input_ids.device)
                    else:
                        current_input_ids = self.tokenizer.encode(
                            current_text+THINK_STR,
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
                        eos_token_id=[self.tokenizer.encode(THINK_STR)[0], self.tokenizer.encode(ANSWER_STR)[0]]
                    )
                    
                    
                    # logits = model(next_outputs.sequences).logits
            
                    # grad_logits = logits[:, :-1, :]


                    next_texts = self.tokenizer.batch_decode(next_outputs.sequences, skip_special_tokens=True)

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
                    if node.is_answer:
                        # 到达叶子节点，计算整条路径的损失
                        path_text = node.text
                        path_outputs = accumulated_outputs + [node.output]
                        steps, answer = self._parse_generation(path_text)
                        reward = self.reward_function(steps, answer, node.labels) 
                        node.value = reward    
                        return reward
                    
                    else:
                        # 准备输入
                        # if node.text.endswith("think"):
                        #     input_code = node.output
                        # else:
                        #     input_text = node.text + "think"
                        
                        input_code = node.output
                        # 获取当前节点的输入ID
                        input_code_length = len(self.tokenizer.encode(node.text, add_special_tokens=False))
                        # 获取模型对所有子节点文本的概率输出
                        # with torch.no_grad():
                        node_logits = node.score

                        node_logits = torch.stack(node_logits, dim=0)
                        # logits = logits[:, -1, :]  # 获取最后一个token的logits
                        node_probs = torch.nn.functional.softmax(node_logits, dim=-1)
                        
                        # 计算每个子节点的条件概率并更新value
                        for child in node.children:
                            # to do 用score 去和logits mask
                            child_logits = child.score
                            child_logits = torch.stack(child_logits, dim=0)
                            child_probs = torch.nn.functional.softmax(child_logits, dim=-1)

                            # child_probs = child_logits 
                            # child_logits = child.grad_logits
                            # # child_logits = torch.stack(child_logits, dim=0)
                            # child_probs = torch.nn.functional.softmax(child_logits, dim=-1)


                            # 获取新生成的token的索引
                            # child_text_ids = self.tokenizer.encode(child.text, add_special_tokens=False)
                            # new_token_ids = child_text_ids[input_code_length:]
                            
                            # 计算序列概率（所有token概率的乘积）
                            sequence_prob = torch.tensor(0.0).to(input_ids.device)
                            for i,pos in enumerate(range(child.input_length,child.output.shape[0])):
                                if child.output[pos]==self.tokenizer.eos_token_id or child.output[pos]==151643:
                                    break
                                token_id = child.output[pos]
                                token_prob = child_probs[i, token_id]
                                # token_prob = child_probs[pos, token_id]
                                sequence_prob += torch.log(token_prob)
                            
                            # 更新子节点的value
                            node.value += torch.exp(sequence_prob) * compute_node_value(child, [])
                            node.sample_weight += torch.exp(sequence_prob)
                        node.value = node.value / node.sample_weight
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
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)
        
        trees = self.sample_generate_and_adv_cal(model, inputs, num_items_in_batch=num_items_in_batch)
        node_num = 0
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
                    loss = self.compute_node_loss(model,node)

                        
                    # 或者方式3：同时检查
                    if not loss.requires_grad:
                        continue
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

        return loss.detach()
    
    def compute_node_loss(self, model, node):
        """计算单个节点的损失
        
        Args:
            model: 模型
            node: 节点
        """
        logits_new = model(node.output.view(1,-1)).logits[:,:-1,:]
        probs = torch.nn.functional.softmax(logits_new, dim=-1)
        token_prob_list = []
        token_prob_olden = []
        sequence_log_new = torch.tensor(0.0).to(logits_new.device)
        for i,pos in enumerate(range(node.input_length-1,node.output.shape[0]-1)):
            if node.output[pos+1]==self.tokenizer.eos_token_id or node.output[pos+1]==151643:
                break
            token_id = node.output[pos+1]
            token_prob = probs[0, pos, token_id]
            token_prob_list.append(token_prob)
            # token_prob = child_probs[pos, token_id]
            sequence_log_new += torch.log(token_prob)
        # sequence_log_old = torch.tensor(0.0).to(logits_new.device)
        logits_old = node.score
        logits_old = torch.stack(logits_old, dim=0)
        old_probs = torch.nn.functional.softmax(logits_old, dim=-1)
        sequence_log_old = torch.tensor(0.0).to(old_probs.device)
        for i,pos in enumerate(range(node.input_length,node.output.shape[0])):
            if node.output[pos]==self.tokenizer.eos_token_id or node.output[pos]==151643:
                break
            token_id = node.output[pos]
            token_prob = old_probs[i, token_id]
            token_prob_olden.append(token_prob)
            sequence_log_old += torch.log(token_prob)
        # 更新子节点的value
        loss =  - (sequence_log_new - sequence_log_old) * node.advantage

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
