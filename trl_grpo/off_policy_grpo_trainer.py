# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, List, Dict
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

import random
from dataclasses import dataclass
import hashlib

class ExperiencePool:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        # 使用字典按prompt存储经验
        self.experiences: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    
    def _get_prompt_key(self, prompt_ids: torch.Tensor) -> str:
        """生成prompt的哈希key"""
        # 将tensor转换为bytes并计算md5哈希值
        prompt_bytes = prompt_ids.cpu().numpy().tobytes()
        return hashlib.md5(prompt_bytes).hexdigest()
    
    def add(self, experience_data: Dict[str, torch.Tensor]):
        """添加新经验到经验池
        
        Args:
            experience_data: 包含各种tensor的字典,必须包含以下键:
                - prompt_ids, prompt_mask, completion_ids, completion_mask
                - ref_per_token_logps, advantages, rewards, rewards_per_func
        """
        # 使用哈希函数生成prompt_key
        prompt_key = self._get_prompt_key(experience_data["prompt_ids"])
        
        if prompt_key not in self.experiences:
            self.experiences[prompt_key] = []
        
        # 将所有tensor移动到CPU并转为Python列表
        experience_dict = {
            key: tensor.cpu().tolist() for key, tensor in experience_data.items()
        }
            
        # 添加新经验到对应prompt组
        self.experiences[prompt_key].append(experience_dict)
        
        # 如果某个prompt组的经验数量超过限制,移除最旧的
        if len(self.experiences[prompt_key]) > self.max_size:
            self.experiences[prompt_key].pop(0)
    
    def sample(self, per_prompt_batch_size: int, keys: List[str] = None) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """按prompt key分组采样经验
        
        Args:
            per_prompt_batch_size: 每个prompt要采样的经验数量
            keys: 指定的prompt key列表,如果为None则从所有prompt中随机采样
            
        Returns:
            Dict[str, List[Dict]]: 按prompt key分组的采样结果
        """
        if not self.experiences:
            return {}
        
        sampled_experiences = {}
        
        if keys is not None:
            # 如果指定了keys,只从这些key对应的经验中采样
            for key in keys:
                if key in self.experiences and self.experiences[key]:
                    samples = random.sample(
                        self.experiences[key],
                        min(per_prompt_batch_size, len(self.experiences[key]))
                    )
                    sampled_experiences[key] = samples
        else:
            # 从所有prompt组中随机采样
            for prompt_key in self.experiences.keys():
                if self.experiences[prompt_key]:
                    samples = random.sample(
                        self.experiences[prompt_key],
                        min(per_prompt_batch_size, len(self.experiences[prompt_key]))
                    )
                    sampled_experiences[prompt_key] = samples
                
        return sampled_experiences
    
    def __len__(self):
        return sum(len(exps) for exps in self.experiences.values())

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        experience_pool_size (int, *optional*, defaults to 10000):
            Maximum size of the experience pool.
        experience_ratio (float, *optional*, defaults to 0.6):
            Ratio of experience pool to use for training.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        experience_pool_size: int = 500,
        experience_ratio: float = 0.6,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # Add experience pool
        self.experience_pool = ExperiencePool(max_size=experience_pool_size)
        self.experience_ratio = experience_ratio

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

    def _move_model_to_vllm(self):
        for param in self.model.parameters():
            param.ds_active_sub_modules.clear()
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            
            # 使用临时变量来避免保留大型张量
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                
                # 清理all_prompts_text和outputs以释放内存
                del all_prompts_text
                del outputs
            else:
                completion_ids = [None] * len(all_prompts_text)
                # 清理all_prompts_text以释放内存
                del all_prompts_text
            
            # 广播完成ID
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # 检查是否有预定答案需要替换
            

            # Pad the completions, and concatenate them with the prompts
            # 这个if 关键是要拿到每一个process中的prompt_completion_ids 和 completion_ids
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            
            # breakpoint()
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            del prompt_completion_ids  # 释放不再需要的张量

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # 释放不再需要的临时张量
        del is_eos, eos_idx, sequence_indices

        # 创建prompt_completion_ids用于计算
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # 计算每个奖励函数的奖励
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                # 释放不再需要的张量
                del reward_inputs
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        
        # 处理预定答案替换逻辑 - 优化为单次收集和广播
        # 先收集所有inputs用于检查替换
        all_inputs = gather_object(inputs)
        completion_ids_gathered = gather_object(completion_ids)


        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards.view(-1)) / (std_grouped_rewards + 1e-4)
        

        
        # 只在主进程中进行替换逻辑
        if self.accelerator.is_main_process:
            for i in range(0, len(all_inputs), self.num_generations):
                if rewards[i:i+self.num_generations].sum() == 0:
                    if 'extra_info' in all_inputs[i] and 'answer' in all_inputs[i]['extra_info']:
                        answer = all_inputs[i]['extra_info']['answer']
                        # 将答案转换为token ids
                        answer_ids = self.processing_class(
                            answer+self.processing_class.eos_token, 
                            return_tensors="pt", 
                            add_special_tokens=False
                        )["input_ids"][0]
                        completion_ids_gathered[i] = answer_ids
                        advantages[i] = 0.4
        
            # 清理不再需要的张量
            del all_inputs

                # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        advantages = broadcast_object_list(advantages, from_process=0)
        advantages = advantages[process_slice]

        
        
        # 广播修改后的completion_ids和rewards
        completion_ids = broadcast_object_list(completion_ids_gathered, from_process=0)
        rewards = broadcast_object_list(rewards, from_process=0)
        
        # 获取当前进程的数据切片
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        
        completion_ids = completion_ids[process_slice]
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        
        # 重建prompt_completion_ids和masks
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        # 重新计算掩码
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        del is_eos, eos_idx, sequence_indices


        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
        '''importance weight'''
    
        # current_logps = gather_object(per_token_logps)
        # padded_completion_mask = gather_object(completion_mask)

        # importance_weights = []
        # for i, (curr_logps, comp_mask) in enumerate(zip(current_logps, padded_completion_mask)):
        #     # 计算有效token数
        #     token_count = comp_mask.sum().item()
        #     if token_count == 0:
        #         importance_weights.append(0.0)
        #         continue
            
        #     # 计算平均logps
        #     avg_curr_logp = (curr_logps * comp_mask).sum() /token_count
        #     # avg_ref_logp = (ref_logps * comp_mask).sum() / token_count
            
        #     # 计算importance weight
        #     # importance_weight = torch.exp(avg_curr_logp - avg_ref_logp)
        #     # p_weiht_exp
        #     importance_weight = torch.exp(avg_curr_logp)
        #     importance_weights.append(importance_weight.item())
        



        # # 转为tensor
        # importance_weights = torch.tensor(importance_weights,device=self.accelerator.device)
        # # importance_weights = importance_weights.view(-1,self.num_generations)
        # mean_grouped_rewards = (importance_weights * rewards).view(-1,self.num_generations).sum(dim=-1) / (importance_weights.view(-1,self.num_generations).sum(dim=-1) + 1e-8)
        # mean_grouped_rewards = mean_grouped_rewards.view(-1,1)
        # # 计算加权方差
        # variance = ((rewards.view(-1,self.num_generations) - mean_grouped_rewards) ** 2 * importance_weights.view(-1,self.num_generations)).sum(dim=-1)
        # std_grouped_rewards = torch.sqrt(variance + 1e-8)

        '''importance weight'''


        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        del mean_grouped_rewards, std_grouped_rewards

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        rewards = rewards[process_slice]
        rewards_per_func = rewards_per_func[process_slice]
        #  将新样本添加到经验池
        # for i in range(prompt_ids.size(0)):
        #     new_experience = {
        #         "prompt_ids": prompt_ids[i],
        #         "prompt_mask": prompt_mask[i],
        #         "completion_ids": completion_ids[i],
        #         "completion_mask": completion_mask[i],
        #         "ref_per_token_logps": ref_per_token_logps[i],
        #         "advantages": advantages[i],
        #         "rewards": rewards[i],
        #         "rewards_per_func": rewards_per_func[i]
        #     }

        #     self.experience_pool.add(new_experience)
    
        # 从经验池中采样
        # if len(self.experience_pool) > 0 and exp_samples_size > 0:
        #     exp_samples = self.experience_pool.sample(exp_samples_size)
            
        #     # 获取最大长度用于padding
        #     max_prompt_len = max(prompt_ids.size(1), 
        #                        max(exp.prompt_ids.size(0) for exp in exp_samples))
        #     max_completion_len = max(completion_ids.size(1),
        #                           max(exp.completion_ids.size(0) for exp in exp_samples))
            
        #     # 对新样本进行padding
        #     padded_prompt_ids = pad([prompt_ids[i] for i in range(prompt_ids.size(0))]+[exp.prompt_ids for exp in exp_samples], 
        #                           padding_value=self.processing_class.pad_token_id,
        #                           padding_side="left")
        #     padded_prompt_mask = pad([prompt_mask[i] for i in range(prompt_mask.size(0))]+[exp.prompt_mask for exp in exp_samples],
        #                            padding_value=0,
        #                            padding_side="left")
        #     padded_completion_ids = pad([completion_ids[i] for i in range(completion_ids.size(0))]+[exp.completion_ids for exp in exp_samples],
        #                               padding_value=self.processing_class.pad_token_id,
        #                               padding_side="right")
        #     padded_completion_mask = pad([completion_mask[i] for i in range(completion_mask.size(0))]+[exp.completion_mask for exp in exp_samples],
        #                                padding_value=0,
        #                                padding_side="right")
        #     padded_ref_per_token_logps = pad([ref_per_token_logps[i] for i in range(ref_per_token_logps.size(0))]+[exp.ref_per_token_logps for exp in exp_samples],
        #                                   padding_value=0,
        #                                   padding_side="right")

        #     # 合并新样本和经验池样本
        #     advantages = torch.cat([advantages] + 
        #                         [exp.advantages.view(-1) for exp in exp_samples], dim=0)
        #     rewards = torch.cat([rewards] + 
        #                           [exp.rewards.view(-1) for exp in exp_samples], dim=0)
        #     rewards_per_func = torch.cat([rewards_per_func] + 
        #                                    [exp.rewards_per_func.view(1,-1) for exp in exp_samples], dim=0)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask":prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "rewards": rewards,
            "rewards_per_func": rewards_per_func
        }

    def advantage_compute(self, model, inputs):
        """使用重要性采样计算advantage
        
        从经验池中为每个prompt采样相关样本,计算baseline和advantage
        
        Args:
            model: 当前模型
            inputs: 包含prompt_ids等的输入数据
            
        Returns:
            torch.Tensor: 计算出的advantage值
        """
        device = self.accelerator.device
        
        # 1. 对每个prompt从经验池中采样
        batch_size = inputs["prompt_ids"].size(0)
        prompt_keys = [self.experience_pool._get_prompt_key(inputs["prompt_ids"][i]) for i in range(batch_size)]
        
        # 每个prompt采样的样本数量
        samples_per_prompt = 8  # 可设为超参数
        
        # 从经验池中采样
        sampled_experiences = [self.experience_pool.sample(samples_per_prompt, keys=prompt_keys)]
        # print('sampled_experiences',sampled_experiences)
        # print('sampled_experiences',sampled_experiences)
        sampled_experiences_list = gather_object(sampled_experiences)
        # print('sampled_experiences_len',len(sampled_experiences))
        # print('sampled_experiences_list',sampled_experiences_list)
        # 如果经验池为空,直接返回输入的advantages
        if not sampled_experiences:
            return inputs["advantages"]
        
        # 2. 为每个prompt收集样本
        prompt_samples = {}  # 按prompt组织样本
        
        # 收集当前batch中的样本
        # for i, key in enumerate(prompt_keys):
        #     if key not in prompt_samples:
        #         prompt_samples[key] = []
            
        #     # 添加当前batch的样本
        #     prompt_samples[key].append({
        #         "prompt_ids": inputs["prompt_ids"][i],
        #         "prompt_mask": inputs["prompt_mask"][i],
        #         "completion_ids": inputs["completion_ids"][i],
        #         "completion_mask": inputs["completion_mask"][i],
        #         "ref_per_token_logps": inputs["ref_per_token_logps"][i],
        #         "rewards": inputs["rewards"][i],
        #         "from_current_batch": True,
        #         "batch_index": i  # 记录在原始batch中的索引
        #     })
        
        # 添加从经验池中采样的样本
        for sampled_experiences in sampled_experiences_list:
            for key, experiences in sampled_experiences.items():
                if key not in prompt_samples:
                    prompt_samples[key] = []
                
                for exp in experiences:
                    prompt_samples[key].append({
                        "prompt_ids": torch.tensor(exp["prompt_ids"]),
                        "prompt_mask": torch.tensor(exp["prompt_mask"]),
                        "completion_ids": torch.tensor(exp["completion_ids"]),
                        "completion_mask": torch.tensor(exp["completion_mask"]),
                        "ref_per_token_logps": torch.tensor(exp["ref_per_token_logps"]),
                        "rewards": torch.tensor(exp["rewards"]),
                        "from_current_batch": False
                    })
            
        # 3. 处理每个prompt组,计算advantage
        advantages = inputs['advantages']
        
        for key, samples in prompt_samples.items():
            # 如果这个prompt组没有当前batch的样本,跳过
            # current_batch_samples = [s for s in samples if s["from_current_batch"]]
            # if not current_batch_samples:
            #     continue
            
            # 准备这个prompt组所有样本的数据进行批处理
            prompt_ids_list = [sample["prompt_ids"] for sample in samples]
            prompt_mask_list = [sample["prompt_mask"] for sample in samples]
            completion_ids_list = [sample["completion_ids"] for sample in samples]
            completion_mask_list = [sample["completion_mask"] for sample in samples]
            ref_logps_list = [sample["ref_per_token_logps"] for sample in samples]
            rewards_list = [sample["rewards"].to(self.accelerator.device) for sample in samples]
            
            # 获取最大长度用于padding
            max_prompt_len = max(ids.size(0) for ids in prompt_ids_list)
            max_completion_len = max(ids.size(0) for ids in completion_ids_list)
            
            # Padding
            padded_prompt_ids = pad(prompt_ids_list, padding_value=self.processing_class.pad_token_id, padding_side="left").to(device)
            padded_prompt_mask = pad(prompt_mask_list, padding_value=0, padding_side="left").to(device)
            padded_completion_ids = pad(completion_ids_list, padding_value=self.processing_class.pad_token_id, padding_side="right").to(device)
            padded_completion_mask = pad(completion_mask_list, padding_value=0, padding_side="right").to(device)
            padded_ref_logps = pad(ref_logps_list, padding_value=0, padding_side="right").to(device)
            
            # 拼接prompt和completion
            input_ids = torch.cat([padded_prompt_ids, padded_completion_ids], dim=1)
            attention_mask = torch.cat([padded_prompt_mask, padded_completion_mask], dim=1)
            
            # 计算当前策略下的logps
            with torch.inference_mode():
                current_logps = self._get_per_token_logps(model, input_ids.to(self.accelerator.device), 
                                                          attention_mask.to(self.accelerator.device), max_completion_len)
            
            # 计算每个样本的importance weight
            importance_weights = []
            for i, (curr_logps, ref_logps, comp_mask) in enumerate(zip(current_logps, padded_ref_logps, padded_completion_mask)):
                # 计算有效token数
                token_count = comp_mask.sum().item()
                if token_count == 0:
                    importance_weights.append(0.0)
                    continue
                
                # 计算平均logps
                avg_curr_logp = (curr_logps * comp_mask).sum() / token_count
                avg_ref_logp = (ref_logps * comp_mask).sum() / token_count
                
                # 计算importance weight
                # importance_weight = torch.exp(avg_curr_logp - avg_ref_logp)
                # p_weiht_exp
                importance_weight = torch.exp(avg_curr_logp)
                importance_weights.append(importance_weight.item())
            
            # 转为tensor
            importance_weights = torch.tensor(importance_weights,device=self.accelerator.device)
            rewards = torch.stack(rewards_list)
            print('rewards',rewards)
            # 归一化权重(防止数值不稳定)
            # normalized_weights = importance_weights / (importance_weights.sum() + 1e-8)
            
            # 计算baseline (加权平均奖励)
            # baseline = (importance_weights * rewards).sum() / len(rewards_list)
            
            #计算baseline (加权概率平均)
            baseline = (importance_weights * rewards).sum() / (importance_weights.sum() + 1e-8)
            
            # 计算加权方差
            variance = ((rewards - baseline) ** 2 * importance_weights).sum()
            std = torch.sqrt(variance + 1e-8)
            
            # 对当前batch中的样本计算advantage
            for i, sample in enumerate(samples):
                if sample["from_current_batch"]:
                    batch_idx = sample["batch_index"]
                    advantages[batch_idx] = (rewards[i] - baseline) / (std + 1e-8)

            for i in range(batch_size):
                if prompt_keys[i] == key:
                    advantages[i] = (inputs["rewards"][i] - baseline) / (std + 1e-8)
        
        
        print('samples',len(samples))
        # 对于没有在经验池中找到相应样本的prompt,保留原始advantage
        # for i in range(batch_size):
        #     key = prompt_keys[i]

        #     if key not or not sampled_experiences[key]:
        #         advantages[i] = inputs["advantages"][i]
        print('advantages',advantages)
        return advantages

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # 计算优势 - 使用重要性采样
        # advantages = self.advantage_compute(model, inputs)

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

def safe_tensor_to_list(tensor):
    """安全地将任何类型的张量转换为Python列表"""
    if tensor.dtype == torch.bfloat16:
        return tensor.cpu().float().tolist()
    return tensor.cpu().tolist()
