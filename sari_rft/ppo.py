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

import wandb
wandb.login(key='WANDB_API_KEY')

import torch
torch.cuda.empty_cache()
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLPPOTrainer, Qwen2VLPPOVLLMTrainer
from trl import PPOConfig, PPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

@dataclass
class PPOScriptArguments(ScriptArguments):
    """
    Script arguments for the PPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """奖励函数：使用符号验证或字符串匹配判断completion是否正确。"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass

        if reward == 0.0:
            try:
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                ground_truth = ground_truth.replace(' ','').replace('_','').lower()
                student_answer = student_answer.replace(' ','').replace('_','').lower()
                if ground_truth in student_answer or student_answer in ground_truth:
                    reward = 1.0
            except Exception:
                pass
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
    return rewards

def format_reward(completions, **kwargs):
    """奖励函数：判断completion是否符合指定格式。"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # 获取奖励函数
    script_args.reward_funcs = ['accuracy','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # 加载数据集
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)

    # 构造对话格式
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    if "image1" in dataset[script_args.dataset_train_split].features or "image2" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLPPOTrainer if not training_args.use_vllm else Qwen2VLPPOVLLMTrainer
    print("using: ", trainer_cls)

    # 初始化PPO Trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # 训练并推送模型到Hub
    trainer.train()

    # 保存和推送到hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    print("PPO训练启动")
    parser = TrlParser((PPOScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
