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
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

import torch
from torch import sigmoid

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

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

from nltk.translate.bleu_score import sentence_bleu
from transformers import CLIPProcessor, CLIPModel
import torch
import re
import pickle

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

from sklearn.metrics.pairwise import cosine_similarity
import gensim
import numpy as np
from torch import sigmoid

def sigmoid_mapped_similarity(similarity):
    # 线性映射 [-1, 1] -> [0, 1]
    similarity_normalized = (similarity + 1) / 2
    # 应用sigmoid函数
    return sigmoid(torch.tensor(similarity_normalized * 10 - 5)).item()  # 使用线性缩放使得sigmoid效果更明显

def new_map_similarity(similarity):
    # return max(0.5*pos(similarity, 2) + 0.5, 0)
    return similarity*similarity

class Word2VecScorer:
    def __init__(self, w2v_model_path='./data/GoogleNews-vectors-negative300.bin'):
        # 加载预训练的Word2Vec模型
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)

    def get_sentence_vector(self, sentence):
        # 将句子转换为词向量的平均值
        words = sentence.split()  # 简单的空格分隔，可以替换为更复杂的分词方法
        word_vectors = []

        for word in words:
            if word in self.w2v_model:
                word_vectors.append(self.w2v_model[word])

        if len(word_vectors) == 0:
            # 如果句子中没有任何可用的单词，返回零向量
            return np.zeros(self.w2v_model.vector_size)

        # 返回句子向量（平均词向量）
        sentence_vector = np.mean(word_vectors, axis=0)
        return sentence_vector

    def metrics(self, answer, gt):
        # 获取answer和gt的句子向量
        answer_vector = self.get_sentence_vector(answer)
        gt_vector = self.get_sentence_vector(gt)

        # 计算余弦相似度
        similarity = cosine_similarity([answer_vector], [gt_vector])[0][0]
        sigmoid_similarity = sigmoid_mapped_similarity(similarity)
        # print(f"answer: {answer}, gt: {gt}, word2vecsimilarity: {similarity}")
        return sigmoid_similarity

class ClipScorer:
    def __init__(self, features_path='./data/gt_features_512.pkl'):
        # 加载预处理的features
        with open(features_path, 'rb') as f:
            self.gt_features = pickle.load(f)
        
    def metrics(self, answer, gt):
        # 只需要处理answer的features
        inputs = clip_processor(text=[answer], return_tensors="pt", padding=True, truncation=True)
        answer_features = clip_model.get_text_features(**inputs)
        
        # 使用预存的gt features
        gt_feature = self.gt_features[gt].to(answer_features.device)  # 确保在同一设备上
        
        # 计算相似度
        clip_score = torch.cosine_similarity(gt_feature, answer_features[0], dim=0).item()
        return clip_score


def metrics(answer, gt):
    # Calculate CLIP-score
    inputs = clip_processor(text=[gt, answer], return_tensors="pt", padding=True, truncation=True)
    text_features = clip_model.get_text_features(**inputs)
    clip_score = torch.cosine_similarity(text_features[0], text_features[1], dim=0).item()  
    return clip_score

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that calculates the sum of BLEU-1, Recall@5, and CLIP-score between the completion and solution."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    lambda_ = 0.5
    scorer = Word2VecScorer()
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Extract answer from solution if it has think/answer tags
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        if reward == 0.0:
            try:
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                # print(f"ground_truth: {ground_truth}")
                
                # Extract answer from content if it has think/answer tags
                # print(f"content:\n {content}")
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                # print(f"student_answer: \n {student_answer}")
                word2vec_score = scorer.metrics(student_answer, ground_truth)
                if ground_truth in student_answer or student_answer in ground_truth:
                    reward = 1.0
                else:
                    reward = word2vec_score
                    
            except Exception:
                pass
                
        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
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
    # Get reward functions
    # import pdb; pdb.set_trace()
    script_args.reward_funcs = ['accuracy','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # import pdb; pdb.set_trace()

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    ### lzy modified
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)


    # Format into conversation
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
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }


    if "image1" in dataset[script_args.dataset_train_split].features or "image2" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
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

    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        # device_map="auto",
        trust_remote_code=True,
    )
    model.train()  # 确保模型处于训练模式
    
    # 设置模型参数需要梯度
    for param in model.parameters():
        param.requires_grad = True

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print("script_args: %s", script_args)
    main(script_args, training_args, model_args)
