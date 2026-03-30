import json
import torch
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import re
import json
import torch
from tqdm import tqdm
import logging

import functools
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
torch.manual_seed(1234)

from nltk.translate.bleu_score import sentence_bleu
from transformers import CLIPProcessor, CLIPModel
import torch
import re
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

selected_relations = ["atlocation", "createdby", "usedfor", "antonym", "madeof", "partof"]

from sklearn.metrics.pairwise import cosine_similarity
import gensim
import numpy as np
from torch import sigmoid
def sigmoid_mapped_similarity(similarity):
    # 线性映射 [-1, 1] -> [0, 1]
    similarity_normalized = (similarity + 1) / 2
    # 应用sigmoid函数
    sigmoid_value = torch.sigmoid(torch.tensor(similarity_normalized * 10 - 5))  # 使用线性缩放使得sigmoid效果更明显
    return sigmoid_value.item()  # 返回float类型

class Word2VecScorer:
    def __init__(self, w2v_model_path='./GoogleNews-vectors-negative300.bin'):
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
        # print("into metrics......")
        answer_vector = self.get_sentence_vector(answer)
        gt_vector = self.get_sentence_vector(gt)

        # # 计算余弦相似度
        similarity = cosine_similarity([answer_vector], [gt_vector])[0][0]
        sigmoid_similarity = sigmoid_mapped_similarity(similarity)

        # 计算Recall@5
        answer_words = answer.split()
        gt_words = gt.split()
        recall_at_5 = sum(1 for word in answer_words if word in gt_words) / 5.0

        # print(f"answer: {answer}, gt: {gt}, word2vecsimilarity: {similarity}, sigmoid_similarity: {sigmoid_similarity}")
        return similarity, sigmoid_similarity, recall_at_5

def parse(answer):
    try:
        match = re.search(r"<answer>(.*?)</answer>", answer)
        answer_content = match.group(1)
        # answer_content = answer_content.replace(' ','').replace('_','').lower()
        return answer_content
    except Exception as e:
        return ""

def judgement(output_text, gt):
    if output_text == gt:
        return True
    else:
        return False
    
def split_dict(image_pairs, rank, world_size):
    # 将字典的键转换为列表
    keys = list(image_pairs.keys())
    
    # 计算每个分片的长度
    import math
    split_length = math.ceil(len(keys) / world_size)
    
    # 获取当前进程的键列表
    split_keys = keys[int(rank * split_length): int((rank + 1) * split_length)]
    
    # 构建子字典
    split_images = {key: image_pairs[key] for key in split_keys}
    
    return split_images

MODEL_PATH = "./share_models/Qwen2-VL-7B-Instruct_GRPO_6_relations_sub_acc_only/checkpoint-72"  # after RL
# MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
def run(rank, world_size, image_pairs):
    device = torch.device(f"cuda:{rank}")
    model_path = MODEL_PATH  # after RL
    model_base = "Qwen/Qwen2-VL-7B-Instruct"
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu", # 一定要指定cpu或者一个cuda，如果用的是auto，会报错Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
    )

    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_base,use_fast=True)
    model = model.to(device)
    model = model.eval()


    split_images = split_dict(image_pairs, rank, world_size)

    text_bleu_score_list = []
    text_recall_at_5_list = []
    text_clip_score_list = []
    image_bleu_score_list = []
    image_recall_at_5_list = []
    image_clip_score_list = []
    word2vec_score_text_list = []
    word2vec_score_image_list = []
    word2vec_score_text_sigmoid_list = []
    word2vec_score_image_sigmoid_list = []
    word2vec_score_text_recall_at_5_list = []
    word2vec_score_image_recall_at_5_list = []
    acc_text = 0
    acc_image = 0
    total = 0

    res = []
    score = Word2VecScorer()
    # 逐条处理 messages
    for pair, pair_path in tqdm(split_images.items()):
        total += 1
        # 构建text单条消息
        message_text = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Given the relationship between \"{pair.split("+")[0]}\"  and  \"{pair.split("+")[1]}\", please solve the analogy. Specifically, answer the following question: What is \"{pair.split("+")[2]}\" to as \"{pair.split("+")[0]}\" is to \"{pair.split("+")[1]}\"?
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:
<think> ... </think> <answer>Use less than 5 words</answer>
Please strictly follow the format.Please answer in English."""
                }
            ]
        }
        # print(f"message_text: {message_text}")
        message_image = {
                "role": "user",
                "content": [
                    {"type": "image", "image": pair_path[2]},
                    {"type": "image", "image": pair_path[0]},
                    {"type": "image", "image": pair_path[1]},
                    {"type": "text", "text": f"""Given the relationship between image2 and image3, please solve the analogy. Specifically, answer the following question: What is image1 to as image2 is to image3?
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:
<think> ... </think> <answer>Use less than 5 words</answer>
Please strictly follow the format.Please answer in English.""",}
                ],
            }
        # print(f"message_image: {message_image}")

        # text -------------------  准备输入  -------------------
        text = processor.apply_chat_template(
            [message_text], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([message_text])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # 推理：生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # image -------------------  准备输入  -------------------
        text = processor.apply_chat_template(
            [message_image], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([message_image])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # 推理：生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_image = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


        # 打印输出
        # print(f"{pair.split('+')[0]} to {pair.split('+')[1]} equals to [WHAT] to {pair.split('+')[3]}")
        gt = pair.split('+')[3]


        answer_text = parse(output_text[0])
        answer_image = parse(output_image[0])
        
        if judgement(answer_text, gt):
            acc_text += 1
        if judgement(answer_image, gt):
            acc_image += 1
        
        # print(f"Output_text: {output_text[0]}\n")
        # print(f"Output_image: {output_image[0]}\n")
        # print(f"Answer_text: {answer_text}\n")
        # print(f"Answer_image: {answer_image}\n")
        # print(f"GT: {gt}\n")
        # text_bleu_score, text_recall_at_5, text_clip_score = metrics(answer_text, gt)
        # image_bleu_score, image_recall_at_5, image_clip_score = metrics(answer_image, gt)
        relation = pair_path[0].split("/")[-3]
        word2vec_score_text, word2vec_score_text_sigmoid, word2vec_score_text_recall_at_5 = score.metrics(answer_text, gt)
        word2vec_score_image, word2vec_score_image_sigmoid, word2vec_score_image_recall_at_5 = score.metrics(answer_image, gt)

        res.append({
            "pair": (pair.split('+')[0], pair.split('+')[1], pair.split('+')[2]), 
            "text": answer_text, "image": answer_image, "gt": gt,
            "word2vec_score_text": word2vec_score_text, "word2vec_score_text_sigmoid": word2vec_score_text_sigmoid,
            "word2vec_score_image": word2vec_score_image, "word2vec_score_image_sigmoid": word2vec_score_image_sigmoid,
            "word2vec_score_text_recall_at_5": word2vec_score_text_recall_at_5, "word2vec_score_image_recall_at_5": word2vec_score_image_recall_at_5,
            "text_think": output_text[0],
            "image_think": output_image[0],
            "relation": relation
        })
        word2vec_score_text_list.append(word2vec_score_text)
        word2vec_score_image_list.append(word2vec_score_image)
        word2vec_score_text_sigmoid_list.append(word2vec_score_text_sigmoid)
        word2vec_score_image_sigmoid_list.append(word2vec_score_image_sigmoid)
        word2vec_score_text_recall_at_5_list.append(word2vec_score_text_recall_at_5)
        word2vec_score_image_recall_at_5_list.append(word2vec_score_image_recall_at_5)

    
    return {
        "acc_text": acc_text,
        "acc_image": acc_image,
        "word2vec_score_text": word2vec_score_text_list,
        "word2vec_score_image": word2vec_score_image_list,
        "word2vec_score_text_sigmoid": word2vec_score_text_sigmoid_list,
        "word2vec_score_image_sigmoid": word2vec_score_image_sigmoid_list,
        "word2vec_score_text_recall_at_5": word2vec_score_text_recall_at_5_list,
        "word2vec_score_image_recall_at_5": word2vec_score_image_recall_at_5_list,
        "res": res,
        "total": total
    }



def main():
    image_pairs = json.load(open("./image_pairs_summation_297_50.json"))
    logger.info("Starting main function...")
    logger.info("into main....")
    multiprocess = torch.cuda.device_count() >= 1
    logger.info(f"cuda devices num = {torch.cuda.device_count()}")
    mp.set_start_method('spawn', force=True)
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, image_pairs=image_pairs)
            result_lists = pool.map(func, range(world_size))

        global_right_text = 0
        global_right_image = 0
        global_results = []
        global_text_bleu_score = []
        global_text_recall_at_5 = []
        global_text_clip_score = []
        global_image_bleu_score = []
        global_image_recall_at_5 = []
        global_image_clip_score = []
        global_word2vec_score_text = []
        global_word2vec_score_image = []
        global_word2vec_score_text_sigmoid = []
        global_word2vec_score_image_sigmoid = []
        global_word2vec_score_text_recall_at_5 = []
        global_word2vec_score_image_recall_at_5 = []

        for i in range(world_size):
            global_right_text += int(result_lists[i]["acc_text"])
            global_right_image += int(result_lists[i]["acc_image"])
            global_results.extend(result_lists[i]["res"])
            global_word2vec_score_text.extend(result_lists[i]["word2vec_score_text"])
            global_word2vec_score_image.extend(result_lists[i]["word2vec_score_image"])
            global_word2vec_score_text_sigmoid.extend(result_lists[i]["word2vec_score_text_sigmoid"])
            global_word2vec_score_image_sigmoid.extend(result_lists[i]["word2vec_score_image_sigmoid"])
            global_word2vec_score_text_recall_at_5.extend(result_lists[i]["word2vec_score_text_recall_at_5"])
            global_word2vec_score_image_recall_at_5.extend(result_lists[i]["word2vec_score_image_recall_at_5"])

        logger.info('text right number: ' + str(global_right_text))  
        logger.info('image right number: ' + str(global_right_image))
        logger.info('text accuracy: ' + str(global_right_text / len(image_pairs)))
        logger.info('image accuracy: ' + str(global_right_image / len(image_pairs)))
        logger.info('word2vec text score: ' + str(sum(global_word2vec_score_text) / len(global_word2vec_score_text)))
        logger.info('word2vec image score: ' + str(sum(global_word2vec_score_image) / len(global_word2vec_score_image)))
        logger.info('word2vec text sigmoid score: ' + str(sum(global_word2vec_score_text_sigmoid) / len(global_word2vec_score_text_sigmoid)))
        logger.info('word2vec image sigmoid score: ' + str(sum(global_word2vec_score_image_sigmoid) / len(global_word2vec_score_image_sigmoid)))
        logger.info('word2vec text recall@5: ' + str(sum(global_word2vec_score_text_recall_at_5) / len(global_word2vec_score_text_recall_at_5)))
        logger.info('word2vec image recall@5: ' + str(sum(global_word2vec_score_image_recall_at_5) / len(global_word2vec_score_image_recall_at_5)))
        # sorted_results = sorted(global_results, key=lambda x: x['word2vec_score_image_sigmoid'], reverse=True)
        with open(f"global_results_format_obly_summation.json", "w") as f:
            json.dump(global_results, f, indent=4, default=str)
        
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()