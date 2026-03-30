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



# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
# from lmdeploy import pipeline, TurbomindEngineConfig
# from lmdeploy.vl import load_image
# from lmdeploy.vl.constants import IMAGE_TOKEN



# # ------------------------- preliminary Intern ------------------------
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# 假设visual7w-telling的dataset.json路径如下
dataset_json_path = "./visual7w-toolkit/datasets/visual7w-telling/dataset.json"

# 读取整个dataset.json
with open(dataset_json_path, "r") as f:
    dataset = json.load(f)

# 按split划分数据集
split2samples = {"train": [], "val": [], "test": []}
for item in dataset["images"]:
    split = item.get("split", "train")
    split2samples[split].append(item)

# 统计每个split的样本数
for split in split2samples:
    print(f"{split} 样本数: {len(split2samples[split])}")

# 以test split为例，构建测试对
test_samples = split2samples["test"]

# 构建测试对，每个pair为(question, multiple_choices, answer, image_path)
test_pairs = []
base_img_path = "./visual7w-toolkit/datasets/visual7w-telling/images"  # 你需要根据实际图片路径修改

for item in test_samples:
    image_id = item["image_id"]
    filename = item["filename"]
    img_path = os.path.join(base_img_path, filename)
    for qa in item["qa_pairs"]:
        question = qa["question"]
        choices = qa["multiple_choices"] + [qa["answer"]]
        answer = qa["answer"]
        qtype = qa.get("type", "")
        qa_id = qa.get("qa_id", "")
        test_pairs.append({
            "image_id": image_id,
            "img_path": img_path,
            "question": question,
            "choices": choices,
            "answer": answer,
            "type": qtype,
            "qa_id": qa_id
        })
test_pairs = test_pairs[:1000]
print(f"test split中QA对数量: {len(test_pairs)}")
print("示例：")
for pair in test_pairs[:3]:
    print(pair)

def split_list(data_list, rank, world_size):
    # 按进程数分割list
    import math
    total = len(data_list)
    split_len = math.ceil(total / world_size)
    start = rank * split_len
    end = min((rank + 1) * split_len, total)
    return data_list[start:end]

def judgement(output_text, gt):
    # 这里假设output_text为模型输出字符串，gt为标准答案
    # 可以根据实际需要调整
    try:
        match = re.search(r"<answer>(.*?)</answer>", output_text)
        answer_content = match.group(1)
        answer_content = answer_content.replace(' ','').replace('_','').replace(".","").lower()
        gt = gt.replace(' ','').replace('_','').replace(".","").lower()
        return answer_content == gt
    except Exception as e:
        return False


def run(rank, world_size, test_pairs):
    device = torch.device(f"cuda:{rank}")
    # ------------------------ InternVL2_5-8B ------------------------
    # 加载模型
    path = 'OpenGVLab/InternVL2_5-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    model = model.to(device)
    model = model.eval()

    acc = 0
    wrong = []
    each_type_right = defaultdict(int)
    each_type_total = defaultdict(int)
    res = []

    split_pairs = split_list(test_pairs, rank, world_size)

    for pair in tqdm(split_pairs):
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        pixel_values = load_image(f'{pair["img_path"]}', max_num=12).to(torch.bfloat16).to(device)
        num_patches_list = [pixel_values.size(0)]
        question = f"""<image>Question: {pair['question']}
Choices: {chr(10).join(pair['choices'])}
Please select the most appropriate answer from the choices above.
Output your reasoning in <think> </think> tags and the final answer in <answer> </answer> tags.
The output format should be:
<think> ... </think> <answer>your answer</answer>
Please strictly follow the format."""
        output_image = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=False)
        print(f"Input: {pair['question']}")
        print(f"Output_image: {output_image}\n")
        print(f"GT: {pair['answer']}\n")
        gt = pair["answer"]
        qtype = pair.get("type", "unknown")
        each_type_total[qtype] += 1
        if judgement(output_image, gt):
            each_type_right[qtype] += 1
            acc += 1
            print("Right")
        else:
            wrong.append(pair)
        res.append({
            "qa_id": pair.get("qa_id", ""),
            "question": pair["question"],
            "choices": pair["choices"],
            "answer": output_image[0],
            "gt": gt,
            "type": qtype
        })

    return acc, wrong, res, each_type_right, each_type_total

def main():
    logger.info("开始主函数...")
    logger.info("进入 main ...")
    multiprocess = torch.cuda.device_count() >= 1
    logger.info(f"cuda 设备数量 = {torch.cuda.device_count()}")
    mp.set_start_method('spawn', force=True)
    if multiprocess:
        logger.info('开始多卡推理')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, test_pairs=test_pairs)
            result_lists = pool.map(func, range(world_size))

        global_right = 0
        global_wrong = []
        global_results = []
        each_type_right = defaultdict(int)
        each_type_total = defaultdict(int)

        for i in range(world_size):
            global_right += int(result_lists[i][0])
            global_wrong.extend(result_lists[i][1])
            global_results.extend(result_lists[i][2])
            for qtype, count in result_lists[i][3].items():
                each_type_right[qtype] += count
            for qtype, count in result_lists[i][4].items():
                each_type_total[qtype] += count

        logger.info('总正确数: ' + str(global_right))
        logger.info('总准确率: ' + str(global_right / len(test_pairs)))
        logger.info('每个类型的正确数量:')
        for qtype, count in each_type_right.items():
            logger.info(f'{qtype}: {count}')
            logger.info(f'{qtype} 准确率: {count / each_type_total[qtype]}')
        logger.info('每个类型的总数:')
        for qtype, count in each_type_total.items():
            logger.info(f'{qtype}: {count}')

        with open("global_results.json", "w", encoding="utf-8") as f:
            json.dump(global_results, f, indent=4, ensure_ascii=False)
        with open("global_wrong.txt", "w", encoding="utf-8") as f:
            for item in global_wrong:
                f.write(str(item) + "\n")
    else:
        logger.info("GPU 数量不足，无法进行多进程推理")

if __name__ == "__main__":
    main()
    rub = 9