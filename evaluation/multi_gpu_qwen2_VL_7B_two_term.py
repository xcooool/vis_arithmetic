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

# 测试数据 —————————————————————————— 所有数据集 ————————————————————————————————

# 选择你要的relationships

# selected_relations = [ "usedfor", "antonym", "madeof", "partof"]

selected_relations = [ "usedfor", "antonym", "madeof", "partof", "irrelevant"]

import os


base_path = "./dataset_test_flux_6_relations"

# 存储所有子文件夹名称的列表
test_pairs = []

# 遍历基础路径下的所有子文件夹
for root, dirs, files in os.walk(base_path):
    # 只处理直接位于base_path下的文件夹
    if root == base_path:
        for dir_name in dirs:
            # 检查文件夹名是否在selected_relations中
            if dir_name in selected_relations:
                relation_path = os.path.join(base_path, dir_name)
                # 遍历关系文件夹下的所有子文件夹
                for subdir in os.listdir(relation_path):
                    pair = subdir.split("+")
                    # 确保分割后有两个元素
                    if len(pair) == 2:
                        test_pairs.append(pair)

# 打印结果
print(f"找到的测试对数: {len(test_pairs)}")
print("测试对示例:")
for pair in test_pairs[:3]:  # 打印前3个测试对作为示例
    print(pair)
print(f"selected_relations: {len(selected_relations)}")


# -------------- Prompt 1 --------------
import os


# 存储所有图片路径的列表
image_pairs = {}

# 遍历每个测试对
for pair in test_pairs:
    # 构建子文件夹名称
    subfolder_name = f"{pair[0]}+{pair[1]}"
    
    # 遍历所有可能的关系文件夹
    for relationship in selected_relations:
        # 构建完整路径
        full_path = os.path.join(base_path, relationship, subfolder_name)
        
        # 如果路径存在
        if os.path.exists(full_path):
            # 获取该文件夹下的所有文件
            pair_path = []
            # 先找到{pair[0]}的图片
            for file in os.listdir(full_path):
                if file.lower().endswith(('.jpg', '.png')) and pair[0] in file:
                    image_path = os.path.join(full_path, file)
                    pair_path.append(image_path)
            # 再找到{pair[1]}的图片
            for file in os.listdir(full_path):
                if file.lower().endswith(('.jpg', '.png')) and pair[1] in file:
                    image_path = os.path.join(full_path, file)
                    pair_path.append(image_path)
            image_pairs[subfolder_name] = pair_path

print(f"找到的图片对数: {len(image_pairs)}")


def judgement(output_text, gt):
    try:
        match = re.search(r"<answer>(.*?)</answer>", output_text)
        answer_content = match.group(1)
        answer_content = answer_content.replace(' ','').replace('_','').lower()
        # judgement
        if answer_content == gt:
            return True
        else:
            return False
    except Exception as e:
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

def run(rank, world_size, image_pairs):
    device = torch.device(f"cuda:{rank}")
    model_path = "./Qwen2-VL-7B-Instruct_GRPO_4_relations_flux_1024"  # after RL
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

    acc_text = 0
    acc_image = 0

    split_images = split_dict(image_pairs, rank, world_size)

    wrong_text = []
    wrong_image = []
    each_relation_text_right = defaultdict(int)
    each_relation_image_right = defaultdict(int)
    each_relation_total = defaultdict(int)

    res = []
    # 逐条处理 messages
    for pair, pair_path in tqdm(split_images.items()):

        # 构建text单条消息
        message_text = {
            "role": "user",
            "content": [
    #             {
    #                 "type": "text",
    #                 "text": f"""What relation is contained between {pair.split('+')[0]} and {pair.split('+')[1]}? Choose only one phrase from: {', '.join(selected_relations)}. Use one word.
    # Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:
    # <think> ... </think> <answer>species name</answer>
    # Please strictly follow the format."""
    #             }
                {
                    "type": "text",
                    "text": f"""What relation is contained between {pair.split('+')[0]} and {pair.split('+')[1]}? Choose only one phrase from: {', '.join(selected_relations)}. Use one word.
    Output the final answer in <answer> </answer> tags.The output answer format should be as follows:
    <answer>relation name</answer>
    Please strictly follow the format."""
                }
            ]
        }
        message_image = {
                "role": "user",
                "content": [
                    {"type": "image", "image": pair_path[0]},
                    {"type": "image", "image": pair_path[1]},
                    {"type": "text", "text": f"""What relation is contained between two images? Choose only one phrase from: {', '.join(selected_relations)}. Use one word.
    Output the final answer in <answer> </answer> tags.The output answer format should be as follows:
    <answer>relation name</answer>
    Please strictly follow the format.""",}
                ],
            }

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
        print(f"Input: {pair.split('+')[0]} and {pair.split('+')[1]}")
        print(f"Output_text: {output_text[0]}\n")
        print(f"Output_image: {output_image[0]}\n")
        gt = pair_path[0].split("/")[-3]
        # print(f"GT: {gt}")

        each_relation_total[gt] += 1
        if judgement(output_text[0], gt):
            print("text right")
            acc_text += 1
            each_relation_text_right[gt] += 1
        else:
            wrong_text.append(pair)
        if judgement(output_image[0], gt):
            print("image right")
            acc_image += 1
            each_relation_image_right[gt] += 1
        else:
            wrong_image.append(pair)
        res.append({"pair": (pair.split('+')[0], pair.split('+')[1]), "text": output_text[0], "image": output_image[0], "gt": gt})

    # print(f"ACCURACY>>>> text = {acc_text / total_len} | image = {acc_image / total_len}")
    # print(f"WRONG>>>> text = {wrong_text} | image = {wrong_image}")
    
    return acc_text, acc_image, wrong_text, wrong_image, res, each_relation_text_right, each_relation_image_right, each_relation_total



def main():
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
        global_wrong_text = []  # 所有text wrong pair
        global_wrong_image = [] # 所有image wrong pair
        each_relation_text_right = defaultdict(int) # 每个relation正确的text的数量
        each_relation_image_right = defaultdict(int) # 每个relation正确的image的数量
        each_relation_total = defaultdict(int) # 每个relation的总数
        for i in range(world_size):
            global_right_text += int(result_lists[i][0])
            global_right_image += int(result_lists[i][1])
            global_results.extend(result_lists[i][4])
            global_wrong_text.extend(result_lists[i][2])
            global_wrong_image.extend(result_lists[i][3])
            for relation, count in result_lists[i][5].items():
                if relation not in each_relation_text_right:
                    each_relation_text_right[relation] = count
                else:
                    each_relation_text_right[relation] += count

            for relation, count in result_lists[i][6].items():
                if relation not in each_relation_image_right:
                    each_relation_image_right[relation] = count
                else:
                    each_relation_image_right[relation] += count
            for relation, count in result_lists[i][7].items():
                if relation not in each_relation_total:
                    each_relation_total[relation] = count
                else:
                    each_relation_total[relation] += count

        logger.info('text right number: ' + str(global_right_text))  
        logger.info('image right number: ' + str(global_right_image))
        logger.info('text accuracy: ' + str(global_right_text / len(image_pairs)))
        logger.info('image accuracy: ' + str(global_right_image / len(image_pairs)))
        logger.info('每个关系的文本正确数量:')
        for relation, count in each_relation_text_right.items():
            logger.info(f'{relation}: {count}')
            logger.info(f'{relation} accuracy: {count / each_relation_total[relation]}')

        logger.info('每个关系的图像正确数量:')
        for relation, count in each_relation_image_right.items():
            logger.info(f'{relation}: {count}')
            logger.info(f'{relation} accuracy: {count / each_relation_total[relation]}')

        logger.info('每个关系的总数:')
        for relation, count in each_relation_total.items():
            logger.info(f'{relation}: {count}')

        with open("global_results_qwen_6_0410.json", "w") as f:
            json.dump(global_results, f, indent=4)
        with open("global_wrong_text.txt", "w") as f:
            for item in global_wrong_text:
                f.write(item + "\n")
        with open("global_wrong_image.txt", "w") as f:
            for item in global_wrong_image:
                f.write(item + "\n")
        
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()