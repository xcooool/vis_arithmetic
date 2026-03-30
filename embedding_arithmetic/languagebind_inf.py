import json
import torch
import re
import os
import re
import json
from tqdm import tqdm
import logging
import os
import re
import json
import torch
from tqdm import tqdm
import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

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
    filename="res.log",
    force=True
)
logger = logging.getLogger(__name__)

# 测试数据 —————————————————————————— 所有数据集 ————————————————————————————————

# 选择你要的relationships

selected_relations = ["atlocation", "createdby", "usedfor", "antonym", "madeof", "partof"]

# selected_relations = [ "usedfor", "antonym", "madeof", "partof"]

import os

# 基础路径
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
logger.info(f"找到的测试对数: {len(test_pairs)}")
logger.info("测试对示例:")
for pair in test_pairs[:3]:  # 打印前3个测试对作为示例
    logger.info(pair)

# -------------- Prompt 1 --------------
import os

image_pairs = {}
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

logger.info(f"找到的图片对数: {len(image_pairs)}")

    
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
    clip_type = {
                'image': 'LanguageBind_Image',
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    acc_text = 0
    acc_image = 0

    split_images = split_dict(image_pairs, rank, world_size)

    wrong_text = []
    wrong_image = []
    each_relation_text_right = defaultdict(int)
    each_relation_image_right = defaultdict(int)
    each_relation_total = defaultdict(int)
    embedding_text = {}
    embedding_image = {}
    res = []
    # 逐条处理 messages
    for pair, pair_path in tqdm(split_images.items()):
        # -------------- 处理text的sub ---------------
        relations = ["used for", "antonym", "made of", "part of"]
        map_to_dir_relations = {
            "used for": "usedfor",
            "antonym": "antonym",
            "made of": "madeof",
            "part of": "partof",
            # "at location": "atlocation",
            # "created by": "createdby"
        }
        pair = [pair.split('+')[1], pair.split('+')[0]]
        image = [pair_path[1], pair_path[0]]
        inputs = {
            'image': to_device(modality_transform['image'](image), device)
        }
        inputs['relations'] = to_device(tokenizer(relations, max_length=10, padding='max_length',
                                                truncation=True, return_tensors='pt'), device)
        inputs['pair'] = to_device(tokenizer(pair, max_length=10, padding='max_length',
                                                truncation=True, return_tensors='pt'), device)
        
        with torch.no_grad():
            embeddings = model(inputs)
        
        logger.info("Text x Text: \n" +
                str(torch.softmax(embeddings['pair'] @ embeddings['relations'].T, dim=-1).detach().cpu().numpy()))
        max_relation = torch.argmax(torch.softmax(embeddings['pair'] @ embeddings['relations'].T, dim=-1), dim=-1)
        output_text = [map_to_dir_relations[relations[i]] for i in max_relation.detach().cpu().numpy()][0]



        logger.info("Image x Text: \n" +
            str(torch.softmax(embeddings['image'] @ embeddings['relations'].T, dim=-1).detach().cpu().numpy()))
        max_relation = torch.argmax(torch.softmax(embeddings['image'] @ embeddings['relations'].T, dim=-1), dim=-1)
        output_image = [map_to_dir_relations[relations[i]] for i in max_relation.detach().cpu().numpy()][0]


        gt = pair_path[0].split("/")[-3]
        logger.info(f"GT: {gt}")
        logger.info(f"output_text: {output_text}")
        logger.info(f"output_image: {output_image}")
        each_relation_total[gt] += 1
        if output_text == gt:
            logger.info("text right")
            acc_text += 1
            each_relation_text_right[gt] += 1
        else:
            wrong_text.append(pair)
        if output_image == gt:
            logger.info("image right")
            acc_image += 1
            each_relation_image_right[gt] += 1
        else:
            wrong_image.append(pair)
        res.append({"pair": (pair[0], pair[1]), "text": output_text, "image": output_image, "gt": gt})

    # print(f"ACCURACY>>>> text = {acc_text / total_len} | image = {acc_image / total_len}")
    # print(f"WRONG>>>> text = {wrong_text} | image = {wrong_image}")
    
    return acc_text, acc_image, wrong_text, wrong_image, res, each_relation_text_right, each_relation_image_right, each_relation_total



def main():
    logger.info("Starting main function...")
    logger.info("into main....")
    multiprocess = torch.cuda.device_count() >= 2
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

        with open(f"global_results_languagebind_sub_{len(selected_relations)}_0411.json", "w") as f:
            json.dump(global_results, f, indent=4)
        # with open("global_wrong_text.txt", "w") as f:
        #     for item in global_wrong_text:
        #         f.write(item + "\n")
        # with open("global_wrong_image.txt", "w") as f:
        #     for item in global_wrong_image:
        #         f.write(item + "\n")
        
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()