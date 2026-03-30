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
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.imagebind_model import ImageBindModel
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

import os
import json

# 假设visual7w-telling的dataset.json路径如下
dataset_json_path = "./data/dataset.json"

# 读取整个dataset.json
with open(dataset_json_path, "r") as f:
    dataset = json.load(f)

# 按split划分数据集
split2samples = {"train": [], "val": [], "test": []}
print(dataset.keys())
for item in dataset["images"]:
    # 每个item已经是字典格式,不需要再次解析
    split = item.get("split", "train")
    split2samples[split].append(item)

# 统计每个split的样本数
for split in split2samples:
    print(f"{split} 样本数: {len(split2samples[split])}")

# 以test split为例，构建测试对
test_samples = split2samples["test"]

# 构建测试对，每个pair为(question, multiple_choices, answer, image_path)
test_pairs = []
base_img_path = "./data/images"  # 你需要根据实际图片路径修改

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
        answer_content = answer_content.replace(' ','').replace('_','').lower()
        gt = gt.replace(' ','').replace('_','').lower()
        return answer_content == gt
    except Exception as e:
        return False
def run(rank, world_size, image_pairs):
    device = torch.device(f"cuda:{rank}")
    clip_type = {
                'image': 'LanguageBind_Image',
    }

        
    model = ImageBindModel.from_pretrained("nielsr/imagebind-huge")
    model.eval()
    model.to(device)

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
        relations = ["used for", "antonym", "made of", "part of", "at location", "created by"]
        map_to_dir_relations = {
            "used for": "usedfor",
            "antonym": "antonym",
            "made of": "madeof",
            "part of": "partof",
            "at location": "atlocation",
            "created by": "createdby"
        }
        pair = [pair.split('+')[1], pair.split('+')[0]]
        image = [pair_path[1], pair_path[0]]
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image, device),
            "pair": data.load_and_transform_text(pair, device),
            "relations": data.load_and_transform_text(relations, device),
        }
        
        with torch.no_grad():
            embeddings = model(inputs)
        
        logger.info("Text x Text: \n" +
                str(torch.softmax(embeddings['pair'] @ embeddings['relations'].T, dim=-1).detach().cpu().numpy()))
        max_relation = torch.argmax(torch.softmax(embeddings['pair'] @ embeddings['relations'].T, dim=-1), dim=-1)
        output_text = [map_to_dir_relations[relations[i]] for i in max_relation.detach().cpu().numpy()][0]



        logger.info("Image x Text: \n" +
            str(torch.softmax(embeddings[ModalityType.VISION] @ embeddings['relations'].T, dim=-1).detach().cpu().numpy()))
        max_relation = torch.argmax(torch.softmax(embeddings[ModalityType.VISION] @ embeddings['relations'].T, dim=-1), dim=-1)
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

        with open(f"global_results_intern_imagebind_{len(selected_relations)}_0411.json", "w") as f:
            json.dump(global_results, f, indent=4)
        
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()