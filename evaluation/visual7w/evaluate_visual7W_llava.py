import json
import torch
import re
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
import os
from tqdm import tqdm
import logging
import functools
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
from PIL import Image

torch.manual_seed(1234)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

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
    model_base = "LanguageBind/Video-LLaVA-7B-hf"
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = VideoLlavaProcessor.from_pretrained(model_base)
    # Fix the vision feature selection strategy - try different strategies if needed
    try:
        processor.vision_feature_select_strategy = "default"
    except:
        try:
            processor.vision_feature_select_strategy = "full"
        except:
            processor.vision_feature_select_strategy = "patch"
    model = model.eval()
    
    # Debug: Print model configuration
    logger.info(f"Model device: {model.device}")
    logger.info(f"Vision feature select strategy: {processor.vision_feature_select_strategy}")
    logger.info(f"Model config: {model.config}")

    acc = 0
    wrong = []
    each_type_right = defaultdict(int)
    each_type_total = defaultdict(int)
    res = []

    split_pairs = split_list(test_pairs, rank, world_size)

    for pair in tqdm(split_pairs):
        # 构造LLaVA风格的prompt
        # 选项用换行分隔
        choices_str = "\n".join(pair['choices'])
        image_prompt = f"""USER: <image>\nQuestion: {pair['question']}\nChoices: {choices_str}\nPlease select the most appropriate answer from the choices above.\nOutput your reasoning in <think> </think> tags and the final answer in <answer> </answer> tags.\nThe output format should be:\n<think> ... </think> <answer>your answer</answer>\nPlease strictly follow the format. ASSISTANT:"""

        # 加载图片
        try:
            image = Image.open(pair["img_path"]).convert("RGB")
        except Exception as e:
            print(f"图片加载失败: {pair['img_path']}, 错误: {e}")
            wrong.append(pair)
            continue

        # 处理输入
        inputs = processor(text=image_prompt, images=image, return_tensors="pt")
        # Move inputs to the same device as the model
        for key, value in inputs.items():
            if hasattr(value, 'to'):
                inputs[key] = value.to(model.device)

        # 推理
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
            output_image = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_image = output_image[0].split("ASSISTANT:")[-1]
        except Exception as e:
            print(f"推理失败: {e}")
            print(f"输入形状: {inputs['input_ids'].shape if 'input_ids' in inputs else 'N/A'}")
            print(f"注意力掩码形状: {inputs['attention_mask'].shape if 'attention_mask' in inputs else 'N/A'}")
            wrong.append(pair)
            continue

        print(f"Input: {pair['question']}")
        print(f"Output_image: {output_image}\n")
        print(f"GT: {pair['answer']}\n")
        gt = pair["answer"]
        qtype = pair.get("type", "unknown")
        each_type_total[qtype] += 1
        if judgement(output_image, gt):
            each_type_right[qtype] += 1
            acc += 1
        else:
            wrong.append(pair)
        res.append({
            "qa_id": pair.get("qa_id", ""),
            "question": pair["question"],
            "choices": pair["choices"],
            "answer": output_image,
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