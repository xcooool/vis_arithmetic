import json
import torch
import re
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
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
from PIL import Image
import functools
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
torch.manual_seed(1234)

from nltk.translate.bleu_score import sentence_bleu
from transformers import CLIPProcessor, CLIPModel
import torch
import re
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
print("load clip model and processor success")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logfile.log',  # 将日志存储到文件中
    filemode='a',  # 追加模式
    force=True
)
logger = logging.getLogger(__name__)

# 测试数据 —————————————————————————— 所有数据集 ————————————————————————————————

# 选择你要的relationships

selected_relations = ["atlocation", "createdby", "usedfor", "antonym", "madeof", "partof"]

# selected_relations = [ "usedfor", "antonym", "madeof", "partof"]
def parse(answer):
    try:
        match = re.search(r"<answer>(.*?)</answer>", answer)
        answer_content = match.group(1)
        # answer_content = answer_content.replace(' ','').replace('_','').lower()
        return answer_content
    except Exception as e:
        return answer
import os



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

def metrics(answer, gt):
    # 截断gt
    # gt = gt[:20]
    # Calculate BLEU-1
    ground_truth_with_prefix = "Image of a" + gt
    student_answer_with_prefix = "Image of a" + answer
    bleu_score = sentence_bleu([ground_truth_with_prefix.split()], student_answer_with_prefix.split(), weights=(1, 0, 0, 0))        

    # Calculate Recall@5
    student_words = student_answer_with_prefix.split()
    recall_at_5 = sum(1 for word in student_words if word in ground_truth_with_prefix.split()) / 5.0

    # Calculate CLIP-score
    # inputs = clip_processor(text=[ground_truth_with_prefix, student_answer_with_prefix], return_tensors="pt", padding=True)
    inputs = clip_processor(text=[gt, answer], return_tensors="pt", padding=True, truncation=True)
    text_features = clip_model.get_text_features(**inputs)
    clip_score = torch.cosine_similarity(text_features[0], text_features[1], dim=0).item()  
    if gt in answer or answer in gt:
        clip_score = 1.0
    # 做了一个映射 之所以选择0.6是因为qwen_original的均值是0.66
    clip_score = (clip_score - 0.6) / 0.4
    clip_score = max(0.0, min(clip_score, 1.0))
    return bleu_score, recall_at_5, clip_score


def run(rank, world_size, image_pairs):
    device = torch.device(f"cuda:{rank}")
    model_base = "LanguageBind/Video-LLaVA-7B-hf"
    # 加载模型
    model = VideoLlavaForConditionalGeneration.from_pretrained(
       model_base,
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        # device_map="cpu", # 一定要指定cpu或者一个cuda，如果用的是auto，会报错Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
    )
    

    # 加载处理器
    processor = VideoLlavaProcessor.from_pretrained(model_base)
    processor.vision_feature_select_strategy = "full"
    model = model.to(device)
    model = model.eval()

    acc_text = 0
    acc_image = 0

    split_images = split_dict(image_pairs, rank, world_size)

    text_bleu_score_list = []
    text_recall_at_5_list = []
    text_clip_score_list = []
    image_bleu_score_list = []
    image_recall_at_5_list = []
    image_clip_score_list = []
    acc_text = 0
    acc_image = 0
    total = 0

    res = []
    # 逐条处理 messages
    for pair, pair_path in tqdm(split_images.items()):
        total += 1

        text_prompt = f""" USER: There will be a relationship contained in first two texts. Please infer: {pair.split("+")[0]} to {pair.split("+")[1]} equals to [WHAT] to {pair.split("+")[3]}? Use a phrase. Answer in Engilish. ASSISTANT: """
        
        image_prompt = f"""USER: <image>\n <image>\n <image>\nThere will be a relationship contained in first two images. Please infer: image1 to image2 equals to [WHAT] to image3? Use a phrase.Answer in Engilish. ASSISTANT:"""

        
        

        # text -------------------  准备输入  -------------------
        inputs = processor(text=text_prompt, return_tensors="pt")
        inputs = inputs.to(device)
        print(f"text inputs的长度: {len(inputs)}")  # 输出inputs的长度

        # 推理：生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=60, use_cache=True)

        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        output_text = output_text[0].split("ASSISTANT:")[-1]

        # image -------------------  准备输入  -------------------
        print(f"pair_path: {pair_path}")
        img_paths = [Image.open(pair_path[0]), Image.open(pair_path[1]), Image.open(pair_path[2])]
        inputs = processor(text=image_prompt,images=img_paths, return_tensors="pt")
        inputs = inputs.to(device)
        print(f"image inputs的长度: {len(inputs)}")  # 输出inputs的长度
        # 推理：生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=80, use_cache=True)

        output_image = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        output_image = output_image[0].split("ASSISTANT:")[-1]
        
        # 打印输出
        # print(f"Input: {pair.split('+')[0]} and {pair.split('+')[1]}")
        # print(f"Output_text: {output_text[0]}\n")
        # print(f"Output_image: {output_image[0]}\n")
        gt = pair.split('+')[2]
        # print(f"GT: {gt}")
        if judgement(output_text, gt):
            acc_text += 1
        if judgement(output_image, gt):
            acc_image += 1
        answer_text = output_text
        answer_image = output_image
        print(f"Output_text: {answer_text}\n")
        print(f"Output_image: {answer_image}\n")
        print(f"GT: {gt}\n")
        text_bleu_score, text_recall_at_5, text_clip_score = metrics(answer_text, gt)
        image_bleu_score, image_recall_at_5, image_clip_score = metrics(answer_image, gt)

        res.append({"pair": (pair.split('+')[0], pair.split('+')[1], pair.split('+')[2]), "text": answer_text, "image": answer_image, "gt": gt, "text_bleu_score": text_bleu_score, "text_recall_at_5": text_recall_at_5, "text_clip_score": text_clip_score, "image_bleu_score": image_bleu_score, "image_recall_at_5": image_recall_at_5, "image_clip_score": image_clip_score})
        text_bleu_score_list.append(text_bleu_score)
        text_recall_at_5_list.append(text_recall_at_5)
        text_clip_score_list.append(text_clip_score)
        image_bleu_score_list.append(image_bleu_score)
        image_recall_at_5_list.append(image_recall_at_5)
        image_clip_score_list.append(image_clip_score)
    # print(f"ACCURACY>>>> text = {acc_text / total_len} | image = {acc_image / total_len}")
    # print(f"WRONG>>>> text = {wrong_text} | image = {wrong_image}")
    return {
        "acc_text": acc_text,
        "acc_image": acc_image,
        "text_bleu_score": text_bleu_score_list,
        "text_recall_at_5": text_recall_at_5_list,
        "text_clip_score": text_clip_score_list,
        "image_bleu_score": image_bleu_score_list,
        "image_recall_at_5": image_recall_at_5_list,
        "image_clip_score": image_clip_score_list,
        "res": res,
        "total": total
    }


def main():
    print("into main....")
    image_pairs = json.load(open("image_pairs_summation.json"))
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
        global_text_bleu_score = []
        global_text_recall_at_5 = []
        global_text_clip_score = []
        global_image_bleu_score = []
        global_image_recall_at_5 = []
        global_image_clip_score = []
        
        for i in range(world_size):
            global_right_text += int(result_lists[i]["acc_text"])
            global_right_image += int(result_lists[i]["acc_image"])
            global_results.extend(result_lists[i]["res"])
            global_text_bleu_score.extend(result_lists[i]["text_bleu_score"])
            global_text_recall_at_5.extend(result_lists[i]["text_recall_at_5"])
            global_text_clip_score.extend(result_lists[i]["text_clip_score"])
            global_image_bleu_score.extend(result_lists[i]["image_bleu_score"])
            global_image_recall_at_5.extend(result_lists[i]["image_recall_at_5"])
            global_image_clip_score.extend(result_lists[i]["image_clip_score"])


        logger.info('text right number: ' + str(global_right_text))  
        logger.info('image right number: ' + str(global_right_image))
        logger.info('text accuracy: ' + str(global_right_text / len(image_pairs)))
        logger.info('image accuracy: ' + str(global_right_image / len(image_pairs)))
        logger.info('text bleu score: ' + str(sum(global_text_bleu_score) / len(global_text_bleu_score)))
        logger.info('text recall at 5: ' + str(sum(global_text_recall_at_5) / len(global_text_recall_at_5)))
        logger.info('text clip score: ' + str(sum(global_text_clip_score) / len(global_text_clip_score)))
        logger.info('image bleu score: ' + str(sum(global_image_bleu_score) / len(global_image_bleu_score)))
        logger.info('image recall at 5: ' + str(sum(global_image_recall_at_5) / len(global_image_recall_at_5)))
        logger.info('image clip score: ' + str(sum(global_image_clip_score) / len(global_image_clip_score)))

        with open("global_results.json", "w") as f:
            json.dump(global_results, f, indent=4)
      
        
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()