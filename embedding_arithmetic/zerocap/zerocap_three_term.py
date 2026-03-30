import argparse
import logging
import torch
import clip
from model.ImageCLIP import CLIPTextGenerator
from model.ImageCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import math
import glob
import itertools
import json
import numpy as np
from accelerate import Accelerator
from accelerate.utils import gather_object
from collections import defaultdict

# 配置日志输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("caption_ops_multi_gpu.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger()

def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--cond_text", type=str, default="An image of ")
    parser.add_argument("--reset_context_delta", action="store_true",
                        help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--multi_gpu", action="store_true")

    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics'])

    parser.add_argument("--caption_img_path", type=str, default='example_images/captions/COCO_val2014_000000008775.jpg',
                        help="Path to image for captioning")

    parser.add_argument("--arithmetics_imgs", nargs="+",
                        default=['example_images/arithmetics/woman2.jpg',
                                 'example_images/arithmetics/king2.jpg',
                                 'example_images/arithmetics/man2.jpg'])
    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])

    args = parser.parse_args()

    return args

def perplexity_score(text, lm_model, lm_tokenizer, device):
    encodings = lm_tokenizer(f'{lm_tokenizer.bos_token + text}', return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    outputs = lm_model(input_ids, labels=target_ids)
    log_likelihood = outputs[0]
    ll = log_likelihood.item()

    return ll

def run(args, img_path):
    if args.multi_gpu:
        text_generator = CLIPTextGenerator_multigpu(**vars(args))
    else:
        text_generator = CLIPTextGenerator(**vars(args))

    image_features = text_generator.get_img_feature([img_path], None)
    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', args.cond_text + captions[best_clip_idx])

def run_arithmetic(args, imgs_path, img_weights, op):
    # 初始化 accelerator
    accelerator = Accelerator()
    text_generator = CLIPTextGenerator(**vars(args))
    text_generator = accelerator.prepare(text_generator)

    dataset_folder = "./dataset_test_flux_4_relations"
    output_json = f"output_results_{dataset_folder.split('/')[-1]}_image_gpt2_medium_4_relations.json"
    
    # 只在主进程中打印日志
    if accelerator.is_main_process:
        logging.info(f"DATASET: {dataset_folder}")
        logging.info("Start processing!.....")

    all_results = []
    relations = [f for f in os.listdir(dataset_folder)]
    from tqdm import tqdm

    if op == "subtraction":
        # 做选择题
        for relation in tqdm(relations, desc="Processing relations", disable=not accelerator.is_main_process):
            results = []
            if accelerator.is_main_process:
                logging.info(f"======================= cur relation is {relation} =======================")
            
            cur_dir = os.path.join(dataset_folder, relation)
            pairs = [f for f in os.listdir(cur_dir)]

            local_batches = get_batches(pairs, args.batch_size * accelerator.num_processes)
            process_batches = local_batches[accelerator.process_index::accelerator.num_processes]

            for batch in tqdm(process_batches, desc=f"Processing batches for {relation}", leave=False):
                batch_info = []
                
                for pair in batch:
                    start, end = pair.split("+")
                    start_img = glob.glob(os.path.join(os.path.join(cur_dir, pair), f"{start}.*"))[0]
                    end_img = glob.glob(os.path.join(os.path.join(cur_dir, pair), f"{end}.*"))[0]
                    
                    batch_info.append((start_img, end_img, start, end, pair))
                
                gt = "An image of " + relation
                
                # 逐对处理图片特征
                for start_img, end_img, start, end, pair in tqdm(batch_info, desc="Processing image pairs", leave=False):
                    image_features = text_generator.get_combined_feature([str(start_img), str(end_img)], [], [1, -1], None)
                    image_features = image_features.to(accelerator.device)
                    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)
                    
                    result = args.cond_text + captions[0]
                    
                    # Calculate metrics
                    smoothing_function = SmoothingFunction()
                    bleu_score = sentence_bleu([gt.split()], result.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function.method4)

                    result_words = result.split()[:5]
                    gt_words = gt.split()
                    recall_5 = int(any(word in result_words for word in gt_words))

                    gt_features = text_generator.clip.encode_text(clip.tokenize(gt).to(accelerator.device))
                    gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)
                    
                    result_features = text_generator.clip.encode_text(clip.tokenize(result).to(accelerator.device))
                    result_features = result_features / result_features.norm(dim=-1, keepdim=True)
                    
                    clip_score = (gt_features @ result_features.t()).item()

                    relation_words = [args.cond_text + 'antonym', args.cond_text + 'part of', args.cond_text + 'made of', args.cond_text + 'used for', args.cond_text + 'at location', args.cond_text + 'created by']
                    words2relation = {args.cond_text + 'antonym': 'antonym', args.cond_text + 'part of': 'partof', args.cond_text + 'made of': 'madeof', args.cond_text + 'used for': 'usedfor', args.cond_text + 'at location': 'atlocation', args.cond_text + 'created by': 'createdby'}
                    
                    relation_scores = []
                    for rel in relation_words:
                        rel_features = text_generator.clip.encode_text(clip.tokenize(rel).to(accelerator.device))
                        rel_features = rel_features / rel_features.norm(dim=-1, keepdim=True)
                        rel_score = (result_features @ rel_features.t()).item()
                        relation_scores.append(rel_score)
                    
                    best_relation = relation_words[np.argmax(relation_scores)]

                    res2append = {
                        "gt": gt,
                        "result": result,
                        "pair": f"{start}+{end}",
                        "start": start,
                        "end": end,
                        "bleu": bleu_score,
                        "recall@5": recall_5, 
                        "clip_score": clip_score,
                        "relation": best_relation,
                        "correct": gt.split()[-1] == words2relation[best_relation]
                    }
                    results.append(res2append)
                    if accelerator.is_main_process:
                        logging.info(" | ".join(f"{key}: {value}" for key, value in res2append.items()))

            # 等待所有进程完成当前relation的处理
            accelerator.wait_for_everyone()
            # 收集所有进程的结果
            gathered_results = gather_object(results)
            if accelerator.is_main_process:
                # 将收集到的结果展平并添加到总结果中
                for res_list in gathered_results:
                    all_results.extend(res_list)
    elif op == "summation":
        for relation in relations:
            if accelerator.is_main_process:
                logging.info(f"======================= cur relation is {relation} =======================")
            
            cur_dir = os.path.join(dataset_folder, relation)
            pairs = [f for f in os.listdir(cur_dir)]
            meta_pairs = list(itertools.combinations(pairs, 2))
            
            # 将meta_pairs分成batch
            batches = get_batches(meta_pairs, args.batch_size)
            
            with accelerator.split_between_processes(batches) as batches_per_process:
                for batch in batches_per_process:
                    batch_images = []
                    batch_info = []
                    
                    for equation in batch:
                        concat = "+"
                        start1_text, end1_text, start2_text, end2_text = equation[0].split(concat)[0], equation[0].split(concat)[1], equation[1].split(concat)[0], equation[1].split(concat)[1]
                        
                        start1_img = glob.glob(os.path.join(os.path.join(cur_dir, equation[0]), f"{start1_text}.*"))[0]
                        end1_img = glob.glob(os.path.join(os.path.join(cur_dir, equation[0]), f"{end1_text}.*"))[0]
                        start2_img = glob.glob(os.path.join(os.path.join(cur_dir, equation[1]), f"{start2_text}.*"))[0]
                        end2_img = glob.glob(os.path.join(os.path.join(cur_dir, equation[1]), f"{end2_text}.*"))[0]
                        
                        four_equations = [
                            (start2_img, end2_img, end1_img),
                            (start1_img, end1_img, end2_img),
                        ]
                        
                        for cmd_ in four_equations:
                            batch_images.extend([str(img) for img in cmd_])
                            batch_info.append((equation, start1_text, start2_text))
                    
                    # 批量处理图片特征
                    image_features = text_generator.get_combined_feature(batch_images, [], [1, -1, 1] * len(batch), None)
                    image_features = image_features.to(accelerator.device)
                    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)
                    
                    # 批量处理结果
                    for i, (equation, start1_text, start2_text) in enumerate(batch_info):
                        gt = [start1_text, start2_text]
                        gt = ["An image of " + x for x in gt]
                        result = args.cond_text + captions[i]
                        
                        smoothing_function = SmoothingFunction()
                        bleu_score = sentence_bleu([gt[i % 2].split()], result.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function.method4)

                        result_words = result.split()[:5]
                        gt_words = gt[i % 2].split()
                        recall_5 = int(any(word in result_words for word in gt_words))

                        gt_text = gt[i % 2]
                        gt_features = text_generator.clip.encode_text(clip.tokenize(gt_text).to(accelerator.device))
                        gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)
                        
                        result_features = text_generator.clip.encode_text(clip.tokenize(result).to(accelerator.device))
                        result_features = result_features / result_features.norm(dim=-1, keepdim=True)
                        
                        clip_score = (gt_features @ result_features.t()).item()
                        
                        res2append = {
                            "gt": gt[i % 2],
                            "result": result,
                            "equation": equation,
                            "bleu": bleu_score,
                            "recall@5": recall_5,
                            "clip_score": clip_score
                        }
                        results.append(res2append)
                        if accelerator.is_main_process:
                            logging.info(" | ".join(f"{key}: {value}" for key, value in res2append.items()))

            accelerator.wait_for_everyone()
            results = gather_object(results)

    # 只在主进程中保存结果
    if accelerator.is_main_process:
        with open(output_json, "w") as json_file:
            json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    args = get_args()

    if args.run_type == 'caption':
        run(args, img_path=args.caption_img_path)
    elif args.run_type == 'arithmetics':
        args.arithmetics_weights = [float(x) for x in args.arithmetics_weights]
        run_arithmetic(args, imgs_path=args.arithmetics_imgs, img_weights=args.arithmetics_weights, op="subtraction")
    else:
        raise Exception('run_type must be caption or arithmetics!')