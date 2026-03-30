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

# 配置日志输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("caption_ops_single_gpu.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger()
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
    # word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda", download_root='./clip_checkpoints', jit=False)
    text_generator = CLIPTextGenerator(**vars(args))

    dataset_folder = "./dataset_test_flux_6_relations"
    output_json = f"output_results_{dataset_folder.split('/')[-1]}.json"
    logging.info(f"DATASET: {dataset_folder}")

    results = []
    bleu_scores = []
    rouge_scores = []
    cosine_similarities = []

    relations = [f for f in os.listdir(dataset_folder)]
    logging.info("Start processing!.....")

    if op == "subtraction":
        # 做选择题
        for relation in relations:
            logging.info(f"======================= cur relation is {relation} =======================")
            cur_dir = os.path.join(dataset_folder, relation)
            pairs = [f for f in os.listdir(cur_dir)] # /start+end
            
            for pair in pairs:
                start, end = pair.split("+")
                start_img = glob.glob(os.path.join(os.path.join(cur_dir, pair), f"{start}.*"))[0]
                end_img = glob.glob(os.path.join(os.path.join(cur_dir, pair), f"{end}.*"))[0]
                
                gt = "An image of " + relation
            
                image_features = text_generator.get_combined_feature([str(start_img), str(end_img)], [], [1, -1], None)
                image_features = image_features.to('cuda')
                captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

                encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to('cuda')) for c in captions]
                encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
                best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()
                result = args.cond_text + captions[best_clip_idx]

                # Calculate BLEU-1 score (unigram precision)
                smoothing_function = SmoothingFunction()
                bleu_score = sentence_bleu([gt.split()], result.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function.method4)

                # Calculate Recall@5
                result_words = result.split()[:5]  # Get first 5 words
                gt_words = gt.split()
                recall_5 = int(any(word in result_words for word in gt_words))

                # Calculate CLIP score for ground truth
                gt_text = gt
                gt_features = text_generator.clip.encode_text(clip.tokenize(gt_text).to('cuda'))
                gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)
                
                # Calculate CLIP score for result
                result_features = text_generator.clip.encode_text(clip.tokenize(result).to('cuda'))
                result_features = result_features / result_features.norm(dim=-1, keepdim=True)
                
                clip_score = (gt_features @ result_features.t()).item()

                # 计算与关系词的clip分数
                relation_words = [args.cond_text + 'antonym', args.cond_text + 'part of', args.cond_text + 'made of', args.cond_text + 'used for']
                words2relation = {args.cond_text + 'antonym': 'antonym', args.cond_text + 'part of': 'partof', args.cond_text + 'made of': 'madeof', args.cond_text + 'used for': 'usedfor'}
                relation_scores = []
                for rel in relation_words:
                    rel_features = text_generator.clip.encode_text(clip.tokenize(rel).to('cuda'))
                    rel_features = rel_features / rel_features.norm(dim=-1, keepdim=True)
                    rel_score = (result_features @ rel_features.t()).item()
                    relation_scores.append(rel_score)
                
                # 找出分数最高的关系词
                best_relation = relation_words[np.argmax(relation_scores)]

                res2append = {
                    "gt": gt, # 这里不需要gt[i],因为在subtract模式下gt就是一个字符串,不是列表
                    "result": result,
                    "pair": f"{start}+{end}", # 需要记录具体是哪两张图片做减法
                    "start": start,
                    "end": end,
                    "bleu": bleu_score,
                    "recall@5": recall_5, 
                    "clip_score": clip_score,
                    "relation": best_relation,
                    "correct": gt.split()[-1] == words2relation[best_relation]
                }
                results.append(res2append)
                logging.info(" | ".join(f"{key}: {value}" for key, value in res2append.items()))
                   

    elif op == "summation":
        for relation in relations:
            logging.info(f"======================= cur relation is {relation} =======================")
            cur_dir = os.path.join(dataset_folder, relation)
            pairs = [f for f in os.listdir(cur_dir)]
            meta_pairs = list(itertools.combinations(pairs, 2))
            len_pairs = len(pairs)
            len_meta_pairs = len(meta_pairs)
            assert math.comb(len_pairs, 2) == len_meta_pairs
            # print(meta_pairs)
            
            for equation in meta_pairs:
                # equation = ("fish_pets", "cars_crash")
                concat = "+"
                start1_text, end1_text, start2_text, end2_text = equation[0].split(concat)[0], equation[0].split(concat)[1], equation[1].split(concat)[0], equation[1].split(concat)[1]
                # print(f"cur_dir = {cur_dir}")
                start1_img = glob.glob(os.path.join(os.path.join(cur_dir, equation[0]), f"{start1_text}.*"))[0]
                end1_img = glob.glob(os.path.join(os.path.join(cur_dir, equation[0]), f"{end1_text}.*"))[0]
                start2_img = glob.glob(os.path.join(os.path.join(cur_dir, equation[1]), f"{start2_text}.*"))[0]
                end2_img = glob.glob(os.path.join(os.path.join(cur_dir, equation[1]), f"{end2_text}.*"))[0]
                
                four_equations = [
                    (start2_img, end2_img, end1_img),
                    # (start1_img, start2_img, end2_img),
                    (start1_img, end1_img, end2_img),
                    # (start2_img, end1_img, start1_img)
                ]
                gt = [start1_text, start2_text]
                gt = ["An image of " + x for x in gt]
            
                for i, cmd_ in enumerate(four_equations):
                    if i == 0 or i == 2:
                        continue
                    image_features = text_generator.get_combined_feature([str(cmd_[0]), str(cmd_[1]), str(cmd_[2])], [], [1, -1, 1], None)
                    image_features = image_features.to('cuda')
                    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

                    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to('cuda')) for c in captions]
                    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
                    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()
                    result =  args.cond_text + captions[best_clip_idx]

                    # ----------------------------------------- METRICS -------------------------------------------
                    # Calculate BLEU-1 score (unigram precision)
                    smoothing_function = SmoothingFunction()
                    bleu_score = sentence_bleu([gt[i].split()], result.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function.method4)

                    # Calculate Recall@5
                    result_words = result.split()[:5]  # Get first 5 words
                    gt_words = gt[i].split()
                    recall_5 = int(any(word in result_words for word in gt_words))

                    # Calculate CLIP score for ground truth
                    gt_text = gt[i]
                    gt_features = text_generator.clip.encode_text(clip.tokenize(gt_text).to('cuda'))
                    gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate CLIP score for result
                    result_features = text_generator.clip.encode_text(clip.tokenize(result).to('cuda'))
                    result_features = result_features / result_features.norm(dim=-1, keepdim=True)
                    
                    clip_score = (gt_features @ result_features.t()).item()

                    
                    res2append = {
                        "gt": gt[i],
                        "result": result,
                        "equation": equation,
                        "bleu": bleu_score,
                        "recall@5": recall_5,
                        "clip_score": clip_score
                    }
                    results.append(res2append)
                    logging.info(" | ".join(f"{key}: {value}" for key, value in res2append.items()))
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