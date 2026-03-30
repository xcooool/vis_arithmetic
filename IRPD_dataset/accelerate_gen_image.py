from collections import defaultdict
import os
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import torch
import time
import fire
import json
import clip
from PIL import Image
import random

START_TIME = time.strftime("%Y%m%d_%H%M%S")
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}



def generate_each_image(pipeline, seed, text, threshold, try_times):
    prompt = f"An image of {text}"
    while try_times > 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipeline(prompt, generator=generator).images[0]
        # calculate the clip score
        clip_score = get_clip_score(image=image, text=prompt)
        if clip_score < threshold and try_times > 0:
            seed = random.randint(0, 2**32 - 1)
            try_times -= 1
        else:
            return image, False
    # this text is not suitalble for drawing should be delete
    return None, True



def get_clip_score(image, text):
# Load the pre-trained CLIP model and the image
    # Preprocess the image and tokenize the text
    global CLIP_MODEL, CLIP_PROCESSOR
    image_input = CLIP_PROCESSOR(image).unsqueeze(0)
    text_input = clip.tokenize([text])
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    CLIP_MODEL = CLIP_MODEL.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = CLIP_MODEL.encode_image(image_input)
        text_features = CLIP_MODEL.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score

# Load your `data` dictionary
def load_data(pairs_dir):
    relation_data_ls = [d for d in os.listdir(pairs_dir)]
    data = defaultdict(list)
    for relation_data in relation_data_ls:
        with open(os.path.join(pairs_dir, relation_data), "r") as file:
            relation = relation_data.split("/")[-1].split(".")[0]
            data[relation] = json.load(file)[relation]

    # with open('./pairs.json', 'r') as file:
    #     data = json.load(file)
    return data

def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches

def main(
    ckpt_id: str = "stabilityai/stable-diffusion-2-base",
    save_dir: str = "./dataset/dataset",
    seed: int = 1,
    batch_size: int = 4,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    dtype: str = "fp16",
    low_mem: int = 0,
    pairs_dir: str = "./dataset_test_flux_4_relations",
    threhold: float = 0.23,
    try_times: int = 3
):
    global CLIP_MODEL, CLIP_PROCESSOR
    CLIP_MODEL, CLIP_PROCESSOR = clip.load('ViT-B/32')
    
    pipeline = StableDiffusionPipeline.from_pretrained(ckpt_id, torch_dtype=DTYPE_MAP[dtype])

    save_dir = save_dir + f"_{START_TIME}"
    data = load_data(pairs_dir)

    distributed_state = Accelerator()
    if low_mem:
        pipeline.enable_model_cpu_offload(gpu_id=distributed_state.device.index)
    else:
        pipeline = pipeline.to(distributed_state.device)

    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created successfully.")
        else:
            print(f"Directory '{save_dir}' already exists.")

    for relation, pairs_ls in data.items():
        print(f"Processing relation: {relation}")
        relation_save_dir = os.path.join(save_dir, relation)
        os.makedirs(relation_save_dir, exist_ok=True)

        data_loader = get_batches(items=pairs_ls, batch_size=batch_size)
            
        count = 0
        for _, pairs_raw in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_prompts = []

            with distributed_state.split_between_processes(pairs_raw) as pairs:

                for pair in pairs:
                    
                    image_start, text_not_suitable_start = generate_each_image(pipeline, seed, pair[0], threhold, try_times)
                    image_end, text_not_suitable_end = generate_each_image(pipeline, seed, pair[1], threhold, try_times)
                    if text_not_suitable_start or text_not_suitable_end:
                        continue # 直接丢弃这个pairs
                    # 创建文件夹名称
                    folder_name = f"{pair[0]}+{pair[1]}"
                    pair_save_dir = os.path.join(relation_save_dir, folder_name)
                    os.makedirs(pair_save_dir, exist_ok=True)

                    # 保存生成的图片
                    image_start.save(os.path.join(pair_save_dir, f"{pair[0]}.png"))
                    image_end.save(os.path.join(pair_save_dir, f"{pair[1]}.png"))

                    input_prompts.extend(pair)

            distributed_state.wait_for_everyone()

    if distributed_state.is_main_process:
        print(f">>> Image Generation Finished. Saved in {save_dir}")


if __name__ == "__main__":

    fire.Fire(main)
