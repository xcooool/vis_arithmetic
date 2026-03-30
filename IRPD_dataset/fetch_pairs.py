import requests
from collections import defaultdict
from itertools import combinations
import pickle
import spacy
import os
import requests
import json
import base64
import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import time
from collections import defaultdict

client = AzureOpenAI(
  azure_endpoint = "",
  api_key = "",
  api_version = ""
)

def isvalid_chatgpt(word1, word2, relation):
    '''
    Check whether this word has some multi meaning and can not easily be drawn
    '''
    try:
        response = client.chat.completions.create(
            model='gpt-35-turbo',
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond in JSON format. The format is: \{\"word1\":{\"polysemous\": true or false, \"easy_drawing\": true or false}, \"word2\": {\"polysemous\": true or false, \"easy_drawing\": true or false}\}. I will input two words and one relation wrapped with tags. For example, the input is \"<word1>Trip</word1> <relation>causedby</relation>  <word2>drug</word2>.\" You should check two words whether is polysemous and can be mapped into a substance in the real world, which is easy-drawing. The word trip has multi-meaning and can not map a substance in the real world. Therefore, the output should be \{\"drug\":{\"polysemous\": false, \"easy_drawing\": true},\"trip\":{\"polysemous\": true, \"easy_drawing\": false}\}"},
                {"role": "user", "content": f"<word1>{word1}</word1> <relation>{relation}</relation> <word2>{word2}</word2>. Please respond in a JSON format."},
            ],
        )
    except Exception as e:
        print(f"request to openai: {e}")
        return False
    print(response.choices[0].message.content)
    try:
        res = json.loads(response.choices[0].message.content.replace("\\", ""))
        
        if word1 not in res.keys() or word2 not in res.keys():
            return False
        
        if res[word1].get("polysemous") == False and res[word2].get("polysemous") == False:
            return True
        else:
            return False

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def isvalid_qwen(word1, word2, relation):
    inputs = tokenizer(f'I will ask you two questions. You must return text with the format of json. First question, is there a clear relation {relation} between {word1} and {word2}? Second question, are {word1} and {word2} both a substance in the real world and easy to draw? For example, the input is word1:cat, word2: bed, relation: atlocation. You should return a json like: \"relation\": \"false\", \"substance\": \"true\". Because it is not strong relation between cat and bed, but cat and bed is a substance in the real world.', return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    res_data = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    print(f"{word1} | {word2} | {relation} | {res_data}")
    try:
        res = json.loads(res_data)
        return res['relation'] and res['substance']
    except Exception as e:
        print(e)

def notExist(word1, word2, relation, path):
    res1 = False
    file_path = os.path.join(path, f"{relation}.json")
    try:
        with open(file_path, 'r') as f:
            pair_list = json.load(f)[relation]
            existing_pairs = set(tuple(pair) for pair in pair_list)
            if not (word1, word2) in existing_pairs and not (word2, word1) in existing_pairs:
                res1 = True
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    return res1 



pairs_d = defaultdict(list)
nlp = spacy.load("en_core_web_sm")

def is_noun(word):
    doc = nlp(word)
    return any(token.pos_ == "NOUN" for token in doc) and not any(token.pos_ == "VERB" for token in doc)

word = "typing_letters"
# print(f"{word} is a noun: {is_noun(word)}")
pair_each_relation = 100
START_TIME = time.strftime("%Y%m%d_%H%M%S")
if not os.path.exists(f'./pairs/pairs_{START_TIME}'):
    os.mkdir(f'./pairs/pairs_{START_TIME}')

for relation in [
    "/r/HasA", 
    "/r/CapableOf", 
    # "/r/AtLocation", 
    "/r/Causes", 
    # "/r/CreatedBy", 
    # "/r/Antonym", 
    "/r/Synonym", 
    # "/r/UsedFor", 
    # "/r/PartOf", 
    # "/r/MadeOf",
    "/r/RelatedTo",
    "/r/IsA",
    "/r/HasSubevent",
    "/r/HasPrerequisite",
    "/r/HasProperty",
    "/r/ObstructedBy",
    "/r/Desires",
    "/r/DistinctFrom",
    "/r/DerivedFrom",
    "/r/HasFirstSubevent",
    "/r/HasLastSubevent",
    "/r/SymbolOf",
    "/r/DefinedAs",
    "/r/LocatedNear",
    "/r/HasContext",
    "/r/ReceivesAction"
    ]:
    limit = 1000
    offset = 0
    processed_relation = relation.split("/")[-1].lower()
    file_path = f'./pairs/pairs_{START_TIME}/{processed_relation}.json'
    while True:
        url = f'https://api.conceptnet.io{relation}?offset={offset}&limit={limit}'
        obj = requests.get(url).json()
        print(f"Fetch {url} success!")
        if offset == 50000: 
            # if offset exceed the limit, then pass
            break
        flag = False
        for edge in obj['edges']:
            start_id = edge['start']['@id']
            end_id = edge['end']['@id']
            if end_id.split("/")[1] == "c" and end_id.split("/")[2] == "en" and start_id.split("/")[2] == "en" and start_id.split("/")[1] == "c":
                word1 = edge['start']['label']
                word2 = edge['end']['label']
                if is_noun(word1) and is_noun(word2) :
                    if (word1, word2) not in pairs_d[processed_relation] and (word2, word1) not in pairs_d[processed_relation] and word1 != word2 and word1 not in word2 and word2 not in word1:
                        train_path = "./pairs/pairs_20241207_183425_train"
                        test_path = "./pairs/pairs_20250320_125954_test_need_easy_drawing"
                        if isvalid_chatgpt(word1, word2, processed_relation) :
                            pairs_d[processed_relation].append((word1, word2))
                            if len(pairs_d[processed_relation]) == pair_each_relation:
                                flag = True
                                break
        if flag:
            break
        offset += limit
    with open(f"{file_path}", 'w') as f:
        json.dump(pairs_d, f)
        pairs_d = defaultdict(list)
        print(f"Saving successfully to {file_path}!.....")

