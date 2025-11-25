##this evalutes the core metric for a given model 
## run on a single gpu - python -m scripts.base_eval 
## using torchrun on eg 8 gpus 
## torchrun --nproc_per_node=8 -m scripts.base_eval

##the script will print the core metric to the console 


import os 
import csv
import time 
import json 
import yaml 
import random 
import shutil 
import zipfile 
import tempfile 
from contextlib import nullcontext 

import torch 
from common import compute_init, compute_cleanup, print0, print_banner, get_base_dir, autodetect_device_type, download_file_with_lock
from tokenizer import HuggingfaceTokenizer
from checkpoint_manager import load_model
from core_eval import evaluate_bpb


#nanochat specific function dealing with I/O
##data needed for core metric 
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle():
    #file path is the path to the eval_bundle.zip file 
    base_dir = get_base_dir()
    eval_bundle_path = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        extracted_bundle_dir = os.path.join(tmp_dir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")



def evaluate_model(model, tokenizer, device, max_per_task=-1):
    ##evaluate the base model on the core benchmark 
    ##max_per_task crop the data to this many examples per task for testing 

    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    ##download the eval bundle to disk and unzip if needed 
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL,"eval_bundle.zip", postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(data_base_path, "eval_meta_data.csv")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    ##load random base line values from the eval metadata 

    random_baseline = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline[task_name] = row['Random baseline']
            random_baseline[task_name] = float(random_baseline)

            ##evaluate each task 
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'], 
            'dataset_uri': task['dataset_uri'], 
            'num_fewshot': task['num_fewshot'][0], 
            'continuation_delimeter' : task.get('continuation_delimeter', '')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        ##load the data for this task 
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        

        ##shuffle the data because in many cases it is in the order we want 
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        
        ##run the evaluation for this task 
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy 
        random_baseline = random_baseline[label]
        centered_result= (accuracy -0.01*random_baseline) / (1.0-0.01*random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")
        core_metric = sum(centered_results.values()) / len(centered_results)
        out = {
            "results": results, 
            "centered_results": centered_results, 
            "core_metric": core_metric
        }
        return out 


    
        
        


        
    
        

