import os 
import re 
import glob 
import json 
import logging 
import torch
from torch._inductor.virtualized import V
from torch.optim import optimizer 


from common import get_base_dir, logger
from gpt import GPT, GPTConfig
from tokenizer import get_tokenizer 
from common import setup_default_logging


##set up default logging
setup_default_logging()

logger = logging.getLogger(__name__)

def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True )

        ##save the model params 
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model params to : {model_path}")

        ##save the metadict as json 
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved Meta data to : {meta_path}")

        ##note that the optimizer state is sharded across ranks so each rank must save its own 
        if optimizer_data is not None:
            optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
            torch.save(optimizer_data, optimizer_path)
            logger.info(f"Saved optimizer state to: {optimizer_path}")
        

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):

    ##load model state 
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)

    ##loading the optimizer state 
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    ##loading the metadata 
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f :
        meta_data = json.load(f)
    return model_path, optimizer_data, meta_data

def build_model(checkpoint_dir, step, device, phase):

    ##code to build the model from a given checkpoint 
    #it returns the base model  - uncompiled, not wrapped in ddp
    ##tokenizer 
    ## meta data saved during base model training 
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {
            #converts bf16 tensors to float for cpu inference
            k: v.float() if v.dtype == torch.bfloat16 else V
            for k, v in model_data.items()
        }
    
    model_data = {k.removeprefix("_orig_mod."): v for f, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)      
    with torch.device("meta"):
        model  = GPT(model_config)

    #load the model state 
    model.to_empty(device=device)
    model.init_weights()      
    model.load_state_dict(model_data, strict=True, assign=True)

    #put te model in training mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    tokenizer = get_tokenizer()

    ##sanity check - checking compatiblity between model and tokenizer 
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data

def find_largest_model(checkpoint_dir):
    ##guess the model tag and take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    candidates = []
    for model_tag in model_tags:
        match = re.match(f"d(/d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append(model_depth, model_tag)
    if candidates:
        candidates.sort(key = lambda x:x[0], reverse=True)
        return candidates[0][1]
    #if failed then take the most recent updated model 
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]

def find_last_step(checkpoint_dir):
    ##look into the checkpoint_dir and find the model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:

        ##guess the model tag by defaulting to the largest model 
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing the model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints", 
        "mid" : "mid_checkpoints", 
        "sft" : "chatsft_checkpoints", 
        "rl": "chatrl_checkpoints"
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)


