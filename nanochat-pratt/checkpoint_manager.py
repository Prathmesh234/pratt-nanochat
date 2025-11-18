import os 
import re 
import glob 
import json 
import logging 
import torch 


from common import get_base_dir, logger
from gpt import GPT, GPTCongig 
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
            