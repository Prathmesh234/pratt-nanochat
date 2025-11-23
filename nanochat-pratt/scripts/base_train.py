import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time 
from contextlib import nullcontext
import wandb 
import torch 

from gpt import GPT, GPTConfig 
from dataloader import tokenizing_distributed_data_loader_with_state, tokenizing_dstributed_data_loader
from common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from tokenizer import get_tokenizer, get_token_bytes
from checkpoint_manager import save_checkpoint, load_checkpoint
from loss_eval import evaluate_bpb
from engine import Engine
