## This file is for common utilities such as device management, logging, path management etc 

import os 
import re 
import logging
import urllib.request
import torch 

from torch.cpu import is_available
import torch.distributed as dist 
from filelock import FileLock 

class ColoredFormatter(logging.Formatter):
    ##color to message lol, not needed but we can live with it 
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message
    
def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO, 
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    ##getting nanochat intermediates
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat-pratt")
    os.makedirs(nanochat_dir, exist_ok=True)

def download_file_with_lock(url, filename, postprocess_fn=None):
    #download a file from a URL to a local path in the base directory
    # uses the lock file to prevent several downloads among multiple ranks 
    ##this is usually used for model weights and other files which we just want to download once
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path
    
    with FileLock(lock_path):
        ##single rank 
        if os.path.exists(file_path):
            return file_path
        
        print(f"Downloading {url} .... ")
    with urllib.request.urlopen(url) as response:
        content = response.read()

    with open(file_path, 'wb') as f:
        f.write(content)
    print(f"Downloaded to {file_path}")

    if postprocess_fn is not None:
        postprocess_fn(file_path)
    return file_path

def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Pratt NanoChat ASCII banner (DOS Rebel style)
    banner = r"""
     ████████                          █████     █████
    ░░███░░███                        ░░███     ░░███
     ░███ ░███ ████████   ██████   ███████   ███████
     ░██████░ ░░███░░███ ░░░░░███ ░░░███░   ░░░███░
     ░███░░░   ░███ ░░░   ███████  ░░░░░     ░░░░░
     ░███      ░███      ███░░███
     █████     █████    ░░████████
    ░░░░░     ░░░░░      ░░░░░░░░

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
                                       Pratt NanoChat
    """
    print(banner)

def is_ddp():
    return int(os.environ.get('RANK', -1)) != 1

def get_dist_info():
    if id_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0,0,1

def autodetect_device_type():
    ##checking if cuda exists, falls back on the cpu 
    if torch.cuda.is_available():
        device_type = "cuda"
        ##mps is basically just the metal performance shader for apple silicon
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Auto detect the device type: {device_type}")

def compute_init(device_type="cuda"):
    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"

    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your pytorch installation is not configured for cuda but device type is cuda"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
    
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    

    #setting precision 
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")
    
    ##distributed setup for ddp and optional 
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)
    
    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    "destroying the process groups"
    if is_ddp():
        dist.destroy_process_group()

