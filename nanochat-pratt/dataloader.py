##now we need to work in batches of sequences in a distributed setting 
## so we would have to create a distributed data loaded with state 
##each gpu needs different data to train on for data parallel 

from ast import List
from collections import deque 
import torch 
import pyarrow.parquet as pq 

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer 


def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device='cuda', resume_state_dict=None):
    #stream pretraining text from parquet files, tokenize, yield training batches 
    #we return the state_dict in every batch 
    ## and then we pass the state dict to resume training 
    assert split in ["train", "val"], "split must be train or val"
    
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pg_idx = resume_state_dict["pd_idx"] if resume_state_dict is not None else 0 
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None 
        pg_idx = resume_pg_idx
        while True:
            while pg_idx < len(parquet_paths):
                filepath = parquet_paths[pg_idx]
                pf = pq.ParquetFile(filepath)

                ##start from the resume point resuming on the same file 
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_groups(rg_idx)
                    batch = rg.column('text').to_pylist() ## each batch is a parquet group eg 1024 rows

                    ## the tokenizer encode might want even smaller batches 
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pg_idx, rg_idx)
                    rg_idx += ddp_world_size ##next row in group 
                pd_idx += 1
    batches = document_batches()

    needed_tokens = B*T + 1
    ## get the tokenizer and the bos tokens 
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_box_token_id()
    ##scratch the buffer holds the tokens for one iteration 
    token_buffer = deque()
    while True:
        ##accumulate tokens for one iteration before yielding 
        while len(token_buffer) < needed_tokens:
            doc_batch, (pg_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        
        ## move the tokens from the deque to the buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        ##cuda supports the memory pinning for async transfers between cpu and gpu
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        ##creating inputs and targets as 1 D tensors 
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]

        ##reshape to 2d and move the GPU async 
        inputs = inputs_cpu.view(B,T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B,T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pg_idx": pg_idx, "rg_idx":rg_idx}
        yield inputs, targets, state_dict
def tokenizing_dstributed_data_loader(*args, **kwargs):
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(**args, **kwargs):
        yield inputs, targets


