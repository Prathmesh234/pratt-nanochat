##number of functions that help with evaluating a base model 


import math 
import torch 
import torch.distributed as dist 


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    ##instead of naive mean loss, this function returns bits per byte  (bpb). 
    ##this is a tokenization vocab size independent metric, you are still comparing apples to apples 
    ## if you change the covab size. 
    ##CEL is a tokenization vocab size dependent metric. Which is an issue because we would be comparing apples to oranges. 
    ## for two different models. Hence, in order to calculate the loss we just use the token_bytes in a pure nats form 

    ##record the losses 
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=get_device())
    batch_iter = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction="none") #(B, T)
        loss2d = loss2d.view(-1) # (B * T)
        y = y.view(-1)
        if (y.int() < 0).any():
            ##mps does not have a kernel for the <0 for int64, only int32
            ##any target token <0, is to be ingnored, do not index token bytes with negatives 
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes2d = torch.where(
                valid, 
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes)
            )
            total_nats += (loss2d * (num_bytes2d>0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d*(num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    
    ##sum reduce all the ranks 
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    
    ##compute the bpb
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float("inf")
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb

    

        