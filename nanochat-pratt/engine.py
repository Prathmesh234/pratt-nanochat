##this is a very important part 
## this is the inference engine 
## everything works around token sequences to the engine 
# the engine returns the next token 

#3 the engine knows nothing about tokenization, purely token id sequences 



import torch 
import torch.nn.functional as F 
import signal 
import warnings 
from contextlib import contextmanager 
from collections import deque 
from common import compute_init, autodetect_device_type
from checkpoint_manager import load_model
from contextlib import nullcontext 



## calculator tool helpers 
@contextmanager 
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"{formula}: timedout after {duration} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield 
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__", {}}, {})
    except Exception as e:
        signal.alarm(0)
        return None 


def use_calculator(expr):
    ##math expressions and .count() like operations 
    expr = expr.replace(",", "")

    #check if it is pure math expression  
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all(x in allowed_chars for x in expr):
        return None 

    ##dangerous pattern 
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None 
    if '.count(' not in expr:
        return None 
    
    #evaluate with timeout 
    return eval_with_timeout(expr)


class KVCache:

    ##works hand in hand with the gpt model to maintain the kv cache 
    ## .pos advances automatically after the last layer of the transformer inserts 

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        ## each of kv is of shape (B, H, T, D) and we have one per layer of the transformer 
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim )
        self.kv_cache = None 
        self.pos = 0 #current position in time in the cache 
    
    def reset(self):
        self.pos = 0
    def get_pos(self):
        return self.pos
    
    def prefill(self, other):
        #prefill given another kv cache, expand along with batch dim 
        #batch 1 prefill and then generate 
        ##multiple samples in parallel from here 

        ##validate the shapes 
        assert self.kv_cache is None, "Cannot prefill a non empty kv cache"
        assert other.kv_cache is not None, "Cannot prefill with a None kv cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape), other.kv_shape):
        # ix 0: num_layers, 1: k/v, 2: batch_size, 3: num_heads, 4: seq_len, 5: head_dim
            if ix in [0,1,3,5]:
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        
        ##initalize hte cache 
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        #copy the data over 
        self.kv_cache[:,:,:,:, :other.pos, :] = other.kv_cache
        ##update the pos 
        self.pos = other.pos
    
    def insert_kv(self, layer_idx, k, v):
        ##lazy intialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        ##insert new keys and values to the cache and return the full cache 
        B, H, T_add, D = k.size()
        t0, t1 = self.pos + T_add
        #grow the cache 
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 ##how much we need plus a buffer value 
            t_needed = (t_needed + 1023) & ~1023 
            additional_shape  = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        
        ##insert the k, v into the cache 
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :,:, t0:t1] = v
        ##return the full cached keys and values 
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        
        ##increment the pos after the last layer of the transformer processes 
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view
    
    @torch.inference_mode()
    def sample_next_token(logits, rng, temperature=1.0, top_k=None):
        #sample the single token from the given logits of the shape (B, vocab_size,) returns B, 1
        assert temperature >= 0.0, "temperature must be non negative"
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            vals = vals / temperature
            probs = F.softmax(vals, dim=-1)
            choice = torch.multinomial(probs, num_samples=1, generator=rng)
            return idx.gather(1, choice)
        else:
            logits = logits/temperature
            probs = F.softmax(logits, dim=-1)
            choice = torch.multinomial(probs, num_samples=1, generator=rng)
            return choice

class RowState:
    #per row tracking during generation 
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] #current token sequence for this row 
        self.forced_tokens = deque() ##queue the tokens 
        self.in_python_block = False #. when in the python block 
        self.python_expr_tokens = [] #tokens in the python expression 
        self.completed = False #when the row is completed 


class Engine:
    
    

            











