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
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        ##same as generate but does single prefill and then clones the kv cache 
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting a list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        ##getting the special tokens 
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special('<|python_start|>')
        python_end = get_special('<|python_end|>')
        output_start = get_special('<|output_start|>')
        output_end = get_special('<|output_end|>')
        assistant_end = get_special('<|assistant_end|>')
        bos = self.tokenizer.get_bos_token_id()


        ##run the prefill of the prompt tokens 
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1, 
            seq_len=len(tokens), 
            **kv_model_kwargs
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)

        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)
        sampled_tokens = next_ids[:, 0].tolist()

        ##replcate the kv cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len 
        kv_cache_decode = KVCache(
            batch_size=num_samples, 
            seq_len = kv_length_hint, 
            **kv_model_kwargs
        )      
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        #main generation loop 
        num_generated = 0
        first_iteration=True 
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break 
            if all(state.completed for state in row_states):
                break

            ##get the sampled tokens either from the prefill or from the forward pass 
            if first_iteration:
                #use the tokens from the prefill 
                sampled_tokens = [sampled_tokens[0]] *num_samples
                first_iteration = False
            else:
                #forward the model and get the next token for each row 
                logits = self.model.forward(ids, kv_cache_decode)
                logits = logits[:,-1,:]
                next_ids = sample_next_token(logits, rng, temperature, top_k)
                sampled_tokens = next_ids[:, 0].tolist()
        

        #process each row - choose the next token, update the state and optional tool use 
        token_column = []
        token_masks = []
        for i, state in enumerate(row_states):
            #select the next token in this row 
            is_forced = len(state.forced_tokens) > 0
            token_masks.append(0 if is_forced else 1)
            next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
            token_column.append(next_token)
            #in the most easy way possible, each sequence in the batch has the row to it
            ##on <|assistant_end|>  or <|bos|> we mark the row as completed 
            if next_token in (assistant_end, bos):
                state.completed = True
            
            ##handling the tool logic 
            if next_token == python_start:
                state.in_python_block = True
                state.python_expr_tokens = []
            elif next_token == python_end and state.in_python_block:
                state.in_python_block = False
                if state.python_expr_tokens:
                    expr = use_calculator(expr)
                    result = use_calculator(expr)
                    if result is not None:
                        result_tokens = self.tokenizer.encode(str(result))
                        state.forced_tokens.append(output_start)
                        state.forced_tokens.extend(result_tokens)
                        state.forced_tokens.append(output_end)
                state.python_expr_tokens = []
            elif state.in_python_block:
                state.python_expr_tokens.append(next_token)
        
        yield token_column, token_masks
        num_generated += 1

        #prepare ids for the next iteration 
        ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        ##we have masks during inference because we want to diffrentiate between the forced tokens and the sampled tokens
        ##this returns a list of token sequences and a list of ints 
        ##terminal tokens are also included

        assistant_end = self.tokenizer.encode_special('<|assistant_end|>')
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        
        for token_column, token_masks in self.generate(tokens, num_samples, *kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        
        return results, masks


if __name__ == "__main__":
    ##quick inline test 
    import time 
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type()
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == "cuda" else nullcontext())

    ##load the model and the tokenizer 
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()

    ##common hyperparam 
    kwargs = dict(max_tokens=64, temperature=0.0)

    #set the starting prompt 
    prompt_tokens= tokenizer.encode("the chemical formula of water is", prepend_bos=bos_token_id)
    ##generate the refernece sequence using the model.generate() function 

    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference generation time: {t1-t0:.2f} seconds")
    reference_ids = generated_tokens

    ##generated tokens with the engine itself 
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) ##this runs on fp32

    torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0]
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine generation time: {t1-t0:.2f} seconds")
    ##compare the two sequences 
    for i in range(len(reference_ids)):
        if refernece_ids[i] != generated_tokens[i]:
            print(f"Mismatch at index {i}")
            break
    print(f"Match : {reference_ids == generated_tokens}")
        


                

    
    

            











