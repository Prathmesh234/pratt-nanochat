##now we are going to building the gpt model 
#pos embeddings 
## unitied weights 
## relu 
## normalization after token embeddings 
## no learned params 
## no bias 
## gqa attention 

import math 
from functools import partial 
from dataclasses import dataclass

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from nanochat-pratt.common import get_dist_info, print0 
from nanochat-pratt.muon import Muon, DistMuon 
from nanochat-pratt.adamw import DistAdamW


@dataclass 
class GPTConfig:
    sequence_len: int= 1024
    vocab_size: int= 50304
    n_layer: int= 12 
    n_head: int = 6
    n_kv_head: int= 6
    n_embd: int = 768 

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2 
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1*cos + x2*sin
    y2 = x1*(-sin) + x2*cos
    out  = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out 

class CasualSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head*self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head*self.head_dim, bias = False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head*self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        ## getting the input for queries, keys and values 
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B,T,self.n_kv_head, self._head_dim)
        v = self.c_v(x).view(B,T,self.n_kv_head, self.head_dim)

        ##applying rotatory embeddings 
        cos, sin = cos_sin
        ##rotating the q and k rotary embeddings
        q,k = apply_rotary_emb(q,cos, sin), apply_rotary_emb(k, cos, sin)
        q,k = norm(q), norm(k)
        ##B,T,H,D -> B,H,T,D
        q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        ##applying the kv cache, insert the key and value into the cache 
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) ##number of queries this forward pass 
        Tk = k.size(2) ##number of keys and values in total 

        ##attention - queries attend to keys and values autoregressively 
        enable_gqa = self.n_head != self.n_kv_head #duplicate the key and value heads to match the query heads 
        if kv_cache is None or Tq==Tk:
            ##if there is no KV Cache then we attend in a simple way 

            ##during training - no kv cache, attend as usual with casual attention 
            y = F.scaled_dot_product_attention(q,k,v, is_casual=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            ##during inference but with a single query (because it is autoregressive)
             y = F.scaled_dot_product_attention(q,k,v, is_causal=False, enable_gqa=enable_gqa)
        else:

            ##during inference and we have a chunk of queries for the forward pass 
            #each query attends to all cached keys and values
            # multiple tokens means - using speculative decoding, batch processing etc 
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len=Tk-Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len]=True 
            
            #then causal attention within the chunk 
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q,k,v, attn_mask=attn_mask, enable_gqa=enable_gqa)
        
        y = y.transpose(1,2).contiguous().view(B,T,-1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=False)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x 

##block of attention with mlp 
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CasualSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    
    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


##GPT 
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd), 
            "h": nn.ModuleList([Block(config, layer_idx)] for layer_idx in range(config.n_layer)), 
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        ## these are somw configs for the rotary pos embeddings 
        ##we usually pre compute the cos and sin table for 10x length in case the model sees more than sequence_len for a particular batch 
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, presistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def init_weights(self):
        self.apply(self.__init_weights)

        torch.nn.init.zeros(self.lm_head.weights)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

        ##init the rotary embeddings 
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        ##cast the embeddings from fp32 to bf16 so it saves memory 
        if self.transformer.wte.weight.device.type=="cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out/fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    ##bump base theta more 
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin
    
    def get_device(self):
        return self.transformer.wte.weight.device
    
    def estimate_flops(self):
        # return the estimated flops per token for the model 
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l,h,q,t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6*(nparams - nparams_embedding) + 12*l*h*q*t
        return num_flops_per_token
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        ##seperate the params into 3 groups - matrix, embedding, lm_head
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params= list(self.lm_head.parameters())

        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        






