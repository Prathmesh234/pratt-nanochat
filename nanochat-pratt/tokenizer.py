import os 
import copy 
from functools import lru_cache

SPECIAL_TOKENS = {
"<|bos|>", 
"<|user_start|>", 
"<|user_end|>", 
"<|assistant_start|>", 
"<|assistant_end|>", 
"<|python_start|>", 
"<|python_end|>", 
"<|output_start|>", 
"<|output_end|>", 
"<|eos|>", 
}

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

##Now we will create generic GPT-4 Style tokenizer 
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingfaceTokenizer:
    "Wrapper around huggingface tokenzier with some custom utilities"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @classmethod
    def from_pretrained(cls, hf_path):
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)
    
    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)
    
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        ##configuring the tokenizer 
        tokenizer = HFTokenizer((BPE(
            byte_fallback=True,
            unk_token=None, 
            fuse_unk=False
        )))
        tokenizer.normalizer = None
    ## pre tokenizer which is gpt 4 like style. The regex pattern used by GPT-4 to split the text into groups before BPE
        gpt4_split_regex =Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(
            special_tokens=SPECIAL_TOKENS,
            vocab_size=vocab_size,
            show_progress=True, 
            min_frequency=0, 
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens
    
    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
    
    def _encode_one(self, text, prepend=None, append=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids
    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)
    
    def get_bos_token_id(self):
        bos = self.tokenizer.token_to_id(text)
    

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self.encode(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid type: {type(text)}")
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    
    def decode(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path=os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


## We are going to skip the rust bpe tokenizer for now, I want to rather focus on the basics and then go to the rust bpe 

def get_tokenizer():
    from common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return HuggingfaceTokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes



    