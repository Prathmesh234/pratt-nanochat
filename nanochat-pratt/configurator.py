import os 
import sys 
from ast import literal_eval

##literal eval in python basically identifies the arguments in the terminal command as python objects 
## this helps when setting the args up 

def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--')
        config_file = arg
        print0(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print0(f.read())
        exec(open(config_file).read())
    else:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            if globals()[key] is not None:
                attempt_type = type(attempt)
                default_type = type(globals()[key])
                assert attempt_type == default_type, f"Type mismatch: {attempt_type} != {default_type}"
            print0(f"Overview: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")


