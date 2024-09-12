import yaml
import termcolor
from easydict import EasyDict as edict


def load_config(fname):
    with open(fname) as file:
        config = edict(yaml.load(file, yaml.FullLoader))
    return config


def print_config(config, level=0):
    
    def green(message,**kwargs): return termcolor.colored(str(message),color="green",attrs=[k for k,v in kwargs.items() if v is True])
    def cyan(message,**kwargs): return termcolor.colored(str(message),color="cyan",attrs=[k for k,v in kwargs.items() if v is True])
    def yellow(message,**kwargs): return termcolor.colored(str(message),color="yellow",attrs=[k for k,v in kwargs.items() if v is True])

    for key,value in sorted(config.items()):
        if isinstance(value,(dict,edict)):
            print("   "*level+cyan("* ")+green(key)+":")
            print_config(value,level+1)
        else:
            print("   "*level+cyan("* ")+green(key)+":",yellow(value))
