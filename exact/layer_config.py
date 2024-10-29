from contextlib import contextmanager
from collections import namedtuple
import torch.nn as nn

# For docs, see: http://internal.cerebras.aws/~gaviag/context_factory.html


def init_contextual_config():
    global config
    EmptyConfig = namedtuple('DefaultConfig', ['Embedding', 'Linear', 'LayerNorm'])
    config = EmptyConfig(Embedding=nn.Embedding, Linear=nn.Linear, LayerNorm=nn.LayerNorm)
    #config = {'Embedding': lc.config.Embedding}
    #print("Setting up config", hash(config))

init_contextual_config()

@contextmanager
def set_contextual_config(**kwargs):
    if "config" not in globals():
        init_contextual_config()
    global config
    original_config = config._replace() # copy
    for k, v in kwargs.items():
        if k not in kwargs:
            kwargs[k] = getattr(config, k) # use the original value if not present
    ConfigClass = namedtuple('Config', kwargs.keys())
    config = ConfigClass(**kwargs)
    #original_config = config.copy()
    #for k, v in kwargs.items():
    #    config[k] = v
    #print(f"New config: {config} {hash(config)}")
    try:
        yield config
    finally:
        config = original_config
        # for k in list(config.keys()):
        #     del config[k]
        # for k, v in original_config.items():
        #     config[k] = v
