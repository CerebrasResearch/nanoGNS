"""
Provides a context manager for managing global PyTorch module configurations in
a safe and explicit way.

This module offers a solution to the common deep learning pattern of using
global variables for configuration, particularly when working with PyTorch
modules. Instead of relying on mutable global state, it provides a context
manager that temporarily modifies module configurations in a controlled scope.

Key Features:
    - Default configurations for common PyTorch modules (Embedding, Linear, LayerNorm)
    - Ability to temporarily override module implementations within a specific context
    - Minimal modification to existing model definitions

Example:
    >>> import torch.nn as nn
    >>> from contextlib import contextmanager
    >>>
    >>> # Override with custom implementations in a specific context
    >>> class CustomLinear(nn.Linear):
    ...     pass
    >>>
    >>> with set_contextual_config(Linear=CustomLinear):
    ...     import my_model  # Must import inside context
    ...     model = my_model.MyModel()  # Uses CustomLinear instead of nn.Linear

Note:
    The module containing your model must be imported inside the context manager
    for the configuration changes to take effect.
"""

from contextlib import contextmanager
from collections import namedtuple
import torch.nn as nn

def init_contextual_config():
    global config
    EmptyConfig = namedtuple('DefaultConfig', ['Embedding', 'Linear', 'LayerNorm'])
    config = EmptyConfig(Embedding=nn.Embedding, Linear=nn.Linear, LayerNorm=nn.LayerNorm)

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
    try:
        yield config
    finally:
        config = original_config
