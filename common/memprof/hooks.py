"""
hooks for memory profiling
"""

import torch.nn as nn

global activations
activations = {}
def _get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def mem_hook():
    """
    get CUDA memory statistics
    """

def _add_module_hooks(hr, module, hook, **hook_kwargs):
    """
    add hook functions to module

    Args:
        hr: (list) a list to store the hook handlers
        module: (nn.Module) add hooks to this module instance
        hook: (callable) a callable hook function object
        hook_kwargs: pointer to keyword arguments to hook function
    """
    assert isinstance(module, nn.Module)
    if callable(hook):
        try:
            # pre hook
            h = module.register_forward_pre_hook(hook(**hook_kwargs))
            hr.append(h)
            # forward hook
            h = module.register_forward_hook(hook(**hook_kwargs))
            hr.append(h)
        except Exception as err:
            print("register hooks failed: {}".format(err))
    else:
        raise ValueError('function is not callable: {}'.format(hook))



