import torch
from torch import nn
from typing import Dict, List

class ActivationSaver:
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def save_activation(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                self.activations[name] = [o.detach().cpu() for o in output]
            else:
                self.activations[name] = output
        return hook
    
    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.save_activation(name))
                self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self) -> Dict:
        return self.activations

def get_model_activations(model: nn.Module, inputs: torch.Tensor, layer_names: List[str]):
    saver = ActivationSaver()
    saver.register_hooks(model, layer_names)
    with torch.no_grad():
        _ = model(inputs)
    activations = saver.get_activations()
    saver.clear_hooks()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return activations