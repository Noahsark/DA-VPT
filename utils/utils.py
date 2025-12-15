"""
Utility functions for DA-VPT - Cleaned and Improved Version
"""

import os
import shutil
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def colorstr(*input_args) -> str:
    """Color a string for terminal output."""
    *args, string = input_args if len(input_args) > 1 else ('blue', 'bold', input_args[0])
    
    colors = {
        'black': '\033[30m', 'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m', 'magenta': '\033[35m',
        'cyan': '\033[36m', 'white': '\033[37m', 'bright_black': '\033[90m',
        'bright_red': '\033[91m', 'bright_green': '\033[92m',
        'bright_yellow': '\033[93m', 'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m', 'bright_cyan': '\033[96m',
        'bright_white': '\033[97m', 'end': '\033[0m',
        'bold': '\033[1m', 'underline': '\033[4m'
    }
    
    return ''.join(colors.get(x, '') for x in args) + f'{string}' + colors['end']


def save_model(args, model: nn.Module, epoch: int, best_acc: float) -> None:
    """Save model checkpoint."""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = save_dir / '_'.join([args.model_name, args.dataset, str(args.task_name)])
    
    model_name = '_'.join([
        args.model_name, args.dataset, str(args.proxy_prompt_len),
        str(args.num_prompts), str(args.task_name),
        str(epoch), f"{best_acc:.4f}"
    ]) + '.ptm'
    
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / model_name
    torch.save(model, model_path)
    
    if not args.quiet_mode:
        print(f"Model successfully saved to: {model_path}")


def load_model(load_path: str, quiet_mode: bool = False) -> Optional[nn.Module]:
    """Load model from checkpoint."""
    try:
        model = torch.load(load_path, map_location='cpu')
        model.eval()
        
        if not quiet_mode:
            print(f"Model successfully loaded from: {load_path}")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def accuracy(output: torch.Tensor, target: torch.Tensor, 
            topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Compute accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: float, args) -> float:
    """Adjust learning rate with cosine annealing after warmup."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1. + math.cos(math.pi * (epoch - args.warmup_epochs) / 
                         (args.epochs - args.warmup_epochs))
        )
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def param_groups_weight_decay(args, model: nn.Module, weight_decay: float = 1e-5, 
                            weight_decay_head: float = 1.0, 
                            no_weight_decay_list: Tuple = ()) -> List[dict]:
    """Create parameter groups with different weight decay settings."""
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    head = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine if parameter should have no weight decay
        no_decay_condition = name in no_weight_decay_list
        if not args.open_bias_decay:
            no_decay_condition = no_decay_condition or name.endswith(".bias") or param.ndim <= 1
        if not args.open_weight_decay:
            no_decay_condition = no_decay_condition or (
                name.endswith(".weight") and param.ndim <= 1
            )
        
        if no_decay_condition:
            no_decay.append(param)
            if not args.quiet_mode:
                print(f"no decay: {name}")
        else:
            if 'head' in name:
                head.append(param)
                if not args.quiet_mode:
                    print(f"head decay: {name}")
            else:
                decay.append(param)
                if not args.quiet_mode:
                    print(f"weight decay: {name}")

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': head, 'weight_decay': weight_decay_head}
    ]


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class FeatureCatcher:
    """Base class for catching intermediate features from models."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.features = {}
        self.hooks = {}

    def _get_hook(self, name: str):
        def hook(model, input, output):
            self.features[name] = output
        return hook

    def register_model_hooks(self, catch_dict: dict):
        """Register forward hooks for specified modules."""
        for name, module in self.model.named_modules():
            if name in catch_dict:
                key = catch_dict[name]
                self.features[key] = None
                self.hooks[key] = module.register_forward_hook(self._get_hook(key))

        missing_features = set(catch_dict.values()) - set(self.hooks.keys())
        if missing_features:
            print(f"Warning: Could not find features: {missing_features}")

    def get_features(self, key: str):
        """Get cached features by key."""
        return self.features.get(key)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()


class RepresentationCatcher(FeatureCatcher):
    """Catch intermediate representations from transformer layers."""
    
    def __init__(self, model: nn.Module, layers_idx: List[int]):
        super().__init__(model)
        self.layers_idx = layers_idx
        catch_dict = self._get_representation_catch_dict(layers_idx)
        self.register_model_hooks(catch_dict)

    def _get_representation_catch_dict(self, idx: List[int]) -> dict:
        """Create dictionary mapping layer names to representation keys."""
        catch_dict = {}
        for i in idx:
            if i == 0:
                catch_dict["patch_embed"] = f"rep{i}"
            else:
                catch_dict[f"blocks.{i-1}"] = f"rep{i}"
        return catch_dict
    
    def get_features(self, idx: Optional[List[int]] = None) -> List[torch.Tensor]:
        """Get features for specified layer indices."""
        if idx is None:
            idx = self.layers_idx
        
        return_single = isinstance(idx, int)
        if return_single:
            idx = [idx]
        
        representations = []
        for i in idx:
            rep = self.features[f"rep{i}"]
            if i == 0:
                # Patch embedding output
                representations.append(rep)
            else:
                # Remove cls token from transformer block output
                representations.append(rep[:, 1:, :])

        return representations[0] if return_single else representations