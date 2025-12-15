import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import random

# class_item_sampler
# sample k for each class
# 1.2.0
class MappingSampler(Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, args, image_dict, num_imgs, has_path=True, **kwargs):
        
        #####
        self.image_dict         = image_dict
        self.classes        = list(self.image_dict.keys())
        self.train_classes  = self.classes
        
        ####
        self.batch_size         = args.batch_size
        self.samples_per_class  = args.init_sample_per_class
        #assert self.batch_size % self.samples_per_class == 0, \
        #'Batch size must be divisible by samples per class'
        # Number samples
        self.sampler_length     = len(self.classes) * self.samples_per_class
        self.has_path           = has_path # if the image_dict values are paths
        
        self.name             = 'class_random_item_sampler'
        self.requires_storage = False
    
    # from set S sample N elements
    # if N <= len(S), sample without replacement
    def _smart_sample(self, S, N):
        M = len(S)
        if N <= M:
            return random.sample(S, N)
        else:
            multiple_samples = N // M
            remaining_samples = N % M
            result = []
            for _ in range(multiple_samples):
                result.extend(S)
            result.extend(random.sample(S, remaining_samples))
            random.shuffle(result)
            return result
        
    def __iter__(self):
        subset = []
        for cls in self.classes:
            sample = self._smart_sample(self.image_dict[cls], self.samples_per_class)
            if self.has_path:
                subset.extend([s[-1] for s in sample])
            else:
                subset.extend(sample)
        assert len(subset) == self.sampler_length
        return iter(subset)
    
    def __len__(self):
        return self.sampler_length