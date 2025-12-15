import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import random
import math


###
class ClassSampler(Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, num_imgs, has_path=True, **kwargs):
        
        #####
        self.image_dict         = image_dict
        self.classes            = list(self.image_dict.keys())
        self.train_classes      = self.classes
        self.has_path           = has_path # if the image_dict values are paths
        
        ####
        self.batch_size         = opt.batch_size
        self.samples_per_class  = opt.samples_per_class
        # Number of batches per epoch
        self.sampler_length     = num_imgs // opt.batch_size  
        assert self.batch_size % self.samples_per_class==0, \
        '#Samples per class must divide batchsize!'
        self.name             = 'class_random_batch_sampler'
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
    
    def _sample(self, subset, draws, classes):
        # Randomly draw classes
        class_keys = self._smart_sample(classes, draws)
        for cls in class_keys:
            sample = self._smart_sample(self.image_dict[cls], self.samples_per_class)
            if self.has_path:
                subset.extend([s[-1] for s in sample])
            else:
                subset.extend(sample)
        return subset
        
    def __iter__(self):
        
        #warning: this cannot garantee that all classes are sampled
        for _ in range(self.sampler_length):
            subset = []
            train_draws = self.batch_size//self.samples_per_class
            subset = self._sample(subset, train_draws, self.train_classes)
            yield subset

    def __len__(self):
        return self.sampler_length
    

class DistributedBatchSampler(ClassSampler):
    def __init__(self, opt, image_dict, num_imgs, rank=None, **kwargs):
        super().__init__(opt, image_dict, num_imgs, **kwargs)

        self.num_replicas = self.opt.world_size
        
        self.rank = rank

        self.epoch = 0

    def __iter__(self):
        
        batches = list(super().__iter__())

        # Determine how many samples to add to make it evenly divisible
        length = len(batches)
        length_per_replica = int(math.ceil(length / self.num_replicas))
        total_length = length_per_replica * self.num_replicas

        # add extra samples by duplicating some batches to make it evenly divisible
        batches += batches[:(total_length - length)]
        assert len(batches) == total_length

        # subsample
        batches = batches[self.rank:total_length:self.num_replicas]
        assert len(batches) == length_per_replica

        return iter(batches)

    def __len__(self):
        return int(math.ceil(len(super()) / float(self.num_replicas)))
    
    def set_epoch(self, epoch):
        self.epoch = epoch