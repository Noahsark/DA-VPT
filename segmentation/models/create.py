from collections import OrderedDict, Counter, deque
from argparse import Namespace

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import torch.backends.cudnn as cudnn
from contextlib import nullcontext
from utils.utils import *

from mmengine.runner import Runner
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.registry import MODELS

from model_creator import generate_mapping

from tqdm import tqdm


MAPPING_BATCH_SIZE = 32

@torch.no_grad()
def generate_semantic_mapping(args: Namespace, model, 
                              data_preprocessor_cfg: ConfigType,
                              train_dataloader_cfg: ConfigType,
                              max_loading_step: int):
    """
    args: dict
        quiet_mode: False
        initial_mapping: kmeans
        proxy_prompt_len: 10
        kmeans_norm: layer    
    """
    
    # load data
    print(colorstr('yellow', "Generating Semantic map... "))    
    
    train_dataloader_cfg['batch_size'] = MAPPING_BATCH_SIZE
    init_loader = Runner.build_dataloader(train_dataloader_cfg)
    
    data_iterator = tqdm(init_loader,
            bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt},'
                        '{elapsed}{postfix}]',
            ncols=96, ascii=True)
    
    #model.saparate_qkv_params()
    model.eval()
    
    preprocessor = MODELS.build(data_preprocessor_cfg)
    
    # initialize mapping
    for step, data in enumerate(data_iterator):
        
        if step >= max_loading_step:
            break
        
        data = preprocessor(data, True)
        
        images = data['inputs'].to('cuda')
        data_samples = data['data_samples']
        
        scene_labels = []
        for i, sample in enumerate(data_samples):
            scene_labels.append(sample.scene_label)
        
        # compute output
        model.forward_cls(images, labels=scene_labels)

    cls_mean_features = model.get_cls_mean_feature()
    
    mapping = None
    centroids = None
    
    if args.proxy_prompt_len > 0:
        mapping, centroids, _ = generate_mapping(args, cls_mean_features, None)
    
    return mapping, centroids


def create_model(args: Namespace, 
                 backbone_cfg: ConfigType,
                 data_preprocessor_cfg: ConfigType,
                 train_dataloader_cfg: ConfigType,
                 max_loading_step: int):
    """ 
    1. Create
    2. Choose tuning_type of model

    Args:
        num_class: int
        tuning_type: [prompt, full, linear]
        cls_loss: False
        learn_bias: False
        bias_q: False
        bias_k: False
        bias_v: False
        bias_fc1: False
        bias_fc2: False
        bias_proj: False
        bias_norm1: False
        bias_norm2: False
        bias_norm: False
        norm_norm1: False
        norm_norm2: False
        norm_norm: False
        train_cls: False
        proxy_prompt_len: 10
    """
    
    args.proxy_prompt_len = backbone_cfg['proxy_prompt_len']
    args.num_prompts = backbone_cfg['num_prompts']
    
    model = MODELS.build(backbone_cfg)
    model.init_weights()
    model.to('cuda')
    
    if args.tuning_type == "full":
        # for name, param in model.named_parameters():
        #     if 'in_proj_weight' not in name and 'in_proj_bias' not in name:
        #         param.requires_grad = True
        
        #Note: do nothing, just leave all parameters to be learnable
        model.print_learnable_parameters()
        
    elif args.tuning_type == "linear":
        # freeze all
        model.frozen_exclude = []
        model._freeze()
        model.print_learnable_parameters()
        
    else: # prompt
        #######################################

        centroids = None
        
        if args.proxy_vpt:
            mapping, centroids = \
                    generate_semantic_mapping(args, model, 
                                data_preprocessor_cfg,
                                train_dataloader_cfg,
                                max_loading_step)
            
            model.reset_for_mapping_update()
            model.update_mapping(mapping, centroids)
        
        learnable_list = ['prompt_token']
        learnable_list.append("proxy_prompt_tokens")
        learnable_list.append("other_prompt_tokens")
        learnable_list.append("vpt_norm")
        
        if args.learn_bias:
            learnable_list.append("bias")
        if args.bias_q:
            learnable_list.append("q.bias")
        if args.bias_k:
            learnable_list.append("k.bias")
        if args.bias_v:
            learnable_list.append("v.bias")
        if args.bias_fc1:
            learnable_list.append("ln1.bias")
        if args.bias_fc2:
            learnable_list.append("ln2.bias")
        if args.bias_proj:
            learnable_list.append("out_proj_bias")
            learnable_list.append("in_proj.bias")
        if args.bias_norm1:
            learnable_list.append("norm1.bias")
        if args.bias_norm2:
            learnable_list.append("norm2.bias")
        if args.bias_norm:
            learnable_list.append("norm.bias")
        if args.norm_norm1:
            learnable_list.append("norm1")
        if args.norm_norm2:
            learnable_list.append("norm2")
        if args.norm_norm:
            learnable_list.append("norm")
        if args.train_cls:
            learnable_list.append("cls_token")
        
        model.set_learnable_parameters(learnable_list)
        model.print_learnable_parameters()
        
    if args.cls_loss:
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features=in_features, out_features=args.num_class)
        trunc_normal_(model.head.weight, std=.02)
        nn.init.constant_(model.head.bias, 0)
        # print model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ': ' + str(list(param.shape)))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of params (M): %.4f' % (num_params / 1.e6))
    
    
    return model


# prompt_drop_rate=0.0,
# proxy_prompt_len=10,
# num_prompts=20,
# num_classes=1055,
# mapping=None,
# proxy_prompt_start_idx=11,
# proxy_prompt_end_idx=11,
# qk_norm=False,
# mask=False,
# vpt_loss_weight=0.05,
# cls_loss_weight=0.05,
# loss_vpt_scale_ratio=5.0,
# vpt_norm='none',