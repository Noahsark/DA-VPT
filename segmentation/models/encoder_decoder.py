from ast import Name
from typing import List, Optional, Union
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors import EncoderDecoder

from .create import create_model


@MODELS.register_module()
class EncoderDecoderVPT(EncoderDecoder):
    
    def __init__(self,
                 args: dict,
                 train_dataloader: dict,
                 max_loading_step: int = 200,
                 **kwargs):
        super(EncoderDecoderVPT, self).__init__(**kwargs)
        
        backbone_cfg = kwargs['backbone']
        self.args = args
        data_preprocessor_cfg = kwargs['data_preprocessor']
        self.backbone = None
        self.backbone = create_model(
                        Namespace(**args), 
                        backbone_cfg, 
                        data_preprocessor_cfg,
                        train_dataloader,
                        max_loading_step)
        
        #self.print_learnable_parameters()
    
    @torch.no_grad()
    def print_learnable_parameters(self, logger) -> None:
        logger.info(" =================== print learnable parameters ====================== ")
        for name, param in self.named_parameters():
            if param.requires_grad:
                logger.info(name + ': ' + str(list(param.shape)))
        # print num of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info('Number of params (M): %.4f' % (num_params / 1.e6))
        logger.info(" ===================================================================== ")
    
    
    def extract_feat(self, inputs: Tensor, 
                     labels = None,
                     update_mapping: bool = False):
        
        # tuple(outs), cls_out, vpt_out, vpt_loss
        x, vpt_loss = self.backbone(inputs, labels, update_mapping)
        if self.with_neck:
            x = self.neck(x)
        
        if self.training:
            return x, vpt_loss
        else:
            return x

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        
        scene_labels = []
        for i, sample in enumerate(data_samples):
            scene_labels.append(sample.scene_label)
        
        losses = dict()
        
        x, vpt_loss = self.extract_feat(inputs, scene_labels)
        losses.update(dict(vpt_loss=vpt_loss))
        
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
