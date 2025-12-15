import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer as timm_vit
from functools import partial
from .criterion import VPTCriterion
from .attn import VPTAttention


class VisionTransformer(timm_vit.VisionTransformer):
    def __init__(self, args, num_classes=200, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
    
        self.args = args
        self.embed_dim = kwargs['embed_dim']
        self.num_heads = kwargs['num_heads']
        self.qkv_bias = kwargs.get('qkv_bias', True)
        self.qk_norm = kwargs.get('qk_norm', False)
        self.attn_drop = kwargs.get('attn_drop_rate', 0)
        self.proj_drop = kwargs.get('proj_drop_rate', 0)
        self.drop_path_rate = kwargs.get('drop_path_rate', 0)
        self.norm_layer = kwargs.get('norm_layer', None)
        self.depth = kwargs.get('depth', 12)
        self.num_classes = num_classes
        self.global_pool = global_pool
        
        #assert self.args.prompt_depth < self.depth, "prompt_depth should be smaller than depth"
        #self.prompt_token = nn.Parameter(
        #        torch.zeros(self.args.prompt_depth, self.args.num_prompts, self.embed_dim), 
        #        requires_grad=True)
        
        self.class_prompt_token = nn.Parameter(
                torch.zeros(self.num_classes, self.embed_dim), requires_grad=True)
        
        self.vpt_criterion = VPTCriterion(args, embed_dim=self.embed_dim)
        self.cls_mean_feature = torch.zeros(self.num_classes, self.embed_dim, device='cuda')
        
        #TODO: check this for visuallization
        self._setup_attn()
    
    
    def _setup_attn(self) -> None:
        for idx in range(self.depth):
            self.blocks[idx].attn = VPTAttention(
                dim=self.embed_dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
                qk_norm=self.qk_norm, attn_drop=self.attn_drop, proj_drop=self.proj_drop,
                norm_layer=self.norm_layer
            )
    
    def separate_qkv_params(self) -> None:
        for idx in range(self.depth):
            self.blocks[idx].attn._separate_qkv_params()
    
    def reset_for_mapping_update(self):
        self.vpt_criterion.reset_num_batches()
        self.cls_mean_feature = torch.zeros(self.num_classes, self.embed_dim, device='cuda')
    
    def get_cls_mean_feature(self):
        return self.cls_mean_feature
    
    
    
    #########=========================== forward ========================###################
    
    def forward_features(self, x, labels=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        
        for blk in self.blocks:
            x = blk(x)
        
        if self.global_pool:
            out = x[:, 1:, :].mean(dim=1) # global pool without cls token
            out = self.norm(out)
        else:
            x = self.norm(x)
            out = x[:, 0] # [cls] token
        
        self.vpt_criterion.create_cls_mean_features(out, labels, self.cls_mean_feature)

        return out
    
    def forward_head(self, x):
        x = self.head(x)
        return x
    
    def forward(self, x, labels=None):
        x = self.forward_features(x)
        
        logits = self.forward_head(x)
        return logits



def create_vit(vit_size = 'B', **kwargs):
    if vit_size=='B':
        patch_size = 16
        embed_dim = 768
        depth = 12
        num_heads = 12
    elif vit_size =='S':
        patch_size = 16
        embed_dim = 384
        depth = 12
        num_heads = 6
    elif vit_size =='L':
        patch_size = 16
        embed_dim = 1024
        depth = 24
        num_heads = 16
    elif vit_size =='H':
        patch_size = 14
        embed_dim = 1280
        depth = 32
        num_heads = 16
    else:
        raise ValueError(f"Unsupported ViT size: {vit_size}")
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, 
            mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model