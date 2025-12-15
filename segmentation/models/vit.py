import math
from curses import def_prog_mode
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as cp
from collections import OrderedDict, Counter, deque
from timm.models.vision_transformer import DropPath, Mlp, LayerScale

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.registry import MODELS
from mmseg.models.utils import PatchEmbed, resize
from mmseg.models import VisionTransformer
from mmseg.models.backbones.vit import TransformerEncoderLayer

from models.vpt_backup import VPTCriterion, L2NormalizationLayer
#from models.attn import VPTAttention

from argparse import Namespace



class VPTAttention(nn.Module):
    # This class implements the multi-head attention mechanism.
    # A class attribute indicating whether a fused attention implementation should be used.
    
    def __init__(
            self,
            dim: int,  # The dimensionality of the input features.
            num_heads: int = 12,  # The number of attention heads.
            qkv_bias: bool = True,  # Whether to add a bias term to the QKV linear transformation.
            qk_norm: bool = False,  # Whether to apply normalization to Q and K.
            attn_drop: float = 0.,  # Dropout rate for attention weights.
            proj_drop: float = 0.,  # Dropout rate after the attention projection.
            norm_layer: nn.Module = nn.LayerNorm,  # The normalization layer to use.
            flash_attn: bool = True,  # Whether to use the Flash attention implementation.
            return_output: bool = False,  # Whether to return the output of the attention layer.
            mask: bool = False,  # Whether to apply a mask to the attention weights.
            prompt_len: int = 0,  # The length of the prompt.
            num_img_tokens: int = 196,  # The number of image tokens.
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads  # Number of attention heads.
        self.head_dim = dim // num_heads  # Dimension per head.
        self.scale = self.head_dim ** -0.5  # Scaling factor for dot-product attention.
        self.flash_attn = flash_attn  # Whether to use the Flash attention implementation.
        self.return_output = return_output  # Whether to return the output of the attention layer.
        
        if mask:
            total_len = num_img_tokens + prompt_len + 1
            self.mask = self._generate_mask(total_len, total_len, prompt_len, 
                                            torch.device('cuda'))
        else:
            self.mask = None
            
        # warning:      Do not change any one of the names. Otherwise the parameter dict
        # warning:      will not be able to load the model.
        
        # TODO: delete self.qkv after copying to save 
        self.in_proj_weight = nn.Parameter(torch.randn(dim * 3, dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3)) if qkv_bias else None
        #note: 1.2.2.6 saparate qkv
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Optional normalization layers for queries and keys.
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()  
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout layer for attention weights.
        self.out_proj = nn.Linear(dim, dim)  # Linear projection layer.
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout layer after projection.
    
    def _print_params(self):
        for name, param in self.named_parameters():
            print(name + ': ' + str(param.sum().item()) + ' ' + str(param.requires_grad))
    
    
    def _saparate_qkv_params(self):
        # copy qkv to q, k, v
        # weight: [3*dim, dim] -> [dim, dim]
        # use assign '=' in order to connect the computational 
        # graph between qkv and q, k, v
        
        with torch.no_grad():
            self.q.weight.data = self.in_proj_weight[:self.q.in_features]
            self.k.weight.data = self.in_proj_weight[self.q.in_features: self.q.in_features + self.k.in_features]
            self.v.weight.data = self.in_proj_weight[self.q.in_features + self.k.in_features:]
            if self.in_proj_bias is not None:
                self.q.bias.data = self.in_proj_bias[:self.q.in_features]
                self.k.bias.data = self.in_proj_bias[self.q.in_features: self.q.in_features + self.k.in_features]
                self.v.bias.data = self.in_proj_bias[self.q.in_features + self.k.in_features:]    
            # recombine q, k, v for assertion
            recombined_qkv = torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)
            recombined_bias = torch.cat([self.q.bias, self.k.bias, self.v.bias], dim=0) \
                if self.in_proj_bias is not None else None
            # Assert to check if the recombined weights and biases match the originals
            assert torch.allclose(recombined_qkv,  self.in_proj_weight, atol=1e-6), "Weights do not match"
            if self.in_proj_bias is not None:
                assert torch.allclose(recombined_bias, self.in_proj_bias, atol=1e-6), "Biases do not match"
            
            if self.in_proj_weight is not None:
                self.in_proj_weight.requires_grad = False
            if self.in_proj_bias is not None:
                self.in_proj_bias.requires_grad = False
            
    
    def _generate_mask(self, L: int, S: int, len: int, device: torch.device) -> torch.Tensor:
        # Generate a mask to prevent attention from some prompts.
        # L: len of query
        # S: len of key
        # len: len of prompt
        mask = torch.zeros(L, S).to(device)
        mask[1:(1+len), 0:(1+len)] = -float('inf')
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # x: [batch size, number of tokens, channel dimension]
        # Apply QKV linear transformation, reshape for multi-head, and permute.
        # After self.qkv: [B, N, 3*C], after reshape: [B, N, 3, num_heads, head_dim], 
        # after permute: [3, B, num_heads, N, head_dim]
        #! saperate q, k, v 
        #_qkv = self.qkv(x) # [B, N, 3*C]
        # qkv = _qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
        # q, k, v = qkv.unbind(0)  # q, k, v: [B, num_heads, N, head_dim]
        
        # for 1.2.2.6 saparate qkv
        self._q = self.q(x) # [B, N, C]
        self._k = self.k(x) # [B, N, C]
        self._v = self.v(x) #  [B, N, C]
        q = self._q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N, head_dim]
        k = self._k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N, head_dim]
        v = self._v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N, head_dim]
        
        q, k = self.q_norm(q), self.k_norm(k)  # Apply optional normalization, q, k: same as above [B, num_heads, N, head_dim]
        
        # ============================== Attention ==============================
        
        if self.flash_attn:
            # `FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning`_
            #! flash attention would not return attn_map
            #! with mask will cause higher latency in some cases, may be a bug
            x = F.scaled_dot_product_attention(
                q, k, v,
                scale=self.scale,
                attn_mask=self.mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            #Note Manual computation of scaled dot-product attention.
            q = q * self.scale  # Scale the queries, q: [B, num_heads, N, head_dim]
            attn = q @ k.transpose(-2, -1)  # Dot product, attn: [B, num_heads, N, N]
            attn += self.mask
            attn = attn.softmax(dim=-1)  # Softmax over the last dimension, attn: same as above [B, num_heads, N, N]
            attn = self.attn_drop(attn)  # Apply dropout, attn: same as above [B, num_heads, N, N]
            # attn: [B, num_heads, N, N] v: [B, num_heads, N, head_dim]
            x = attn @ v  # Weighted sum, x: [B, num_heads, N, head_dim]
        
        # =======================================================================
        
        # Reshape and permute back to match the input dimension, then project.
        x = x.transpose(1, 2).reshape(B, N, C)
        attn_out = x
        
        x = self.out_proj(x)  # Linear projection, x: [B, N, C], same as input dimension
        x = self.proj_drop(x)  # Apply dropout, x: same as above [B, N, C]
        
        if self.return_output:
            return x, attn_out
        else:
            return x


class VPTBlock(BaseModule):
    def __init__(self, 
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False,
                 mask=False, 
                 prompt_len=10, 
                 qk_norm=False, 
                 num_img_tokens=1024,
                 ):
        super().__init__()
        
        # override the default values
        # dim=dim,
        # num_heads=num_heads, 
        # qkv_bias=qkv_bias,
        # qk_norm=qk_norm, 
        # attn_drop=attn_drop, 
        # proj_drop=proj_drop,
        # norm_layer=norm_layer,
        # return_output=True, #NOTE: return attn_out
        # mask=mask,
        # prompt_len=prompt_len
        
        attn_cfg.update(
            dict(
                dim=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                qkv_bias=qkv_bias,
                return_output=True,
                qk_norm=qk_norm,
                mask=mask,
                prompt_len=prompt_len,
                num_img_tokens=num_img_tokens
                )
            )
        
        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        
        self.drop_path = DropPath(attn_drop_rate) if attn_drop_rate > 0. else nn.Identity()
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        
        self.attn = VPTAttention(**attn_cfg)
        self.ffn = FFN(**ffn_cfg)
        self.with_cp = with_cp
    
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)
    
    def forward(self, x):
        
        def _inner_forward(x):
            #! critical bug here!!!
            identity = x
            x, attn_out = self.attn(self.norm1(x))
            x = identity + self.drop_path(x)
            x = self.ffn(self.norm2(x), identity=x)
            return x, attn_out

        if self.with_cp and x.requires_grad:
            x, attn_out = cp.checkpoint(_inner_forward, x)
        else:
            x, attn_out = _inner_forward(x)
        return x, attn_out
    
    

@MODELS.register_module()
class VisionTransformerVPT(VisionTransformer):
    def __init__(self, 
                    prompt_drop_rate=0.0,
                    proxy_prompt_len=10,
                    num_prompts=20,
                    num_classes=150,
                    mapping=None,
                    proxy_prompt_start_idx=11,
                    proxy_prompt_end_idx=11,
                    qk_norm=False,
                    mask=False,
                    vpt_loss_weight=0.05,
                    cls_loss_weight=0.05,
                    loss_vpt_scale_ratio=5.0,
                    vpt_norm='none',
                    proxy_vpt=True,
                 **kwargs):
        super(VisionTransformerVPT, self).__init__(
            **kwargs)

        # ======================== Add VPT specific components ========================
        img_size = kwargs['img_size']
        patch_size = kwargs['patch_size']
        num_layers = kwargs['num_layers']
        embed_dims = kwargs['embed_dims']
        drop_rate = kwargs.get('drop_rate', 0.0)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.0)
        drop_path_rate = kwargs.get('drop_path_rate', 0.0)
        num_heads = kwargs.get('num_heads', 16)
        mlp_ratio = kwargs.get('mlp_ratio', 4)
        qkv_bias = kwargs.get('qkv_bias', True)
        num_fcs = kwargs.get('num_fcs', 2)
        act_cfg = kwargs.get('act_cfg', dict(type='GELU'))
        norm_cfg = kwargs.get('norm_cfg', dict(type='LN'))
        with_cp = kwargs.get('with_cp', False)
        qk_norm = qk_norm
        
        self.depth = num_layers
        self.embed_dim = embed_dims
        self.num_classes = num_classes
        self.num_img_tokens = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # this setting is fixed
        self.args = Namespace(  proxy_prompt_len=proxy_prompt_len,
                            num_classes=num_classes,
                            test_vpt_loss_off=False,
                            test_cls_loss_off=False,
                            vpt_loss_weight=vpt_loss_weight,
                            cls_loss_weight=cls_loss_weight,
                            loss_oproxy_pos_alpha=32,
                            loss_oproxy_pos_delta=0.1,
                            loss_oproxy_neg_alpha=32,
                            loss_oproxy_neg_delta=-0.1,
                            loss_vpt_scale_ratio=loss_vpt_scale_ratio,
                            patch_type='output',
                            vpt_type_for_patch_ml='vpt',
                            vpt_type_for_cls_ml='vpt',
                            cls_type='query',
                            init_divide_by8=True,
                            quiet_mode=False,
                            initial_mapping='kmeans',
                            criterion='"proxyanchor"',
                            proxy_vpt=proxy_vpt,
                         )

        self.vpt_criterion = VPTCriterion(self.args, embed_dim=embed_dims, mapping=mapping)
        
        # 1.2.3.1 get mean feature
        self.cls_mean_feature = torch.zeros(num_classes, embed_dims, device='cuda')
        
        # 1.2.5.0
        self.prompt_dropout = nn.Dropout(prompt_drop_rate)
        # define p_idxs, o_idxs, p_depth, o_depth, p_start, p_len, o_len
        # guided idx = [p_start, p_start + p_depth)
        
        self.p_start = proxy_prompt_start_idx
        self.p_end = proxy_prompt_end_idx
        self.p_depth = self.p_end - self.p_start + 1
        
        # fix the case when p_start + p_depth > depth
        if self.p_start + self.p_depth > self.depth:
            self.p_depth = self.depth - self.p_start
        self.o_depth = self.depth - self.p_depth
        
        self.p_idxs = list(range(self.p_start, self.p_start + self.p_depth))
        self.o_idxs = list(range(0, self.p_start)) + list(range(self.p_start + self.p_depth, self.depth))
        assert len(self.p_idxs) == self.p_depth
        assert len(self.o_idxs) == self.o_depth
        assert len(self.p_idxs) + len(self.o_idxs) == self.depth
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        
        #TODO: create block
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                VPTBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    qk_norm=qk_norm,
                    mask=mask,
                    prompt_len=proxy_prompt_len \
                    if i in self.p_idxs \
                    else num_prompts,
                    num_img_tokens=self.num_img_tokens),
                )
        
        if not self.args.quiet_mode:
            print(f"Proxy prompt idxs: {self.p_idxs}")
            print(f"Other prompt idxs: {self.o_idxs}")
        
        self.o_len = num_prompts
        if not self.args.initial_mapping == 'all_classes':
            self.p_len = self.args.proxy_prompt_len
        else:
            self.p_len = num_classes
            
        if self.args.init_divide_by8:
            self.divider = 8.0
        else:
            self.divider = 1.0
        
        self.proxy_prompt_tokens = nn.Parameter(nn.init.xavier_uniform_(
                torch.randn(self.p_depth, self.p_len, embed_dims)
                ) / self.divider, requires_grad=True)
        self.other_prompt_tokens = nn.Parameter(nn.init.xavier_uniform_(
                torch.randn(self.o_depth, self.o_len, embed_dims)
                ) / self.divider, requires_grad=True)
        assert self.proxy_prompt_tokens.size()[0] + self.other_prompt_tokens.size()[0] == self.depth
        
        # 1.2.3.3
        if vpt_norm == 'l2':
            self.vpt_norm = L2NormalizationLayer(dim=-1, eps=1e-6)
        elif vpt_norm == 'layer':
            self.vpt_norm = nn.LayerNorm(embed_dims, eps=1e-6)
        else:
            self.vpt_norm = nn.Identity()

    #==================================== Tools =======================================
    
    def init_weights(self):
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') in ['Pretrained', 'Pretrained_Part']:
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            #! attn.attn -> attn
            for k, v in list(checkpoint.items()):
                if 'attn.attn' in k:
                    ks = k.split('.')
                    ks.pop(2)
                    nk = '.'.join(ks)
                    #checkpoint[nk] = checkpoint.pop(k)
                    checkpoint[nk] = v # safe modify ordered dict
                    del checkpoint[k] # safe delete ordered dict
                     
            if self.init_cfg.get('type') == 'Pretrained':
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

            elif self.init_cfg.get('type') == 'Pretrained_Part':
                state_dict = checkpoint.copy()
                para_prefix = 'image_encoder'
                prefix_len = len(para_prefix) + 1
                for k, v in checkpoint.items():
                    state_dict.pop(k)
                    if para_prefix in k:
                        state_dict[k[prefix_len:]] = v

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print_log(msg=f'Resize the pos_embed shape from '
                              f'{state_dict["pos_embed"].shape} to '
                              f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            load_state_dict(self, state_dict, strict=False, logger=None)
            
        self.saparate_qkv_params()
    
    
    #NOTE: only set param_list, other parameters will be freeze
    def set_learnable_parameters(self, param_list) -> None:
        for name, param in self.named_parameters():
            names = name.split('.')
            #NOTE: first name: blocks, norm, head
            if names[0] in param_list:
                param.requires_grad = True
            #NOTE: last name: weight, bias
            elif names[-1] in param_list:
                param.requires_grad = True
            #NOTE: last two names: fc2.bias
            elif len(names) > 1 and '.'.join([names[-2], 
                        names[-1]]) in param_list:
                param.requires_grad = True
            #NOTE: last 3 names attn.proj.bias
            elif len(names) > 2 and '.'.join([names[-3], 
                        names[-2], names[-1]]) in param_list:
                param.requires_grad = True
            #NOTE: 3rd name
            elif len(names) > 2 and names[2] in param_list:
                param.requires_grad = True
            #NOTE: full name: blocks.8.mlp.fc2.bias
            elif name in param_list:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
    def print_learnable_parameters(self) -> None:
        # print learnable parameters
        if not self.args.quiet_mode:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name + ': ' + str(list(param.shape)))
        
        # print num of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Number of params (M): %.4f' % (num_params / 1.e6))
    
    def saparate_qkv_params(self) -> None:
        for idx in range(self.depth):
            self.layers[idx].attn._saparate_qkv_params()
    
    def reset_for_mapping_update(self):
        self.vpt_criterion.reset_num_batches()
        self.cls_mean_feature = torch.zeros(self.num_classes, self.embed_dim, device='cuda')
    
    def update_mapping(self, mapping, centroids):
        if len(Counter(mapping.tolist())) > 1:
            self.vpt_criterion.update_mapping(mapping)
            self.vpt_criterion.set_centroids(centroids)
        else:
            self.vpt_criterion.set_centroids(None)
            if not self.args.quiet_mode:
                print("Warning: skip updating mapping")
        
    def get_centroids(self):
        return self.vpt_criterion.get_centroids()
    
    def get_cls_mean_feature(self):
        return self.cls_mean_feature
    
    # ==================================== Forward ====================================
    
    def forward_cls(self, inputs, labels):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if self.pre_norm:
            x = self.pre_ln(x)

        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
        
        cls_out = x[:, 0]    
        self.vpt_criterion.create_cls_mean_features(cls_out, labels, self.cls_mean_feature)
        
        return cls_out
    
    
    def forward_vpt(self, batch, vpt, labels, blk, attn_out, **kwargs):
        
        if self.training and self.p_len > 0:
            assert labels is not None
            assert vpt.size() == (1, self.p_len, self.embed_dim)
            
            vpt_loss = self.vpt_criterion(batch, vpt, labels,
                                    blk.attn, attn_out, **kwargs)
        else:
            vpt_loss = torch.Tensor([0.0]).to(batch.device)
            
        return vpt_loss
    
    def forward_features(self, x, labels=None, get_attn=False, get_tokens=False):
        
        B = x.shape[0] #
        
        # x = self.patch_embed(x) # B x 196 x 768
        # cls_tokens = self.cls_token.expand(B, -1, -1) # B x 1 x 768
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed # add position embedding 1 x 197 x 786
        
        x, hw_shape = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)
        if self.pre_norm:
            x = self.pre_ln(x)
        
        B, N, C = x.shape
        num_img_tokens = N - 1
        #! test img are not fixed
        #assert self.num_img_tokens == num_img_tokens
        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]
        
        vpt_loss = torch.Tensor([0.0]).to(x.device)
        
        attn_collector = []
        token_collector = []
        outs = []
        # define p_idxs, o_idxs, p_depth, o_depth, p_start, p_len, o_len
        for idx, layer in enumerate(self.layers):
            
            x_cls = x[:, :1, :]
            x_img = x[:, (-num_img_tokens):, :]
            
            if idx in self.p_idxs:
                local_idx = self.p_idxs.index(idx)
                prompt_tokens = self.vpt_norm(self.proxy_prompt_tokens[local_idx]).unsqueeze(0)
            else:
                local_idx = self.o_idxs.index(idx)
                prompt_tokens = self.vpt_norm(self.other_prompt_tokens[local_idx]).unsqueeze(0)
            
            _expanded_tokens = prompt_tokens.expand(B, -1, -1) # B x N x 768
            _expanded_tokens = self.prompt_dropout(_expanded_tokens)
            
            _x = torch.cat((
                x_cls,
                _expanded_tokens,  # B x N x 768
                x_img
                ), dim=1)
            x, attn_out = layer(_x)
            
            # aggregate the loss
            if self.args.proxy_vpt and idx in self.p_idxs:
                vpt_loss += self.forward_vpt(_x, prompt_tokens, labels, layer, attn_out)
            
            # if get_attn:
            #     attn_collector.append(blk.attn.full_attn)
            # if get_tokens:
            #     token_collector.append(_x)
            
            # select output features
            if idx in self.out_indices:
                out = x[:, (-num_img_tokens):, :]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
            
        if self.p_depth > 0:
            vpt_loss /= self.p_depth
        
        # [cls, prompts, img]
        # vpt_out: the other tokens except cls token, or the cls token in MAE
        B, N, D = x.shape
        
        if self.final_norm:
            x = self.norm1(x)
        
        cls_out = x[:, 0] # [cls] token
        vpt_out = x[:, 1:, :].mean(dim=1)
        
        # if get_attn:
        #     return attn_collector
        # if get_tokens:
        #     return token_collector
        
        return tuple(outs), cls_out, vpt_out, vpt_loss
    
    def forward(self, x, labels=None, update_mapping=False):
        x, cls_out, vpt_out, vpt_loss = self.forward_features(x, labels)
        if update_mapping:
            self.vpt_criterion.create_cls_mean_features(cls_out, labels, self.cls_mean_feature)
            
        # vpt_logits = None
        # if self.args.vpt_cls_loss:
        #     vpt_logits = self.vpt_head(vpt_out)
        # TODO: add vpt_logits to return
        #logits = self.head(x)
        return x, vpt_loss




