"""
Improved VPT (Visual Prompt Tuning) Implementation
Fixed critical issues and improved code quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer as timm_vit
from timm.models.vision_transformer import DropPath, Mlp, LayerScale
from functools import partial
from typing import Optional, Tuple, List
import logging

from .criterion import VPTCriterion
from .attn import VPTAttention

logger = logging.getLogger(__name__)


class VPTBlock(nn.Module):
    """Improved VPT Block with better error handling and documentation."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        mask: bool = False,
        batch_first: bool = True,
        prompt_len: int = 0,
        num_img_tokens: int = 196,
        **kwargs,
    ) -> None:
        super().__init__()
        
        if not batch_first:
            raise ValueError("Only batch_first=True is supported")
        
        self.norm1 = norm_layer(dim)
        self.attn = VPTAttention(
            dim=dim,
            num_heads=num_heads, 
            qkv_bias=qkv_bias,
            qk_norm=qk_norm, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            return_output=True,
            mask=mask,
            prompt_len=prompt_len,
            num_img_tokens=num_img_tokens
        )
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with improved error handling."""
        try:
            x_out, attn_out = self.attn(self.norm1(x))
            x = x + self.drop_path1(self.ls1(x_out))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, attn_out
        except RuntimeError as e:
            logger.error(f"Error in VPTBlock forward pass: {e}")
            raise


class PromptVisionTransformer(timm_vit.VisionTransformer):
    """Improved Prompt Vision Transformer with better error handling and memory management."""
    
    def __init__(
        self, 
        args, 
        mapping: Optional[torch.Tensor] = None, 
        prompt_init: Optional[torch.Tensor] = None, 
        last_prompt_init: Optional[torch.Tensor] = None, 
        num_classes: int = 200, 
        num_img_tokens: int = 196,
        global_pool: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.args = args
        self.num_prompts = args.num_prompts
        self.embed_dim = kwargs['embed_dim']
        self.num_heads = kwargs['num_heads']
        self.qkv_bias = kwargs.get('qkv_bias', True)
        self.qk_norm = kwargs.get('qk_norm', False)
        self.mlp_ratio = kwargs.get('mlp_ratio', 4)
        self.init_values = kwargs.get('init_values', None)
        self.attn_drop = kwargs.get('attn_drop_rate', 0)
        self.proj_drop = kwargs.get('proj_drop_rate', 0)
        self.drop_path_rate = kwargs.get('drop_path_rate', 0)
        self.norm_layer = kwargs.get('norm_layer', None)
        self.depth = kwargs.get('depth', 12)
        self.block_fn = kwargs.get('block_fn', VPTBlock)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_img_tokens = num_img_tokens
        
        self.device = self.args.device
        self.vpt_criterion = VPTCriterion(args, embed_dim=self.embed_dim, mapping=mapping)
        self.cls_mean_feature = torch.zeros(
            self.num_classes, self.embed_dim, device=self.device
        )
        
        self.prompt_dropout = nn.Dropout(self.args.prompt_drop_rate)
        
        # Calculate prompt indices and depths
        self._setup_prompt_indices()
        
        # Create prompt tokens
        self._initialize_prompt_tokens()
        
        # Setup normalization
        self._setup_normalization()
        
        # Rebuild blocks with VPT support
        self._rebuild_blocks()
        
        if not self.args.quiet_mode:
            self._print_prompt_info()
    
    def _setup_prompt_indices(self) -> None:
        """Setup prompt indices and depths."""
        self.p_start = self.args.proxy_prompt_start_idx
        self.p_end = self.args.proxy_prompt_end_idx
        self.p_depth = self.p_end - self.p_start + 1
        
        # Ensure prompt depth doesn't exceed model depth
        if self.p_start + self.p_depth > self.depth:
            self.p_depth = self.depth - self.p_start
            logger.warning(f"Adjusted prompt depth to {self.p_depth} to fit model depth")
        
        self.o_depth = self.depth - self.p_depth
        
        self.p_idxs = list(range(self.p_start, self.p_start + self.p_depth))
        self.o_idxs = (list(range(0, self.p_start)) + 
                      list(range(self.p_start + self.p_depth, self.depth)))
        
        # Validation
        assert len(self.p_idxs) == self.p_depth
        assert len(self.o_idxs) == self.o_depth
        assert len(self.p_idxs) + len(self.o_idxs) == self.depth
    
    def _initialize_prompt_tokens(self) -> None:
        """Initialize prompt tokens with proper scaling."""
        self.o_len = self.num_prompts
        
        if self.args.initial_mapping != 'all_classes':
            self.p_len = self.args.proxy_prompt_len
        else:
            self.p_len = self.num_classes
        
        divider = 8.0 if self.args.init_divide_by8 else 1.0
        
        # Initialize proxy prompt tokens
        self.proxy_prompt_tokens = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.randn(self.p_depth, self.p_len, self.embed_dim)
            ) / divider, 
            requires_grad=True
        )
        
        # Initialize other prompt tokens
        self.other_prompt_tokens = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.randn(self.o_depth, self.o_len, self.embed_dim)
            ) / divider, 
            requires_grad=True
        )
        
        # Validation
        assert (self.proxy_prompt_tokens.size(0) + 
                self.other_prompt_tokens.size(0) == self.depth)
    
    def _setup_normalization(self) -> None:
        """Setup VPT normalization layer."""
        if self.args.vpt_norm == 'l2':
            self.vpt_norm = L2NormalizationLayer(dim=-1, eps=1e-6)
        elif self.args.vpt_norm == 'layer':
            self.vpt_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        else:
            self.vpt_norm = nn.Identity()
    
    def _rebuild_blocks(self) -> None:
        """Rebuild transformer blocks with VPT support."""
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        
        self.blocks = nn.Sequential(*[
            VPTBlock(
                dim=self.embed_dim, 
                num_heads=self.num_heads, 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, 
                qk_norm=self.qk_norm, 
                proj_drop=self.proj_drop,
                attn_drop=self.attn_drop, 
                init_values=self.init_values, 
                drop_path=dpr[i],
                norm_layer=self.norm_layer, # type: ignore
                mask=self.args.mask,
                prompt_len=self.args.proxy_prompt_len if i in self.p_idxs else self.args.num_prompts,
                num_img_tokens=self.num_img_tokens,
            )
            for i in range(self.depth)
        ])
    
    def _print_prompt_info(self) -> None:
        """Print prompt configuration information."""
        print(f"Proxy prompt indices: {self.p_idxs}")
        print(f"Other prompt indices: {self.o_idxs}")
    
    def set_learnable_parameters(self, param_list: List[str]) -> None:
        """Set which parameters should be learnable."""
        for name, param in self.named_parameters():
            names = name.split('.')
            
            # Check various naming patterns
            should_learn = (
                names[0] in param_list or
                names[-1] in param_list or
                (len(names) > 1 and '.'.join([names[-2], names[-1]]) in param_list) or
                (len(names) > 2 and '.'.join([names[-3], names[-2], names[-1]]) in param_list) or
                (len(names) > 2 and names[2] in param_list) or
                name in param_list
            )
            
            param.requires_grad = should_learn
    
    def print_learnable_parameters(self) -> None:
        """Print information about learnable parameters."""
        if not self.args.quiet_mode:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {list(param.shape)}")
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of learnable params (M): {num_params / 1e6:.4f}')
    
    def separate_qkv_params(self) -> None:
        """Separate QKV parameters for fine-grained control."""
        for idx in range(self.depth):
            self.blocks[idx].attn._separate_qkv_params()
    
    def reset_for_mapping_update(self) -> None:
        """Reset statistics for mapping update."""
        self.vpt_criterion.reset_num_batches()
        self.cls_mean_feature = torch.zeros(
            self.num_classes, self.embed_dim, device=self.device
        )
    
    def update_mapping(self, mapping, centroids) -> None:
        """Update class-to-prompt mapping."""
        try:
            if len(torch.unique(mapping)) > 1:
                self.vpt_criterion.update_mapping(mapping)
                self.vpt_criterion.set_centroids(centroids)
            else:
                self.vpt_criterion.set_centroids(None)
                if not self.args.quiet_mode:
                    logger.warning("Skipping mapping update - all classes map to same prompt")
        except Exception as e:
            logger.error(f"Error updating mapping: {e}")
            raise
    
    def get_centroids(self) -> Optional[torch.Tensor]:
        """Get current centroids."""
        return self.vpt_criterion.get_centroids()
    
    def get_cls_mean_feature(self) -> torch.Tensor:
        """Get class mean features."""
        return self.cls_mean_feature
    
    def forward_vpt(
        self, 
        batch: torch.Tensor, 
        vpt: torch.Tensor, 
        labels: torch.Tensor, 
        blk: VPTBlock, 
        attn_out: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for VPT loss computation."""
        if not self.training or self.p_len == 0:
            return torch.tensor(0.0, device=batch.device)
        
        if labels is None:
            raise ValueError("Labels are required for VPT loss computation during training")
        
        if vpt.size() != (1, self.p_len, self.embed_dim):
            raise ValueError(f"Expected VPT shape (1, {self.p_len}, {self.embed_dim}), "
                           f"got {vpt.size()}")
        
        try:
            return self.vpt_criterion(batch, vpt, labels, blk.attn, attn_out, **kwargs)
        except Exception as e:
            logger.error(f"Error in VPT forward pass: {e}")
            return torch.tensor(0.0, device=batch.device)
    
    def forward_features( # type: ignore
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        get_attn: bool = False, 
        get_tokens: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through feature extraction layers."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        vpt_loss = torch.tensor(0.0, device=x.device)
        attn_collector = [] if get_attn else None
        token_collector = [] if get_tokens else None
        
        # Forward through transformer blocks
        for idx, blk in enumerate(self.blocks):
            x_cls = x[:, :1, :]
            x_img = x[:, -self.num_img_tokens:, :]
            
            # Get appropriate prompt tokens
            if idx in self.p_idxs:
                local_idx = self.p_idxs.index(idx)
                prompt_tokens = self.vpt_norm(self.proxy_prompt_tokens[local_idx]).unsqueeze(0)
            else:
                local_idx = self.o_idxs.index(idx)
                prompt_tokens = self.vpt_norm(self.other_prompt_tokens[local_idx]).unsqueeze(0)
            
            # Expand and apply dropout
            expanded_tokens = prompt_tokens.expand(B, -1, -1)
            expanded_tokens = self.prompt_dropout(expanded_tokens)
            
            # Concatenate tokens
            _x = torch.cat((x_cls, expanded_tokens, x_img), dim=1)
            
            # Forward through block
            x, attn_out = blk(_x)

            # Compute VPT loss for proxy prompts
            if self.args.proxy_vpt and idx in self.p_idxs:
                block_vpt_loss = self.forward_vpt(_x, prompt_tokens, labels, blk, attn_out)
                vpt_loss = vpt_loss + block_vpt_loss  # Avoid in-place operation
            
            # Collect attention/tokens if requested
            if get_attn:
                attn_collector.append(blk.attn.full_attn)
            if get_tokens:
                token_collector.append(_x)
        
        # Average VPT loss over proxy layers
        if self.p_depth > 0:
            vpt_loss = vpt_loss / self.p_depth
        
        # Final processing
        B, N, D = x.shape
        if self.global_pool:
            x = self.norm(x)
            if self.args.mae_pooling == 'img':
                out = x[:, (N - self.num_img_tokens):, :].mean(dim=1)
            elif self.args.mae_pooling == 'all':
                out = x[:, 1:, :].mean(dim=1)
            else:
                out = x[:, 1:, :].mean(dim=1)
            vpt_out = x[:, 0]
        else:
            x = self.norm(x)
            out = x[:, 0]  # CLS token
            vpt_out = x[:, 1:, :].mean(dim=1)
        
        # Return requested outputs
        if get_attn:
            return attn_collector
        if get_tokens:
            return token_collector
        
        return out, vpt_out, vpt_loss
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        update_mapping: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Main forward pass."""
        try:
            x, vpt_out, vpt_loss = self.forward_features(x, labels)
            
            if update_mapping and labels is not None:
                self.vpt_criterion.create_cls_mean_features(x, labels, self.cls_mean_feature)
            
            # VPT classification head
            vpt_logits = None
            if self.args.vpt_cls_loss:
                vpt_logits = self.vpt_head(vpt_out)
            
            # Main classification head
            logits = self.head(x)
            
            return logits, vpt_loss, vpt_logits
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("CUDA out of memory in forward pass")
                torch.cuda.empty_cache()
            raise


class L2NormalizationLayer(nn.Module):
    """L2 normalization layer."""
    
    def __init__(self, dim: int = 1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


def create_vpt_vit(vit_size: str = 'B', **kwargs) -> PromptVisionTransformer:
    """Create VPT Vision Transformer with specified size."""
    size_configs = {
        'B': {'patch_size': 16, 'num_img_tokens': 196, 'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'L': {'patch_size': 16, 'num_img_tokens': 1024, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
        'S': {'patch_size': 16, 'num_img_tokens': 196, 'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'H': {'patch_size': 14, 'num_img_tokens': 256, 'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
    }
    
    if vit_size not in size_configs:
        raise ValueError(f"Unsupported ViT size: {vit_size}. Supported: {list(size_configs.keys())}")
    
    config = size_configs[vit_size]
    
    model = PromptVisionTransformer(
        patch_size=config['patch_size'], 
        embed_dim=config['embed_dim'], 
        depth=config['depth'], 
        num_heads=config['num_heads'], 
        mlp_ratio=4, 
        qkv_bias=True, 
        num_img_tokens=config['num_img_tokens'],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )
    
    return model