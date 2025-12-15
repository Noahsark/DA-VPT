import torch
import torch.nn as nn
from torch.jit import Final
import torch.nn.functional as F

#NUM_IMG_TOKENS = 196


class VPTAttention(nn.Module):
    # This class implements the multi-head attention mechanism.
    fused_attn: Final[bool]
    # A class attribute indicating whether a fused attention implementation should be used.
    
    def __init__(
            self,
            dim: int,  # The dimensionality of the input features.
            num_heads: int = 12,  # The number of attention heads.
            qkv_bias: bool = False,  # Whether to add a bias term to the QKV linear transformation.
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
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Optional normalization layers for queries and keys.
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()  
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout layer for attention weights.
        self.proj = nn.Linear(dim, dim)  # Linear projection layer.
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout layer after projection.
    
    def _print_params(self):
        for name, param in self.named_parameters():
            print(name + ': ' + str(param.sum().item()) + ' ' + str(param.requires_grad))
    
    
    def _separate_qkv_params(self):
        # copy qkv to q, k, v
        # weight: [3*dim, dim] -> [dim, dim]
        # use assign '=' in order to connect the computational 
        # graph between qkv and q, k, v
        with torch.no_grad():
            self.q.weight.data = self.qkv.weight[:self.q.in_features]
            self.k.weight.data = self.qkv.weight[self.q.in_features: self.q.in_features + self.k.in_features]
            self.v.weight.data = self.qkv.weight[self.q.in_features + self.k.in_features:]
            if self.qkv.bias is not None:
                self.q.bias.data = self.qkv.bias[:self.q.in_features]
                self.k.bias.data = self.qkv.bias[self.q.in_features: self.q.in_features + self.k.in_features]
                self.v.bias.data = self.qkv.bias[self.q.in_features + self.k.in_features:]    
            # recombine q, k, v for assertion
            recombined_qkv = torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)
            recombined_bias = torch.cat([self.q.bias, self.k.bias, self.v.bias], dim=0) \
                if self.qkv.bias is not None else None
            # Assert to check if the recombined weights and biases match the originals
            assert torch.allclose(recombined_qkv, self.qkv.weight, atol=1e-6), "Weights do not match"
            if self.qkv.bias is not None:
                assert torch.allclose(recombined_bias, self.qkv.bias, atol=1e-6), "Biases do not match"
    
    
    def _generate_mask(self, L: int, S: int, len: int, device: torch.device) -> torch.Tensor:
        # Generate a mask to prevent attention from some prompts.
        # L: len of query
        # S: len of key
        # len: len of prompt
        mask = torch.zeros(L, S).to(device)
        mask[1:(1+len), 0:(1+len)] = -float('inf')
        return mask
    
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape  # x: [batch size, number of tokens, channel dimension]
        # Apply QKV linear transformation, reshape for multi-head, and permute.
        # After self.qkv: [B, N, 3*C], after reshape: [B, N, 3, num_heads, head_dim], 
        # after permute: [3, B, num_heads, N, head_dim]
        #! saperate q, k, v 
        #_qkv = self.qkv(x) # [B, N, 3*C]
        # qkv = _qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
        # q, k, v = qkv.unbind(0)  # q, k, v: [B, num_heads, N, head_dim]

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
        
        x = self.proj(x)  # Linear projection, x: [B, N, C], same as input dimension
        x = self.proj_drop(x)  # Apply dropout, x: same as above [B, N, C]
        
        if self.return_output:
            return x, attn_out
        else:
            return x


