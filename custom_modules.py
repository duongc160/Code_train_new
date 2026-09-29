"""
Custom modules for YOLO11 - Vision Transformer (ViT) integration
Simplified version compatible with Ultralytics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViT(nn.Module):
    """
    Lightweight Vision Transformer for YOLO integration.
    Uses 1x1 Conv instead of patch embedding to preserve resolution.
    
    Args:
        c1: Input channels (auto-injected by YOLO from previous layer)
        c2: Output channels
        num_heads: Number of attention heads (default: 4)
        depth: Number of transformer blocks (default: 2)
    """
    
    def __init__(self, c1, c2, num_heads=4, depth=2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        embed_dim = c2
        
        # Input projection (1x1 conv to preserve resolution)
        self.input_proj = nn.Conv2d(c1, embed_dim, kernel_size=1)
        self.norm_pre = nn.LayerNorm(embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
            }))
        
        # Output projection
        self.output_proj = nn.Conv2d(embed_dim, c2, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, embed_dim, H, W)
        
        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        x = self.norm_pre(x)
        
        # Transformer blocks
        for block in self.blocks:
            # Self-attention with residual
            x_norm = block['norm1'](x)
            attn_out, _ = block['attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            
            # MLP with residual
            x = x + block['mlp'](block['norm2'](x))
        
        # Reshape back to feature map
        x = x.transpose(1, 2).reshape(B, self.c2, H, W)
        x = self.output_proj(x)
        
        return x


def register_custom_modules():
    """Register custom modules to Ultralytics."""
    from ultralytics.nn import modules
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.tasks as tasks
    
    # Register ViT to all required locations
    setattr(modules, 'ViT', ViT)
    setattr(block, 'ViT', ViT)
    setattr(tasks, 'ViT', ViT)
    
    print("✅ Custom modules registered: ViT")


# Auto-register when imported
if __name__ != "__main__":
    register_custom_modules()
