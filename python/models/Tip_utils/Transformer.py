'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on Vision Transformer and BERT
* Based on AViT https://github.com/siyi-wind/AViT/blob/main/Models/Transformer/ViT_adapters.py
* Based on BLIP https://github.com/salesforce/BLIP/blob/main/models/med.py
'''
import logging

import sys


'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on Vision Transformer and BERT
* Based on AViT https://github.com/siyi-wind/AViT/blob/main/Models/Transformer/ViT_adapters.py
* Based on BLIP https://github.com/salesforce/BLIP/blob/main/models/med.py
'''
from typing import Dict, List, Tuple
import traceback
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
logger = logging.getLogger('MelanomaAnalysis')


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# RESNET50
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, mask=None, visualize=False):
        try:
            B, N, C = x.shape
            if self.with_qkv:
               qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
               q, k, v = qkv[0], qkv[1], qkv[2]
            else:
               qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
               q, k, v = qkv, qkv, qkv

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                attn = attn + mask

            attn = attn.softmax(dim=-1)
            if self.save_attention:
                self.save_attention_map(attn)
            if self.save_gradients:
                attn.register_hook(self.save_attn_gradients)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            if self.with_qkv:
               x = self.proj(x)
               x = self.proj_drop(x)
               
            return (x, attn) if visualize else x
            
        except Exception as e:
            print(f"\nError in Attention forward:")
            print(f"Input shape: {x.shape}")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.with_qkv = with_qkv
#         if self.with_qkv:
#            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#            self.proj = nn.Linear(dim, dim)
#            self.proj_drop = nn.Dropout(proj_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.save_attention = False
#         self.save_gradients = False

#     def save_attn_gradients(self, attn_gradients):
#         self.attn_gradients = attn_gradients
        
#     def get_attn_gradients(self):
#         return self.attn_gradients
    
#     def save_attention_map(self, attention_map):
#         self.attention_map = attention_map
        
#     def get_attention_map(self):
#         return self.attention_map

#     def forward(self, x, mask=None):
#         print("\nIn Attention forward:")
#         print(f"Input shape: {x.shape}")
#         B, N, C = x.shape
        
#         if self.with_qkv:
#             qkv = self.qkv(x)
#             print(f"After QKV projection shape: {qkv.shape}")
#             qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
#             print(f"After reshape shape: {qkv.shape}")
#             qkv = qkv.permute(2, 0, 3, 1, 4)
#             print(f"After permute shape: {qkv.shape}")
#             q, k, v = qkv[0], qkv[1], qkv[2]
#             print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
#         else:
#             qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#             q, k, v = qkv, qkv, qkv
#             print(f"Without QKV - Q,K,V shapes: {q.shape}")

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         print(f"Attention weights shape: {attn.shape}")

#         if mask is not None:
#             attn = attn + mask
            
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         print(f"After attention output shape: {x.shape}")

#         if self.with_qkv:
#             x = self.proj(x)
#             x = self.proj_drop(x)
#             print(f"Final output shape: {x.shape}")
            
#         return x


class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = k_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.kv_proj = nn.Linear(k_dim, k_dim*2, bias=qkv_bias)
        self.q_proj = nn.Linear(q_dim, k_dim)
        self.proj = nn.Linear(k_dim, k_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, q, k, visualize=False):
        try:
            B, N_k, K = k.shape
            _, N_q, _ = q.shape
            kv = self.kv_proj(k).reshape(B, N_k, 2, self.num_heads, K//self.num_heads).permute(2, 0, 3, 1, 4)  
            k, v = kv[0], kv[1]  # (B,H,N,C)
            q = self.q_proj(q).reshape(B, N_q, self.num_heads, K//self.num_heads).permute(0, 2, 1, 3)  # (B,H,N,C)
            
            # 数值稳定性检查
            if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
                print("Warning: NaN detected in attention inputs")
                q = torch.nan_to_num(q)
                k = torch.nan_to_num(k)
                v = torch.nan_to_num(v)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            
            if self.save_attention:
                self.save_attention_map(attn)
            if self.save_gradients:
                attn.register_hook(self.save_attn_gradients)
            attn = self.attn_drop(attn)

            out = (attn @ v).transpose(1, 2).reshape(B, N_q, K)
            out = self.proj(out)
            out = self.proj_drop(out)
            
            return (out, attn) if visualize else out
            
        except Exception as e:
            print(f"\nError in CrossAttention forward:")
            print(f"Input shapes: q: {q.shape}, k: {k.shape}")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise

# RESNET50
class Block(nn.Module):
    def __init__(self, dim, num_heads=8, is_cross_attention=False, encoder_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = 0.5
        self.norm1 = norm_layer(dim)
        self.is_cross_attention = is_cross_attention
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        if self.is_cross_attention:
           self.cross_attn = CrossAttention(
               q_dim=dim, k_dim=encoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
           self.cross_norm = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, encoder_hidden_states=None, mask=None, visualize=False):
        try:
            if not visualize:
                # Self attention with residual
                x_attn = self.attn(self.norm1(x), mask=mask)
                x = x + self.drop_path(x_attn)
                
                # Cross attention if enabled
                if self.is_cross_attention:
                    assert encoder_hidden_states is not None
                    x_cross = self.cross_attn(self.cross_norm(x), encoder_hidden_states)
                    x = x + self.drop_path(x_cross)
                
                # MLP with residual
                x_norm = self.norm2(x)
                x_mlp = self.mlp(x_norm)
                x = x + self.drop_path(x_mlp)
                
                return x
            else:
                # Visualization mode
                x_norm = self.norm1(x)
                x_attn, self_attn = self.attn(x_norm, mask=mask, visualize=True)
                x = x + self.drop_path(x_attn)
                
                cross_attn = None
                if self.is_cross_attention:
                    assert encoder_hidden_states is not None
                    x_cross, cross_attn = self.cross_attn(self.cross_norm(x), encoder_hidden_states, visualize=True)
                    x = x + self.drop_path(x_cross)
                    
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x, {'self_attn': self_attn, 'cross_attn': cross_attn}
                
        except Exception as e:
            print(f"\nError in Block forward:")
            print(f"Input shape: {x.shape}")
            if encoder_hidden_states is not None:
                print(f"Encoder states shape: {encoder_hidden_states.shape}")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise

# # VIT
# class Block(nn.Module):
#     def __init__(self, dim, num_heads=8, is_cross_attention=False, encoder_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.scale = 0.5
#         self.norm1 = norm_layer(dim)
#         self.is_cross_attention = is_cross_attention
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         if self.is_cross_attention:
#            self.cross_attn = CrossAttention(
#                q_dim=dim, k_dim=encoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#            self.cross_norm = norm_layer(dim)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, encoder_hidden_states=None, mask=None, visualize=False):
#         print("\nIn Block forward:")
#         print(f"Input x shape: {x.shape}")
#         if encoder_hidden_states is not None:
#             print(f"Encoder states shape: {encoder_hidden_states.shape}")
        
#         if not visualize:
#             norm_x = self.norm1(x)
#             print(f"After norm1 shape: {norm_x.shape}")
#             x_attn = self.attn(norm_x, mask=mask)
#             print(f"After self-attention shape: {x_attn.shape}")
#             x = x + self.drop_path(x_attn)
            
#             if self.is_cross_attention:
#                 x_cross = self.cross_attn(self.cross_norm(x), encoder_hidden_states)
#                 print(f"After cross-attention shape: {x_cross.shape}")
#                 x = x + self.drop_path(x_cross)
            
#             x_norm = self.norm2(x)
#             print(f"Before MLP shape: {x_norm.shape}")
#             x_mlp = self.mlp(x_norm)
#             print(f"After MLP shape: {x_mlp.shape}")
#             x = x + self.drop_path(x_mlp)
#             return x

class TabularTransformerEncoder(nn.Module):
   def __init__(self, args: Dict, cat_lengths_tabular: List, con_lengths_tabular: List) -> None:
       super().__init__()
       
       # Basic attributes
       self.num_cat = len(cat_lengths_tabular)  
       self.num_con = len(con_lengths_tabular)
       self.num_unique_cat = sum(cat_lengths_tabular)
       self.embedding_dim = args.tabular_embedding_dim
       
       # Category offsets
       cat_offsets = torch.tensor([0] + cat_lengths_tabular[:-1]).cumsum(0)
       self.register_buffer('cat_offsets', cat_offsets, persistent=False)
       
       # Embeddings
       self.cat_embedding = nn.Embedding(self.num_unique_cat, args.tabular_embedding_dim)
       self.con_proj = nn.Linear(1, args.tabular_embedding_dim)
       
       # Special tokens
       self.cls_token = nn.Parameter(torch.zeros(1, 1, args.tabular_embedding_dim))
       self.mask_special_token = nn.Parameter(torch.zeros(1, 1, args.tabular_embedding_dim))
       
       # Column name embeddings
       pos_ids = torch.arange(self.num_cat + self.num_con + 1)
       self.register_buffer('pos_ids', pos_ids, persistent=False)
       self.column_embedding = nn.Embedding(self.num_cat + self.num_con + 1, args.tabular_embedding_dim)
       
       # Layer norm and dropout 
       self.norm = nn.LayerNorm(args.tabular_embedding_dim)
       self.dropout = nn.Dropout(args.embedding_dropout) if args.embedding_dropout > 0. else nn.Identity()
       
       # Transformer blocks (4 layers per paper)
       self.transformer_blocks = nn.ModuleList([
           Block(dim=args.tabular_embedding_dim,
                 num_heads=8,
                 drop=args.drop_rate,
                 is_cross_attention=False) 
           for _ in range(4)
       ])

       if not args.pretrain.enabled:
           trunc_normal_(self.cls_token, std=.02)
           trunc_normal_(self.mask_special_token, std=.02)
           self.apply(self._init_weights)
           
   def _init_weights(self, m):
       if isinstance(m, (nn.Linear, nn.Embedding)):
           trunc_normal_(m.weight, std=.02)
           if isinstance(m, nn.Linear) and m.bias is not None:
               nn.init.constant_(m.bias, 0)
       elif isinstance(m, nn.LayerNorm):
           nn.init.constant_(m.bias, 0)
           nn.init.constant_(m.weight, 1.0)

   def embedding(self, x: torch.Tensor, mask_special: torch.Tensor = None) -> torch.Tensor:
       # Input validation
       logger.info(x.shape[1])
       logger.info(self.num_cat + self.num_con)
       assert x.dim() == 2, f"Expected 2D input tensor, got {x.dim()}D" 
       assert x.shape[1] == self.num_cat + self.num_con

       # Category features
       cat_features = x[:, :self.num_cat]
       cat_indices = []
       for i in range(self.num_cat):
           curr_feat = cat_features[:, i].long()
           start_idx = self.cat_offsets[i]
           end_idx = self.cat_offsets[i + 1] if i + 1 < len(self.cat_offsets) else self.num_unique_cat
           curr_feat = torch.clamp(curr_feat + start_idx, 0, end_idx - 1)
           cat_indices.append(curr_feat)
       cat_x = self.cat_embedding(torch.stack(cat_indices, dim=1))

       # Continuous features
       con_features = x[:, self.num_cat:].unsqueeze(-1)
       con_features = torch.nan_to_num(con_features, 0.0)
       con_features = torch.clamp(con_features, -1e6, 1e6)
       con_x = self.con_proj(con_features)

       # Combine features
       x = torch.cat([cat_x, con_x], dim=1)

       # Apply mask token if needed
       if mask_special is not None:
           mask_special = mask_special.unsqueeze(-1)
           mask_tokens = self.mask_special_token.expand(x.shape[0], x.shape[1], -1)
           x = mask_special * mask_tokens + (~mask_special) * x

       # Add CLS token and column embeddings  
       cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
       x = torch.cat([cls_tokens, x], dim=1)
       x = x + self.column_embedding(self.pos_ids)

       return self.dropout(self.norm(x))

   def forward(self, x: torch.Tensor, mask: torch.Tensor = None, mask_special: torch.Tensor = None) -> torch.Tensor:
       if mask is not None:
           assert mask.shape[1] == x.shape[1], f"Mask shape {mask.shape} does not match input shape {x.shape}"
           
       x = self.embedding(x, mask_special=mask_special)

       # Create attention mask
       if mask is not None:
           B, N = mask.shape
           cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
           mask = torch.cat([cls_mask, mask], dim=1)
           mask = mask[:, None, :].repeat(1, N+1, 1)
           mask = mask & ~torch.eye(N+1, dtype=torch.bool, device=mask.device)[None, :, :]
           mask = mask[:, None, :, :] * (-1e9)

       # Apply transformer blocks
       for block in self.transformer_blocks:
           x = block(x, mask=mask)

       return x

class MultimodalTransformerEncoder(nn.Module):
    def __init__(self, args: Dict) -> None:
        super().__init__()
        
        # 正确设置交叉注意力参数
        self.num_heads = args.num_heads if hasattr(args, 'num_heads') else 8
        self.mlp_ratio = args.mlp_ratio if hasattr(args, 'mlp_ratio') else 4.0
        
        # 投影层和归一化层
        self.image_proj = nn.Linear(args.embedding_dim, args.multimodal_embedding_dim)
        self.image_norm = nn.LayerNorm(args.multimodal_embedding_dim)
        self.tabular_proj = (nn.Linear(args.tabular_embedding_dim, args.multimodal_embedding_dim) 
                           if args.tabular_embedding_dim != args.multimodal_embedding_dim 
                           else nn.Identity())
        self.tabular_norm = nn.LayerNorm(args.multimodal_embedding_dim)  # 添加这一行
        
        # Transformer blocks with cross attention
        self.transformer_blocks = nn.ModuleList([
            Block(
                dim=args.multimodal_embedding_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                is_cross_attention=True,
                encoder_dim=args.multimodal_embedding_dim,
                qkv_bias=True,
                drop=args.drop_rate if hasattr(args, 'drop_rate') else 0.0,
                attn_drop=args.attn_drop_rate if hasattr(args, 'attn_drop_rate') else 0.0
            ) for _ in range(args.multimodal_transformer_num_layers)
        ])
        
        self.norm = nn.LayerNorm(args.multimodal_embedding_dim)

    def forward(self, tabular_features: torch.Tensor, image_features: torch.Tensor, visualize: bool = False) -> torch.Tensor:
        try:
            # 处理图像特征维度
            if len(image_features.shape) == 4:
                B, C, H, W = image_features.shape
                image_features = image_features.reshape(B, C, -1).permute(0, 2, 1)
                
            # 投影和归一化
            image_features = self.image_proj(image_features)
            image_features = self.image_norm(image_features)
            
            # 表格特征处理
            tabular_features = self.tabular_proj(tabular_features)
            tabular_features = self.tabular_norm(tabular_features)

            # Transformer处理
            if not visualize:
                x = tabular_features
                for i, transformer_block in enumerate(self.transformer_blocks):
                    x_prev = x
                    x = transformer_block(x, encoder_hidden_states=image_features)
                    # 残差连接
                    if torch.isnan(x).any():
                        print(f"NaN detected in transformer block {i}")
                        x = x_prev
                x = self.norm(x)
                return x
            else:
                x = tabular_features
                attns = []
                for transformer_block in self.transformer_blocks:
                    x, attn = transformer_block(x, encoder_hidden_states=image_features, visualize=True)
                    attns.append(attn)
                x = self.norm(x)
                return x, attns
                
        except Exception as e:
            print(f"\nError in MultimodalTransformerEncoder forward:")
            print(f"Tabular features shape: {tabular_features.shape}")
            print(f"Image features shape: {image_features.shape}")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise

class TabularPredictor(nn.Module):
    def __init__(self, args: Dict, cat_lengths_tabular: List[int], con_lengths_tabular: List[int], num_unique_cat: int = None) -> None:
        super().__init__()
        self.num_cat = len(cat_lengths_tabular)  # 2 for ISIC
        self.num_con = len(con_lengths_tabular)  # 30 for ISIC
        
        # 保存每个类别特征的类别数
        self.cat_lengths = cat_lengths_tabular
        
        # 为每个类别特征创建独立的分类器
        self.cat_classifiers = nn.ModuleList([
            nn.Linear(args.tabular_embedding_dim, length) 
            for length in cat_lengths_tabular
        ])

        # 连续值回归器
        self.con_regressor = nn.Linear(args.tabular_embedding_dim, 1)
        
        # Debug信息
        print(f"TabularPredictor initialized:")
        print(f"Number of categorical features: {self.num_cat}")
        print(f"Category lengths: {cat_lengths_tabular}")
        print(f"Number of continuous features: {self.num_con}")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        
        # Remove CLS token if present
        if x.dim() == 3:  # [B, N, D]
            x = x[:, 1:]  # Remove CLS token

        # 类别预测
        cat_outputs = []
        for i, classifier in enumerate(self.cat_classifiers):
            if x.dim() == 3:
                cat_feat = x[:, i]  # [B, D]
            else:
                cat_feat = x  # [B, D]
            cat_out = classifier(cat_feat)  # [B, num_classes]
            cat_outputs.append(cat_out)

        cat_output = torch.stack(cat_outputs, dim=1)  # [B, num_cat, max_classes]

        # 连续特征预测
        if x.dim() == 3:
            con_input = x[:, self.num_cat:]  # [B, num_con, D]
        else:
            con_input = x  # [B, D]
            
        con_output = self.con_regressor(con_input)

        return cat_output, con_output
        
   