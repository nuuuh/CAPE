import torch.nn as nn
from .transformer import TransformerBlock
from .embedding import time_embedding

from .layers.pos_encoding import *
from .layers.basics import *
from .layers.attention import *

class PatchTST(nn.Module):

    def __init__(self, patch_len, horizon, num_patches, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.num_features = patch_len
        self.num_preds = horizon
        self.hidden = d_model
        self.n_layers = n_layers
        self.attn_heads = n_heads

        self.feed_forward_hidden = d_model * 4

        self.embedding = time_embedding(self.num_features, d_model, num_patches=num_patches, emb_type=None)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_model * 4, dropout) for _ in range(n_layers)])

        # self.decomposition = series_decomp(kernel_size=3) 
    

    def forward(self, x, time, mask=None): 
        # import ipdb; ipdb.set_trace()
        x = self.embedding(input_sequence=x, time=time)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x


# # original
# class PatchTST(nn.Module):
#     """
#     Output dimension: 
#          [bs x target_dim x nvars] for prediction
#          [bs x target_dim] for regression
#          [bs x target_dim] for classification
#          [bs x num_patch x n_vars x patch_len] for pretrain
#     """
#     def __init__(self, patch_len:int, num_patches:int, 
#                  n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
#                   attn_dropout:float=0., dropout:float=0., act:str="gelu", 
#                  res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
#                  pe:str='zeros', learn_pe:bool=True,
#                 verbose:bool=False, **kwargs):
#         super().__init__()
#         # Backbone
#         c_in = 1
#         self.hidden = d_model
#         self.num_patches = num_patches
#         self.patch_len = patch_len
#         self.backbone = PatchTSTEncoder(c_in, num_patch=num_patches, patch_len=patch_len, 
#                                 n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
#                                 shared_embedding=shared_embedding, d_ff=d_ff,
#                                 attn_dropout=attn_dropout, dropout=dropout, act=act, 
#                                 res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
#                                 pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

#     def forward(self, z):                             
#         """
#         z: tensor [bs x num_patch x n_vars x patch_len]
#         """   
#         # import ipdb; ipdb.set_trace()
        
#         z = self.backbone(z.unsqueeze(-2))  # z: [bs x nvars x d_model x num_patch]

#         return z.squeeze().transpose(1,2)


# class PatchTSTEncoder(nn.Module):
#     def __init__(self, c_in, num_patch, patch_len, 
#                  n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
#                  d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
#                  res_attention=True, pre_norm=False,
#                  pe='zeros', learn_pe=True, verbose=False, **kwargs):

#         super().__init__()
#         self.n_vars = c_in
#         self.num_patch = num_patch
#         self.patch_len = patch_len
#         self.d_model = d_model
#         self.shared_embedding = shared_embedding        

#         # Input encoding: projection of feature vectors onto a d-dim vector space
#         if not shared_embedding: 
#             self.W_P = nn.ModuleList()
#             for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
#         else:
#             self.W_P = nn.Linear(patch_len, d_model)      

#         # Positional encoding
#         self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

#         # Residual dropout
#         self.dropout = nn.Dropout(dropout)

#         # Encoder
#         self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
#                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
#                                     store_attn=store_attn)

#     def forward(self, x) -> Tensor:          
#         """
#         x: tensor [bs x num_patch x nvars x patch_len]
#         """
#         bs, num_patch, n_vars, patch_len = x.shape
#         # Input encoding
#         if not self.shared_embedding:
#             x_out = []
#             for i in range(n_vars): 
#                 z = self.W_P[i](x[:,:,i,:])
#                 x_out.append(z)
#             x = torch.stack(x_out, dim=2)
#         else:
#             x = self.W_P(x)                                                      # x: [bs x num_patch x nvars x d_model]
#         x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

#         u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
#         u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

#         # Encoder
#         z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
#         z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
#         z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x num_patch]

#         return z
    
    
# # Cell
# class TSTEncoder(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff=None, 
#                         norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
#                         res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
#         super().__init__()

#         self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
#                                                       attn_dropout=attn_dropout, dropout=dropout,
#                                                       activation=activation, res_attention=res_attention,
#                                                       pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
#         self.res_attention = res_attention

#     def forward(self, src:Tensor):
#         """
#         src: tensor [bs x q_len x d_model]
#         """
#         output = src
#         scores = None
#         if self.res_attention:
#             for mod in self.layers: output, scores = mod(output, prev=scores)
#             return output
#         else:
#             for mod in self.layers: output = mod(output)
#             return output



# class TSTEncoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
#                  norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
#                 activation="gelu", res_attention=False, pre_norm=False):
#         super().__init__()
#         assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
#         d_k = d_model // n_heads
#         d_v = d_model // n_heads

#         # Multi-Head attention
#         self.res_attention = res_attention
#         self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

#         # Add & Norm
#         self.dropout_attn = nn.Dropout(dropout)
#         if "batch" in norm.lower():
#             self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
#         else:
#             self.norm_attn = nn.LayerNorm(d_model)

#         # Position-wise Feed-Forward
#         self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
#                                 get_activation_fn(activation),
#                                 nn.Dropout(dropout),
#                                 nn.Linear(d_ff, d_model, bias=bias))

#         # Add & Norm
#         self.dropout_ffn = nn.Dropout(dropout)
#         if "batch" in norm.lower():
#             self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
#         else:
#             self.norm_ffn = nn.LayerNorm(d_model)

#         self.pre_norm = pre_norm
#         self.store_attn = store_attn


#     def forward(self, src:Tensor, prev:Optional[Tensor]=None):
#         """
#         src: tensor [bs x q_len x d_model]
#         """
#         # Multi-Head attention sublayer
#         if self.pre_norm:
#             src = self.norm_attn(src)
#         ## Multi-Head attention
#         if self.res_attention:
#             src2, attn, scores = self.self_attn(src, src, src, prev)
#         else:
#             src2, attn = self.self_attn(src, src, src)
#         if self.store_attn:
#             self.attn = attn
#         ## Add & Norm
#         src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
#         if not self.pre_norm:
#             src = self.norm_attn(src)

#         # Feed-forward sublayer
#         if self.pre_norm:
#             src = self.norm_ffn(src)
#         ## Position-wise Feed-Forward
#         src2 = self.ff(src)
#         ## Add & Norm
#         src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
#         if not self.pre_norm:
#             src = self.norm_ffn(src)

#         if self.res_attention:
#             return src, scores
#         else:
#             return src
