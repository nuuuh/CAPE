import torch
from torch import nn
from .TS_lib.Transformer_EncDec import Encoder, EncoderLayer
from .TS_lib.SelfAttention_Family import FullAttention, AttentionLayer
from .TS_lib.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, patch_len, horizon, num_patches, d_model, n_layers, n_heads, stride=4, dropout=0.1):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = num_patches*patch_len
        self.pred_len = horizon

        self.hidden = d_model

        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    4*d_model,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(n_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        )
        
        # Output projection - will be set dynamically on first forward pass
        self.out = None
        self.horizon = horizon
        self.dropout_layer = nn.Dropout(dropout)


    def forward(self, x_enc, time=None, dec_time=None, mask=None):
        # x_enc: [batch, num_patches, patch_len] or [batch, features, seq_len]
        enc_out = self.patch_embedding(x_enc)
        # Encoder
        # enc_out: [batch, num_patches, d_model]
        enc_out, attns = self.encoder(enc_out)
        
        # Flatten and project to horizon
        batch_size = enc_out.shape[0]
        enc_flat = enc_out.flatten(1)  # [batch, num_patches * d_model]
        
        # Initialize output layer on first forward pass (to handle variable num_patches)
        if self.out is None or self.out.in_features != enc_flat.shape[1]:
            self.out = nn.Linear(enc_flat.shape[1], self.horizon).to(enc_out.device)
        
        output = self.out(enc_flat)  # [batch, horizon]
        output = self.dropout_layer(output)
        
        return output
