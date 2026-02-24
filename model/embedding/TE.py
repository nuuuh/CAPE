import torch
import torch.nn as nn
from .position import PositionalEncoding
from .time import *

from .time import (
    LearnablePositionalEncodingSmall,
    LearnablePositionalEncodingBig,
    LearnablePositionalEncodingHybrid,
    T2vCos,
    T2vSin,
    GaussianPositionalEncoding
)


time_embeddings = {
                    "t2v_sin": T2vSin,
                    "t2v_cos": T2vCos,
                    "fully_learnable_big": LearnablePositionalEncodingBig,
                    "fully_learnable_small": LearnablePositionalEncodingSmall,
                    "gaussian": GaussianPositionalEncoding,
                    "hybrid": LearnablePositionalEncodingHybrid,
                }

class time_embedding(nn.Module):
    def __init__(self, patch_len, embedding_dim, num_patches=None, emb_type=None):
        
        super().__init__()
        self.input = nn.Linear(in_features=patch_len, out_features=embedding_dim)
        if emb_type is None:
            self.time_embedding = PositionalEncoding(d_model=embedding_dim, max_len=53)
        else:
            self.time_embedding = time_embeddings[emb_type](in_features=1, out_features=embedding_dim)

        # self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embedding_dim
        self.num_patches = num_patches
        # import ipdb; ipdb.set_trace()
        W_pos = torch.zeros((num_patches, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
        self.te = nn.Parameter(W_pos, requires_grad=True)

    
    def forward(self, input_sequence, time):
        # import ipdb; ipdb.set_trace()
        x = self.input(input_sequence)

        # x = x + self.te
        x = x + self.time_embedding(torch.LongTensor(range(x.shape[1])).unsqueeze(0).repeat(x.shape[0],1))

        return x


        # batch_size = input_sequence.size(0)
        # seq_length = input_sequence.size(1)
        # obs_embed = self.input(input_sequence)  # [batch_size, seq_length, embedding_dim]
        # x = obs_embed.repeat(1, 1, 2)           # [batch_size, seq_length, embedding_dim*2]
        # for i in range(batch_size):
        #     x[i, :, self.embed_size:] = self.position(doy_sequence[i, :])     # [seq_length, embedding_dim]
        # x = self.dropout(x)
        
        # return x
