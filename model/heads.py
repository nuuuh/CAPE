import torch
import torch.nn as nn

from .layers.pos_encoding import *
from .layers.basics import *
from .layers.attention import *

class representation_head(nn.Module):
    def __init__(self, model, hidden_size, patch_size, dropout):
        super().__init__()
        self.model = model
        self.head = nn.Linear(hidden_size, patch_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time, dec_time=None, mask=None):
        # import ipdb; ipdb.set_trace()
        x, time = self.model(x, time, dec_time, mask)

        self.latent = x
        x = self.head(x)
        x = self.dropout(x)
        return x, time

class forecast_head(nn.Module):
    def __init__(self, model, hidden_size, num_patches, horizon, dropout):
        super().__init__()
        self.model = model
        self.head = nn.Linear(int(hidden_size*num_patches), horizon)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time=None, dec_time=None):
        x, time = self.model(x, time, dec_time)
        # import ipdb; ipdb.set_trace()
        x = x.flatten(1)
        self.latent = x
        x = self.head(x)
        x = self.dropout(x)
        return x, time




class representation_head_(nn.Module):
    def __init__(self, model, hidden_size, patch_size, dropout):
        super().__init__()
        self.model = model
        self.head = nn.Linear(hidden_size, patch_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time, dec_time=None, mask=None):
        # import ipdb; ipdb.set_trace()
        x, attns, time, ngm = self.model(x, time, dec_time, mask)

        self.latent = x
        x = self.head(x)
        x = self.dropout(x)
        return x, attns, time, ngm

class forecast_head_(nn.Module):
    def __init__(self, model, hidden_size, num_patches, horizon, dropout):
        super().__init__()
        self.model = model
        self.head = nn.Linear(int(hidden_size), horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time=None, dec_time=None, DFE_value=None, compute_aux_losses=True, return_ngm=True, layer_index=0):
        x, attns, time, ngm = self.model(x, time, dec_time, DFE_value=DFE_value, compute_aux_losses=compute_aux_losses, return_ngm=return_ngm, layer_index=layer_index)
        # import ipdb; ipdb.set_trace()
        # x = x.flatten(1)
        x = x[:,-1,:]
        self.latent = x
        x = self.head(x)
        x = self.dropout(x)
        return x, attns, time, ngm

# class RegressionHead(nn.Module):
#     def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
#         super().__init__()
#         self.y_range = y_range
#         self.flatten = nn.Flatten(start_dim=1)
#         self.dropout = nn.Dropout(head_dropout)
#         self.linear = nn.Linear(n_vars*d_model, output_dim)

#     def forward(self, x):
#         """
#         x: [bs x nvars x d_model x num_patch]
#         output: [bs x output_dim]
#         """
#         x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
#         x = self.flatten(x)         # x: bs x nvars * d_model
#         x = self.dropout(x)
#         y = self.linear(x)         # y: bs x output_dim
#         if self.y_range: y = SigmoidRange(*self.y_range)(y)        
#         return y


# class PredictionHead(nn.Module):
#     def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
#         super().__init__()

#         self.individual = individual
#         self.n_vars = n_vars
#         self.flatten = flatten
#         head_dim = d_model*num_patch

#         if self.individual:
#             self.linears = nn.ModuleList()
#             self.dropouts = nn.ModuleList()
#             self.flattens = nn.ModuleList()
#             for i in range(self.n_vars):
#                 self.flattens.append(nn.Flatten(start_dim=-2))
#                 self.linears.append(nn.Linear(head_dim, forecast_len))
#                 self.dropouts.append(nn.Dropout(head_dropout))
#         else:
#             self.flatten = nn.Flatten(start_dim=-2)
#             self.linear = nn.Linear(head_dim, forecast_len)
#             self.dropout = nn.Dropout(head_dropout)


#     def forward(self, x):                     
#         """
#         x: [bs x nvars x d_model x num_patch]
#         output: [bs x forecast_len x nvars]
#         """
#         if self.individual:
#             x_out = []
#             for i in range(self.n_vars):
#                 z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
#                 z = self.linears[i](z)                    # z: [bs x forecast_len]
#                 z = self.dropouts[i](z)
#                 x_out.append(z)
#             x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
#         else:
#             x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
#             x = self.dropout(x)
#             x = self.linear(x)      # x: [bs x nvars x forecast_len]
#         return x.transpose(2,1)     # [bs x forecast_len x nvars]


# class PretrainHead(nn.Module):
#     def __init__(self, d_model, patch_len, dropout):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(d_model, patch_len)

#     def forward(self, x):
#         """
#         x: tensor [bs x nvars x d_model x num_patch]
#         output: tensor [bs x nvars x num_patch x patch_len]
#         """

#         x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
#         x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
#         x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
#         return x