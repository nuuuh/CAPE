import torch.nn as nn
import torch
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, mode='SUM'):
        "Apply residual connection to any sublayer with the same size."
        if mode=='SUM':
            return x + self.dropout(sublayer(self.norm(x)))
        elif mode == 'EXPEM_SUM':
            # import ipdb; ipdb.set_trace()
            if type(x) is list:
                return x[0] + self.dropout(self.norm(sublayer(x)))
            else:
                return x + self.dropout(self.norm(sublayer(x)))
        elif mode == 'EXPEM_CAT':
            # import ipdb; ipdb.set_trace()
            if type(x) is list:
                return torch.cat([x[0], self.dropout(self.norm(sublayer(x)))], dim=1)
            else:
                return torch.cat([x, self.dropout(self.norm(sublayer(x)))], dim=1)
