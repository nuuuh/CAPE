import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from .TS_lib.Embed import DataEmbedding
from .TS_lib.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, seq_len, horizon, num_feats, d_model, n_layers, top_k, num_kernels, d_ff, dropout=0.1):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = horizon
        self.hidden = d_model
        self.model = nn.ModuleList([TimesBlock(self.seq_len, self.pred_len, top_k, d_model, d_ff, num_kernels)
                                    for _ in range(n_layers)])
        self.enc_embedding = DataEmbedding(num_feats, d_model, embed_type='fixed', freq=None,
                                           dropout=dropout)
        self.layer = n_layers
        self.layer_norm = nn.LayerNorm(d_model)

        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len) 
        # self.projection = nn.Linear(d_model, c_out, bias=True)

    def forecast(self, x_enc, time):
        # tmp
        # import ipdb; ipdb.set_trace()
        time = None
        # embedding
        enc_out = self.enc_embedding(x_enc, time)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # # porject back
        # dec_out = self.projection(enc_out)
        # import ipdb; ipdb.set_trace()
        enc_out = enc_out[:, -self.pred_len : , :]
        return enc_out
    

    def forward(self, x_enc, time, dec_time, mask=None):
        enc_out = self.forecast(x_enc, time)
        return enc_out, time