import torch
import torch.nn as nn
import torch.nn.functional as F

from .TS_lib.Embed import DataEmbedding, DataEmbedding_wo_pos
from .TS_lib.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .TS_lib.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, seq_len, pred_len, label_len, num_feats, d_model, n_layers, n_heads, activation, moving_avg, embed, freq, d_ff, factor, dropout=0.1):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.hidden = d_model
        d_layers = 1

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(num_feats, d_model, embed, freq,
                                                  dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding_wo_pos(num_feats, d_model, embed, freq,
                                                    dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    num_feats,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, num_feats, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def forward(self, x_enc, time, dec_time, mask=None):
        # import ipdb; ipdb.set_trace()
        # tmp
        time = dec_time = None

        x_dec = torch.zeros_like(x_enc[:, -self.pred_len:, :]).float()
        x_dec = torch.cat([x_enc[:, self.seq_len-self.label_len : , :], x_dec], dim=1).float().cuda()

        dec_out = self.forecast(x_enc, time, x_dec, dec_time)
        # import ipdb; ipdb.set_trace()
        dec_out = dec_out[:, -self.pred_len:, :].squeeze(-1)

        return dec_out, time

