import torch
import torch.nn as nn
import torch.nn.functional as F

from .TS_lib.Embed import DataEmbedding
from .TS_lib.AutoCorrelation import AutoCorrelationLayer
from .TS_lib.FourierCorrelation import FourierBlock, FourierCrossAttention
from .TS_lib.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .TS_lib.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Fedformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, seq_len, pred_len, label_len, num_feats, d_model, n_layers, n_heads, activation, moving_avg, embed, freq, d_ff, dropout=0.1, version='fourier', mode_select='random', modes=32):
        """  
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Fedformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.hidden = d_model

        d_layers = 1

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(moving_avg)
        self.enc_embedding = DataEmbedding(num_feats, d_model, embed, freq,
                                           dropout)
        self.dec_embedding = DataEmbedding(num_feats, d_model, embed, freq,
                                           dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=d_model,
                                                  out_channels=d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=d_model,
                                                      out_channels=d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select,
                                                      num_heads=n_heads)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
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
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        # import ipdb; ipdb.set_trace()
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # import ipdb; ipdb.set_trace()
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def forward(self, x_enc, time, dec_time, mask=None):
        # tmp
        time = dec_time = None

        x_dec = torch.zeros_like(x_enc[:, -self.pred_len:, :]).float()
        x_dec = torch.cat([x_enc[:, self.seq_len-self.label_len : , :], x_dec], dim=1).float().cuda()

        dec_out = self.forecast(x_enc, time, x_dec, dec_time)

        dec_out = dec_out[:, -self.pred_len:, :].squeeze(-1)
        return dec_out, time

