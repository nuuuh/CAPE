import torch
import torch.nn as nn
import torch.nn.functional as F

from .TS_lib.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .TS_lib.SelfAttention_Family import ProbAttention, AttentionLayer
from .TS_lib.Embed import DataEmbedding


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, seq_len, pred_len, label_len, num_feats, d_model, n_layers, n_heads, activation, distil, embed, freq, d_ff, factor, dropout=0.1):
        super(Informer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        self.hidden = d_model

        d_layers=1

        # Embedding
        self.enc_embedding = DataEmbedding(num_feats, d_model, embed, freq,
                                           dropout)
        self.dec_embedding = DataEmbedding(num_feats, d_model, embed, freq,
                                           dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(n_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, num_feats, bias=True)
        )

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out  # [B, L, D]


    def forward(self, x_enc, time, dec_time, mask=None):
        # import ipdb; ipdb.set_trace()
        # tmp
        time = dec_time = None

        x_dec = torch.zeros_like(x_enc[:, -self.pred_len:, :]).float()
        x_dec = torch.cat([x_enc[:, self.seq_len-self.label_len : , :], x_dec], dim=1).float().cuda()

        dec_out = self.long_forecast(x_enc, time, x_dec, dec_time)
        # import ipdb; ipdb.set_trace()
        dec_out = dec_out[:, -self.pred_len:, :].squeeze(-1)

        return dec_out, time
        

