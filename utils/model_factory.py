import torch
from model import (
    representation_head, forecast_head,
    PatchTST, DlinearModel, GRUModel, LSTMModel,
    PEM, EXPEM, TimesNet, TimeMixer,
    Informer, Fedformer, Autoformer, Moment,
    EINN, EpiDeep, CAPE, create_cape_model,
    NBeatsModel, CNNModel,
)


def get_model(config, mode='test'):
    try:
        num_patches = int((config.lookback - config.patch_len) / config.stride + 1)
    except:
        print('NaN patches!')

    if 'NBeats' in config.model or 'NBEATS' in config.model:
        backbone = NBeatsModel(config.lookback, config.horizon, config.num_features,
                               config.hidden_size, num_layers=4, num_stacks=2,
                               num_blocks=3, dropout=config.dropout)
    elif 'CNN' in config.model or 'TCN' in config.model:
        backbone = CNNModel(config.lookback, config.horizon, config.num_features,
                            config.hidden_size, num_layers=getattr(config, 'layers', 4),
                            kernel_size=3, dropout=config.dropout)
    elif 'PatchTST' in config.model:
        backbone = PatchTST(patch_len=config.patch_len,
                            horizon=config.horizon,
                            num_patches=num_patches,
                            d_model=config.hidden_size,
                            n_layers=config.layers,
                            n_heads=config.attn_heads,
                            dropout=config.dropout)
    elif 'EXPEM' in config.model:
        backbone = EXPEM(patch_len=config.patch_len,
                         horizon=config.horizon,
                         num_patches=num_patches,
                         d_model=config.hidden_size,
                         n_layers=config.layers,
                         n_heads=config.attn_heads,
                         dropout=config.dropout,
                         envs=config.num_envs)
    elif 'PEM' in config.model:
        backbone = PEM(patch_len=config.patch_len,
                       horizon=config.horizon,
                       num_patches=num_patches,
                       d_model=config.hidden_size,
                       n_layers=config.layers,
                       n_heads=config.attn_heads,
                       dropout=config.dropout)
    elif 'Dlinear' in config.model:
        backbone = DlinearModel(config.lookback, config.horizon, config.num_features)
    elif 'GRU' in config.model:
        backbone = GRUModel(config.lookback, config.horizon, config.num_features,
                            config.hidden_size, config.dropout, False)
    elif 'LSTM' in config.model:
        backbone = LSTMModel(config.lookback, config.horizon, config.num_features,
                             config.hidden_size, config.dropout, False)
    elif 'TimesNet' in config.model:
        config.layers = 2
        config.hidden_size = 768
        backbone = TimesNet(config.lookback, config.horizon, config.num_features,
                            config.hidden_size, config.layers, top_k=5, num_kernels=6,
                            d_ff=config.hidden_size, dropout=config.dropout)
    elif 'TimeMixer' in config.model:
        backbone = TimeMixer(config.lookback, config.horizon, config.num_features,
                             config.hidden_size, config.layers, use_norm=True,
                             embed='fixed', freq=None, decomp_method='moving_avg',
                             down_sampling_method='avg', down_sampling_window=2,
                             down_sampling_layers=3, channel_independence=1,
                             moving_avg=5, top_k=5, d_ff=2*config.hidden_size,
                             dropout=config.dropout)
    elif 'Informer' in config.model:
        backbone = Informer(config.lookback, config.horizon, config.label_len,
                            config.num_features, config.hidden_size, config.layers,
                            config.attn_heads, activation='gelu', distil=False,
                            embed='fixed', freq=None, d_ff=2*config.hidden_size,
                            factor=3, dropout=config.dropout)
    elif 'Fedformer' in config.model:
        backbone = Fedformer(config.lookback, config.horizon, config.label_len,
                             config.num_features, config.hidden_size, config.layers,
                             config.attn_heads, activation='gelu', moving_avg=5,
                             embed='fixed', freq=None, d_ff=2*config.hidden_size,
                             dropout=config.dropout)
    elif 'Autoformer' in config.model:
        backbone = Autoformer(config.lookback, config.horizon, config.label_len,
                              config.num_features, config.hidden_size, config.layers,
                              config.attn_heads, activation='gelu', moving_avg=5,
                              embed='fixed', freq=None, d_ff=2*config.hidden_size,
                              factor=3, dropout=config.dropout)
    elif 'Moment' in config.model or 'MOMENT' in config.model:
        backbone = Moment(config.horizon, config.moment_mode)
    elif 'EINN' in config.model:
        backbone = EINN(config.lookback, config.horizon, config.num_features,
                        config.hidden_size, config.dropout)
    elif 'EpiDeep' in config.model:
        backbone = EpiDeep(config.lookback, config.horizon, config.num_features,
                           config.hidden_size, config.dropout)
    elif 'CAPE' in config.model:
        backbone = create_cape_model(config)
    else:
        raise ValueError(f"Model '{config.model}' not implemented")

    if hasattr(backbone, 'EXO'):
        if config.num_envs != 0 and mode == 'test':
            pass

    return backbone