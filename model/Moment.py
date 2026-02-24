import torch
from torch import nn
from .momentfm import MOMENTPipeline

# Try to import peft for fine-tuning support (optional)
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class Moment(nn.Module):
    """
    MOMENT foundation model wrapper for time series forecasting.
    
    Supports two modes:
    - 'zero_shot' or 'linear_probing': Frozen encoder, zero-shot inference (no training)
    - 'full_finetuning': All weights trainable with LoRA
    """
    def __init__(self, horizon, moment_mode='zero_shot'):
        super().__init__()
        self.pred_len = horizon
        self.moment_mode = moment_mode
        
        # Determine if this is zero-shot mode
        is_zero_shot = moment_mode in ['zero_shot', 'linear_probing']
        
        self.model = MOMENTPipeline.from_pretrained(
                                                    "AutonLab/MOMENT-1-large", 
                                                    model_kwargs={
                                                        'task_name': 'forecasting',
                                                        'forecast_horizon': horizon,
                                                        # Freeze encoder and embedder for zero-shot
                                                        'freeze_encoder': True if is_zero_shot else False,
                                                        'freeze_embedder': True if is_zero_shot else False,
                                                        # Disable gradient checkpointing for zero-shot
                                                        'enable_gradient_checkpointing': False, 
                                                    },
                                                )
        self.model.init()
        
        # Only apply LoRA for fine-tuning mode (not zero-shot)
        if moment_mode == 'full_finetuning' and PEFT_AVAILABLE:
            lora_config = LoraConfig(
                                    r=64,
                                    lora_alpha=32,
                                    target_modules=["q", "v"],
                                    lora_dropout=0.05,
                                    )
            self.model = get_peft_model(self.model, lora_config)
            print('LoRA enabled for fine-tuning')
            self.model.print_trainable_parameters()
        else:
            print(f'MOMENT loaded in zero-shot mode (no training)')

        self.hidden = 1024

    def forward(self, x_enc, time=None, dec_time=None, mask=None):
        """
        Forward pass for MOMENT model.
        
        Args:
            x_enc: Input tensor [batch, seq_len, features] or [batch, features, seq_len]
            time: Optional time tensor (for compatibility, not used)
            dec_time: Optional decoder time (for compatibility, not used)
            mask: Optional mask (for compatibility, not used)
            
        Returns:
            predictions: [batch, horizon]
        """
        device = x_enc.device
        
        # Handle input shape - expect [batch, seq_len, features]
        if x_enc.dim() == 2:
            x_enc = x_enc.unsqueeze(-1)  # [batch, seq_len, 1]
        
        x_enc = x_enc.transpose(1, 2)  # [batch, features, seq_len]
        
        # Pad to MOMENT's expected size (512)
        inputs = torch.zeros(x_enc.shape[0], x_enc.shape[1], 512, device=device)
        inputs[:, :, :x_enc.shape[-1]] = x_enc

        input_mask = torch.zeros(x_enc.shape[0], 512, dtype=torch.bool, device=device)
        input_mask[:, :x_enc.shape[-1]] = True
        
        enc_out = self.model(x_enc=inputs, input_mask=input_mask)

        return enc_out.forecast[:, -1, :]  # [batch, horizon]