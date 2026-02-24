import torch
from torch import nn
import numpy as np

# =============================================================================
# CHRONOS (original Bolt models)
# =============================================================================
try:
    from chronos import ChronosBoltPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    print("Warning: chronos-forecasting not installed. Run: pip install chronos-forecasting")

# =============================================================================
# CHRONOS 2.0 (newer T5-based models)
# =============================================================================
try:
    from chronos import BaseChronosPipeline
    # Chronos2 uses BaseChronosPipeline for T5-based models
    CHRONOS2_AVAILABLE = True
except ImportError:
    CHRONOS2_AVAILABLE = False
    # Silent - will use same pipeline as Chronos if available

# =============================================================================
# MOIRAI 2.0 (Salesforce foundation model)
# =============================================================================
try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    MOIRAI_AVAILABLE = True
except ImportError:
    MOIRAI_AVAILABLE = False
    # print("Warning: uni2ts not installed. Run: pip install uni2ts")

# =============================================================================
# LAG-LLAMA (Foundation model with lag features)
# =============================================================================
try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
    LAGLLAMA_AVAILABLE = True
except ImportError:
    LAGLLAMA_AVAILABLE = False
    # print("Warning: lag-llama not installed. Run: pip install lag-llama")

# =============================================================================
# TIMESFM (Google's TimesFM foundation model)
# =============================================================================
try:
    import timesfm
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False
    # print("Warning: timesfm not installed. Run: pip install timesfm")


class Chronos(nn.Module):
    """
    Chronos wrapper for time series forecasting.
    
    Uses Chronos-Bolt models which are faster and more efficient than the original Chronos.
    Available models:
    - amazon/chronos-bolt-tiny (9M params)
    - amazon/chronos-bolt-mini (21M params)
    - amazon/chronos-bolt-small (48M params)
    - amazon/chronos-bolt-base (205M params)
    """
    
    def __init__(self, horizon, model_name="amazon/chronos-bolt-small", device="cuda"):
        super().__init__()
        self.pred_len = horizon
        self.model_name = model_name
        self._device = device
        
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Please install it with: pip install chronos-forecasting"
            )
        
        # Load the Chronos-Bolt pipeline
        # Using Chronos-Bolt for faster inference (up to 250x faster than original)
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        
        # Hidden size for compatibility (not directly used, but needed for some interfaces)
        self.hidden = 256
        
        print(f"Loaded Chronos model: {model_name}")
    
    def forward(self, x_enc, time=None, dec_time=None, mask=None):
        """
        Forward pass for Chronos model.
        
        Args:
            x_enc: Input tensor [batch, seq_len, features] or [batch, seq_len]
            time: Optional time tensor (for compatibility, not used by Chronos)
            dec_time: Optional decoder time (for compatibility, not used)
            mask: Optional mask (for compatibility, not used)
            
        Returns:
            predictions: [batch, horizon]
        """
        # Handle input shape - expect [batch, seq_len, features] or [batch, seq_len]
        if x_enc.dim() == 3:
            # Take first feature channel if multiple features
            x_enc = x_enc[:, :, 0]  # [batch, seq_len]
        
        batch_size, seq_len = x_enc.shape
        device = x_enc.device
        
        # Chronos-Bolt expects input tensor of shape [batch, seq_len]
        # and returns [batch, num_quantiles, prediction_length]
        with torch.no_grad():
            # Move input to CPU for Chronos pipeline
            forecast = self.pipeline.predict(
                inputs=x_enc.cpu(),  # [batch, seq_len]
                prediction_length=self.pred_len,
            )
            # forecast shape: [batch, num_quantiles, prediction_length]
            # num_quantiles is 9 for official Chronos-Bolt (0.1, 0.2, ..., 0.9)
            
            # Quantile indices: 0=0.1, 1=0.2, 2=0.3, 3=0.4, 4=0.5, 5=0.6, 6=0.7, 7=0.8, 8=0.9
            predictions = forecast.mean(dim=1)  # [batch, prediction_length] - mean across quantiles
        
        return predictions.to(device)
    
    def predict_quantiles(self, x_enc, quantile_levels=[0.1, 0.5, 0.9]):
        """
        Generate probabilistic forecasts with quantiles.
        
        Args:
            x_enc: Input tensor [batch, seq_len, features] or [batch, seq_len]
            quantile_levels: List of quantile levels to predict
            
        Returns:
            quantiles: Dict mapping quantile level to [batch, horizon] predictions
        """
        if x_enc.dim() == 3:
            x_enc = x_enc[:, :, 0]
        
        device = x_enc.device
        
        # Chronos-Bolt returns [batch, num_quantiles, prediction_length]
        # Quantiles are [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=x_enc.cpu(),
                prediction_length=self.pred_len,
            )
            # forecast: [batch, 9, pred_len]
        
        # Map requested quantile levels to Chronos quantile indices
        quantile_to_idx = {0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 
                          0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8}
        
        quantiles = {}
        for q in quantile_levels:
            if q in quantile_to_idx:
                quantiles[q] = forecast[:, quantile_to_idx[q], :].to(device)
            else:
                # Interpolate for non-standard quantiles
                quantiles[q] = forecast[:, 4, :].to(device)  # fallback to median
        
        return quantiles
    
    def train(self, mode=True):
        """Override train mode - Chronos is always in eval mode (frozen)."""
        # Chronos models are pretrained and used for zero-shot inference
        # We keep the model in eval mode
        self.training = mode
        return self
    
    def parameters(self, recurse=True):
        """Return empty iterator - Chronos has no trainable parameters."""
        return iter([])
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        # Pipeline handles its own device placement
        return self


# =============================================================================
# CHRONOS 2.0 (T5-based models - newer architecture)
# =============================================================================

class Chronos2(nn.Module):
    """
    Chronos 2.0 wrapper for time series forecasting.
    
    Chronos2 uses T5-based architecture with improved zero-shot capabilities.
    Available models:
    - amazon/chronos-t5-tiny (8M params)
    - amazon/chronos-t5-mini (20M params)
    - amazon/chronos-t5-small (46M params)
    - amazon/chronos-t5-base (200M params)
    - amazon/chronos-t5-large (710M params)
    """
    
    def __init__(self, horizon, model_name="amazon/chronos-t5-small", device="cuda"):
        super().__init__()
        self.pred_len = horizon
        self.model_name = model_name
        self._device = device
        
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Please install it with: pip install chronos-forecasting"
            )
        
        # Load the Chronos T5 pipeline using BaseChronosPipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        
        self.hidden = 256
        print(f"Loaded Chronos2 model: {model_name}")
    
    def forward(self, x_enc, time=None, dec_time=None, mask=None):
        """
        Forward pass for Chronos2 model.
        
        Args:
            x_enc: Input tensor [batch, seq_len, features] or [batch, seq_len]
            
        Returns:
            predictions: [batch, horizon]
        """
        if x_enc.dim() == 3:
            x_enc = x_enc[:, :, 0]
        
        device = x_enc.device
        
        with torch.no_grad():
            # Chronos T5 returns samples: [batch, num_samples, prediction_length]
            forecast = self.pipeline.predict(
                inputs=x_enc.cpu(),
                prediction_length=self.pred_len,
                num_samples=20,  # Generate 20 samples for probabilistic forecast
            )
            # Take mean across samples: [batch, prediction_length]
            predictions = forecast.mean(dim=1)
        
        return predictions.to(device)
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def parameters(self, recurse=True):
        return iter([])
    
    def to(self, device):
        self._device = device
        return self


# =============================================================================
# MOIRAI 2.0 (Salesforce foundation model)
# =============================================================================

class Moirai(nn.Module):
    """
    Moirai 2.0 wrapper for time series forecasting.
    
    Moirai is Salesforce's universal time series forecasting model.
    Available models:
    - Salesforce/moirai-1.0-R-small (14M params)
    - Salesforce/moirai-1.0-R-base (91M params)
    - Salesforce/moirai-1.0-R-large (311M params)
    - Salesforce/moirai-1.1-R-small (newer version)
    - Salesforce/moirai-1.1-R-base
    - Salesforce/moirai-1.1-R-large
    """
    
    def __init__(self, horizon, model_name="Salesforce/moirai-1.1-R-small", device="cuda", context_length=None):
        super().__init__()
        self.pred_len = horizon
        self.model_name = model_name
        self._device = device
        self._context_length = context_length  # Will be set dynamically if None
        
        if not MOIRAI_AVAILABLE:
            raise ImportError(
                "uni2ts is not installed. "
                "Please install it with: pip install uni2ts"
            )
        
        # Load Moirai module only - we create MoiraiForecast dynamically
        self.module = MoiraiModule.from_pretrained(model_name)
        self.module = self.module.to(device)
        self._model_cache = {}  # Cache for different context lengths
        
        self.hidden = 256
        print(f"Loaded Moirai model: {model_name}")
    
    def _get_model_for_context(self, context_length):
        """Get or create MoiraiForecast model for given context length."""
        if context_length not in self._model_cache:
            # Moirai requires patch_size to divide context_length evenly
            # Use patch_size=16 (default) or smaller if context is short
            # Find largest valid patch_size <= 32 that divides context_length
            valid_patch_sizes = [32, 16, 8, 4, 2, 1]
            patch_size = 1
            for ps in valid_patch_sizes:
                if context_length >= ps and context_length % ps == 0:
                    patch_size = ps
                    break
            
            model = MoiraiForecast(
                module=self.module,
                prediction_length=self.pred_len,
                context_length=context_length,
                patch_size=patch_size,
                num_samples=20,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            model = model.to(self._device)
            model.eval()
            self._model_cache[context_length] = model
        return self._model_cache[context_length]
    
    def forward(self, x_enc, time=None, dec_time=None, mask=None):
        """
        Forward pass for Moirai model.
        
        Args:
            x_enc: Input tensor [batch, seq_len, features] or [batch, seq_len]
            
        Returns:
            predictions: [batch, horizon]
        """
        if x_enc.dim() == 3:
            x_enc = x_enc[:, :, 0]
        
        batch_size, seq_len = x_enc.shape
        device = x_enc.device
        
        # Get model with correct context length
        model = self._get_model_for_context(seq_len)
        
        with torch.no_grad():
            # Moirai expects specific input format
            # past_target: [batch, seq_len, target_dim]
            # past_observed_target: [batch, seq_len, target_dim] - MUST match past_target shape
            # past_is_pad: [batch, seq_len]
            past_target = x_enc.unsqueeze(-1)  # [batch, seq_len, 1]
            past_observed_target = torch.ones_like(past_target, dtype=torch.bool)  # [batch, seq_len, 1]
            past_is_pad = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            
            # Run prediction
            forecast = model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
            )
            # forecast: [batch, num_samples, prediction_length]
            predictions = forecast.mean(dim=1)  # [batch, prediction_length]
        
        return predictions
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def parameters(self, recurse=True):
        return iter([])
    
    def to(self, device):
        self._device = device
        if hasattr(self, 'model'):
            self.model = self.model.to(device)
        return self


# =============================================================================
# LAG-LLAMA (Lag-based LLM for time series)
# =============================================================================

class LagLlama(nn.Module):
    """
    Lag-Llama wrapper for time series forecasting.
    
    Lag-Llama is a foundation model for probabilistic time series forecasting
    based on LLaMA architecture with lag features.
    
    NOTE: LagLlama requires minimum context length of 1093 due to its lag features.
    For epidemic data with shorter sequences, this model may not be suitable.
    
    Available models:
    - lag-llama (default, ~7M params)
    """
    
    MIN_CONTEXT_LENGTH = 1093  # Max lag (1092) + 1
    
    def __init__(self, horizon, model_name="time-series-foundation-models/Lag-Llama", device="cuda"):
        super().__init__()
        self.pred_len = horizon
        self.model_name = model_name
        self._device = device
        
        if not LAGLLAMA_AVAILABLE:
            raise ImportError(
                "lag-llama is not installed. "
                "Please install it with: pip install lag-llama"
            )
        
        # Load Lag-Llama using HuggingFace
        from huggingface_hub import hf_hub_download
        import torch
        from lag_llama.gluon.lightning_module import LagLlamaLightningModule
        
        # Download checkpoint
        ckpt_path = hf_hub_download(repo_id=model_name, filename="lag-llama.ckpt")
        
        # Load checkpoint to get hyperparameters
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        hyper_params = ckpt.get('hyper_parameters', {})
        self.lags_seq = hyper_params.get('model_kwargs', {}).get('lags_seq', [])
        self.max_lag = max(self.lags_seq) if self.lags_seq else 0
        
        # Load the lightning module directly (it reads hparams from checkpoint)
        self.model = LagLlamaLightningModule.load_from_checkpoint(
            ckpt_path, 
            prediction_length=horizon,
            map_location=torch.device(device)
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        self.hidden = 256
        print(f"Loaded LagLlama model: {model_name} (requires min context: {self.max_lag + 1})")
    
    def forward(self, x_enc, time=None, dec_time=None, mask=None):
        """
        Forward pass for LagLlama model.
        
        Args:
            x_enc: Input tensor [batch, seq_len, features] or [batch, seq_len]
            
        Returns:
            predictions: [batch, horizon]
        """
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        
        if x_enc.dim() == 3:
            x_enc = x_enc[:, :, 0]
        
        batch_size, seq_len = x_enc.shape
        device = x_enc.device
        
        # LagLlama needs context >= max_lag + 1
        min_len = self.max_lag + 1
        if seq_len < min_len:
            # Pad with repetition of first value
            pad_len = min_len - seq_len
            padding = x_enc[:, :1].expand(batch_size, pad_len)
            x_enc_padded = torch.cat([padding, x_enc], dim=1)
        else:
            x_enc_padded = x_enc
        
        # Use the predictor interface via GluonTS
        all_preds = []
        for i in range(batch_size):
            data = x_enc_padded[i].cpu().numpy()
            
            # Create a simple ListDataset
            ds = ListDataset(
                [{"start": pd.Timestamp("2020-01-01"), "target": data}],
                freq="D"
            )
            
            # Create transformation and predictor
            transformation = self.model.create_transformation() if hasattr(self.model, 'create_transformation') else None
            
            # Use model's forward with proper inputs
            with torch.no_grad():
                past_target = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
                past_observed = torch.ones_like(past_target)
                
                # Create time features (zeros for simplicity)
                past_time_feat = torch.zeros(1, len(data), 1, device=device)
                future_time_feat = torch.zeros(1, self.pred_len, 1, device=device)
                
                # Try direct forward with the internal model
                try:
                    output = self.model.model(
                        past_target=past_target,
                        past_observed_values=past_observed,
                        past_time_feat=past_time_feat,
                        future_time_feat=future_time_feat
                    )
                    # output is distribution params [batch, seq_len + pred_len, n_params]
                    # Take the last pred_len predictions, first param (loc)
                    preds = output[:, -self.pred_len:, 0]
                    all_preds.append(preds)
                except Exception as e:
                    # Fallback: return zeros
                    print(f"LagLlama forward failed: {e}")
                    all_preds.append(torch.zeros(1, self.pred_len, device=device))
        
        predictions = torch.cat(all_preds, dim=0)
        return predictions.to(device)
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def parameters(self, recurse=True):
        return iter([])
    
    def to(self, device):
        self._device = device
        return self


# =============================================================================
# TIMESFM (Google's Time Series Foundation Model)
# =============================================================================

class TimesFM(nn.Module):
    """
    TimesFM wrapper for time series forecasting.
    
    TimesFM is Google's patched-decoder style foundation model for time series.
    
    Available models:
    - google/timesfm-1.0-200m-pytorch (200M params, default)
    """
    
    def __init__(self, horizon, model_name="google/timesfm-1.0-200m-pytorch", device="cuda"):
        super().__init__()
        self.pred_len = horizon
        self.model_name = model_name
        self._device = device
        
        if not TIMESFM_AVAILABLE:
            raise ImportError(
                "timesfm is not installed. "
                "Please install it with: pip install timesfm"
            )
        
        import timesfm
        
        # Initialize TimesFM with new API (using hparams and checkpoint classes)
        hparams = timesfm.TimesFmHparams(
            context_len=512,  # Max context length
            horizon_len=horizon,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="gpu" if "cuda" in device else "cpu",
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            version="torch",  # Use PyTorch version
            huggingface_repo_id=model_name,
        )
        
        self.tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
        
        self.hidden = 256
        print(f"Loaded TimesFM model: {model_name}")
    
    def forward(self, x_enc, time=None, dec_time=None, mask=None):
        """
        Forward pass for TimesFM model.
        
        Args:
            x_enc: Input tensor [batch, seq_len, features] or [batch, seq_len]
            
        Returns:
            predictions: [batch, horizon]
        """
        if x_enc.dim() == 3:
            x_enc = x_enc[:, :, 0]
        
        batch_size, seq_len = x_enc.shape
        device = x_enc.device
        
        # TimesFM expects numpy arrays
        inputs = x_enc.cpu().numpy()
        
        # Predict
        with torch.no_grad():
            # TimesFM batch predict
            point_forecast, _ = self.tfm.forecast(
                inputs,
                freq=[0] * batch_size,  # 0 = unknown frequency
            )
        
        predictions = torch.tensor(point_forecast[:, :self.pred_len], dtype=torch.float32, device=device)
        return predictions
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def parameters(self, recurse=True):
        return iter([])
    
    def to(self, device):
        self._device = device
        return self


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_chronos_online(test_tokens, num_input: int, num_output: int,
                            token_size: int, device: str = "cuda"):
    """
    Evaluate Chronos baseline for online forecasting.
    
    Args:
        test_tokens: Test tokens array (num_tokens, token_size)
        num_input: Number of input tokens
        num_output: Number of output tokens
        token_size: Token dimension size
        device: Device to run on
        
    Returns:
        mse: Mean squared error (or None if failed)
        mae: Mean absolute error (or None if failed)
        targets: Target values (or None if failed)
    """
    from tqdm import tqdm
    
    if not CHRONOS_AVAILABLE:
        print("  Chronos not available, skipping...")
        return None, None, None
    
    chronos = Chronos(horizon=num_output * token_size,
                      model_name="amazon/chronos-bolt-small", device=device)
    
    total = num_input + num_output
    samples = [(test_tokens[i:i+num_input].flatten(),
                test_tokens[i+num_input:i+total].flatten())
               for i in range(len(test_tokens) - total + 1)]
    
    if len(samples) == 0:
        return None, None, None
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inp, out in tqdm(samples, desc="CHRONOS", leave=False):
            x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
            pred = chronos(x).cpu().numpy()[0].reshape(num_output, token_size)
            all_preds.append(pred)
            all_targets.append(out.reshape(num_output, token_size))
    
    # Use LAST token for evaluation
    preds = np.array(all_preds)[:, -1, :]
    targets = np.array(all_targets)[:, -1, :]
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    return mse, mae, targets


def evaluate_chronos2_online(test_tokens, num_input: int, num_output: int,
                             token_size: int, device: str = "cuda",
                             model_name: str = "amazon/chronos-t5-small"):
    """
    Evaluate Chronos2 (T5-based) baseline for online forecasting.
    """
    from tqdm import tqdm
    
    if not CHRONOS_AVAILABLE:
        print("  Chronos2 not available, skipping...")
        return None, None, None
    
    try:
        chronos2 = Chronos2(horizon=num_output * token_size,
                           model_name=model_name, device=device)
    except Exception as e:
        print(f"  Chronos2 failed to load: {e}")
        return None, None, None
    
    total = num_input + num_output
    samples = [(test_tokens[i:i+num_input].flatten(),
                test_tokens[i+num_input:i+total].flatten())
               for i in range(len(test_tokens) - total + 1)]
    
    if len(samples) == 0:
        return None, None, None
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inp, out in tqdm(samples, desc="CHRONOS2", leave=False):
            x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
            pred = chronos2(x).cpu().numpy()[0].reshape(num_output, token_size)
            all_preds.append(pred)
            all_targets.append(out.reshape(num_output, token_size))
    
    preds = np.array(all_preds)[:, -1, :]
    targets = np.array(all_targets)[:, -1, :]
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    return mse, mae, targets


def evaluate_moirai_online(test_tokens, num_input: int, num_output: int,
                           token_size: int, device: str = "cuda",
                           model_name: str = "Salesforce/moirai-1.1-R-small"):
    """
    Evaluate Moirai baseline for online forecasting.
    """
    from tqdm import tqdm
    
    if not MOIRAI_AVAILABLE:
        print("  Moirai not available (install uni2ts), skipping...")
        return None, None, None
    
    try:
        moirai = Moirai(horizon=num_output * token_size,
                        model_name=model_name, device=device)
    except Exception as e:
        print(f"  Moirai failed to load: {e}")
        return None, None, None
    
    total = num_input + num_output
    samples = [(test_tokens[i:i+num_input].flatten(),
                test_tokens[i+num_input:i+total].flatten())
               for i in range(len(test_tokens) - total + 1)]
    
    if len(samples) == 0:
        return None, None, None
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inp, out in tqdm(samples, desc="MOIRAI", leave=False):
            x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
            pred = moirai(x).cpu().numpy()[0].reshape(num_output, token_size)
            all_preds.append(pred)
            all_targets.append(out.reshape(num_output, token_size))
    
    preds = np.array(all_preds)[:, -1, :]
    targets = np.array(all_targets)[:, -1, :]
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    return mse, mae, targets


class ChronosFinetune(nn.Module):
    """
    Chronos with a trainable linear head for fine-tuning.
    
    Uses Chronos-Bolt as a frozen feature extractor and adds a trainable
    linear layer on top for domain-specific adaptation.
    """
    
    def __init__(self, horizon, model_name="amazon/chronos-bolt-small", device="cuda"):
        super().__init__()
        self.pred_len = horizon
        self.model_name = model_name
        self._device = device
        
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Please install it with: pip install chronos-forecasting"
            )
        
        # Load the frozen Chronos model
        self.chronos = Chronos(horizon, model_name, device)
        
        # Add trainable linear head
        # Maps from Chronos predictions to refined predictions
        self.linear_head = nn.Linear(horizon, horizon)
        
        # Initialize to identity mapping
        nn.init.eye_(self.linear_head.weight)
        nn.init.zeros_(self.linear_head.bias)
        
        self.hidden = 256
    
    def forward(self, x_enc, time=None, dec_time=None, mask=None):
        """Forward pass with trainable head."""
        # Get Chronos predictions (frozen)
        with torch.no_grad():
            chronos_pred = self.chronos(x_enc, time, dec_time, mask)
        
        # Apply trainable linear head
        output = self.linear_head(chronos_pred)
        
        return output
    
    def parameters(self, recurse=True):
        """Return only linear head parameters."""
        return self.linear_head.parameters()
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        self.chronos = self.chronos.to(device)
        self.linear_head = self.linear_head.to(device)
        return self
