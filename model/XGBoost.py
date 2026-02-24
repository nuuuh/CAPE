"""
XGBoost Time Series Forecasting Wrapper

Wraps XGBoost regressor for time series forecasting with lag features.
Uses a similar interface to neural network models for compatibility.
"""

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None


class XGBoostModel:
    """
    XGBoost model for time series forecasting.
    
    Creates lag features from input sequence and uses XGBoost regressor.
    Supports multi-step forecasting via iterative or direct strategy.
    
    Parameters
    ----------
    num_timesteps_input : int
        Number of input timesteps (lookback window)
    num_timesteps_output : int
        Number of output timesteps (forecast horizon)
    num_features : int
        Number of features per timestep
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Boosting learning rate
    strategy : str
        'direct' (one model per horizon) or 'recursive' (iterative)
    """
    def __init__(self, num_timesteps_input, num_timesteps_output, num_features=1,
                 n_estimators=100, max_depth=6, learning_rate=0.1, 
                 strategy='direct', random_state=42):
        
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.num_features = num_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.strategy = strategy
        self.random_state = random_state
        
        self.models = []
        self.is_fitted = False
    
    def _create_base_model(self):
        """Create a base XGBoost regressor."""
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
    
    def _prepare_features(self, X):
        """
        Convert input tensor to feature matrix.
        
        Args:
            X: (batch, timesteps, features) or (batch, timesteps)
        
        Returns:
            features: (batch, timesteps * features)
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim == 3:
            batch_size = X.shape[0]
            X = X.reshape(batch_size, -1)
        
        return X
    
    def fit(self, X, y):
        """
        Fit the XGBoost model(s).
        
        Args:
            X: (batch, timesteps, features) input sequences
            y: (batch, horizon) or (batch, horizon, 1) target values
        """
        X_features = self._prepare_features(X)
        
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        if y.ndim == 3:
            y = y.squeeze(-1)
        
        self.models = []
        
        if self.strategy == 'direct':
            # Train one model per output timestep
            for h in range(self.num_timesteps_output):
                model = self._create_base_model()
                model.fit(X_features, y[:, h] if y.ndim > 1 else y)
                self.models.append(model)
        else:
            # Single model for recursive prediction
            model = self._create_base_model()
            # For recursive, train to predict next step only
            if y.ndim > 1:
                y_next = y[:, 0]
            else:
                y_next = y
            model.fit(X_features, y_next)
            self.models.append(model)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: (batch, timesteps, features) input sequences
        
        Returns:
            predictions: (batch, horizon) predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_features = self._prepare_features(X)
        batch_size = X_features.shape[0]
        
        predictions = np.zeros((batch_size, self.num_timesteps_output))
        
        if self.strategy == 'direct':
            for h in range(self.num_timesteps_output):
                predictions[:, h] = self.models[h].predict(X_features)
        else:
            # Recursive prediction
            current_input = X_features.copy()
            for h in range(self.num_timesteps_output):
                pred = self.models[0].predict(current_input)
                predictions[:, h] = pred
                # Update input by shifting and adding prediction
                if h < self.num_timesteps_output - 1:
                    current_input = np.roll(current_input, -self.num_features, axis=1)
                    current_input[:, -self.num_features:] = pred.reshape(-1, 1)
        
        return predictions
    
    def __call__(self, X, time=None, dec_time=None, mask=None):
        """Forward pass compatible with neural network interface."""
        predictions = self.predict(X)
        return torch.tensor(predictions, dtype=torch.float32), time
    
    def forward(self, X, time=None, dec_time=None, mask=None):
        """Forward pass compatible with neural network interface."""
        return self(X, time, dec_time, mask)
    
    def to(self, device):
        """Dummy method for compatibility with PyTorch models."""
        return self
    
    def eval(self):
        """Dummy method for compatibility with PyTorch models."""
        return self
    
    def train(self, mode=True):
        """Dummy method for compatibility with PyTorch models."""
        return self
    
    def parameters(self):
        """Dummy method - returns empty iterator for compatibility."""
        return iter([])
    
    def state_dict(self):
        """Return model state for saving."""
        import pickle
        return {'models': pickle.dumps(self.models), 'is_fitted': self.is_fitted}
    
    def load_state_dict(self, state_dict):
        """Load model state."""
        import pickle
        self.models = pickle.loads(state_dict['models'])
        self.is_fitted = state_dict['is_fitted']
    
    def initialize(self):
        """Reset the model."""
        self.models = []
        self.is_fitted = False


class XGBoostWrapper(torch.nn.Module):
    """
    PyTorch wrapper for XGBoost to enable training in standard pipeline.
    Note: This is a special case - the actual training uses XGBoost's fit().
    """
    def __init__(self, num_timesteps_input, num_timesteps_output, num_features=1,
                 n_estimators=100, max_depth=6, learning_rate=0.1):
        super(XGBoostWrapper, self).__init__()
        self.xgb_model = XGBoostModel(
            num_timesteps_input, num_timesteps_output, num_features,
            n_estimators, max_depth, learning_rate
        )
        # Dummy parameter for optimizer compatibility
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self._train_data = {'X': [], 'y': []}
    
    def forward(self, x, time=None, dec_time=None, mask=None):
        if self.training:
            # During training, just return zeros - actual training is in fit()
            batch_size = x.shape[0]
            return torch.zeros(batch_size, self.xgb_model.num_timesteps_output), time
        else:
            return self.xgb_model(x, time, dec_time, mask)
    
    def fit_xgb(self, X, y):
        """Fit the underlying XGBoost model."""
        self.xgb_model.fit(X, y)
    
    def initialize(self):
        self.xgb_model.initialize()


# Alias for compatibility
XGBoost = XGBoostModel
