"""
Statistical Models for Evaluation
==================================

Contains baseline statistical models:
- ARIMAModel: ARIMA with bootstrap uncertainty
- SARIMAModel: Seasonal ARIMA
- SIRModel: SIR compartmental model with parameter uncertainty
- NaiveModel: Naive baseline (repeat last value)
"""

import numpy as np
import warnings
from typing import Tuple, Optional


class ARIMAModel:
    """
    ARIMA model wrapper with uncertainty estimation via bootstrap.
    
    Uses statsmodels ARIMA implementation with residual bootstrap
    for uncertainty quantification.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (2, 0, 2), 
                 num_bootstrap: int = 20):
        """
        Args:
            order: ARIMA order (p, d, q)
            num_bootstrap: Number of bootstrap samples for uncertainty
        """
        self.order = order
        self.num_bootstrap = num_bootstrap
        self.fitted_model = None
    
    def fit(self, data: np.ndarray) -> bool:
        """Fit ARIMA model to data"""
        from statsmodels.tsa.arima.model import ARIMA
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_model = ARIMA(data, order=self.order).fit()
            return True
        except Exception as e:
            print(f"ARIMA fit failed: {e}")
            return False
    
    def predict(self, horizon: int) -> np.ndarray:
        """Generate point forecast"""
        if self.fitted_model is None:
            return np.zeros(horizon)
        
        try:
            forecast = self.fitted_model.forecast(steps=horizon)
            return forecast
        except:
            return np.zeros(horizon)
    
    def predict_with_uncertainty(self, data: np.ndarray, horizon: int, 
                                 num_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty via bootstrap.
        
        Args:
            data: Historical data to fit on
            horizon: Number of steps to forecast
            num_samples: Number of bootstrap samples (default: self.num_bootstrap)
            
        Returns:
            mean_pred: Mean prediction across bootstrap samples
            std_pred: Standard deviation (uncertainty)
        """
        from statsmodels.tsa.arima.model import ARIMA
        
        n_samples = num_samples or self.num_bootstrap
        predictions = []
        
        # Bootstrap: resample data and refit
        for _ in range(n_samples):
            try:
                # Residual bootstrap
                indices = np.random.choice(len(data), size=len(data), replace=True)
                boot_data = data[indices]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(boot_data, order=self.order).fit()
                pred = model.forecast(steps=horizon)
                predictions.append(pred)
            except:
                continue
        
        if len(predictions) < 2:
            # Fallback: use original model
            self.fit(data)
            mean_pred = self.predict(horizon)
            return mean_pred, np.zeros_like(mean_pred)
        
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred



class SARIMAModel:
    """
    Seasonal ARIMA (SARIMA) model.
    Attempts to model seasonality which may not exist in short epidemic windows.
    Often overfits or fails on non-seasonal epidemic data.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 4),
                 num_bootstrap: int = 20):
        self.order = order
        self.seasonal_order = seasonal_order
        self.num_bootstrap = num_bootstrap
        self.fitted_model = None
    
    def fit(self, data: np.ndarray) -> bool:
        """Fit SARIMA model to data"""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_model = SARIMAX(
                    data, 
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False, maxiter=50)
            return True
        except Exception as e:
            # Fallback to simple ARIMA
            try:
                from statsmodels.tsa.arima.model import ARIMA
                self.fitted_model = ARIMA(data, order=self.order).fit()
                return True
            except:
                return False
    
    def predict(self, horizon: int) -> np.ndarray:
        """Generate point forecast"""
        if self.fitted_model is None:
            return np.zeros(horizon)
        try:
            forecast = self.fitted_model.forecast(steps=horizon)
            return np.array(forecast)
        except:
            return np.zeros(horizon)



class SIRModel:
    """
    Classic SIR compartmental model for epidemic forecasting.
    
    Uses scipy's odeint for solving differential equations.
    Uncertainty via parameter sampling from fitted distribution.
    """
    
    def __init__(self, beta: float = 0.3, gamma: float = 0.1, 
                 population: float = 1.0, num_samples: int = 50):
        """
        Args:
            beta: Infection rate (transmission parameter)
            gamma: Recovery rate
            population: Total population (for normalization)
            num_samples: Number of parameter samples for uncertainty
        """
        self.beta = beta  # Infection rate
        self.gamma = gamma  # Recovery rate
        self.population = population
        self.num_samples = num_samples
    
    def _sir_derivatives(self, y, t, beta, gamma, N):
        """SIR differential equations"""
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    def fit(self, data: np.ndarray, fit_window: int = 14):
        """
        Estimate SIR parameters from observed infection data
        using least squares optimization.
        
        Args:
            data: Observed infection counts
            fit_window: Number of recent points to use for fitting
        """
        from scipy.optimize import minimize
        from scipy.integrate import odeint
        
        # Normalize data
        I_obs = data[-fit_window:] if len(data) > fit_window else data
        I_obs = np.maximum(I_obs, 1e-6)  # Avoid zeros
        
        # Initial conditions (rough estimate)
        I0 = I_obs[0]
        R0 = 0.0
        S0 = self.population - I0 - R0
        
        def loss(params):
            beta, gamma = params
            if beta <= 0 or gamma <= 0:
                return 1e10
            
            t = np.arange(len(I_obs))
            try:
                solution = odeint(self._sir_derivatives, [S0, I0, R0], t, 
                                 args=(beta, gamma, self.population))
                I_pred = solution[:, 1]
                return np.sum((I_pred - I_obs) ** 2)
            except:
                return 1e10
        
        # Optimize parameters
        result = minimize(loss, [self.beta, self.gamma], 
                         bounds=[(0.01, 2.0), (0.01, 1.0)],
                         method='L-BFGS-B')
        
        if result.success:
            self.beta, self.gamma = result.x
        
        return self
    
    def predict(self, initial_I: float, horizon: int) -> np.ndarray:
        """Generate point forecast"""
        from scipy.integrate import odeint
        
        I0 = max(initial_I, 1e-6)
        S0 = self.population - I0
        R0 = 0.0
        
        t = np.arange(horizon + 1)
        solution = odeint(self._sir_derivatives, [S0, I0, R0], t,
                         args=(self.beta, self.gamma, self.population))
        
        return solution[1:, 1]  # Return I compartment
    
    def predict_with_uncertainty(self, initial_I: float, horizon: int,
                                 beta_std: float = 0.05, gamma_std: float = 0.02
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty via parameter sampling.
        
        Args:
            initial_I: Initial infection value
            horizon: Number of steps to forecast
            beta_std: Standard deviation for beta sampling
            gamma_std: Standard deviation for gamma sampling
            
        Returns:
            mean_pred: Mean prediction
            std_pred: Standard deviation (uncertainty)
        """
        predictions = []
        
        for _ in range(self.num_samples):
            # Sample parameters from normal distribution
            beta_sample = max(0.01, np.random.normal(self.beta, beta_std))
            gamma_sample = max(0.01, np.random.normal(self.gamma, gamma_std))
            
            try:
                from scipy.integrate import odeint
                I0 = max(initial_I, 1e-6)
                S0 = self.population - I0
                R0 = 0.0
                
                t = np.arange(horizon + 1)
                solution = odeint(self._sir_derivatives, [S0, I0, R0], t,
                                 args=(beta_sample, gamma_sample, self.population))
                predictions.append(solution[1:, 1])
            except:
                continue
        
        if len(predictions) < 2:
            mean_pred = self.predict(initial_I, horizon)
            return mean_pred, np.zeros_like(mean_pred)
        
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_naive_online(test_tokens, num_input: int, num_output: int,
                          token_size: int):
    """
    Evaluate Naive baseline for online forecasting.
    
    Args:
        test_tokens: Test tokens array (num_tokens, token_size)
        num_input: Number of input tokens
        num_output: Number of output tokens
        token_size: Token dimension size
        
    Returns:
        mse: Mean squared error
        mae: Mean absolute error
    """
    total = num_input + num_output
    all_preds, all_targets = [], []
    
    for i in range(len(test_tokens) - total + 1):
        inp = test_tokens[i:i+num_input].flatten()
        out = test_tokens[i+num_input:i+total].flatten()
        # Repeat last value
        pred = np.repeat(inp[-1], num_output * token_size)
        all_preds.append(pred.reshape(num_output, token_size))
        all_targets.append(out.reshape(num_output, token_size))
    
    preds = np.array(all_preds)[:, -1, :]
    targets = np.array(all_targets)[:, -1, :]
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    return mse, mae


def evaluate_arima_online(test_tokens, num_input: int, num_output: int,
                          token_size: int, order: Tuple[int, int, int] = (2, 1, 2)):
    """
    Evaluate ARIMA baseline for online forecasting.
    
    Args:
        test_tokens: Test tokens array (num_tokens, token_size)
        num_input: Number of input tokens
        num_output: Number of output tokens
        token_size: Token dimension size
        order: ARIMA order (p, d, q)
        
    Returns:
        mse: Mean squared error (or None if failed)
        mae: Mean absolute error (or None if failed)
    """
    total = num_input + num_output
    all_preds, all_targets = [], []
    
    for i in range(len(test_tokens) - total + 1):
        inp = test_tokens[i:i+num_input].flatten()
        out = test_tokens[i+num_input:i+total].flatten()
        
        model = ARIMAModel(order=order)
        if model.fit(inp):
            pred = model.predict(num_output * token_size)
        else:
            pred = np.repeat(inp[-1], num_output * token_size)
        
        all_preds.append(pred.reshape(num_output, token_size))
        all_targets.append(out.reshape(num_output, token_size))
    
    preds = np.array(all_preds)[:, -1, :]
    targets = np.array(all_targets)[:, -1, :]
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    return mse, mae


class NaiveModel:
    """
    Naive baseline: repeat last observed value.
    
    Uses historical variance for uncertainty estimation.
    """
    
    def __init__(self):
        self.last_value = None
        self.historical_std = 0.1
    
    def fit(self, data: np.ndarray):
        """Fit model (store last value and compute variance)"""
        self.last_value = data[-1] if len(data) > 0 else 0
        
        # Compute historical standard deviation
        if len(data) > 4:
            self.historical_std = np.std(data[-10:]) if len(data) >= 10 else np.std(data)
        else:
            self.historical_std = 0.1
        
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Generate point forecast (repeat last value)"""
        return np.repeat(self.last_value, horizon)
    
    def predict_with_uncertainty(self, data: np.ndarray, horizon: int
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty using historical variance.
        
        Args:
            data: Historical data
            horizon: Number of steps to forecast
            
        Returns:
            mean_pred: Mean prediction (repeated last value)
            std_pred: Standard deviation (historical variance)
        """
        self.fit(data)
        mean_pred = self.predict(horizon)
        std_pred = np.full(horizon, self.historical_std)
        
        return mean_pred, std_pred
