from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any, Protocol
import numpy as np
import pandas as pd
from dataclasses import dataclass

from dataclasses import dataclass
import numpy as np

@dataclass
class MarketNoiseParameters:
    """Core parameters for funding market dynamics."""
    volatility: float      # Annualized volatility (sigma)
    trend: float           # Drift/expected return (mu)
    momentum: float        # Jump probability (lambda)
    market_impact: float   # Jump size magnitude (scale for jumps)

class AdaptiveNoiseGenerator:
    def __init__(self, params: MarketNoiseParameters):
        self.params = params
        self._validate_parameters()
        
    def _validate_parameters(self):
        if not (0 <= self.params.volatility <= 1):
            raise ValueError("Volatility must be in [0, 1]")
        if not (-0.5 <= self.params.trend <= 0.5):
            raise ValueError("Drift must be in [-0.5, 0.5]")
        if not (0 <= self.params.momentum <= 1):
            raise ValueError("Jump probability must be in [0, 1]")
        if not (0 <= self.params.market_impact <= 1):
            raise ValueError("Jump size must be in [0, 1]")
    
    def generate_noise(
        self,
        n_periods: int,
        random_state: np.random.RandomState,
    ) -> np.ndarray:
        """
        Generate paths using GBM with jumps and clear impact of trend.
        
        Parameters:
        - n_periods: Number of time steps (years).
        - random_state: Numpy random state for reproducibility.
        
        Returns:
        - Returns array of size n_periods.
        """
        # Generate normal random variables for stochastic term
        Z = random_state.normal(0, 1, n_periods)

        # Poisson process for jumps 
        n_jumps = random_state.poisson(self.params.momentum * n_periods)
        jump_times = random_state.choice(n_periods, size=n_jumps)

        # Lognormal jump sizes
        jump_sizes = random_state.lognormal(
        mean=-0.5 * self.params.market_impact**2,  # Ensures E[jump] = 1
        sigma=self.params.market_impact,
        size=n_jumps
        )

        # Initialize returns with diffusion
        returns = self.params.trend + self.params.volatility * Z

        # Add jumps at random times
        for i, t in enumerate(jump_times):
            returns[t] += jump_sizes[i] - 1  # Subtract 1 to center jumps around 0

        return returns
    
    
def calibrate_parameters(historical_prices: np.ndarray) -> MarketNoiseParameters:
    prices = np.array(historical_prices, dtype=np.float64)
    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)
    valid_returns = log_returns[np.isfinite(log_returns)]
    
    if len(valid_returns) == 0 or np.all(np.isnan(valid_returns)):
        return MarketNoiseParameters(
            volatility=0.05, trend=0.02, momentum=0.03, market_impact=0.1
        )
    
    # Estimate diffusion volatility (excluding jumps)
    volatility = np.std(valid_returns[
    (np.abs(valid_returns) >= np.percentile(np.abs(valid_returns), 25)) & 
    (np.abs(valid_returns) <= np.percentile(np.abs(valid_returns), 75))
])
    
    # Trend estimation (same as before)
    time_points = np.arange(len(log_prices))
    slopes = [(log_prices[j] - log_prices[i]) / (time_points[j] - time_points[i])
             for i in range(len(time_points)) 
             for j in range(i + 1, len(time_points))]
    trend = np.median(slopes)
    
    # Jump frequency (lambda)
    extreme_returns = np.abs(valid_returns) > 1.2 * volatility
    print("extreme", extreme_returns)
    momentum = np.mean(extreme_returns)
    
    # Jump size volatility
    jump_returns = valid_returns[extreme_returns]
    market_impact = np.std(jump_returns) if len(jump_returns) > 0 else 0.1
    

    # Cap all parameters between 0 and 1
    volatility = min(volatility, 0.3)
    trend = min(max(np.median(slopes), -0.2), 0.2)  # Cap trend at ±20%
    momentum = min(np.mean(extreme_returns), 0.3)    # Cap at 0.3 jumps per year
    market_impact = min(np.std(jump_returns) if len(jump_returns) > 0 else 0.1, 0.5)  # Cap impact at 50%

    return MarketNoiseParameters(
        volatility=volatility,
        trend=trend,
        momentum=momentum,
        market_impact=market_impact
    )

# Protocol for prediction models
class PredictionModel(Protocol):
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        """Protocol method for predictions"""
        pass

    def fit(self, *args, **kwargs) -> Any:
        """Protocol method for fitting"""
        pass

class NoiseGenerator(ABC):
    """Abstract base class for noise generation strategies"""
    @abstractmethod
    def generate_noise(self, n_periods: int, random_state: np.random.RandomState) -> np.ndarray:
        pass



class MonteCarloSimulator:
    """Monte Carlo simulation class that works with any prediction model"""
    
    def __init__(
        self, 
        model: PredictionModel,
        noise_generator: NoiseGenerator,
        forecast_column: str = 'Forecast',
        seed: Optional[int] = None
    ):
        self.model = model
        self.noise_generator = noise_generator
        self.forecast_column = forecast_column
        self.random_state = np.random.RandomState(seed)

    def run_simulation(
        self, 
        future_periods: int, 
        n_simulations: int = 1000,
        **model_kwargs
    ) -> pd.DataFrame:
        """Run Monte Carlo simulation"""
        try:
            # Get base forecast from the model
            base_forecast = self.model.predict(future_periods, **model_kwargs)
            
            # Initialize array for simulations
            simulations = np.zeros((n_simulations, future_periods))
            
            # Run Monte Carlo simulations
            for i in range(n_simulations):
                noise = self.noise_generator.generate_noise(future_periods, self.random_state)
                simulations[i] = base_forecast[self.forecast_column].values * (1 + noise)
            
            # Calculate statistics
            intervals = {
                'mean': np.mean(simulations, axis=0),
                'std': np.std(simulations, axis=0),
                'lower_95': np.percentile(simulations, 2.5, axis=0),
                'upper_95': np.percentile(simulations, 97.5, axis=0),
                'lower_50': np.percentile(simulations, 25, axis=0),
                'upper_50': np.percentile(simulations, 75, axis=0)
            }
            
            # Add results to the forecast DataFrame
            result_df = base_forecast.copy()
            for key, values in intervals.items():
                result_df[f'MC_{key}'] = values
                
            return result_df
            
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {str(e)}")

