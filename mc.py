from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any, Protocol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


# Protocol for prediction models
class PredictionModel(Protocol):
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        """Protocol method for predictions"""
        pass

    def fit(self, *args, **kwargs) -> Any:
        """Protocol method for fitting"""
        pass

@dataclass
class NoiseParameters:
    """Dataclass for noise parameters"""
    base_volatility: float = 0.1
    trend_volatility: float = 0.05
    seasonality_volatility: float = 0.03
    growth_factor: float = 0.02

class NoiseGenerator(ABC):
    """Abstract base class for noise generation strategies"""
    @abstractmethod
    def generate_noise(self, n_periods: int, random_state: np.random.RandomState) -> np.ndarray:
        pass

class MultiComponentNoise(NoiseGenerator):
    """Implements multi-component noise generation with growth factor"""
    def __init__(self, noise_params: NoiseParameters):
        self.noise_params = noise_params

    def generate_noise(self, n_periods: int, random_state: np.random.RandomState) -> np.ndarray:
        # Base noise component
        base_noise = random_state.normal(
            0, 
            self.noise_params.base_volatility, 
            n_periods
        )
        
        # Trend noise with growth factor
        trend_factor = np.linspace(0, 1, n_periods)
        trend_noise = random_state.normal(
            0, 
            self.noise_params.trend_volatility, 
            n_periods
        ) * trend_factor
        
        # Seasonal noise component
        seasonal_cycle = np.sin(2 * np.pi * np.arange(n_periods) / 12)
        seasonal_noise = random_state.normal(
            0, 
            self.noise_params.seasonality_volatility, 
            n_periods
        ) * seasonal_cycle
        
        # Calculate cumulative growth factor
        cumulative_growth = np.array([(1 + self.noise_params.growth_factor) ** i for i in range(n_periods)])
        
        # Combine all components with growth
        total_noise = base_noise + trend_noise + seasonal_noise
        return (total_noise + 1) * cumulative_growth - 1

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

