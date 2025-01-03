import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from HybridPredictor import EnhancedHybridPredictor
import pandas as pd
from ltsm import DealSizePredictor

class StochasticHybridPredictor(EnhancedHybridPredictor):
    def __init__(self, 
                 data_path: str, 
                 deal_predictor=None,  # Make deal_predictor optional
                 target_col: str = 'Total Funding',
                 column_mapping: Optional[Dict[str, str]] = None,
                 noise_params: Optional[Dict[str, float]] = None):
        """
        Initialize StochasticHybridPredictor with optional noise parameters.
        
        Args:
            noise_params (dict): Dictionary containing noise parameters:
                - base_volatility: Base volatility level (default: 0.1)
                - trend_volatility: Volatility in trend (default: 0.05)
                - seasonality_volatility: Volatility in seasonal components (default: 0.03)
        """
        self.use_nn = deal_predictor is not None
        super().__init__(data_path, deal_predictor, target_col, column_mapping)
        
        # Default noise parameters
        self.noise_params = noise_params or {
            'base_volatility': 0.1,
            'trend_volatility': 0.05,
            'seasonality_volatility': 0.03
        }
        
        # Initialize random state
        self.random_state = np.random.RandomState()
        
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility"""
        self.random_state = np.random.RandomState(seed)
        
    def generate_stochastic_noise(self, n_periods: int) -> np.ndarray:
        """Generate multi-component stochastic noise."""
        # Base random noise
        base_noise = self.random_state.normal(
            0, 
            self.noise_params['base_volatility'], 
            n_periods
        )
        
        # Trend noise (increases with time)
        trend_factor = np.linspace(0, 1, n_periods)
        trend_noise = self.random_state.normal(
            0, 
            self.noise_params['trend_volatility'], 
            n_periods
        ) * trend_factor
        
        # Seasonal noise
        seasonal_cycle = np.sin(2 * np.pi * np.arange(n_periods) / 12)
        seasonal_noise = self.random_state.normal(
            0, 
            self.noise_params['seasonality_volatility'], 
            n_periods
        ) * seasonal_cycle
        
        return base_noise + trend_noise + seasonal_noise
    
    def prepare_future_features(self, future_years: int) -> pd.DataFrame:
        """Override parent method to handle missing NN predictor"""
        # Get last values for features
        last_values = self.df[self.feature_cols].iloc[-1]
        future_X = pd.DataFrame([last_values] * future_years, 
                              columns=self.feature_cols)
        
        # Add time-based features
        last_year = int(self.df['Year'].max())
        future_years_range = range(last_year + 1, last_year + future_years + 1)
        
        future_X['Year_Squared'] = np.array(future_years_range) ** 2
        future_X['Year_Cubed'] = np.array(future_years_range) ** 3
        future_X['Trend'] = np.arange(len(self.df), len(self.df) + future_years)
        future_X['Cycle'] = np.sin(2 * np.pi * np.arange(future_years) / 12)
        
        # Add zero NN trend if no predictor available
        future_X['NN_Trend'] = np.zeros(future_years)
        
        return future_X
    
    def predict_with_uncertainty(self, 
                               future_years: int = 3, 
                               n_simulations: int = 1000) -> pd.DataFrame:
        """Generate forecast with uncertainty intervals through Monte Carlo simulation."""
        try:
            # Get base forecast
            base_forecast = super().predict(future_years)
            
            # Initialize array for simulations
            simulations = np.zeros((n_simulations, future_years))
            
            # Run Monte Carlo simulations
            for i in range(n_simulations):
                noise = self.generate_stochastic_noise(future_years)
                simulations[i] = base_forecast['Forecast'].values * (1 + noise)
                
            # Calculate confidence intervals
            intervals = {
                'mean': np.mean(simulations, axis=0),
                'std': np.std(simulations, axis=0),
                'lower_95': np.percentile(simulations, 2.5, axis=0),
                'upper_95': np.percentile(simulations, 97.5, axis=0),
                'lower_50': np.percentile(simulations, 25, axis=0),
                'upper_50': np.percentile(simulations, 75, axis=0)
            }
            
            # Create output DataFrame
            forecast_df = base_forecast.copy()
            for key, values in intervals.items():
                forecast_df[f'Stochastic_{key}'] = values
                
            return forecast_df
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
    
    def plot_forecast(self, forecast_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot the forecast with uncertainty bands."""
        plt.figure(figsize=figsize)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 12
        
        # Plot historical data
        historical_years = self.df['Year']
        historical_values = self.df[self.target_col]
        plt.plot(historical_years, historical_values, 'b-', label='Historical Data', linewidth=2)
        
        # Plot forecast mean
        forecast_years = forecast_df['Year']
        plt.plot(forecast_years, forecast_df['Stochastic_mean'], 'r--', 
                label='Forecast Mean', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(forecast_years, 
                        forecast_df['Stochastic_lower_95'],
                        forecast_df['Stochastic_upper_95'],
                        alpha=0.2, color='red',
                        label='95% Confidence Interval')
        
        plt.fill_between(forecast_years,
                        forecast_df['Stochastic_lower_50'],
                        forecast_df['Stochastic_upper_50'],
                        alpha=0.3, color='red',
                        label='50% Confidence Interval')
        
        # Customize plot
        plt.title(f'Stochastic Forecast with Uncertainty Bands\nTarget: {self.target_col}',
                 fontsize=14, pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel(self.target_col, fontsize=12)
        plt.legend(loc='best', fontsize=10)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()

# Example usage
if __name__ == "__main__":
    try:
        # Initialize without neural network for now
        predictor = StochasticHybridPredictor(
            'data.csv',
            deal_predictor=None,  # Skip neural network component
            noise_params={
                'base_volatility': 0.15,
                'trend_volatility': 0.08,
                'seasonality_volatility': 0.05
            }
        )
        
        # Fit the model
        predictor.fit()
        
        # Generate forecast with uncertainty
        forecast = predictor.predict_with_uncertainty(future_years=3, n_simulations=1000)
        
        # Plot the results
        predictor.plot_forecast(forecast)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")