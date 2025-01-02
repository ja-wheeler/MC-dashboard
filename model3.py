import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler

class RobustAdaptivePredictor:
    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()
        self.scaler = RobustScaler()
        self.smoothing_params = None
        self.trend = None
        self.last_values = None
        
    def preprocess(self, series):
        """Handle zeros and log transform the data"""
        # Convert to numpy array if not already
        series = np.array(series)
        # Handle zeros using numpy operations
        min_nonzero = np.min(series[series > 0])
        adjusted_series = np.where(series == 0, min_nonzero / 10, series)
        # Log transform
        return np.log1p(adjusted_series)
        
    def fit(self, smoothing_window=3):
        """Fit the model using robust techniques"""
        y = self.df['Total Funding'].values
        
        # Preprocess the data
        y_processed = self.preprocess(y)
        y_scaled = self.scaler.fit_transform(y_processed.reshape(-1, 1)).flatten()
        
        # Calculate adaptive smoothing parameters
        diffs = np.abs(np.diff(y_scaled))
        self.smoothing_params = 1 / (1 + np.exp(-diffs))  # Sigmoid function
        
        # Store last values for prediction
        self.last_values = y_scaled[-smoothing_window:]
        
        # Calculate robust trend
        x = np.arange(len(y_scaled))
        # Use Theil-Sen regression for robust trend estimation
        slope, intercept = stats.theilslopes(y_scaled, x)[:2]
        self.trend = slope
        
        return self
        
    def predict(self, future_periods: int) -> pd.DataFrame:
        """Generate predictions with uncertainty estimates"""
        last_year = self.df['Year'].max()
        future_years = np.arange(last_year + 1, last_year + future_periods + 1)
        
        # Initialize predictions
        predictions = []
        last_value = np.mean(self.last_values)
        
        for i in range(future_periods):
            # Combine adaptive smoothing with trend
            pred = last_value + self.trend
            
            # Add some controlled randomness for realistic variations
            noise = np.random.normal(0, abs(self.trend) * 0.1)
            pred += noise
            
            predictions.append(pred)
            last_value = pred
        
        # Transform predictions back to original scale
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        # Inverse of log transformation
        predictions = np.expm1(predictions)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'Forecast': predictions,
        })
        
        return forecast_df

class RobustAdaptivePredictorWrapper:
    def __init__(self, data: pd.DataFrame):
        self.model = RobustAdaptivePredictor(data)
        self.df = data
        
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, future_periods: int, **kwargs) -> pd.DataFrame:
        return self.model.predict(future_periods)
    
    def get_model_params(self) -> dict:
        return {
            'trend': self.model.trend,
            'smoothing_params': self.model.smoothing_params,
            'last_values': self.model.last_values
        }
    
    def set_random_seed(self, seed: int):
        np.random.seed(seed)