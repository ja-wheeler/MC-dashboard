import pandas as pd
import numpy as np

class WeightedTrendPredictor:
    """A robust predictor that combines trend analysis with weighted recent growth"""
    
    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()
        self.target_col = 'Total Funding'
        self.weights = None
        self.trend_coefficients = None
        self.recent_growth_rate = None
        
    def calculate_weighted_growth(self, values, window=10):
        """Calculate weighted average growth rate with a more balanced weighting over 10 years"""
        if len(values) < 2:
            return 0
            
        # Calculate year-over-year growth rates
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] > 0:  # Avoid division by zero
                growth_rate = (values[i] - values[i-1]) / values[i-1]
                # Clip extreme growth rates to reduce outlier impact
                growth_rate = np.clip(growth_rate, -1.0, 2.0)  # Limit to between -100% and +200%
                growth_rates.append(growth_rate)
            else:
                growth_rates.append(0)
        
        if not growth_rates:
            return 0
            
        # Use last 'window' years with a more gradual weighting scheme
        recent_rates = growth_rates[-window:]
        # Use square root for more gradual weight increase
        weights = np.sqrt(np.linspace(0.5, 1.0, len(recent_rates)))
        weights = weights / weights.sum()  # Normalize weights
        
        # Calculate weighted average excluding extreme outliers
        weighted_growth = np.average(recent_rates, weights=weights)
        
        return weighted_growth
    
    def fit(self, window=5):
        """Fit the model using trend analysis and weighted growth rates"""
        # Prepare data
        y = self.df[self.target_col].values
        X = np.arange(len(y)).reshape(-1, 1)
        
        # Fit polynomial trend (degree 2 for robustness)
        self.trend_coefficients = np.polyfit(X.flatten(), y, 2)
        
        # Calculate weighted growth rate from recent years
        self.recent_growth_rate = self.calculate_weighted_growth(y, window)
        
        return self
    
    def predict(self, future_periods: int) -> pd.DataFrame:
        """Generate predictions combining trend and growth components"""
        last_year = self.df['Year'].max()
        future_years = np.arange(last_year + 1, last_year + future_periods + 1)
        
        # Generate base predictions using trend
        X_future = np.arange(len(self.df), len(self.df) + future_periods)
        trend_pred = np.polyval(self.trend_coefficients, X_future)
        
        # Apply weighted growth rate influence
        growth_factor = np.array([(1 + self.recent_growth_rate) ** i 
                                for i in range(1, future_periods + 1)])
        
        # Combine predictions (60% trend, 40% growth)
        base_value = self.df[self.target_col].iloc[-1]
        growth_pred = base_value * growth_factor
        
        final_pred = 0.6 * trend_pred + 0.4 * growth_pred
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'Forecast': final_pred,
            'Trend_Component': trend_pred,
            'Growth_Component': growth_pred
        })
        
        return forecast_df

class WeightedTrendPredictorWrapper:
    """Wrapper for WeightedTrendPredictor to make it compatible with MC simulation"""
    
    def __init__(self, data: pd.DataFrame):
        self.model = WeightedTrendPredictor(data)
        self.df = data
        
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, future_periods: int, **kwargs) -> pd.DataFrame:
        return self.model.predict(future_periods)