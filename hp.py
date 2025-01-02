import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from rf7 import FundingForecaster

class HybridPredictor:
    def __init__(self, data_path, target_col='Total Funding'):
        self.df = pd.read_csv(data_path)
        self.target_col = target_col
        self.feature_forecaster = FundingForecaster(data_path)
        self.scaler = StandardScaler()
        
    def prepare_features(self, future_years):
        """Prepare feature matrix using all available features"""
        # First get forecasts for all features using previous model
        self.feature_forecaster.fit_and_forecast(future_years=future_years)
        
        # Prepare historical feature matrix
        feature_cols = [col for col in self.df.columns 
                    if col != self.target_col and col != 'Year']
        
        X = self.df[feature_cols].copy()
        
        # Add engineered features
        X['Year_Squared'] = self.df['Year'] ** 2
        X['Year_Cubed'] = self.df['Year'] ** 3
        X['Trend'] = np.arange(len(self.df))
        X['Cycle'] = np.sin(2 * np.pi * X['Trend'] / 12)
        
        return X
    
    def fit_residual_model(self, X):
        """Fit XGBoost model on residuals"""
        X_scaled = self.scaler.fit_transform(X)
        self.residual_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        self.residual_model.fit(X_scaled, self.residuals)
        return self.residual_model
    
    def fit_arima(self, order=(1,1,1)):
        """Fit ARIMA model"""
        self.arima_model = ARIMA(self.df[self.target_col], order=order)
        self.arima_results = self.arima_model.fit()
        
        # Get ARIMA predictions and residuals
        self.arima_predictions = self.arima_results.fittedvalues
        self.residuals = self.df[self.target_col] - self.arima_predictions
        
        return self.arima_predictions, self.residuals
    
    def prepare_future_features(self, future_years):
        """Prepare feature matrix for future predictions"""
        future_features = {}
        
        # Get base forecasts
        for col in self.feature_forecaster.forecast_columns:
            if col != self.target_col:
                future_features[col] = self.feature_forecaster.forecasts[col]['forecast']
        
        future_X = pd.DataFrame(future_features)
        
        # Add engineered features
        last_year = int(self.df['Year'].max())
        future_years_range = range(last_year + 1, last_year + future_years + 1)
        
        future_X['Year_Squared'] = np.array(future_years_range) ** 2
        future_X['Year_Cubed'] = np.array(future_years_range) ** 3
        future_X['Trend'] = np.arange(len(self.df), len(self.df) + future_years)
        future_X['Cycle'] = np.sin(2 * np.pi * future_X['Trend'] / 12)
        
        return future_X
    
    def predict(self, future_years=10):
        """Generate predictions"""
        # First fit the models
        self.fit_arima()
        X = self.prepare_features(future_years)
        self.fit_residual_model(X)
        
        # Get ARIMA forecast
        arima_forecast = self.arima_results.forecast(steps=future_years)
        
        # Get feature-based residual predictions
        future_X = self.prepare_future_features(future_years)
        future_X_scaled = self.scaler.transform(future_X)
        residual_forecast = self.residual_model.predict(future_X_scaled)
        
        # Combine forecasts
        final_forecast = arima_forecast + residual_forecast
        
        # Create output dataframe
        last_year = int(self.df['Year'].max())
        future_years_range = range(last_year + 1, last_year + future_years + 1)
        
        forecast_df = pd.DataFrame({
            'Year': future_years_range,
            'Forecast': final_forecast
        })
        
        return forecast_df

    def plot_results(self, forecast_df):
        """Plot historical data and forecast"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=self.df['Year'],
            y=self.df[self.target_col],
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Forecast'],
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{self.target_col} - Forecast Results',
            xaxis_title='Year',
            yaxis_title='Value',
            hovermode='x'
        )
        
        return fig

# Usage example
if __name__ == "__main__":
    # Initialize predictor
    predictor = HybridPredictor('data.csv')
    
    # Generate forecast
    forecast_df = predictor.predict(future_years=10)
    
    # Plot results
    fig = predictor.plot_results(forecast_df)
    fig.show()
    
    # Print forecast
    print("\nForecast Results:")
    print(forecast_df)