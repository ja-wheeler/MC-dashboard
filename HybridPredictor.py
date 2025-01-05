import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
from ltsm import DealSizePredictor
warnings.filterwarnings('ignore')
from old.base_models import FundingForecaster

class HybridPredictor:
    def __init__(self, data, deal_predictor, target_col='Total Funding', 
                column_mapping=None):
        # Accept either DataFrame or path
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()  # Make a copy of the input DataFrame
        
        print(f"Loaded dataset with {len(self.df)} rows")
        
        self.target_col = target_col
        self.deal_predictor = deal_predictor
        self.scaler = StandardScaler()
        
        self.column_mapping = column_mapping or {
            'Deal Size': 'Total Funding',
            'Deal Count': 'Number of Rounds'
        }
        
        self.feature_cols = [col for col in self.df.columns 
                        if col != self.target_col and col != 'Year']
    
    def prepare_future_features(self, future_years):
        print(f"Preparing features for {future_years} future years")
        
        # Get NN predictions for future periods
        nn_preds = self.get_nn_predictions(future_years)
        print(f"NN predictions shape: mean={nn_preds['mean'].shape}")
        
        # Create base DataFrame with original features
        last_values = self.df[self.feature_cols].iloc[-1]
        future_X = pd.DataFrame([last_values] * future_years, 
                              columns=self.feature_cols)
        print(f"Future features shape: {future_X.shape}")
        
        # Add time-based features
        last_year = int(self.df['Year'].max())
        future_years_range = range(last_year + 1, last_year + future_years + 1)
        
        future_X['Year_Squared'] = np.array(future_years_range) ** 2
        future_X['Year_Cubed'] = np.array(future_years_range) ** 3
        future_X['Trend'] = np.arange(len(self.df), len(self.df) + future_years)
        future_X['Cycle'] = np.sin(2 * np.pi * np.arange(future_years) / 12)
        
        # Add NN trend
        nn_trend = pd.Series(nn_preds['mean']).pct_change().fillna(0)
        if len(nn_trend) != len(future_X):
            print(f"Warning: NN trend length ({len(nn_trend)}) != future_X length ({len(future_X)})")
            # Adjust length
            nn_trend = nn_trend[:future_years]
            if len(nn_trend) < future_years:
                nn_trend = pd.Series([0] * future_years)
        
        future_X['NN_Trend'] = nn_trend.values
        print(f"Final future features shape: {future_X.shape}")
        
        return future_X
    
    def get_nn_predictions(self, future_years):
        deal_size_col, deal_count_col = self.get_mapped_columns(['Deal Size', 'Deal Count'])
        
        last_sequence = self.deal_predictor.scaler.transform(
            self.df.iloc[-self.deal_predictor.sequence_length:][
                [deal_size_col, deal_count_col]
            ]
        )
        
        return self.deal_predictor.predict_with_intervals(
            last_sequence, 
            n_future=future_years
        )
    
    def get_mapped_columns(self, cols):
        return [self.column_mapping.get(col, col) for col in cols]
    
    def prepare_features(self, future_years=0):
        X = self.df[self.feature_cols].copy()
        
        # Add engineered features
        X['Year_Squared'] = self.df['Year'] ** 2
        X['Year_Cubed'] = self.df['Year'] ** 3
        X['Trend'] = np.arange(len(self.df))
        X['Cycle'] = np.sin(2 * np.pi * np.arange(len(self.df)) / 12)
        
        deal_size_col, deal_count_col = self.get_mapped_columns(['Deal Size', 'Deal Count'])
        
        try:
            historical_nn = self.deal_predictor.model.predict(
                self.deal_predictor.scaler.transform(
                    self.df[[deal_size_col, deal_count_col]]
                ).reshape(-1, self.deal_predictor.sequence_length, 2)
            )
            
            nn_trend = pd.Series(historical_nn.flatten()[-len(self.df):])
            nn_trend = nn_trend.pct_change().fillna(0)
            
            if len(nn_trend) != len(self.df):
                print(f"Warning: Historical NN trend length mismatch. Padding/trimming to match.")
                if len(nn_trend) < len(self.df):
                    padding = [0] * (len(self.df) - len(nn_trend))
                    nn_trend = pd.Series(padding + nn_trend.tolist())
                else:
                    nn_trend = nn_trend[-len(self.df):]
            
            X['NN_Trend'] = nn_trend.values
            
        except Exception as e:
            print(f"Warning: Could not generate NN predictions: {str(e)}")
            X['NN_Trend'] = np.zeros(len(self.df))
        
        return X
    
    def fit(self, arima_order=(1,0,1)):
        print("Fitting models...")
        
        # Fit ARIMA
        self.arima_model = ARIMA(self.df[self.target_col], order=arima_order)
        self.arima_results = self.arima_model.fit()
        
        # Get ARIMA predictions and residuals
        self.arima_predictions = self.arima_results.fittedvalues
        self.residuals = self.df[self.target_col] - self.arima_predictions
        
        # Prepare features and fit XGBoost on residuals
        X = self.prepare_features()
        print(f"Training feature shape: {X.shape}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.residual_model = XGBRegressor(
            n_estimators=50,  # Much fewer trees
            learning_rate=0.1,
            max_depth=3,      # Shallower trees
            min_child_weight=3,  # Prevent overfitting
            subsample=0.8, 
            colsample_bytree=0.8
        )
        self.residual_model.fit(X_scaled, self.residuals)
        
        return self
    
    def predict(self, future_years=4):
        print(f"\nGenerating {future_years} year forecast...")
        
        # Get ARIMA forecast
        arima_forecast = self.arima_results.forecast(steps=future_years)
        arima_forecast = np.array(arima_forecast)  # Convert to numpy array
        print(f"ARIMA forecast length: {len(arima_forecast)}")
        print(f"ARIMA forecast range: {arima_forecast.min():.2f} to {arima_forecast.max():.2f}")
        
        # Get feature-based residual predictions
        future_X = self.prepare_future_features(future_years)
        print(f"Future features prepared: {future_X.shape}")
        
        future_X_scaled = self.scaler.transform(future_X)
        residual_forecast = self.residual_model.predict(future_X_scaled)
        print(f"Residual forecast length: {len(residual_forecast)}")
        print(f"Residual forecast range: {residual_forecast.min():.2f} to {residual_forecast.max():.2f}")
        
        # Combine forecasts
        final_forecast = arima_forecast + residual_forecast
        print(f"Final forecast range: {final_forecast.min():.2f} to {final_forecast.max():.2f}")
        
        # Combine forecasts
        final_forecast = arima_forecast + residual_forecast
        
        # Create output dataframe
        last_year = int(self.df['Year'].max())
        future_years_range = range(last_year + 1, last_year + future_years + 1)
        
        # Ensure numpy arrays for all components
        years = list(future_years_range)
        final_forecast = np.array(final_forecast)
        arima_forecast = np.array(arima_forecast)
        residual_forecast = np.array(residual_forecast)
        nn_trend = np.array(future_X['NN_Trend'].values)
        
        # Create DataFrame with explicit index
        forecast_df = pd.DataFrame({
            'Year': years,
            'Forecast': final_forecast.astype(float),
            'ARIMA_Component': arima_forecast.astype(float),
            'Residual_Component': residual_forecast.astype(float),
            'NN_Trend': nn_trend.astype(float)
        }, index=range(len(years)))
        
        print(f"Forecast DataFrame shape: {forecast_df.shape}")
        
        return forecast_df

    def get_feature_importance(self):
        """Get feature importance while handling non-string feature names"""
        try:
            features = [str(name) for name in self.residual_model.feature_names_in_]
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': self.residual_model.feature_importances_
            })
        except AttributeError:
            # Fallback if feature_names_in_ is not available
            importance_df = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(self.residual_model.feature_importances_))],
                'Importance': self.residual_model.feature_importances_
            })
            
        return importance_df.sort_values('Importance', ascending=False)
