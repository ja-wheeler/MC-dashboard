
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
from processor import DataProcessor
warnings.filterwarnings('ignore')

class EnhancedFundingPredictor:
    def __init__(self, data, target_col='Total Funding'):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        
        self.df = self.df.replace('-', 0)
        for col in self.df.columns:
            if col != 'Year':
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Loaded dataset with {len(self.df)} rows")
        
        self.target_col = target_col
        self.scaler = RobustScaler(quantile_range=(10, 90))
        self.funding_types = ['Seed Stage Funding', 'Early Stage Funding', 
                            'Late Stage Funding']

    def add_engineered_features(self, df):
        """Modified feature engineering for startup funding data"""
        features = df.copy()
        
        total_funding = features[self.funding_types].sum(axis=1)
        for funding_type in self.funding_types:
            features[f'{funding_type}_Ratio'] = (
                features[funding_type] / total_funding
            ).fillna(0)
        
        features['Year_Over_Year_Growth'] = features[self.target_col].pct_change()
        features['Round_Growth'] = features['Number of Rounds'].pct_change()
        
        for col in ['Total Funding', 'Number of Rounds']:
            features[f'{col}_MA3'] = features[col].rolling(3, min_periods=1).mean()
            features[f'{col}_Std3'] = features[col].rolling(3, min_periods=1).std()
        
        for stage in ['Seed Stage', 'Early Stage', 'Late Stage']:
            features[f'Avg_{stage}_Size'] = (
                features[f'{stage} Funding'] / 
                features[f'{stage} Rounds']
            ).fillna(0)
        
        return features
    
    def prepare_features(self, df=None):
        if df is None:
            df = self.df
            
        features = self.add_engineered_features(df)
        
        numerical_cols = features.select_dtypes(
            include=['int64', 'float64']
        ).columns
        numerical_cols = [col for col in numerical_cols 
                         if col != self.target_col and col != 'Year']
        
        if not hasattr(self, 'feature_names'):
            self.feature_names = numerical_cols
        
        return features[self.feature_names]
    
    def fit(self):
        """Fit enhanced hybrid model with state space component"""
        print("Fitting enhanced hybrid model...")
        
        # Log transform the target for better state space modeling
        y = np.log(self.df[self.target_col])
        
        # State Space Model with trend and stochastic volatility
        self.ss_model = UnobservedComponents(
            y,
            level='local linear trend',  # Allows for time-varying trend
            trend=True,
            cycle=False,  # No cyclical component for yearly data
            irregular=True,  # Captures random variations
            stochastic_volatility=True  # Allows for varying uncertainty
        )
        
        self.ss_results = self.ss_model.fit(method='powell', disp=False)
        
        # Get state space predictions and residuals
        self.ss_predictions = np.exp(self.ss_results.fittedvalues)
        self.residuals = self.df[self.target_col] - self.ss_predictions
        
        # Prepare features for Random Forest
        X = self.prepare_features()
        print(f"Training features shape: {X.shape}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Enhanced Random Forest with robust settings
        self.residual_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_split=3,
            min_samples_leaf=2,
            bootstrap=True,
            random_state=42
        )
        self.residual_model.fit(X_scaled, self.residuals)
        
        return self
    
    def predict(self, future_years=2):
        """Generate predictions with enhanced uncertainty estimation"""
        print(f"\nGenerating {future_years} year forecast...")
        
        # Get state space forecast in log space
        ss_forecast = self.ss_results.forecast(future_years)
        
        # Use rolling standard deviation for better uncertainty estimation
        rolling_std = pd.Series(self.ss_results.resid).rolling(window=3, min_periods=1).std()
        model_std = np.clip(rolling_std.mean(), 0.1, 0.25)  # Constrain between 10-25%
        
        ss_forecast = np.exp(ss_forecast)  # Transform back to original scale
        
        # Prepare future features
        future_df = pd.DataFrame({
            'Year': range(
                int(self.df['Year'].max() + 1),
                int(self.df['Year'].max() + future_years + 1)
            )
        })
        
        # Copy last year's values for initial future features
        for col in self.df.columns:
            if col != 'Year':
                future_df[col] = self.df[col].iloc[-1]
        
        future_X = self.prepare_features(future_df)
        future_X_scaled = self.scaler.transform(future_X)
        
        # Enhanced bootstrap predictions with progressive uncertainty
        n_bootstraps = 500  # Increased for better stability
        all_predictions = []
        
        last_value = self.df[self.target_col].iloc[-1]
        trend = np.mean(np.diff(np.log(self.df[self.target_col].values[-3:])))  # Recent trend
        
        for _ in range(n_bootstraps):
            # Progressive uncertainty for each year
            yearly_std = np.array([model_std * (i + 1) for i in range(future_years)])
            
            # Bootstrap state space component with trend-aware noise
            noise = np.random.normal(trend, yearly_std)
            cumulative_noise = np.cumsum(noise)
            ss_bootstrap = ss_forecast * np.exp(cumulative_noise)
            
            # Ensure reasonable bounds based on historical volatility
            ss_bootstrap = np.clip(
                ss_bootstrap,
                last_value * 0.7 ** np.arange(1, future_years + 1),  # Maximum annual decline
                last_value * 1.5 ** np.arange(1, future_years + 1)   # Maximum annual growth
            )
            
            # Bootstrap residual predictions
            residual_bootstrap = self.residual_model.predict(future_X_scaled)
            
            # Combine predictions with scaled residuals
            combined_prediction = ss_bootstrap + residual_bootstrap * np.exp(-0.5 * np.arange(future_years))
            all_predictions.append(combined_prediction)
        
        predictions_array = np.array(all_predictions)
        final_forecast = np.median(predictions_array, axis=0)
        lower_bound = np.percentile(predictions_array, 10, axis=0)  # More conservative bounds
        upper_bound = np.percentile(predictions_array, 90, axis=0)
        
        forecast_df = pd.DataFrame({
            'Year': range(
                int(self.df['Year'].max() + 1),
                int(self.df['Year'].max() + future_years + 1)
            ),
            'Forecast': final_forecast,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'SS_Component': ss_forecast,
            'Residual_Component': final_forecast - ss_forecast
        })
        
        return forecast_df

    def get_model_diagnostics(self):
        """Calculate comprehensive model diagnostics"""
        # In-sample predictions
        y_pred = self.ss_predictions + self.residual_model.predict(
            self.scaler.transform(self.prepare_features())
        )
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(self.df[self.target_col], y_pred)
        rmse = np.sqrt(mean_squared_error(self.df[self.target_col], y_pred))
        
        # Calculate prediction intervals coverage
        residuals = self.df[self.target_col] - y_pred
        std_residuals = np.std(residuals)
        coverage_80 = np.mean(
            np.abs(residuals) <= 1.28 * std_residuals
        )
        coverage_95 = np.mean(
            np.abs(residuals) <= 1.96 * std_residuals
        )
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'R2': self.residual_model.score(
                self.scaler.transform(self.prepare_features()),
                self.residuals
            ),
            'Coverage_80': coverage_80,
            'Coverage_95': coverage_95,
            'Residual_Autocorr': np.corrcoef(residuals[:-1], residuals[1:])[0,1]
        }

if __name__ == "__main__":
    # Load and process data using original DataProcessor
    df = pd.read_csv("total.csv", sep=";")
    proc = DataProcessor()
    data = proc._clean_dataframes(df)
    data = proc._post_process_data(data)

    # Initialize and fit the model
    predictor = EnhancedFundingPredictor(data)
    predictor.fit()

    # Make predictions
    forecast = predictor.predict(future_years=2)
    print("\nForecast:")
    print(forecast)

    # Get model diagnostics
    diagnostics = predictor.get_model_diagnostics()
    print("\nModel Diagnostics:")
    print(diagnostics)

 