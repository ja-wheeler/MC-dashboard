import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
from processor import DataProcessor  # Keep the original data processor
warnings.filterwarnings('ignore')

class EnhancedFundingPredictor:
    def __init__(self, data, target_col='Total Funding'):
        """
        Initialize the enhanced predictor while maintaining original data processing
        
        Parameters:
        data: DataFrame or string (path to CSV)
        target_col: string, column name for the target variable
        """
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
        
        # Define funding types as in original
        self.funding_types = ['Seed Stage Funding', 'Early Stage Funding', 
                            'Late Stage Funding']
        
    def calculate_funding_concentration(self, row):
        """Calculate Herfindahl index for funding concentration"""
        values = [row[col] for col in self.funding_types]
        total = sum(values)
        if total == 0:
            return 0
        shares = [v/total for v in values]
        return sum(share * share for share in shares if share > 0)

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
        """Prepare features maintaining original structure"""
        if df is None:
            df = self.df
            
        features = self.add_engineered_features(df)
        
        # Select numerical columns only
        numerical_cols = features.select_dtypes(
            include=['int64', 'float64']
        ).columns
        numerical_cols = [col for col in numerical_cols 
                         if col != self.target_col and col != 'Year']
        
        # Store feature names for consistency
        if not hasattr(self, 'feature_names'):
            self.feature_names = numerical_cols
        
        return features[self.feature_names]
    
    def fit(self):
        """Fit enhanced hybrid model"""
        print("Fitting enhanced hybrid model...")
        
        # Improved ETS model with robust settings
        self.ets_model = ExponentialSmoothing(
            self.df[self.target_col],
            seasonal_periods=2,  # Reduced seasonality for small dataset
            trend='additive',    # More stable for volatile data
            seasonal='additive', 
            initialization_method='estimated'
        )
        self.ets_results = self.ets_model.fit()
        
        # Get ETS predictions and residuals
        self.ets_predictions = self.ets_results.fittedvalues
        self.residuals = self.df[self.target_col] - self.ets_predictions
        
        # Prepare features for Random Forest
        X = self.prepare_features()
        print(f"Training features shape: {X.shape}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Enhanced Random Forest with robust settings
        self.residual_model = RandomForestRegressor(
            n_estimators=50,     # Reduced to prevent overfitting
            max_depth=3,         # Simplified tree structure
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
        
        # Get ETS forecast
        ets_forecast = self.ets_results.forecast(future_years)
        
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
        
        # Prepare features for residual prediction
        future_X = self.prepare_features(future_df)
        future_X_scaled = self.scaler.transform(future_X)
        
        # Enhanced bootstrap predictions
        n_bootstraps = 200
        all_predictions = []
        
        for _ in range(n_bootstraps):
            # Bootstrap ETS component
            ets_bootstrap = (ets_forecast * 
                           np.random.normal(1, 0.1, size=len(ets_forecast)))
            
            # Bootstrap residual predictions
            indices = np.random.randint(
                0, self.residual_model.n_estimators,
                size=self.residual_model.n_estimators
            )
            bootstrap_predictions = []
            
            for idx in indices:
                prediction = self.residual_model.estimators_[idx].predict(
                    future_X_scaled
                )
                bootstrap_predictions.append(prediction)
            
            residual_bootstrap = np.mean(bootstrap_predictions, axis=0)
            all_predictions.append(ets_bootstrap + residual_bootstrap)
        
        # Calculate robust forecasts and intervals
        predictions_array = np.array(all_predictions)
        final_forecast = np.median(predictions_array, axis=0)
        lower_bound = np.percentile(predictions_array, 5, axis=0)
        upper_bound = np.percentile(predictions_array, 95, axis=0)
        
        # Create output dataframe
        forecast_df = pd.DataFrame({
            'Year': range(
                int(self.df['Year'].max() + 1),
                int(self.df['Year'].max() + future_years + 1)
            ),
            'Forecast': final_forecast,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'ETS_Component': ets_forecast,
            'Residual_Component': final_forecast - ets_forecast
        })
        
        return forecast_df
    
    def get_feature_importance(self):
        """Get robust feature importance scores"""
        importance_scores = []
        
        # Calculate feature importance across multiple random seeds
        for seed in range(10):
            rf = RandomForestRegressor(
                n_estimators=50,
                random_state=seed
            )
            rf.fit(
                self.scaler.transform(self.prepare_features()),
                self.residuals
            )
            importance_scores.append(rf.feature_importances_)
        
        # Average importance scores
        mean_importance = np.mean(importance_scores, axis=0)
        std_importance = np.std(importance_scores, axis=0)
        
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_importance,
            'Std': std_importance
        }).sort_values('Importance', ascending=False)
    
    def get_model_diagnostics(self):
        """Calculate comprehensive model diagnostics"""
        # In-sample predictions
        y_pred = self.ets_predictions + self.residual_model.predict(
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

    # Get feature importance
    importance = predictor.get_feature_importance()
    print("\nFeature Importance:")
    print(importance)