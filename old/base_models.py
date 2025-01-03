import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FundingForecaster:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.forecast_columns = [
            'Seed Stage Rounds', 'Early Stage Rounds', 'Late Stage Rounds',
            'Total Funding', 'Number of Rounds', 'Total Number of Companies',
            'total_Seed Stage Funding', 'total_Early Stage Funding', 'total_Late Stage Funding'
        ]
        self.models = {
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_models = {}
        self.forecasts = {}

    def engineer_features(self):
        """Create additional features for better forecasting"""
        self.df['Year_Squared'] = self.df['Year'] ** 2
        self.df['Year_Cubed'] = self.df['Year'] ** 3
        
        # Add cyclical features
        self.df['Trend'] = np.arange(len(self.df))
        self.df['Cycle'] = np.sin(2 * np.pi * self.df['Trend'] / 12)
        
        return self.df.copy()

    def detect_outliers(self, column):
        """Detect and handle outliers using IQR method"""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.df[column] = self.df[column].clip(lower_bound, upper_bound)

    def select_best_model(self, X, y, column):
        """Select the best performing model using cross-validation"""
        best_score = float('-inf')
        best_model = None
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                
        return best_model

    def calculate_confidence_intervals(self, model, X_future, X_train, y_train):
        """Calculate confidence intervals for forecasts"""
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        std_err = np.sqrt(mse)
        
        n = len(y_train)
        dof = n - 2
        t_value = stats.t.ppf(0.975, dof)
        
        ci = t_value * std_err
        
        forecast = model.predict(X_future)
        lower_bound = forecast - ci
        upper_bound = forecast + ci
        
        return lower_bound, forecast, upper_bound

    def fit_and_forecast(self, future_years=5):
        """Fit models and generate forecasts with confidence intervals"""
        engineered_df = self.engineer_features()
        
        feature_columns = ['Year', 'Year_Squared', 'Year_Cubed', 'Trend', 'Cycle']
        X = engineered_df[feature_columns].values
        
        # Fix: Convert max year to integer before using in range
        last_year = int(engineered_df['Year'].max())
        future_years_range = np.array(range(
            last_year + 1,
            last_year + future_years + 1
        ))
        
        # Create future feature matrix
        X_future = np.column_stack([
            future_years_range,
            future_years_range ** 2,
            future_years_range ** 3,
            np.arange(len(X), len(X) + len(future_years_range)),
            np.sin(2 * np.pi * np.arange(len(X), len(X) + len(future_years_range)) / 12)
        ])

        for column in self.forecast_columns:
            # Handle outliers
            self.detect_outliers(column)
            
            y = engineered_df[column].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_future_scaled = self.scaler.transform(X_future)
            
            # Select and fit best model
            best_model = self.select_best_model(X_scaled, y, column)
            best_model.fit(X_scaled, y)
            
            # Calculate confidence intervals
            lower_bound, forecast, upper_bound = self.calculate_confidence_intervals(
                best_model, X_future_scaled, X_scaled, y
            )
            
            self.forecasts[column] = {
                'historical': y,
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'years': future_years_range
            }
            
            self.best_models[column] = best_model

    def plot_interactive_forecast(self, column_name):
        """Create interactive plot using plotly"""
        historical_years = self.df['Year'].values
        forecast_data = self.forecasts[column_name]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_years,
            y=forecast_data['historical'],
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['years'],
            y=forecast_data['forecast'],
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_data['years'].tolist() + forecast_data['years'].tolist()[::-1],
            y=forecast_data['upper_bound'].tolist() + forecast_data['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f'{column_name} - Historical and Forecast with Confidence Intervals',
            xaxis_title='Year',
            yaxis_title='Value',
            hovermode='x'
        )
        
        return fig

    def export_results(self, output_path='forecast_results.xlsx'):
        """Export forecasts and metrics to Excel"""
        with pd.ExcelWriter(output_path) as writer:
            # Export historical data
            self.df.to_excel(writer, sheet_name='Historical_Data', index=False)
            
            # Export forecasts
            forecast_df = pd.DataFrame()
            for column in self.forecast_columns:
                forecast_data = self.forecasts[column]
                temp_df = pd.DataFrame({
                    'Year': forecast_data['years'],
                    f'{column}_Forecast': forecast_data['forecast'],
                    f'{column}_Lower_Bound': forecast_data['lower_bound'],
                    f'{column}_Upper_Bound': forecast_data['upper_bound']
                })
                if forecast_df.empty:
                    forecast_df = temp_df
                else:
                    forecast_df = forecast_df.merge(temp_df, on='Year')
            
            forecast_df.to_excel(writer, sheet_name='Forecasts', index=False)

# Usage example
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = FundingForecaster('data.csv')
    
    # Fit models and generate forecasts
    forecaster.fit_and_forecast(future_years=5)
    
    # Create interactive plots
    for column in forecaster.forecast_columns:
        fig = forecaster.plot_interactive_forecast(column)
        fig.show()
    
    # Export results
    forecaster.export_results()