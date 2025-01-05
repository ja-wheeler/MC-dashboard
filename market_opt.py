from opt import EnhancedFundingPredictor
from ltsm import DealSizePredictor
import pandas as pd
import logging 
from processor import DataProcessor

class EnhancedFundingPredictorWithMarket(EnhancedFundingPredictor):
    def __init__(self, data, deal_size_model_path=None, target_col='Total Funding'):
        super().__init__(data, target_col)
        self.deal_predictor = None
        if deal_size_model_path:
            self.deal_predictor = DealSizePredictor()
            self.deal_predictor.load(deal_size_model_path)
        self.market_feature_names = ['Market_Signal', 'Market_Signal_Ratio', 'Market_Signal_Growth']
            
    def add_market_features(self, df, is_future=False):
        """Add market signal features from DealSizePredictor"""
        features = df.copy()
        
        if self.deal_predictor:
            try:
                if is_future:
                    # For future predictions, use the last sequence from historical data
                    last_sequence = self.deal_predictor.scaler.transform(
                        self.df[['Total Funding', 'Number of Rounds']]
                        .iloc[-self.deal_predictor.sequence_length:].values
                    )
                    pred = self.deal_predictor.predict_with_intervals(
                        last_sequence, 
                        n_future=len(df)
                    )
                    market_signals = pred['mean']
                else:
                    # Historical predictions
                    market_signals = []
                    for i in range(len(df) - self.deal_predictor.sequence_length):
                        sequence_data = self.deal_predictor.scaler.transform(
                            df[['Total Funding', 'Number of Rounds']]
                            .iloc[i:i + self.deal_predictor.sequence_length].values
                        )
                        pred = self.deal_predictor.predict_with_intervals(sequence_data, n_future=1)
                        market_signals.append(pred['mean'][0])
                    
                    # Pad the initial years
                    padding = [market_signals[0]] * self.deal_predictor.sequence_length
                    market_signals = padding + market_signals
                
                # Add features
                features['Market_Signal'] = market_signals
                features['Market_Signal_Ratio'] = features['Market_Signal'] / features['Total Funding']
                features['Market_Signal_Growth'] = pd.Series(market_signals).pct_change().fillna(0)
                
            except Exception as e:
                logging.warning(f"Could not add market features: {str(e)}")
                # Add zero-filled market features if prediction fails
                for feature in self.market_feature_names:
                    features[feature] = features[feature] if feature in features else 0.0
                
        return features
    
    def prepare_features(self, df=None):
        """Override prepare_features to include market features"""
        if df is None:
            df = self.df
            
        # First add engineered features
        features = self.add_engineered_features(df)
        
        # Then add market features with appropriate flag
        is_future = df is not None and len(df) != len(self.df)
        features = self.add_market_features(features, is_future=is_future)
        
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

if __name__=="__main__":
    df = pd.read_csv("total.csv", sep=";")
    proc = DataProcessor()
    data = proc._clean_dataframes(df)
    data = proc._post_process_data(data)

    # Initialize and fit the model with market features
    predictor = EnhancedFundingPredictorWithMarket(
        data, 
        deal_size_model_path='models/'
    )
    predictor.fit()

    # Make predictions
    forecast = predictor.predict(future_years=2)
    print("\nForecast:")
    print(forecast)