import pandas as pd
from processor import DataProcessor
from typing import Optional
from opt import EnhancedFundingPredictor
from market_opt import EnhancedFundingPredictorWithMarket


class NnHybridPredictorWrapper:
    """Wrapper class for EnhancedFundingPredictorWithMarket to make it compatible with MC simulation"""
    def __init__(self, data: pd.DataFrame):
        # Initialize DealSizePredictor
        self.deal_predictor = DealSizePredictor()
        self.deal_predictor.load('models/')  # Using the same path as in the original code
        
        # Initialize the actual predictor with DataFrame directly
        self.model = EnhancedFundingPredictorWithMarket(
            data=data,
            deal_size_model_path='models/',
            target_col='Total Funding'
        )
        
        # Store the dataframe for visualization
        self.df = data

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, future_periods: int, **kwargs) -> pd.DataFrame:
        return self.model.predict(future_years=future_periods)
    

class EnhancedPredictorWrapper:
    """Wrapper class for EnhancedFundingPredictor to make it compatible with MC simulation"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 target_col: str = 'Total Funding',
                 processor: Optional[DataProcessor] = None):
        """
        Initialize wrapper with data preprocessing
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw input data
        target_col : str, optional
            Target column for predictions
        processor : DataProcessor, optional
            Custom data processor instance
        """
        # Initialize data processor
        self.processor = processor if processor is not None else DataProcessor()
        
        # Clean and process data
        cleaned_data = self.processor._clean_dataframes(data)
        processed_data = self.processor._post_process_data(cleaned_data)
        
        # Initialize predictor with processed data
        self.model = EnhancedFundingPredictor(
            data=processed_data,
            target_col=target_col
        )
        
        # Store processed data
        self.df = processed_data
    
    def fit(self, *args, **kwargs):
        """Fit the underlying model"""
        self.model.fit(*args, **kwargs)
        return self
    
    def predict(self, future_periods: int, **kwargs) -> pd.DataFrame:
        """
        Generate predictions
        
        Parameters:
        -----------
        future_periods : int
            Number of periods to forecast
        """
        return self.model.predict(future_years=future_periods)
    
    def get_diagnostics(self) -> dict:
        """Get model performance metrics"""
        return self.model.get_model_diagnostics()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance analysis"""
        return self.model.get_feature_importance()
    

from HybridPredictor import HybridPredictor
from ltsm import DealSizePredictor

class HybridPredictorWrapper:
    """Wrapper class for EnhancedHybridPredictor to make it compatible with MC simulation"""
    def __init__(self, data: pd.DataFrame):  # Changed from data_path: str
        # Initialize DealSizePredictor
        self.deal_predictor = DealSizePredictor()
        self.deal_predictor.load('production_model')
        
        # Initialize the actual predictor with DataFrame directly
        self.model = HybridPredictor(
            data=data,  # Pass DataFrame instead of path
            deal_predictor=self.deal_predictor,
            target_col='Total Funding'
        )
        
        # Store the dataframe for visualization
        self.df = data  # Store the input DataFrame directly

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, future_periods: int, **kwargs) -> pd.DataFrame:
        return self.model.predict(future_years=future_periods)