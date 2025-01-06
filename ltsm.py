import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging
import os
import json
# Add these imports at the top
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DealSizePredictor:
    def __init__(self, model_dir='models', logs_dir='logs'):
        """
        Initialize the predictor with model and logging directories.
        
        Args:
            model_dir (str): Directory to save model artifacts
            logs_dir (str): Directory to save logs
        """
        self.model = None
        self.scaler = RobustScaler()
        self.sequence_length = 5
        self.recent_years_cutoff = 2000
        
        # Create directories if they don't exist
        self.model_dir = Path(model_dir)
        self.logs_dir = Path(logs_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Model metadata
        self.metadata = {
            'creation_date': None,
            'last_training_date': None,
            'input_features': None,
            'performance_metrics': None,
            'data_stats': None
        }
    
    def setup_logging(self):
        """Configure logging with both file and console handlers."""
        self.logger = logging.getLogger('DealSizePredictor')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.logs_dir / 'model.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def prepare_data(self, file_path):
        """
        Prepare and validate data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            tuple: Scaled data and yearly deals DataFrame
        """
        print("HELLO")
        try:

            # Read CSV
            df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')

            print(df.columns)
            
            # Validate required columns
            required_cols = ['Deal Date', 'Deal Size']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Need: {required_cols}")
            
            # Convert Deal Date and Deal Size
            df['Deal Date'] = pd.to_datetime(df['Deal Date'], errors='coerce')
            df['Deal Size'] = pd.to_numeric(df['Deal Size'].astype(str).str.replace(',', '.'), 
                                          errors='coerce')
            
            # Check for sufficient data
            if len(df) < 100:
                self.logger.warning("Dataset might be too small for reliable predictions")
            
            # Group by year
            yearly_deals = df.groupby(df['Deal Date'].dt.year).agg({
                'Deal Size': ['mean', 'count', 'std']
            }).reset_index()
            
            yearly_deals.columns = ['Year', 'Average_Deal_Size', 'Deal_Count', 'Deal_Size_Std']
            yearly_deals = yearly_deals[yearly_deals['Year'] >= self.recent_years_cutoff].copy()
            
            # Store data statistics
            self.metadata['data_stats'] = {
                'n_records': int(len(df)),
                'year_range': [
                    int(yearly_deals['Year'].min()),
                    int(yearly_deals['Year'].max())
                ],
                'avg_deals_per_year': float(yearly_deals['Deal_Count'].mean()),
                'total_deal_count': int(yearly_deals['Deal_Count'].sum())
            }
            
            # Handle missing values
            yearly_deals = yearly_deals.fillna(method='ffill').fillna(method='bfill')
            yearly_deals = yearly_deals.fillna(0)
            
            # Scale features
            scaled_data = self.scaler.fit_transform(yearly_deals[['Average_Deal_Size', 
                                                                'Deal_Count']])
            
            self.logger.info("Data preparation completed successfully")
            return scaled_data, yearly_deals
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def create_sequences(self, data):
        """
        Create sequences for time series prediction with validation.
        
        Args:
            data (np.array): Input data array
            
        Returns:
            tuple: X sequences, y values, and sample weights
        """
        try:
            X, y = [], []
            n_sequences = len(data) - self.sequence_length
            
            if n_sequences < 10:
                self.logger.warning("Very few sequences available for training")
            
            weights = np.exp(np.linspace(-1, 0, n_sequences))
            
            for i in range(n_sequences):
                X.append(data[i:(i + self.sequence_length)])
                y.append(data[i + self.sequence_length, 0])
                
            return np.array(X), np.array(y), weights
            
        except Exception as e:
            self.logger.error(f"Error in sequence creation: {str(e)}")
            raise
    
    def build_model(self, input_shape):
        """
        Build and compile the neural network model.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            keras.Model: Compiled model
        """
        try:
            model = Sequential([
                Flatten(input_shape=input_shape),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='huber')
            
            self.metadata['creation_date'] = datetime.now().isoformat()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error in model building: {str(e)}")
            raise



    # Add this method to the DealSizePredictor class
    def plot_results(self, yearly_deals, forecast_df, save_path=None):
        """
        Create visualizations for historical data and predictions.
        
        Args:
            yearly_deals (pd.DataFrame): Historical yearly deals data
            forecast_df (pd.DataFrame): Forecasted values with confidence intervals
            save_path (str, optional): Path to save the plots
        """
        try:
            # Set the style
            plt.style.use('Solarize_Light2')
            
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)
            
            # 1. Historical and Predicted Deal Sizes
            ax1 = fig.add_subplot(gs[0, :])
            
            # Plot historical data
            ax1.plot(yearly_deals['Year'], yearly_deals['Average_Deal_Size'], 
                    marker='o', label='Historical', color='blue')
            
            # Plot predictions with confidence interval
            ax1.plot(forecast_df['Year'], forecast_df['Predicted_Deal_Size'], 
                    marker='s', label='Predicted', color='red', linestyle='--')
            ax1.fill_between(forecast_df['Year'], 
                            forecast_df['Lower_Bound'],
                            forecast_df['Upper_Bound'],
                            alpha=0.2, color='red',
                            label='95% Confidence Interval')
            
            ax1.set_title('Historical and Predicted Deal Sizes Over Time')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Average Deal Size')
            ax1.legend()
            ax1.grid(True)
            
            # 2. Deal Count Over Time
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.bar(yearly_deals['Year'], yearly_deals['Deal_Count'], 
                    color='skyblue', alpha=0.7)
            ax2.set_title('Number of Deals per Year')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Deal Count')
            ax2.grid(True)
            
            # 3. Deal Size Distribution
            ax3 = fig.add_subplot(gs[1, 1])
            sns.histplot(data=yearly_deals, x='Average_Deal_Size', bins=15, 
                        kde=True, ax=ax3, color='skyblue')
            ax3.set_title('Distribution of Average Deal Sizes')
            ax3.set_xlabel('Average Deal Size')
            ax3.set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error in plotting results: {str(e)}")
            raise
    
    def train(self, X_train, y_train, sample_weights, validation_split=0.2, 
            epochs=200, batch_size=32):
        """
        Train the model with cross-validation and early stopping.
        """
        try:
            if len(sample_weights) != len(X_train):
                raise ValueError("Sample weights length must match training samples")
            
            # Updated checkpoint path to use .keras extension
            checkpoint = ModelCheckpoint(
                self.model_dir / 'model.keras',  # Changed from .h5 to .keras
                monitor='val_loss',
                save_best_only=True
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=100,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )
            
            self.metadata['last_training_date'] = datetime.now().isoformat()
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict_with_intervals(self, last_sequence, n_future=6, n_simulations=100):
        """
        Make predictions with confidence intervals using Monte Carlo dropout.
        
        Args:
            last_sequence (np.array): Input sequence for prediction
            n_future (int): Number of future periods to predict
            n_simulations (int): Number of Monte Carlo simulations
            
        Returns:
            dict: Predictions with confidence intervals
        """
        try:
            predictions = []
            current_sequence = last_sequence.copy()
            
            # Run multiple predictions with dropout enabled
            for _ in range(n_simulations):
                pred_sequence = current_sequence.copy()
                future_preds = []
                
                for _ in range(n_future):
                    pred = self.model(pred_sequence.reshape(1, self.sequence_length, 2), 
                                    training=True)
                    future_preds.append(pred.numpy()[0, 0])
                    
                    # Update sequence
                    pred_sequence = np.roll(pred_sequence, -1, axis=0)
                    pred_sequence[-1] = [pred.numpy()[0, 0], pred_sequence[-1, 1]]
                
                predictions.append(future_preds)
            
            predictions = np.array(predictions)
            
            # Calculate mean and confidence intervals
            mean_preds = np.mean(predictions, axis=0)
            lower_bound = np.percentile(predictions, 5, axis=0)
            upper_bound = np.percentile(predictions, 95, axis=0)
            
            # Transform back to original scale
            mean_preds = self.scaler.inverse_transform(
                np.column_stack([mean_preds, np.zeros_like(mean_preds)]))[:, 0]
            lower_bound = self.scaler.inverse_transform(
                np.column_stack([lower_bound, np.zeros_like(lower_bound)]))[:, 0]
            upper_bound = self.scaler.inverse_transform(
                np.column_stack([upper_bound, np.zeros_like(upper_bound)]))[:, 0]
            
            return {
                'mean': mean_preds,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def save(self, path):
        """Save the model and its metadata."""
        try:
            save_path = Path(path)
            save_path.mkdir(exist_ok=True)
            
            # Save model
            save_model(self.model, save_path / 'model.keras')
            
            # Save scaler
            np.save(save_path / 'scaler.npy', 
                [self.scaler.center_, self.scaler.scale_])
            
            # Convert numpy types to Python types for JSON serialization
            metadata_copy = self.metadata.copy()
            if metadata_copy['data_stats']:
                metadata_copy['data_stats'] = {
                    k: (int(v) if isinstance(v, np.integer) 
                    else float(v) if isinstance(v, np.floating)
                    else [int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x for x in v]
                    if isinstance(v, (list, np.ndarray))
                    else v)
                    for k, v in metadata_copy['data_stats'].items()
                }
            
            # Save metadata
            with open(save_path / 'metadata.json', 'w') as f:
                json.dump(metadata_copy, f)
            
            self.logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path):
        """Load the model and its metadata."""
        try:
            load_path = Path(path)
            
            # Load model with .keras extension
            self.model = load_model(load_path / 'model.keras')  # Changed from .h5 to .keras
            
            # Load scaler
            scaler_params = np.load(load_path / 'scaler.npy')
            self.scaler.center_ = scaler_params[0]
            self.scaler.scale_ = scaler_params[1]
            
            # Load metadata
            with open(load_path / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            self.logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    """Main function to demonstrate usage."""
    try:
        # Initialize predictor
        predictor = DealSizePredictor()
        
        # Prepare data
        scaled_data, yearly_deals = predictor.prepare_data('pitchbook.csv')
        
        # Create sequences
        X, y, weights = predictor.create_sequences(scaled_data)
        
        # Build and train model
        predictor.model = predictor.build_model((predictor.sequence_length, 2))
        history = predictor.train(X, y, weights)
        
        # Make predictions with confidence intervals
        last_sequence = scaled_data[-predictor.sequence_length:]
        predictions = predictor.predict_with_intervals(last_sequence)
        
        # Create forecast DataFrame
        forecast_years = pd.date_range(
            start=str(int(yearly_deals['Year'].max()) + 1),
            periods=6,
            freq='YE'
        ).year
        
        forecast_df = pd.DataFrame({
            'Year': forecast_years,
            'Predicted_Deal_Size': predictions['mean'],
            'Lower_Bound': predictions['lower_bound'],
            'Upper_Bound': predictions['upper_bound']
        })
        
        # Save model
        predictor.save('models')
        
        # Print results
        print("\nHistorical Data:")
        print(yearly_deals)
        print("\nForecast with Confidence Intervals:")
        print(forecast_df)

                 # Add visualization
        predictor.plot_results(
            yearly_deals,
            forecast_df,
            save_path='deal_size_analysis.png'
        )

        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()