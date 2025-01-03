import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple
import numpy as np
from mc import MonteCarloSimulator, MultiComponentNoise, NoiseParameters, HybridPredictorWrapper
from weighted_growth import WeightedTrendPredictorWrapper
from processor import DataProcessor

class StreamlitMonteCarloApp:
    """Enhanced Streamlit app with model selection and parameter controls"""
    
    def __init__(self):
        st.set_page_config(page_title="Monte Carlo Forecast", layout="wide")
        self.available_models = {
            'Enhanced Hybrid Predictor': HybridPredictorWrapper,
            'Weighted Trend Predictor': WeightedTrendPredictorWrapper
        }
        
    def create_model_controls(self) -> Dict:
        """Create controls for model selection and parameters"""
        st.sidebar.header("Model Configuration")
        
        # Model selection
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=list(self.available_models.keys())
        )
        
        uploaded_files = st.sidebar.file_uploader(
            "Upload 4 CSV files",
            type=['csv'],
            accept_multiple_files=True
            )
                    
        # Simulation parameters
        st.sidebar.subheader("Simulation Parameters")
        future_periods = st.sidebar.slider(
            "Future Periods",
            min_value=1,
            max_value=10,
            value=3
        )
        
        n_simulations = st.sidebar.slider(
            "Number of Simulations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
        
        # Noise parameters
        st.sidebar.subheader("Noise Parameters")
        base_volatility = st.sidebar.slider(
            "Base Volatility",
            min_value=0.01,
            max_value=0.5,
            value=0.15,
            step=0.01
        )
        
        trend_volatility = st.sidebar.slider(
            "Trend Volatility",
            min_value=0.01,
            max_value=0.3,
            value=0.08,
            step=0.01
        )
        
        seasonality_volatility = st.sidebar.slider(
            "Seasonality Volatility",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01
        )
            # Add growth factor control
        growth_factor = st.sidebar.slider(
            "Annual Growth Factor",
            min_value=-0.10,
            max_value=0.30,
            value=0.02,
            step=0.01,
            help="Expected annual growth rate (e.g., 0.02 = 2% growth)"
        )
        
        # Random seed
        seed = st.sidebar.number_input(
            "Random Seed",
            value=42,
            min_value=0
        )
        
        return {
            'selected_model': selected_model,
            'uploaded_files': uploaded_files,
            'future_periods': future_periods,
            'n_simulations': n_simulations,
            'base_volatility': base_volatility,
            'trend_volatility': trend_volatility,
            'seasonality_volatility': seasonality_volatility,
            'growth_factor': growth_factor,
            'seed': seed
        }
    
    def create_visualization_controls(self) -> Dict:
        """Create controls for visualization customization"""
        st.sidebar.header("Visualization Controls")
        
        return {
            'show_95_ci': st.sidebar.checkbox(
                "Show 95% Confidence Interval",
                value=True
            ),
            'show_50_ci': st.sidebar.checkbox(
                "Show 50% Confidence Interval",
                value=True
            ),
            'plot_height': st.sidebar.slider(
                "Plot Height",
                min_value=400,
                max_value=1000,
                value=600,
                step=50
            ),
            'line_width': st.sidebar.slider(
                "Line Width",
                min_value=1,
                max_value=5,
                value=2,
                step=1
            )
        }
    
    def run_simulation(self, model_config: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Run Monte Carlo simulation with selected parameters"""
        file_mapping = {}
        try:
            uploaded_files = model_config['uploaded_files']
 
            
            # Read the uploaded file
            #print("file mapping",file_mapping)
            processor = DataProcessor()
            data = processor.process_funding_data(uploaded_files)
            #print("Data loaded successfully:", data.shape)
            
            # Initialize the selected model
            ModelClass = self.available_models[model_config['selected_model']]
            print(f"ModelClass type: {type(ModelClass)}")  # Debug model class type
            
            model = ModelClass(data)
            print(f"Model instance type: {type(model)}")  # Debug model instance type
            print(f"Model attributes: {dir(model)}")      # Debug available methods/attributes
            
            print("About to fit model...")
            model.fit()
            print("Model fitted successfully")
            
            # Debug model state after fitting
            print(f"Model after fit - has df?: {'df' in dir(model)}")
            if hasattr(model, 'df'):
                print(f"Model df type: {type(model.df)}")
            
            # Create noise parameters with debug
            print("Creating noise parameters...")
            noise_params = NoiseParameters(
                base_volatility=model_config['base_volatility'],
                trend_volatility=model_config['trend_volatility'],
                seasonality_volatility=model_config['seasonality_volatility'],
                growth_factor=model_config['growth_factor']  # Add growth factor
            )
            print(f"Noise params type: {type(noise_params)}")
            
            print("Creating noise generator...")
            noise_generator = MultiComponentNoise(noise_params)
            print(f"Noise generator type: {type(noise_generator)}")
            
            # Debug simulator creation
            print("About to create simulator...")
            simulator = MonteCarloSimulator(
                model=model,
                noise_generator=noise_generator,
                seed=model_config['seed']
            )
            print(f"Simulator type: {type(simulator)}")
            print(f"Simulator attributes: {dir(simulator)}")
            
            # Debug simulation run
            print(f"About to run simulation with periods={model_config['future_periods']}, n_sims={model_config['n_simulations']}")
            try:
                forecast = simulator.run_simulation(
                    future_periods=model_config['future_periods'],
                    n_simulations=model_config['n_simulations']
                )
                print("Simulation completed successfully")
            except Exception as sim_error:
                print(f"Simulation error details: {str(sim_error)}")
                print(f"Simulation error type: {type(sim_error)}")
                raise  # Re-raise to be caught by outer try/except
                
            return model.df, forecast
                
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Error type: {type(e)}")
            # Get full traceback
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            st.error(f"Error during simulation: {str(e)}")
            return None, None
        
    def run_app(self):
        """Main application loop"""
        st.title("Monte Carlo Simulation Dashboard")
        
        # Get model and visualization configurations
        model_config = self.create_model_controls()
        viz_controls = self.create_visualization_controls()
        
        # Add a "Run Simulation" button
        if st.sidebar.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                historical_data, forecast = self.run_simulation(model_config)
                
                if historical_data is not None and forecast is not None:
                    # Create visualization
                    visualizer = StreamlitMonteCarloVisualizer()
                    
                    # Plot forecast
                    visualizer.plot_forecast(
                        historical_data=historical_data,
                        forecast_df=forecast,
                        date_column='Year',
                        target_column='Total Funding',
                        controls=viz_controls
                    )
                    
                    # Plot distribution
                    visualizer.plot_simulation_distribution(
                        forecast_df=forecast,
                        target_column='Total Funding'
                    )
                    
                    # Add download button for results
                    st.download_button(
                        label="Download Forecast Data",
                        data=forecast.to_csv(index=False),
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )

class StreamlitMonteCarloVisualizer:
    """Handles visualization of Monte Carlo simulation results"""
    
    def plot_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        controls: Dict
    ) -> None:
        """Create interactive visualization of Monte Carlo forecast"""
        
        st.subheader(f"Monte Carlo Forecast Analysis: {target_column}")
        
        # Create the base plot using plotly
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data[date_column],
                y=historical_data[target_column],
                name="Historical Data",
                line=dict(color="blue", width=controls['line_width'])
            )
        )
        
        # Add forecast mean
        fig.add_trace(
            go.Scatter(
                x=forecast_df[date_column],
                y=forecast_df['MC_mean'],
                name="Forecast Mean",
                line=dict(color="red", width=controls['line_width'], dash='dash')
            )
        )
        
        # Add 95% confidence interval
        if controls['show_95_ci']:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df[date_column],
                    y=forecast_df['MC_upper_95'],
                    mode='lines',
                    name='95% CI Upper',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df[date_column],
                    y=forecast_df['MC_lower_95'],
                    mode='lines',
                    name='95% Confidence Interval',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)'
                )
            )
        
        # Add 50% confidence interval
        if controls['show_50_ci']:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df[date_column],
                    y=forecast_df['MC_upper_50'],
                    mode='lines',
                    name='50% CI Upper',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df[date_column],
                    y=forecast_df['MC_lower_50'],
                    mode='lines',
                    name='50% Confidence Interval',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Monte Carlo Forecast with Uncertainty Bands<br>Target: {target_column}",
            xaxis_title=date_column,
            yaxis_title=target_column,
            height=controls['plot_height'],
            hovermode='x unified',
            showlegend=True,
            template="plotly_white"
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Add statistics section
        st.subheader("Forecast Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Latest Values")
            latest_stats = {
                "Last Historical Value": historical_data[target_column].iloc[-1],
                "Final Forecast (Mean)": forecast_df['MC_mean'].iloc[-1],
                "Final 95% CI": f"({forecast_df['MC_lower_95'].iloc[-1]:.2f}, {forecast_df['MC_upper_95'].iloc[-1]:.2f})"
            }
            st.table(pd.Series(latest_stats))
        
        with col2:
            st.markdown("### Growth Statistics")
            total_growth = (forecast_df['MC_mean'].iloc[-1] / historical_data[target_column].iloc[-1] - 1) * 100
            avg_growth = (total_growth / len(forecast_df)) if len(forecast_df) > 0 else 0
            
            growth_stats = {
                "Total Forecasted Growth": f"{total_growth:.2f}%",
                "Average Period Growth": f"{avg_growth:.2f}%",
                "Forecast Periods": len(forecast_df)
            }
            st.table(pd.Series(growth_stats))

    def plot_simulation_distribution(
        self,
        forecast_df: pd.DataFrame,
        target_column: str,
        period: Optional[int] = None
    ) -> None:
        """Plot distribution of simulation results for a specific period"""
        
        if period is None:
            period = len(forecast_df) - 1
            
        st.subheader(f"Simulation Distribution (Period {period + 1})")
        
        # Create histogram of simulation results
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=forecast_df.iloc[period],
                nbinsx=50,
                name="Distribution"
            )
        )
        
        fig.update_layout(
            title=f"Distribution of Simulated Values for Period {period + 1}",
            xaxis_title=target_column,
            yaxis_title="Frequency",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    app = StreamlitMonteCarloApp()
    app.run_app()