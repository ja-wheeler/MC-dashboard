import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple
import numpy as np
from mc import MonteCarloSimulator, AdaptiveNoiseGenerator, MarketNoiseParameters, calibrate_parameters
from processor import DataProcessor
from wrappers import EnhancedPredictorWrapper, NnHybridPredictorWrapper

class StreamlitMonteCarloApp:
    """Enhanced Streamlit app with model selection and parameter controls"""
    
    def __init__(self):
        st.set_page_config(page_title="Monte Carlo Forecast", layout="wide")
        self.available_models = {
            'Base model': EnhancedPredictorWrapper, 
            'Market adjusted model': NnHybridPredictorWrapper
        }
        
    def create_model_controls(self) -> Dict:
        """Create controls for model selection and parameters"""
        st.sidebar.header("Model Configuration")
        
        # Keep model selection and file upload
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=list(self.available_models.keys())
        )
        
        uploaded_files = st.sidebar.file_uploader(
            "Upload 4 CSV files",
            type=['csv'],
            accept_multiple_files=True
        )
        
        st.sidebar.subheader("Simulation Settings")
        future_periods = st.sidebar.slider(
            "Future Periods (Years)",
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

        st.sidebar.subheader("Market Parameters")
        
        # Simplified market parameters for GBM
        volatility = st.sidebar.slider(
            "Market Volatility (σ)",
            min_value=0.01,
            max_value=1.0,
            value=0.15,
            help="Annual volatility of returns"
        )
        
        drift = st.sidebar.slider(
            "Market Drift (μ)",
            min_value=-0.5,
            max_value=0.5,
            value=0.05,
            help="Expected annual return"
        )
        
        jump_prob = st.sidebar.slider(
            "Jump Probability",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            help="Probability of extreme events"
        )
        
        jump_size = st.sidebar.slider(
            "Jump Size",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            help="Average magnitude of jumps"
        )
        
        # Auto-calibration option
        auto_calibrate = st.sidebar.checkbox(
            "Auto-calibrate from data",
            value=True,
            help="Learn parameters from historical data"
        )
        
        seed = st.sidebar.number_input(
            "Random Seed",
            value=42,
            min_value=0
        )
        
        return {
            'selected_model': selected_model,
            'future_periods': future_periods,
            'n_simulations': n_simulations,
            'uploaded_files': uploaded_files,
            'volatility': volatility,
            'drift': drift,
            'jump_prob': jump_prob,
            'jump_size': jump_size,
            'auto_calibrate': auto_calibrate,
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
        try:
            uploaded_files = model_config['uploaded_files']
            
            # Keep data processing
            processor = DataProcessor()
            data = processor.process_funding_data(uploaded_files)
            
            # Initialize the selected model
            ModelClass = self.available_models[model_config['selected_model']]
            model = ModelClass(data)
            model.fit()
            
            # Create Monte Carlo parameters
            if model_config['auto_calibrate']:
                historical_returns = data['Total Funding']
                print(historical_returns)
                mc_params = calibrate_parameters(historical_returns)
            else:
                mc_params = MarketNoiseParameters(
                    volatility=model_config['volatility'],
                    trend=model_config['drift'],
                    momentum=model_config['jump_prob'],
                    market_impact=model_config['jump_size'],
                )

            # Create simulator with GBM
            print('PARAMS', mc_params)
            noise_generator = AdaptiveNoiseGenerator(mc_params)
            simulator = MonteCarloSimulator(model, noise_generator=noise_generator, seed=model_config['seed'])
            
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
                
            return model.df, forecast, model
                
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Error type: {type(e)}")
            st.error(f"Error during simulation: {str(e)}")
            return None, None, None
        
    def run_app(self):
        """Main application loop"""
        st.title("Monte Carlo Simulation Dashboard")
        
        # Get model and visualization configurations
        model_config = self.create_model_controls()
        viz_controls = self.create_visualization_controls()
        
        # Add a "Run Simulation" button
        if st.sidebar.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                historical_data, forecast, model = self.run_simulation(model_config)
                
                if historical_data is not None and forecast is not None:
                    # Create visualization
                    visualizer = StreamlitMonteCarloVisualizer()
                    
                    # Plot forecast
                    visualizer.plot_forecast(
                        historical_data=historical_data,
                        forecast_df=forecast,
                        model = model,
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
        model,  # Make sure this parameter is added
        date_column: str,
        target_column: str,
        controls: Dict
    ) -> None:
        """Create interactive visualization of Monte Carlo forecast"""
        
        st.subheader(f"Monte Carlo Forecast Analysis: {target_column}")
        
        # Create the base plot using plotly
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=historical_data[date_column],
                y=historical_data[target_column],
                name="Historical Data",
                line=dict(color="blue", width=controls['line_width'])
            )
        )

        # Get historical predictions
        historical_predictions = model.model.get_model_diagnostics()['predicted_values']

        # Add full prediction line (orange)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([historical_data[date_column], forecast_df[date_column]]),
                y=pd.concat([historical_predictions, forecast_df['MC_mean']]),
                name="Model Predictions",
                line=dict(color="orange", width=controls['line_width'])
            )
        )

        # Add running cumulative average 
        historical_cumulative_avg = historical_data[target_column].expanding().mean()
        
        full_cumulative_avg = pd.concat([historical_cumulative_avg])

        fig.add_trace(
        go.Scatter(
            x=pd.concat([historical_data[date_column], forecast_df[date_column]]),
            y=full_cumulative_avg,
            name="Running Historical Average",
            line=dict(color="gray", width=controls['line_width'], dash='dot')
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
        
        # Fix to access the correct column
        plot_data = forecast_df['MC_mean'] if period >= len(forecast_df) else forecast_df.iloc[period]
        
        fig.add_trace(
            go.Histogram(
                x=plot_data,
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