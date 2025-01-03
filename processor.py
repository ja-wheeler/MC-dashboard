import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

class DataProcessor:
    """Handles all data processing and preparation for Monte Carlo simulation."""
    
    def __init__(self, relevant_columns: Optional[List[str]] = None):
        self.relevant_columns = relevant_columns or [
            'Year', 'Seed Stage Rounds', 'Early Stage Rounds', 'Late Stage Rounds', 
            'Total Funding', 'Number of Rounds', 'Total Number of Companies',
            "Seed Stage Funding", "Early Stage Funding", "Late Stage Funding"
        ]
        self.processed_data = None

    def process_funding_data(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        """
        Process and merge funding data from multiple CSV files.
        
        Parameters:
        -----------
        file_paths : dict
            Dictionary containing file paths for each data type:
            {
                'num_stages': path_to_num_fund_stage.csv,
                'total_funding': path_to_total_fund_stage.csv,
                'funding_rounds': path_to_tot_num_funding_rounds.csv,
                'companies_founded': path_to_num_companies.csv
            }
        """
        # Load datasets
        dataframes = self._load_dataframes(file_paths)
        
        # Process each dataframe
        dataframes = self._clean_dataframes(dataframes)
        
        # Merge and process final dataset
        #merged_data = self._merge_dataframes(dataframes)

        #print(merged_data.columns)
        
        # Final processing steps
        self.processed_data = self._post_process_data(dataframes)

        print(self.processed_data.columns)
        
        return self.processed_data

    def _load_dataframes(self, files):
        """Load dataframes from a list of UploadedFile objects."""
        dataframes = []
        for file in files:
            try:
                print(f"Loading file: {file.name}")  # Debug
                df = pd.read_csv(file, sep=';')  # Adjust separator as needed
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading file {file.name}: {e}")
                dataframes.append(None)  # Add a placeholder for failed loads
        return dataframes[0]

    def _clean_dataframes(self, df):
        # Strip whitespace from column names
        #(key)
        df.columns = df.columns.str.strip()
        #print(df.columns)
        
        # Strip whitespace from string elements
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Remove empty or unnamed columns
        df.drop(columns=df.columns[df.columns.str.contains('^Unnamed|^$', regex=True)], 
            inplace=True)
        
        df['Year'] = df['Year'].astype(str).str.strip()
        
            
        return df

    def _post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform final processing steps on merged data."""
        # Handle missing values and empty strings
        data.fillna(0, inplace=True)
        data.replace("", 0, inplace=True)

        # Convert numeric columns while preserving the Year column
        year_column = data['Year']
        numeric_columns = data.drop('Year', axis=1).apply(pd.to_numeric, errors='coerce')
        data = pd.concat([year_column, numeric_columns], axis=1)

        # Remove columns where all entries are zero
        data = data.loc[:, (data != 0).any(axis=0)]

        # Convert Year to integer
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce').astype('Int64')

        # Filter data from year 2000 onwards and sort
        filtered_data = data[data['Year'] >= 2000].sort_values('Year')
        filtered_data = filtered_data.replace({np.nan: 0, 'nan': 0, 'NaN': 0, '': 0})
        filtered_data = filtered_data.interpolate()
        
        print(filtered_data.columns)
        # Select relevant columns
        filtered_data = filtered_data[self.relevant_columns]
        
        # Save processed data
        filtered_data.to_csv('filename.csv', index=False)
        
        return filtered_data

    def prepare_for_simulation(self, target_column: str = 'Total Funding') -> pd.DataFrame:
        """Prepare processed data for Monte Carlo simulation."""
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_funding_data first.")
            
        if target_column not in self.processed_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        simulation_data = self.processed_data.copy()
        
        # Convert Year to numeric and set as index
        simulation_data['Year'] = pd.to_numeric(simulation_data['Year'])
        simulation_data = simulation_data.set_index('Year')
        
        # Verify target column exists after setting index
        if target_column not in simulation_data.columns:
            raise ValueError(f"Target column '{target_column}' not found after processing")
            
        return simulation_data

    def get_historical_growth_rates(self, target_column: str = 'Total Funding',
                                  outlier_threshold: float = 10.0) -> pd.Series:
        """Calculate historical growth rates for the target column."""
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_funding_data first.")
            
        growth_rates = self.processed_data[target_column].pct_change().dropna()
        return growth_rates[np.abs(growth_rates) < outlier_threshold]