# preprocessing.py
import pandas as pd
import numpy as np
from enum import Enum

class HealthMeasure(Enum):
    OVERALL = 'Overall'
    HYPERTENSION = 'Hypertension'
    STRESS = 'Stress'
    WELLNESS = 'Wellness'
    BMI = 'BMI'

class StaffHealthPreprocessor:
    def __init__(self, data_path: str):
        """
        Initialize the health data preprocessor
        
        Parameters:
        data_path (str): Path to the CSV file with staff health data
        """
        # Load the dataset
        self.df = pd.read_csv(data_path)
        
        # Convert date column to datetime
        self.df["date"] = pd.to_datetime(self.df["date"])

        # Define value mappings
        self.stress_map = {
            "low": 1, 
            "normal": 2, 
            "mild": 3, 
            "high": 4,
            "very high": 5
        }
        self.wellness_map = {
            "high": 1,
            "medium": 2.5,
            "low": 5
        }
        self.hypertension_map = {
            "low": 1, 
            "medium": 2.5, 
            "high": 5
        }
        self.bmi_map = {
            "normal": 1, 
            "underweight": 2, 
            "overweight": 3, 
            "obese": 5
        }

        # Column name mappings
        self.measurement_mappings = {
            'Overall': 'overallHealth',
            'Hypertension': 'original_hypertension',
            'Stress': 'original_stress',
            'Wellness': 'original_wellness',
            'BMI': 'original_bmi'
        }
        
        # Standardized display names
        self.display_names = {
            'Overall': 'Overall Health',
            'Hypertension': 'Hypertension Risk',
            'Stress': 'Stress Level',
            'Wellness': 'Wellness Level',
            'BMI': 'BMI'
        }
        
        # Process the data
        self._categorize_bmi()
        self._handle_missing_values()
        self._process_age_ranges()
        self.apply_mappings()
    
    def _categorize_bmi(self):
        """Categorize BMI values into descriptive categories"""
        def categorize_bmi(bmi):
            if bmi is None or pd.isna(bmi):
                return ""
            elif bmi < 18.5:
                return "underweight"
            elif 18.5 <= bmi < 25:
                return "normal"
            elif 25 <= bmi < 30:
                return "overweight"
            else:
                return "obese"
        
        # Save original values before any mappings
        try:
            self.df['original_bmi'] = self.df['bmi'].apply(categorize_bmi).replace('', np.nan)
        except: # To manage missing BMI data specifically for kiosk usage
            self.df['bmi'] = [22]*len(self.df)
            self.df['original_bmi'] = self.df['bmi'].apply(categorize_bmi).replace('', np.nan)
        self.df['original_stress'] = self.df['stressLevel'].fillna('').replace('', np.nan)
        self.df['original_wellness'] = self.df['wellnessLevel'].fillna('').replace('', np.nan)
        self.df['original_hypertension'] = self.df['hypertensionRisk'].fillna('').replace('', np.nan)


        # Managing missing values: Staff ID and Date in sequential order
        self.df = self.df.sort_values(by=['staff_id','date'])
        self.df.reset_index(drop=True, inplace=True)
    
    def _handle_missing_values(self, method: int = 0):
        """Handle missing values in the data
        
        Parameters:
        method (int): 0 for forward/backward fill, other values for fixed value fill
        """
        if method == 0:
            # Forward fill: Fill missing values with the last known non-missing value
            self.df['original_hypertension'] = self.df['original_hypertension'].ffill()
            self.df['original_stress'] = self.df['original_stress'].ffill()
            self.df['original_wellness'] = self.df['original_wellness'].ffill()
            self.df['original_bmi'] = self.df['original_bmi'].ffill()

            # Backward fill: Fill missing values with the next known non-missing value
            self.df['original_hypertension'] = self.df['original_hypertension'].bfill()
            self.df['original_stress'] = self.df['original_stress'].bfill()
            self.df['original_wellness'] = self.df['original_wellness'].bfill()
            self.df['original_bmi'] = self.df['original_bmi'].bfill()
        else:
            # Fixed Value fill: Fill Missing Values with a Fixed Value
            self.df['original_hypertension'] = self.df['original_hypertension'].fillna("medium")
            self.df['original_stress'] = self.df['original_stress'].fillna("normal")
            self.df['original_wellness'] = self.df['original_wellness'].fillna("medium")
            self.df['original_bmi'] = self.df['original_bmi'].fillna("normal")
    
    def _process_age_ranges(self):
        """Process age ranges with correct labels"""
        self.df['age_range'] = pd.cut(
            self.df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
    
    def apply_mappings(self):
        """Apply numeric mappings to categorical data and calculate overall health"""
        # Determine Overall Health category
        def overall_health_category(overall_health_value):
            if overall_health_value <= 2:
                return 'healthy'
            elif 2 < overall_health_value <= 3:
                return 'mild'
            elif 3 < overall_health_value <= 4:
                return 'elevated'
            else:
                return 'risky'

        # Apply mappings
        self.df['stressLevel'] = self.df['original_stress'].map(self.stress_map)
        self.df['wellnessLevel'] = self.df['original_wellness'].map(self.wellness_map)
        self.df['hypertensionRisk'] = self.df['original_hypertension'].map(self.hypertension_map)
        self.df['bmi'] = self.df['original_bmi'].map(self.bmi_map)
        
        # Calculate overall health if not present
        if 'overallHealth' not in self.df.columns:
            # Using the weighted formula for overall health
            self.df['overallHealth'] = (
                0.25 * self.df['stressLevel'] +
                0.35 * self.df['wellnessLevel'] +
                0.25 * self.df['hypertensionRisk'] +
                0.15 * self.df['bmi']
            )
            self.df['overallHealth'] = self.df['overallHealth'].apply(overall_health_category)
        
        return self.df
    
    def get_processed_data(self):
        """Return the fully processed dataframe
        
        Returns:
        pd.DataFrame: Processed health data
        """
        return self.df
    
    def get_measurement_mappings(self):
        """Return the measurement mappings
        
        Returns:
        dict: Mapping between health measures and dataframe columns
        """
        return self.measurement_mappings
    
    def get_display_names(self):
        """Return the display names for health measures
        
        Returns:
        dict: Mapping between health measures and display names
        """
        return self.display_names