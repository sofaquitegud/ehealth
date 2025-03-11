import pandas as pd
from enum import Enum
from typing import Dict

class HealthMeasure(Enum):
    OVERALL = 'Overall'
    HYPERTENSION = 'Hypertension'
    STRESS = 'Stress'
    WELLNESS = 'Wellness'
    BMI = 'BMI'

class VisualizationCategory(Enum):
    OVERALL = 'Overall'
    AGE_RANGE = 'Age range'
    GENDER = 'Gender type'
    BMI = 'BMI'

class TimePeriod(Enum):
    WEEKLY = 'Weekly'
    MONTHLY = 'Monthly'
    QUARTERLY = 'Quarterly'
    YEARLY = 'Yearly'

class ReportType(Enum):
    LATEST = 'Latest'
    TRENDING = 'Trending'

class DataPreprocessor:
    def __init__(self, data_path: str):
        """
        Initialize the preprocessor with staff health data
        
        Parameters:
        data_path (str): Path to the CSV file with staff health data
        """
        # Load the dataset
        self.df = pd.read_csv(data_path)
        
        # Convert date column to datetime
        self.df["date"] = pd.to_datetime(self.df["date"])
        
        # Save original values before any mappings
        self.df['original_hypertension'] = self.df['hypertensionRisk']
        self.df['original_stress'] = self.df['stressLevel']
        self.df['original_wellness'] = self.df['wellnessLevel']
        self.df['original_bmi'] = self.df['bmi']
        
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
        
        # Column name mappings (use same column names for both reports)
        self.measurement_mappings = {
            'Overall': 'overallHealth',
            'Hypertension': 'original_hypertension',
            'Stress': 'original_stress',
            'Wellness': 'original_wellness',
            'BMI': 'original_bmi'
        }
        
        # Standardized display names for both reports
        self.display_names = {
            'Overall': 'Overall Health',
            'Hypertension': 'Hypertension Risk',
            'Stress': 'Stress Level',
            'Wellness': 'Wellness Level',
            'BMI': 'BMI'
        }
        
        # Visualization category mappings
        self.visualization_mappings = {
            'Overall': 'overall',
            'Age range': 'age_range',
            'Gender type': 'gender',
            'BMI': 'original_bmi'
        }
        
        # Process age ranges
        self._process_age_ranges()
        
        # Apply mappings
        self.apply_mappings()
        
    def _process_age_ranges(self):
        """Process age ranges with correct labels"""
        self.df['age_range'] = pd.cut(
            self.df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
    
    def apply_mappings(self):
        """Apply numeric mappings to categorical data"""
        self.df['stressLevel'] = self.df['original_stress'].map(self.stress_map)
        self.df['wellnessLevel'] = self.df['original_wellness'].map(self.wellness_map)
        self.df['hypertensionRisk'] = self.df['original_hypertension'].map(self.hypertension_map)
        self.df['bmi'] = self.df['original_bmi'].map(self.bmi_map)
        
        # Calculate overall health if not present
        if 'overallHealth' not in self.df.columns:
            # Using the correct weighted formula for overall health
            self.df['overallHealth'] = (
                0.25 * self.df['stressLevel'] +
                0.35 * self.df['wellnessLevel'] +
                0.25 * self.df['hypertensionRisk'] +
                0.15 * self.df['bmi']
            )

            # Round to 2 decimal places but keep integers unchanged
            self.df['overallHealthValue'] = self.df['overallHealth'].apply(
                lambda x: int(x) if x.is_integer() else round(x, 2)
            )

            # Classify overall health based on the calculated value
            categories = ['healthy', 'mild', 'elevated', 'risky']

            self.df['overallHealth'] = pd.cut(
                self.df['overallHealthValue'],
                bins=[0, 2, 3, 4, 5],
                labels=categories,
                include_lowest=True
            )

            # Override calculated category if any critical factor exceeds 4
            override_condition = (self.df[['stressLevel', 'wellnessLevel', 'hypertensionRisk']].max(axis=1) > 4)
            self.df.loc[override_condition, 'overallHealthValue'] = 5
            self.df.loc[override_condition, 'overallHealth'] = 'risky'
            
        return self.df
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get most recent data for each staff member"""
        return self.df.sort_values('date').groupby('staff_id').last().reset_index()
    
    def get_data(self) -> pd.DataFrame:
        """Return the processed dataframe"""
        return self.df
    
    def get_mappings(self) -> Dict:
        """Return all the mappings used in preprocessing"""
        return {
            "measurement_mappings": self.measurement_mappings,
            "display_names": self.display_names,
            "visualization_mappings": self.visualization_mappings,
            "stress_map": self.stress_map,
            "wellness_map": self.wellness_map,
            "hypertension_map": self.hypertension_map,
            "bmi_map": self.bmi_map
        }