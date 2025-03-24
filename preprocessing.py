# preprocessing.py

import pandas as pd
import numpy as np
import psycopg2
from enum import Enum
from typing import Dict, Optional, Any

# Constants
DEFAULT_BMI_VALUE = 22.0
MODE_KIOSK = "kiosk"
MODE_MOBILE = "mobile"
ERROR_BMI_KIOSK = "BMI analysis not available in kiosk mode."


class HealthMeasure(Enum):
    OVERALL = "Overall"
    HYPERTENSION = "Hypertension"
    STRESS = "Stress"
    WELLNESS = "Wellness"
    BMI = "BMI"


class VisualizationCategory(Enum):
    OVERALL = "Overall"
    AGE_RANGE = "Age_range"
    GENDER = "Gender"
    BMI = "BMI"


class TimePeriod(Enum):
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    YEARLY = "Yearly"


class ReportType(Enum):
    LATEST = "Latest"
    TRENDING = "Trending"


class HealthDataPreprocessor:
    # Mapping constants
    STRESS_MAP = {"low": 1, "normal": 2, "mild": 3, "high": 4, "very high": 5}
    WELLNESS_MAP = {"high": 1, "medium": 2.5, "low": 5}
    HYPERTENSION_MAP = {"low": 1, "medium": 2.5, "high": 5}
    BMI_MAP = {"normal": 1, "underweight": 2, "overweight": 3, "obese": 5}

    # Age range boundaries
    AGE_BINS = [0, 25, 35, 45, 55, 100]
    AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "55+"]

    def __init__(
        self, 
        data_path: str, 
        mode: str = MODE_MOBILE,
        db_config: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the health data preprocessor

        Parameters:
        data_path (str): Path to the CSV file with staff health data
        mode (str): Analysis mode - 'kiosk' or 'mobile'
        db_config (dict, optional): Database configuration for PostgreSQL connection
        """
        # Validate and store the mode
        self.mode = mode.lower()
        if self.mode not in [MODE_KIOSK, MODE_MOBILE]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be '{MODE_KIOSK}' or '{MODE_MOBILE}'"
            )
            
        # Store database configuration
        self.db_config = db_config

        # Load and preprocess the dataset
        self.df = self._load_and_preprocess_data(data_path)

        # Initialize mappings
        self._initialize_mappings()

    def _load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess the dataset"""
        # Load data
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            print(f"Failed to load CSV data: {e}")
            if self.db_config:
                try:
                    conn = psycopg2.connect(**self.db_config)

                    if self.mode == MODE_MOBILE:
                        query = """
                            SELECT created_at AS date, username AS staff_id, gender, age, bmi, stressLevel, wellnessLevel, hypertensionRisk
                            FROM mobile_table
                            """
                    else:
                        query = """
                            SELECT created_at AS date, email AS staff_id, stressLevel, wellnessLevel, hypertensionRisk
                            FROM kiosk_table
                        """

                    df = pd.read_sql_query(query, conn)
                    conn.close()
                    print(f"Successfully loaded data from database in {self.mode} mode")
                except Exception as db_error:
                    raise ValueError(
                        f"Failed to load data from both CSV and database: {db_error}"
                    )
            else:
                raise ValueError(
                    "Database configuration not provided and CSV file not found"
                )

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Handle demographic data based on mode
        df = self._handle_demographic_data(df)

        # Handle BMI based on mode
        df = self._handle_bmi_data(df)

        # Process categorical BMI
        df["original_bmi"] = df["bmi"].apply(self._categorize_bmi).replace("", np.nan)

        # Save original values for other health measures
        df["original_stress"] = df["stressLevel"].fillna("").replace("", np.nan)
        df["original_wellness"] = df["wellnessLevel"].fillna("").replace("", np.nan)
        df["original_hypertension"] = (
            df["hypertensionRisk"].fillna("").replace("", np.nan)
        )

        # Sort data by staff_id and date
        df = df.sort_values(by=["staff_id", "date"])
        df.reset_index(drop=True, inplace=True)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Process age ranges
        df["age_range"] = pd.cut(
            df["age"],
            bins=self.AGE_BINS,
            labels=self.AGE_LABELS,
        )

        # Apply numeric mappings and calculate overall health
        df = self._apply_value_mappings(df)

        return df
        
    def _handle_demographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle demographic data (age and gender) based on mode"""
        df_copy = df.copy()

        if self.mode == MODE_KIOSK:
            if "age" not in df_copy.columns:
                df_copy["age"] = 35  # Default value
            if "gender" not in df_copy.columns:
                df_copy["gender"] = "unknown"
        elif self.mode == MODE_MOBILE:
            if "age" not in df_copy.columns:
                print("Warning: Age data missing in mobile mode. Using default values.")
                df_copy["age"] = 35
            if "gender" not in df_copy.columns:
                print(
                    "Warning: Gender data missing in mobile mode. Using default values."
                )
                df_copy["gender"] = "unknown"

        return df_copy

    def _handle_bmi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle BMI data based on mode"""
        df_copy = df.copy()

        if self.mode == MODE_KIOSK:
            # For kiosk mode, BMI is not available
            df_copy["bmi"] = DEFAULT_BMI_VALUE
            if "bmi" in df.columns:
                print(
                    "Warning: BMI data present but in kiosk mode. Using placeholder values."
                )
        elif "bmi" not in df.columns:
            # Handle missing BMI column in mobile mode
            print("Warning: BMI data missing in mobile mode. Using estimated values.")
            df_copy["bmi"] = DEFAULT_BMI_VALUE

        return df_copy

    @staticmethod
    def _categorize_bmi(bmi):
        """Categorize BMI value"""
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

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in health metrics"""
        df_copy = df.copy()

        # Health measure columns to process
        health_columns = [
            "original_hypertension",
            "original_stress",
            "original_wellness",
            "original_bmi",
        ]

        # Forward fill followed by backward fill
        for col in health_columns:
            df_copy[col] = df_copy[col].ffill().bfill()

        return df_copy

    def _initialize_mappings(self):
        """Initialize all mappings used in the analysis"""
        # Measurement column mappings
        self.measurement_mappings = {
            "Overall": "overallHealth",
            "Hypertension": "original_hypertension",
            "Stress": "original_stress",
            "Wellness": "original_wellness",
            "BMI": "original_bmi",
        }

        # Display names for reports
        self.display_names = {
            "Overall": "OverallHealth",
            "Hypertension": "HypertensionRisk",
            "Stress": "StressLevel",
            "Wellness": "WellnessLevel",
            "BMI": "BMI",
        }

        # Visualization category mappings
        self.visualization_mappings = {
            "Overall": "overall",
            "Age_range": "age_range",
            "Gender": "gender",
            "BMI": "original_bmi",
        }

    def _apply_value_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply numeric mappings to categorical data and calculate overall health"""
        df_copy = df.copy()

        # Apply numeric mappings
        df_copy["stressLevel"] = df_copy["original_stress"].map(self.STRESS_MAP)
        df_copy["wellnessLevel"] = df_copy["original_wellness"].map(self.WELLNESS_MAP)
        df_copy["hypertensionRisk"] = df_copy["original_hypertension"].map(
            self.HYPERTENSION_MAP
        )
        df_copy["bmi"] = df_copy["original_bmi"].map(self.BMI_MAP)

        # Calculate overall health if not present
        if "overallHealth" not in df_copy.columns:
            overall_health_values = (
                0.25 * df_copy["stressLevel"]
                + 0.35 * df_copy["wellnessLevel"]
                + 0.25 * df_copy["hypertensionRisk"]
                + 0.15 * df_copy["bmi"]
            )

            df_copy["overallHealth"] = overall_health_values.apply(
                self._categorize_overall_health
            )

        return df_copy

    @staticmethod
    def _categorize_overall_health(value: float) -> str:
        """Categorize overall health value"""
        if value <= 2:
            return "healthy"
        elif 2 < value <= 3:
            return "mild"
        elif 3 < value <= 4:
            return "elevated"
        else:
            return "risky"

    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed dataframe"""
        return self.df
