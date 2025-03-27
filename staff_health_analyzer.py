# Import Libraries
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import psycopg2
from enum import Enum
from typing import Dict, Optional, Any, List
from functools import lru_cache
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


class StaffHealthAnalyzer:
    # Mapping constants
    STRESS_MAP = {"low": 1, "normal": 2, "mild": 3, "high": 4, "very high": 5}
    WELLNESS_MAP = {"high": 1, "medium": 2.5, "low": 5}
    HYPERTENSION_MAP = {"low": 1, "medium": 2.5, "high": 5}
    BMI_MAP = {"normal": 1, "underweight": 2, "overweight": 3, "obese": 5}

    # Age range boundaries
    AGE_BINS = [0, 25, 35, 45, 55, 100]
    AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "55+"]

    # System prompt for health recommendations
    RECOMMENDATIONS_PROMPT = """  
    Act as a workplace health advisor tasked with generating actionable recommendations to improve employee health based on data insights. Analyze the provided data and generate prioritized recommendations.
    """

    def __init__(
        self,
        data_path: str,
        api_key: Optional[str] = None,
        mode: str = MODE_MOBILE,
        db_config: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the health analyzer with staff health data

        Parameters:
        data_path (str): Path to the CSV file with staff health data
        api_key (str, optional): OpenAI API key for natural language summaries and recommendations
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

        # Initialize OpenAI client if API key is provided
        self.openai_client = self._initialize_openai_client(api_key)

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
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)

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

        # LLM system prompt
        self.SYSTEM_PROMPT = """Act as a health data analyst tasked with summarizing the health status of employees in a company. Below are the specifications for the report:
        *Report Type*: {report_type}  
        *Health Measurement*: {health_measurement}  
        *Visualization Category*: {visualization_category}  
        *Data*:{data}
        
        ### Input Data Structure
        - *Report Type*:
          - *Latest*: Snapshot of current health metrics.
          - *Trending*: Trends over time (e.g., weekly/monthly changes).
        - *Health Measurement*:
          - Overall | Hypertension | Stress | Wellness | BMI
          - Overall represents the cumulative overall metrics that are derived from 
        - *Visualization Category*:
          - If Report Type = latest
          - *Visualization Category* : Overall | By Age Range | By Gender | By BMI
          - If Report Type = trending
          - *Visualization Category* : weekly | monthly | quarterly | yearly
        - *Data*:
          - Corresponding data of {report_type} {health_measurement} and {visualization_category} in table format
        
        ### Task
        Analyze the health data provided below and generate a concise summary that:
        1. Highlights key findings for the *{health_measurement}* metric.
        2. Compares trends or current status based on the *{report_type}* type.
        
        ### Input Data
        {data}
        
        ### Expected Output Format
        - A 3-5 sentence summary in plain English.
        - Focus on clarity, relevance, and data-driven conclusions."""

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

    def _initialize_openai_client(self, api_key: Optional[str]) -> Optional[OpenAI]:
        """Initialize OpenAI client if API key is provided"""
        if not api_key:
            # Try to get API key from environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None

        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            return None

    def _validate_request(self, health_measure: str, category: str = None) -> None:
        """Validate if the requested report is available in the current mode"""
        if self.mode == MODE_KIOSK:
            if health_measure == "BMI" or category == "BMI":
                raise ValueError(ERROR_BMI_KIOSK)
            # Add validation for demographic analysis in kiosk mode
            if category in ["Age_range", "Gender"]:
                raise ValueError(f"{category} analysis not available in kiosk mode.")

    @lru_cache(maxsize=32)
    def get_latest_data(self) -> pd.DataFrame:
        """Get most recent data for each staff member"""
        return self.df.sort_values("date").groupby("staff_id").last().reset_index()

    def generate_latest_report(
        self, health_measure: str, visualization_cat: str
    ) -> pd.DataFrame:
        """
        Generate a latest snapshot report for given health measure and visualization category

        Parameters:
        health_measure (str): Health measure to report on (Overall, Hypertension, Stress, Wellness, BMI)
        visualization_cat (str): Category to visualize by (Overall, Age range, Gender type, BMI)

        Returns:
        pd.DataFrame: Report data
        """
        # Validate request
        self._validate_request(health_measure, visualization_cat)

        # Get the latest data
        data = self.get_latest_data()

        # Get measure column name and display name
        measure_col = self.measurement_mappings[health_measure]
        display_name = self.display_names[health_measure]

        # Get visualization category column
        viz_col = self.visualization_mappings[visualization_cat]

        # Set the appropriate category name
        category_name = (
            "Category" if visualization_cat == "Overall" else visualization_cat
        )

        # Generate report based on visualization category
        if viz_col == "overall":
            return self._generate_overall_report(
                data, measure_col, display_name, category_name
            )
        else:
            return self._generate_category_report(
                data, measure_col, viz_col, display_name, category_name
            )

    def _generate_overall_report(
        self,
        data: pd.DataFrame,
        measure_col: str,
        display_name: str,
        category_name: str,
    ) -> pd.DataFrame:
        """Generate report for Overall visualization category"""
        counts = data[measure_col].value_counts()

        return pd.DataFrame(
            {
                category_name: "Overall",
                display_name: counts.index.tolist(),
                "Count": counts.values,
                "Percentage": (counts.values / counts.sum() * 100).round(2),
            }
        )

    def _generate_category_report(
        self,
        data: pd.DataFrame,
        measure_col: str,
        viz_col: str,
        display_name: str,
        category_name: str,
    ) -> pd.DataFrame:
        """Generate report for a specific visualization category"""
        # Create crosstab
        crosstab = pd.crosstab(data[measure_col], data[viz_col])
        percentages = crosstab.div(crosstab.sum(axis=0), axis=1) * 100

        # Create report with original categorical values
        report = []
        for measure_val in crosstab.index:
            for group_val in crosstab.columns:
                report.append(
                    {
                        category_name: group_val,
                        display_name: measure_val,
                        "Count": crosstab.loc[measure_val, group_val],
                        "Percentage": percentages.loc[measure_val, group_val].round(2),
                    }
                )

        return pd.DataFrame(report)

    def generate_trending_report(
        self, health_measure: str, time_period: str
    ) -> pd.DataFrame:
        """
        Generate a trending report for given health measure over a time period

        Parameters:
        health_measure (str): Health measure to report on (Overall, Hypertension, Stress, Wellness, BMI)
        time_period (str): Time period for trend analysis (Weekly, Monthly, Quarterly, Yearly)

        Returns:
        pd.DataFrame: Trend data
        """
        # Validate request
        self._validate_request(health_measure)

        # Get measure column name and display name
        measure_col = self.measurement_mappings[health_measure]
        display_name = self.display_names[health_measure]

        # Create filtered dataframe with time period column
        df_filtered = self._filter_by_time_period(time_period)

        # Process the data to generate the trending report
        return self._process_trending_data(df_filtered, measure_col, display_name)

    def _filter_by_time_period(self, time_period: str) -> pd.DataFrame:
        """Filter data and add time_period column based on selected time period"""
        df = self.df.copy()

        if time_period == "Weekly":
            # Filter data for the last 6 weeks
            df = df[df["date"] >= df["date"].max() - pd.Timedelta(weeks=5)].reset_index(
                drop=True
            )
            df["time_period"] = df["date"].dt.to_period("W")
            groupby_col = "time_period"

        elif time_period == "Monthly":
            # Filter data for the last 6 months
            df = df[
                df["date"] >= df["date"].max() - pd.DateOffset(months=5)
            ].reset_index(drop=True)
            df["time_period"] = df["date"].dt.to_period("M")
            groupby_col = "time_period"

        elif time_period == "Quarterly":
            # Filter data for the last 6 quarters
            df = df[
                df["date"] >= df["date"].max() - pd.DateOffset(months=6 * 3 - 2)
            ].reset_index(drop=True)
            df["time_period"] = (
                df["date"]
                .dt.to_period("Q")
                .apply(lambda x: x.start_time.strftime("%Y-%m-%d"))
            )
            groupby_col = "time_period"

        elif time_period == "Yearly":
            # Filter data for the last 6 years
            df = df[
                df["date"] >= df["date"].max() - pd.DateOffset(years=5)
            ].reset_index(drop=True)
            df["time_period"] = df["date"].dt.year
            groupby_col = "time_period"
        else:
            raise ValueError(f"Invalid time period: {time_period}")

        return df, groupby_col

    def _process_trending_data(
        self, filtered_data: tuple, measure_col: str, display_name: str
    ) -> pd.DataFrame:
        """Process filtered data to generate trending report"""
        df, groupby_col = filtered_data

        # Get the latest date in each time period
        latest_dates = df.groupby(groupby_col)["date"].max().reset_index()

        # Merge with the original dataset to keep only the latest records per time period
        df_latest = df.merge(latest_dates, on=[groupby_col, "date"])

        # Define all possible categories for measure
        all_measure_categories = df[measure_col].unique()
        all_periods = df_latest[groupby_col].unique()

        # Create a complete MultiIndex with all combinations
        multi_index = pd.MultiIndex.from_product(
            [all_periods, all_measure_categories], names=[groupby_col, measure_col]
        )

        # Group by measure and time period, then count occurrences
        df_trend = (
            df_latest.groupby([groupby_col, measure_col])
            .size()
            .reset_index(name="Count")
        )

        # Reindex to ensure missing categories are filled with zero
        df_trend = (
            df_trend.set_index([groupby_col, measure_col])
            .reindex(multi_index, fill_value=0)
            .reset_index()
        )

        # Calculate percentage within each time period
        total_count = df_trend.groupby(groupby_col)["Count"].transform("sum")
        df_trend["Percentage"] = ((df_trend["Count"] / total_count) * 100).round(2)

        # Convert time_period to string for consistent output
        df_trend[groupby_col] = df_trend[groupby_col].astype(str)

        # Sort by time period for better visualization
        df_trend = df_trend.sort_values(by=[groupby_col, measure_col])

        # Rename the measure_col column to use the display name for consistency with latest report
        return df_trend.rename(columns={measure_col: display_name})

    def generate_all_reports(self, report_type: str) -> Dict[str, pd.DataFrame]:
        """
        Generate all required reports of a specific type

        Parameters:
        report_type (str): 'Latest' or 'Trending'

        Returns:
        Dict[str, pd.DataFrame]: Dictionary of all reports with appropriate keys
        """
        reports = {}
        required_combinations = []

        if report_type == "Latest":
            # Generate combinations for latest reports
            for measure in HealthMeasure:
                # Skip BMI measure in kiosk mode
                if self.mode == MODE_KIOSK and measure.value == "BMI":
                    continue

                for category in VisualizationCategory:
                    # Skip BMI category in kiosk mode
                    if self.mode == MODE_KIOSK and category.value == "BMI":
                        continue

                    # Exclude BMI|BMI which is not in the requirements
                    if not (measure.value == "BMI" and category.value == "BMI"):
                        required_combinations.append((measure.value, category.value))

            # Generate each report
            for measure, category in required_combinations:
                key = f"{measure}_{category}".lower().replace(" ", "_")
                try:
                    reports[key] = self.generate_latest_report(
                        health_measure=measure, visualization_cat=category
                    )
                except ValueError as e:
                    print(f"Skipping {key}: {str(e)}")

        elif report_type == "Trending":
            # Generate combinations for trending reports
            for measure in HealthMeasure:
                # Skip BMI measure in kiosk mode
                if self.mode == MODE_KIOSK and measure.value == "BMI":
                    continue

                for period in TimePeriod:
                    required_combinations.append((measure.value, period.value))

            # Generate each report
            for measure, period in required_combinations:
                key = f"{measure}_{period}".lower().replace(" ", "_")
                try:
                    reports[key] = self.generate_trending_report(
                        health_measure=measure, time_period=period
                    )
                except ValueError as e:
                    print(f"Skipping {key}: {str(e)}")
        else:
            raise ValueError(f"Invalid report type: {report_type}")

        return reports

    def format_table(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to formatted markdown-style table with percentage formatting"""
        if df.empty:
            return "No data available."

        # Format percentages if column exists
        if "Percentage" in df.columns:
            df = df.copy()
            df["Percentage"] = df["Percentage"].apply(lambda x: f"{x:.2f}%")

        return df.to_markdown(index=False)

    def get_report_summary(
        self, report_type: str, health_measure: str, category: str, data: pd.DataFrame
    ) -> str:
        """
        Generate a natural language summary of the report using LLM if available

        Parameters:
        report_type (str): 'Latest' or 'Trending'
        health_measure (str): Health measure type
        category (str): Specific categorization or time period
        data (pd.DataFrame): Report data

        Returns:
        str: Natural language summary or message if LLM not available
        """
        if not self.openai_client:
            return (
                "OpenAI client not available. Set API key to enable natural language summaries and recommendations."
            )

        formatted_table = self.format_table(data)

        formatted_prompt = self.SYSTEM_PROMPT.format(
            report_type=report_type,
            health_measurement=health_measure,
            visualization_category=category,
            data=formatted_table,
        )

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ]
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_health_recommendations(
        self,
        report_type: str,
        health_measurement: str,
        visualization_category: str,
        data: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        Generate health recommendations based on the analysis data using OpenAI

        Parameters:
        report_type (str): Type of report ('Latest' or 'Trending')
        health_measurement (str): Health measure being analyzed
        visualization_category (str): Category for visualization
        data (pd.DataFrame): Analysis data

        Returns:
        Optional[Dict]: Health recommendations in JSON format or None if OpenAI client is not available
        """
        if not self.openai_client:
            return None

        try:
            # Format data for the prompt
            formatted_data = self.format_table(data)

            response = self.openai_client.responses.create(
                model="o3-mini",  # Use your preferred model
                input=[
                    {
                        "role": "system", 
                        "content": self.RECOMMENDATIONS_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": f"""
                        Report Type: {report_type}
                        Health Measurement: {health_measurement}
                        Visualization Category: {visualization_category}
                        Data: {formatted_data}
                        """
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "health_recommendations",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "recommendations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "action": {"type": "string"},
                                            "action_keyword": {"type": "string"}, 
                                            "action_details": {"type": "string"},
                                            "target_group": {"type": "string"},
                                            "priority": {
                                                "type": "integer"
                                            }
                                        },
                                        "required": ["action","action_keyword", "action_details", "target_group", "priority"],
                                        "additionalProperties": False
                                    }
                                },
                                "report_metadata": {
                                    "type": "object",
                                    "properties": {
                                        "report_type": {
                                            "type": "string",
                                            "enum": ["Latest", "Trending"]
                                        },
                                        "health_measurement": {
                                            "type": "string",
                                            "enum": ["Overall", "Hypertension", "Stress", "Wellness", "BMI"]
                                        },
                                        "visualization_category": {
                                            "type": "string",
                                            "enum": ["Overall", "By Age Range", "By Gender", "By BMI", "yearly", "quarterly","monthly", "weekly"]
                                        }
                                    },
                                    "required": ["report_type", "health_measurement", "visualization_category"],
                                    "additionalProperties": False
                                }
                            },
                            "required": ["recommendations", "report_metadata"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            # Parse the output
            return json.loads(response.output_text)

        except Exception as e:
            print(f"Error generating health recommendations: {e}")
            return None

    def run_analysis(
        self,
        report_type: str,
        health_measure: str,
        category: str,
        with_summary: bool = False,
        enable_display: bool = False,
        with_recommendations: bool = False,
    ) -> Dict:
        """
        Run a complete analysis for the specified parameters

        Parameters:
        report_type (str): 'Latest' or 'Trending'
        health_measure (str): Health measure type (e.g., 'Overall', 'Hypertension')
        category (str): Category for latest report or time period for trending report
        with_summary (bool): Whether to include natural language summary
        enable_display (bool): Whether to display visualization
        with_recommendations (bool): Whether to include health recommendations

        Returns:
        Dict: Dictionary containing report data and optional summary/recommendations
        """
        # Validate request
        self._validate_request(health_measure, category)

        result = {
            "report_type": report_type,
            "health_measure": health_measure,
            "category": category,
            "mode": self.mode,
        }

        # Generate appropriate report
        if report_type == "Latest":
            data = self.generate_latest_report(health_measure, category)
        elif report_type == "Trending":
            data = self.generate_trending_report(health_measure, category)
        else:
            raise ValueError(f"Invalid report type: {report_type}")

        result["data"] = data

        # Add summary if requested and available
        if with_summary:
            result["summary"] = self.get_report_summary(
                report_type, health_measure, category, data
            )

        # Add health recommendations if requested and available
        if with_recommendations:
            recommendations = self.generate_health_recommendations(
                report_type, health_measure, category, data
            )
            if recommendations:
                result["recommendations"] = recommendations

        # Display visualization if requested
        if enable_display:
            self._display_visualization(result)

        return result

    def _display_visualization(self, result: Dict):
        """Display visualization based on report type"""
        report_type = result["report_type"]
        df_plot = result["data"]

        # Check if dataframe is not empty
        if df_plot.empty:
            print("No data to visualize.")
            return

        # Get column names
        time_column = df_plot.columns[0]  # First column (category or time period)
        category_column = df_plot.columns[1]  # Health measure category

        plt.figure(figsize=(12, 6))

        # Get unique categories
        unique_categories = df_plot[category_column].unique()

        # Define color palette
        colors = plt.get_cmap("tab10", len(unique_categories))

        # Choose visualization type based on report type
        if report_type == "Trending":
            self._create_trending_visualization(
                df_plot, time_column, category_column, unique_categories, colors
            )
        else:  # Latest
            self._create_latest_visualization(
                df_plot, time_column, category_column, unique_categories, colors
            )

        # Show plot
        plt.tight_layout()
        plt.show()

    def _create_trending_visualization(
        self,
        df: pd.DataFrame,
        time_column: str,
        category_column: str,
        unique_categories: np.ndarray,
        colors,
    ):
        """Create visualization for trending reports"""
        # Plot each category separately
        for idx, category in enumerate(unique_categories):
            category_data = df[df[category_column] == category]
            plt.plot(
                category_data[time_column],
                category_data["Percentage"],
                marker="o",
                linestyle="-",
                label=f"{category}",
                color=colors(idx),
            )

        # Formatting
        plt.xlabel(time_column, fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.title(f"Trend Analysis: {category_column} Over Time", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")

    def _create_latest_visualization(
        self,
        df: pd.DataFrame,
        category_column: str,
        measure_column: str,
        unique_categories: np.ndarray,
        colors,
    ):
        """Create visualization for latest reports"""
        # Set width for bars
        bar_width = 0.2
        positions = np.arange(len(df[category_column].unique()))  # Positions for bars

        # Create bars for each category
        for idx, category in enumerate(unique_categories):
            category_data = df[df[measure_column] == category]

            plt.bar(
                positions + (idx * bar_width),
                category_data["Percentage"],
                width=bar_width,
                label=f"{category}",
                color=colors(idx),
            )

        # Formatting
        plt.xlabel(category_column, fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.title(
            f"Latest Analysis: {measure_column} by {category_column}", fontsize=14
        )
        plt.xticks(
            positions + (bar_width * (len(unique_categories) / 2)),
            df[category_column].unique(),
            rotation=45,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")


# Main execution section
if __name__ == "__main__":

    # Get database configuration from environment variables
    db_config = {
        "host": os.getenv("host"),
        "port": os.getenv("port"),
        "dbname": os.getenv("dbname"),
        "user": os.getenv("user"),
        "password": os.getenv("pass"),
    }

    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # For mobile data
    mobile_analyzer = StaffHealthAnalyzer(
        data_path=os.getenv("STAFF_HEALTH_DATA_MOBILE"),
        mode="mobile",
        api_key=api_key,
        db_config=db_config,
    )

    # For kiosk data
    kiosk_analyzer = StaffHealthAnalyzer(
        data_path=os.getenv("STAFF_HEALTH_DATA_KIOSK"),
        mode="kiosk",
        api_key=api_key,
        db_config=db_config,
    )

    # Generate report
    generated_report = mobile_analyzer.run_analysis(
        report_type="Latest",  # Latest | Trending
        health_measure="Overall",  # Overall | BMI | Hypertension | Stress | Wellness
        category="Age_range",  # Age_range, Gender, BMI, Overall | Weekly, Monthly, Quarterly, Yearly
        with_summary=True,  # Set to True if have an API key
        enable_display=True,  # Set to True to display visualization
        with_recommendations=True,  # Set to True to include health recommendations
    )

    print("Generated Report:")
    print(generated_report["data"])

    # Print summary if available
    if "summary" in generated_report:
        print("\nSummary:")
        print(generated_report["summary"])

    # Print recommendations if available
    if "recommendations" in generated_report:
        print("\nRecommendations:")
        print(generated_report["recommendations"])