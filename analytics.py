# analytics.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache
from typing import Dict, Optional, Union, List, Any, Tuple
from preprocessing import (
    HealthDataPreprocessor,
    HealthMeasure,
    VisualizationCategory,
    TimePeriod,
    ReportType,
    MODE_KIOSK,
    MODE_MOBILE,
    ERROR_BMI_KIOSK,
)


class StaffHealthAnalyzer:
    def __init__(
        self, 
        data_path: str, 
        api_key: Optional[str] = None, 
        mode: str = MODE_MOBILE,
    ):
        """
        Initialize the health analyzer with staff health data

        Parameters:
        data_path (str): Path to the CSV file with staff health data
        api_key (str, optional): OpenAI API key for natural language summaries
        mode (str): Analysis mode - 'kiosk' or 'mobile'
        """
        # Initialize the preprocessor and get processed data
        self.preprocessor = HealthDataPreprocessor(data_path, mode)
        self.df = self.preprocessor.get_processed_data()
        self.mode = mode

        # Initialize LLM if API key is provided
        self.llm = self._initialize_llm(api_key)

        # Get mappings from preprocessor
        self.measurement_mappings = self.preprocessor.measurement_mappings
        self.display_names = self.preprocessor.display_names
        self.visualization_mappings = self.preprocessor.visualization_mappings

        # Define LLM system prompt
        self.system_prompt = """Act as a health data analyst tasked with summarizing the health status of employees in a company. Below are the specifications for the report:
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

    def _initialize_llm(self, api_key: Optional[str]) -> Any:
        """Initialize LLM if API key is provided"""
        if not api_key:
            return None

        try:
            from langchain_openai import ChatOpenAI

            os.environ["OPENAI_API_KEY"] = api_key
            return ChatOpenAI(model="gpt-4o")
        except ImportError:
            print("Import unsuccessful. Please `pip install langchain_openai`")
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

    def _filter_by_time_period(self, time_period: str) -> Tuple[pd.DataFrame, str]:
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
        self, filtered_data: Tuple[pd.DataFrame, str], measure_col: str, display_name: str
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
        if not self.llm:
            return (
                "LLM not available. Set API key to enable natural language summaries."
            )

        formatted_table = self.format_table(data)

        formatted_prompt = self.system_prompt.format(
            report_type=report_type,
            health_measurement=health_measure,
            visualization_category=category,
            data=formatted_table,
        )

        try:
            return self.llm.invoke(formatted_prompt).content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def run_analysis(
        self,
        report_type: str,
        health_measure: str,
        category: str,
        with_summary: bool = False,
        enable_display: bool = False,
    ) -> Dict:
        """
        Run a complete analysis for the specified parameters

        Parameters:
        report_type (str): 'Latest' or 'Trending'
        health_measure (str): Health measure type (e.g., 'Overall', 'Hypertension')
        category (str): Category for latest report or time period for trending report
        with_summary (bool): Whether to include natural language summary
        enable_display (bool): Whether to display visualization

        Returns:
        Dict: Dictionary containing report data and optional summary
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
