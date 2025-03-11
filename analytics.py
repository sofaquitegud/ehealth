# staff_health_analyzer.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, Optional
from enum import Enum

try:
    from langchain_openai import ChatOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print('Import unsuccessful. Please `pip install langchain_openai`')

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

class StaffHealthAnalyzer:
    def __init__(self, processed_data: pd.DataFrame, measurement_mappings: dict, display_names: dict, api_key: Optional[str] = None):
        """
        Initialize the health analyzer with preprocessed staff health data
        
        Parameters:
        processed_data (pd.DataFrame): Preprocessed staff health data
        measurement_mappings (dict): Mapping between health measures and dataframe columns
        display_names (dict): Mapping between health measures and display names
        api_key (str, optional): OpenAI API key for natural language summaries
        """
        # Store processed data
        self.df = processed_data
        
        # Store mappings
        self.measurement_mappings = measurement_mappings
        self.display_names = display_names
        
        # Visualization category mappings
        self.visualization_mappings = {
            'Overall': 'overall',
            'Age range': 'age_range',
            'Gender type': 'gender',
            'BMI': 'original_bmi'
        }
        
        # Initialize LLM if API key is provided
        self.llm = None
        if api_key and LLM_AVAILABLE:
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = ChatOpenAI(model="gpt-4o")
        
        # Define system prompt for LLM
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

    def get_latest_data(self) -> pd.DataFrame:
        """Get most recent data for each staff member"""
        return self.df.sort_values('date').groupby('staff_id').last().reset_index()
    
    def generate_latest_report(self, health_measure: str, visualization_cat: str) -> pd.DataFrame:
        """
        Generate a latest snapshot report for given health measure and visualization category
        
        Parameters:
        health_measure (str): Health measure to report on (Overall, Hypertension, Stress, Wellness, BMI)
        visualization_cat (str): Category to visualize by (Overall, Age range, Gender type, BMI)
        
        Returns:
        pd.DataFrame: Report data
        """
        # Get the latest data
        data = self.get_latest_data()
        
        # Get measure column name
        measure_col = self.measurement_mappings[health_measure]
        
        # Get display name for the health measure
        display_name = self.display_names[health_measure]
        
        # Get visualization category column
        viz_col = self.visualization_mappings[visualization_cat]
        
        # Set the appropriate category name
        if visualization_cat == 'Overall':
            category_name = 'Category'
        else:
            # Use the proper category name based on visualization_cat
            category_name = visualization_cat
        
        # Handle Overall visualization category
        if viz_col == 'overall':
            counts = data[measure_col].value_counts()
            
            report = pd.DataFrame({
                category_name: 'Overall',
                display_name: counts.index.tolist(),  # Use original categorical values
                'Count': counts.values,
                'Percentage': (counts.values / counts.sum() * 100).round(2)
            })
        else:
            # Create crosstab
            crosstab = pd.crosstab(data[measure_col], data[viz_col])
            percentages = crosstab.div(crosstab.sum(axis=0), axis=1) * 100
            
            # Create report with original categorical values
            report = []
            for measure_val in crosstab.index:
                for group_val in crosstab.columns:
                    report.append({
                        category_name: group_val,
                        display_name: measure_val,  # Use display_name for column name
                        'Count': crosstab.loc[measure_val, group_val],
                        'Percentage': percentages.loc[measure_val, group_val].round(2)
                    })
            report = pd.DataFrame(report)
        
        return report
    
    def generate_trending_report(self, health_measure: str, time_period: str) -> pd.DataFrame:
        """
        Generate a trending report for given health measure over a time period
        
        Parameters:
        health_measure (str): Health measure to report on (Overall, Hypertension, Stress, Wellness, BMI)
        time_period (str): Time period for trend analysis (Weekly, Monthly, Quarterly, Yearly)
        
        Returns:
        pd.DataFrame: Trend data
        """
        # Get measure column name
        measure_col = self.measurement_mappings[health_measure]
        display_name = self.display_names[health_measure]
        
        # Create time period column based on selection
        df = self.df.copy()
        
        if time_period == 'Weekly':
            # Filter data for the last 6 weeks
            df = df[df['date'] >= df['date'].max() - pd.Timedelta(weeks=5)].reset_index()

            df['time_period'] = df['date'].dt.to_period('W')
            groupby_col = 'time_period'

        elif time_period == 'Monthly':
            # Filter data for the last 6 months
            df = df[df['date'] >= df['date'].max() - pd.DateOffset(months=5)].reset_index()

            df['time_period'] = df['date'].dt.to_period('M')
            groupby_col = 'time_period'

        elif time_period == 'Quarterly':
            # Filter data for the last 6 quarters
            df = df[df['date'] >= df['date'].max() - pd.DateOffset(months=6*3-2)].reset_index()

            df['time_period'] = df['date'].dt.to_period('Q').apply(lambda x: x.start_time.strftime('%Y-%m-%d'))
            groupby_col = 'time_period'

        elif time_period == 'Yearly':
            # Filter data for the last 6 years
            df = df[df['date'] >= df['date'].max() - pd.DateOffset(years=5)].reset_index()

            df['time_period'] = df['date'].dt.year
            groupby_col = 'time_period'
        else:
            raise ValueError(f"Invalid time period: {time_period}")
        
        # Get the latest date in each time period
        latest_dates = df.groupby(groupby_col)['date'].max().reset_index()
        
        # Merge with the original dataset to keep only the latest records per time period
        df_latest = df.merge(latest_dates, on=[groupby_col, 'date'])
        
        # Define all possible categories for measure
        all_measure_categories = df[measure_col].unique()
        all_periods = df_latest[groupby_col].unique()
        
        # Create a complete MultiIndex with all combinations
        multi_index = pd.MultiIndex.from_product(
            [all_periods, all_measure_categories], 
            names=[groupby_col, measure_col]
        )
        
        # Group by measure and time period, then count occurrences
        df_trend = df_latest.groupby([groupby_col, measure_col]).size().reset_index(name='Count')
        
        # Reindex to ensure missing categories are filled with zero
        df_trend = df_trend.set_index([groupby_col, measure_col]).reindex(multi_index, fill_value=0).reset_index()
        
        # Calculate percentage within each time period based on total count in that period
        total_count = df_trend.groupby(groupby_col)['Count'].transform('sum')
        df_trend['Percentage'] = ((df_trend['Count']/total_count)*100).round(2)
        
        # Convert time_period to string for consistent output
        df_trend[groupby_col] = df_trend[groupby_col].astype(str)
        
        # Sort by time period for better visualization
        df_trend = df_trend.sort_values(by=[groupby_col, measure_col])
        
        # Rename the measure_col column to use the display name for consistency with latest report
        df_trend = df_trend.rename(columns={measure_col: display_name})
        
        return df_trend
    
    def generate_all_latest_reports(self, health_measures, visualization_categories) -> Dict[str, pd.DataFrame]:
        """Generate all required latest report combinations
        
        Parameters:
        health_measures (list): List of health measures
        visualization_categories (list): List of visualization categories
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary of all reports with keys formatted as 'measure_category'
        """
        reports = {}
        
        # Define required combinations based on the inputs
        # Excluding BMI|BMI which is not in the requirements
        required_combinations = [
            (measure, category)
            for measure in health_measures
            for category in visualization_categories
            if not (measure == 'BMI' and category == 'BMI')
        ]
        
        # Generate each report
        for measure, category in required_combinations:
            key = f"{measure}_{category}".lower().replace(" ", "_")
            reports[key] = self.generate_latest_report(
                health_measure=measure,
                visualization_cat=category
            )
            
        return reports
    
    def generate_all_trending_reports(self, health_measures, time_periods) -> Dict[str, pd.DataFrame]:
        """Generate all required trending report combinations
        
        Parameters:
        health_measures (list): List of health measures
        time_periods (list): List of time periods
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary of all reports with keys formatted as 'measure_timeperiod'
        """
        reports = {}
        
        # Define required combinations
        required_combinations = [
            (measure, period)
            for measure in health_measures
            for period in time_periods
        ]
        
        # Generate each report
        for measure, period in required_combinations:
            key = f"{measure}_{period}".lower().replace(" ", "_")
            reports[key] = self.generate_trending_report(
                health_measure=measure,
                time_period=period
            )
            
        return reports
    
    def format_table(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to formatted markdown-style table with percentage formatting"""
        # Format percentages if column exists
        if 'Percentage' in df.columns:
            df = df.copy()
            df['Percentage'] = df['Percentage'].apply(lambda x: f"{x:.2f}%")
        
        return df.to_markdown(index=False)
    
    def get_report_summary(self, report_type: str, health_measure: str, category: str, data: pd.DataFrame) -> str:
        """
        Generate a natural language summary of the report using LLM if available
        
        Parameters:
        report_type (str): 'Latest' or 'Trending'
        health_measure (str): Health measure type
        category (str): Specific categorization or time period
        data (pd.DataFrame): Report data
        
        Returns:
        str: Natural language summary or empty string if LLM not available
        """
        if not self.llm:
            return "LLM not available. Set API key to enable natural language summaries."
        
        formatted_table = self.format_table(data)
        
        formatted_prompt = self.system_prompt.format(
            report_type=report_type,
            health_measurement=health_measure,
            visualization_category=category,
            data=formatted_table
        )
        
        try:
            return self.llm.invoke(formatted_prompt).content
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def run_analysis(self, report_type: str, health_measure: str, category: str, with_summary: bool = False, enable_display: bool = False) -> Dict:
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
        result = {"report_type": report_type, "health_measure": health_measure, "category": category}
        
        if report_type == 'Latest':
            data = self.generate_latest_report(health_measure, category)
        elif report_type == 'Trending':
            data = self.generate_trending_report(health_measure, category)
        else:
            raise ValueError(f"Invalid report type: {report_type}")
        
        result["data"] = data
        
        if with_summary and self.llm:
            result["summary"] = self.get_report_summary(report_type, health_measure, category, data)

        if enable_display:
            self.visualize_report(result, report_type)
        
        return result
    
    def visualize_report(self, result: Dict, report_type: str):
        """
        Visualize the report data
        
        Parameters:
        result (Dict): Report data dictionary
        report_type (str): 'Latest' or 'Trending'
        """
        df_plot = result["data"]

        # Check if dataframe is not empty
        if not df_plot.empty:
            time_column = df_plot.columns[0]  # The first column (time period e.g., year_week)
            category_column = df_plot.columns[1]  # The health measure category

            plt.figure(figsize=(12, 6))

            # Get unique categories
            unique_categories = df_plot[category_column].unique()

            # Define color palette
            colors = plt.get_cmap("tab10", len(unique_categories))

            if report_type == 'Trending':
                # Plot each category separately
                for idx, category in enumerate(unique_categories):
                    category_data = df_plot[df_plot[category_column] == category]
                    plt.plot(category_data[time_column], category_data["Percentage"],
                             marker="o", linestyle="-", label=f"{category}", color=colors(idx))

                # Formatting
                plt.xlabel(time_column, fontsize=12)
                plt.ylabel("Percentage (%)", fontsize=12)
                plt.title(f"Trend Analysis: {category_column} Over {time_column}", fontsize=14)
                plt.xticks(rotation=45)
                plt.grid(True, linestyle="--", alpha=0.6)

            else:
                # Set width for bars
                bar_width = 0.2
                positions = np.arange(len(df_plot[time_column].unique()))  # Positions for bars

                # Create bars for each category
                for idx, category in enumerate(unique_categories):
                    category_data = df_plot[df_plot[category_column] == category]

                    plt.bar(positions + (idx * bar_width), category_data["Percentage"],
                            width=bar_width, label=f"{category}", color=colors(idx))

                # Formatting
                plt.xlabel(time_column, fontsize=12)
                plt.ylabel("Percentage (%)", fontsize=12)
                plt.title(f"Latest Analysis: {category_column} Over {time_column}", fontsize=14)
                plt.xticks(positions + (bar_width * (len(unique_categories) / 2)), df_plot[time_column].unique(), rotation=45)
                plt.grid(axis="y", linestyle="--", alpha=0.6)

            # Add legend
            plt.legend(loc="upper right")

            # Show plot
            plt.show()