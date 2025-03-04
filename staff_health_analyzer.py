# Import Libraries
import pandas as pd
from typing import Dict, Optional
from enum import Enum
import os
try:
    from langchain_openai import ChatOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print('Import unsuccessful. Please `pip install langchain_openai`')

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

class StaffHealthAnalyzer:
    def __init__(self, data_path: str, api_key: Optional[str] = None):
        """
        Initialize the health analyzer with staff health data
        
        Parameters:
        data_path (str): Path to the CSV file with staff health data
        api_key (str, optional): OpenAI API key for natural language summaries
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
            
        return self.df
    
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
            # Get the max date
            max_date = df['date'].max()
            
            # Generate the last 6 weeks based on ISO calendar
            weeks_range = pd.date_range(
                end=max_date, 
                periods=6, 
                freq='W-MON'  # Use Monday as the start of the week per ISO 8601
            )
            
            # Filter data for these 6 weeks
            df = df[df['date'].isin(weeks_range)].reset_index(drop=True)
            
            # Create time period label in YYYY-MM-DD format
            def format_week_period(date):
                return date.strftime('%Y-%m-%d')
            
            df['time_period'] = df['date'].apply(format_week_period)
            groupby_col = 'time_period'

        elif time_period == 'Monthly':
            # Filter data for the last 6 months
            df = df[df['date'] >= df['date'].max() - pd.DateOffset(months=6)].reset_index()

            df['time_period'] = df['date'].dt.to_period('M')
            groupby_col = 'time_period'

        elif time_period == 'Quarterly':
            # Use ISO 8601 caledar for quarters
            max_date = df['date'].max()
            max_year = max_date.year

            # Generate the last 6 quarters starting from the max date's year
            # Use ISO 8601 which defines Q1 as Jan-Mar, Q2 as Apr-Jun, etc.
            quarters = pd.period_range(
                start=f"{max_year}Q1",
                end=f"{max_year}Q4",
                freq='Q'
            )

            # Find the last 6 quarters
            last_6_quarters = quarters[-6:]

            # Filter data to only include these quarters
            df = df[df['date'].dt.to_period('Q').isin(last_6_quarters)].reset_index()

            # Format time_period as YY-MM-DD using the last date of each quarter
            def format_quarter_period(row):
                quarter_last_day = row['date'].to_period('Q').end_time
                return quarter_last_day.strftime("%y-%m-%d")

            df['time_period'] = df.apply(format_quarter_period, axis=1)
            groupby_col = 'time_period'

        elif time_period == 'Yearly':
            # Filter data for the last 6 years
            df = df[df['date'] >= df['date'].max() - pd.DateOffset(years=6)].reset_index()

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
        df_trend['Percentage'] = ((df_trend['Count'] / total_count) * 100).round(2)
        
        # Convert time_period to string for consistent output
        df_trend[groupby_col] = df_trend[groupby_col].astype(str)
        
        # Sort by time period for better visualization
        df_trend = df_trend.sort_values(by=[groupby_col, measure_col])
        
        # Rename the measure_col column to use the display name for consistency with latest report
        df_trend = df_trend.rename(columns={measure_col: display_name})
        
        return df_trend
    
    def generate_all_latest_reports(self) -> Dict[str, pd.DataFrame]:
        """Generate all required latest report combinations
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary of all reports with keys formatted as 'measure_category'
        """
        reports = {}
        
        # Define required combinations based on the requirement table
        # Excluding BMI|BMI which is not in the requirements
        required_combinations = [
            (measure.value, category.value)
            for measure in HealthMeasure
            for category in VisualizationCategory
            if not (measure.value == 'BMI' and category.value == 'BMI')
        ]
        
        # Generate each report
        for measure, category in required_combinations:
            key = f"{measure}_{category}".lower().replace(" ", "_")
            reports[key] = self.generate_latest_report(
                health_measure=measure,
                visualization_cat=category
            )
            
        return reports
    
    def generate_all_trending_reports(self) -> Dict[str, pd.DataFrame]:
        """Generate all required trending report combinations
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary of all reports with keys formatted as 'measure_timeperiod'
        """
        reports = {}
        
        # Define required combinations
        required_combinations = [
            (measure.value, period.value)
            for measure in HealthMeasure
            for period in TimePeriod
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
    
    def run_analysis(self, report_type: str, health_measure: str, category: str, with_summary: bool = False) -> Dict:
        """
        Run a complete analysis for the specified parameters
        
        Parameters:
        report_type (str): 'Latest' or 'Trending'
        health_measure (str): Health measure type (e.g., 'Overall', 'Hypertension')
        category (str): Category for latest report or time period for trending report
        with_summary (bool): Whether to include natural language summary
        
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
        
        return result


# Main execution section
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = StaffHealthAnalyzer(
        data_path='staff_health_data.csv',
        api_key=os.getenv("OPENAI_API_KEY")
    )

    """
    # Make directories for storing all latest and trending reports
    os.makedirs('reports/latest', exist_ok=True)
    os.makedirs('reports/trending', exist_ok=True)
    """

    # Generate report
    generated_report = analyzer.run_analysis(
        report_type='Trending', # Latest | Trending
        health_measure='Hypertension', # Overall | BMI | Hypertension | Stress | Wellness
        category='Weekly', # (Age range, Gender type, BMI) (Weekly, Monthly, Quarterly, Yearly)
        with_summary=False  # Set to True if have an API key
    )

    print('Generated Report:')
    print(generated_report['data'])

    # Print summary if available
    if 'summary' in generated_report:
        print('\nSummary:')
        print(generated_report['summary'])

    """
    # Generate and save all latest reports
    print('Generating and saving latest reports...')
    all_latest_reports = analyzer.generate_all_latest_reports()
    for key, report in all_latest_reports.items():
        filename = f"reports/latest/{key}.csv"
        report.to_csv(filename, index=False)
        print(f"Saved {filename}")

    # Generate and save all trending reports
    print('Generating and saving trending reports...')
    all_trending_reports = analyzer.generate_all_trending_reports()
    for key, report in all_trending_reports.items():
        filename = f"reports/trending/{key}.csv"
        report.to_csv(filename, index=False)
        print(f"Saved {filename}")

    print('\nAll reports have been saved successfully!')

    # Generate all latest reports
    all_latest_reports = analyzer.generate_all_latest_reports()

    print(f"\nGenerated {len(all_latest_reports)} latest reports")

    # Print details of each report
    for key, report in all_latest_reports.items():
        print(f"\n--- {key} ---")
        print(f"Shape: {report.shape}")
        print(report)
        print("...")
    
    # Generate all trending reports
    all_trending_reports = analyzer.generate_all_trending_reports()

    print(f"\nGenerated {len(all_trending_reports)} trending reports")

    # Print details of each report
    for key, report in all_trending_reports.items():
        print(f"\n--- {key} ---")
        print(f"Shape: {report.shape}")
        print(report)
        print("...")
    """