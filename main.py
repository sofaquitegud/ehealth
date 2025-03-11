# main.py
import os
from preprocessing import StaffHealthPreprocessor, HealthMeasure
from analytics import StaffHealthAnalyzer, VisualizationCategory, TimePeriod, ReportType

def main():
    # Initialize the preprocessor
    preprocessor = StaffHealthPreprocessor(data_path='staff_health_data(1).csv')
    
    # Get processed data and mappings
    processed_data = preprocessor.get_processed_data()
    measurement_mappings = preprocessor.get_measurement_mappings()
    display_names = preprocessor.get_display_names()
    
    # Initialize the analyzer with preprocessed data
    analyzer = StaffHealthAnalyzer(
        processed_data=processed_data,
        measurement_mappings=measurement_mappings,
        display_names=display_names,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Example 1: Generate a single report
    report = analyzer.run_analysis(
        report_type='Trending',      # 'Latest' or 'Trending'
        health_measure='Overall',     # 'Overall', 'BMI', 'Hypertension', 'Stress', 'Wellness'
        category='Monthly',           # For Latest: 'Overall', 'Age range', 'Gender type', 'BMI'
                                                    # For Trending: 'Weekly', 'Monthly', 'Quarterly', 'Yearly'
        with_summary=False,
        enable_display=True
    )
    
    print("\nGenerated Report:")
    print(report['data'])
    
    if 'summary' in report:
        print("\nSummary:")
        print(report['summary'])
    
    # Example 2: Generate all latest reports
    """
    print("\nGenerating all latest reports...")
    health_measures = [measure.value for measure in HealthMeasure]
    viz_categories = [category.value for category in VisualizationCategory]
    
    all_latest_reports = analyzer.generate_all_latest_reports(
        health_measures=health_measures,
        visualization_categories=viz_categories
    )
    
    print(f"Generated {len(all_latest_reports)} latest reports")
    
    # Example 3: Generate all trending reports
    print("\nGenerating all trending reports...")
    time_periods = [period.value for period in TimePeriod]
    
    all_trending_reports = analyzer.generate_all_trending_reports(
        health_measures=health_measures,
        time_periods=time_periods
    )
    
    print(f"Generated {len(all_trending_reports)} trending reports")
    
    # Example 4: Save reports to files
    os.makedirs('reports/latest', exist_ok=True)
    os.makedirs('reports/trending', exist_ok=True)
    
    for key, report in all_latest_reports.items():
        filename = f"reports/latest/{key}.csv"
        report.to_csv(filename, index=False)
        print(f"Saved {filename}")
    
    for key, report in all_trending_reports.items():
        filename = f"reports/trending/{key}.csv"
        report.to_csv(filename, index=False)
        print(f"Saved {filename}")
    
    print("\nAll reports have been saved successfully!")"
    """

if __name__ == "__main__":
    main()