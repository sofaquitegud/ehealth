import os
from preprocessing import DataPreprocessor
from analytics import StaffHealthAnalyzer

# Main execution section
if __name__ == "__main__":
    # Initialize the preprocessor
    preprocessor = DataPreprocessor(data_path='staff_health_data.csv')
    
    # Initialize the analyzer with preprocessed data
    analyzer = StaffHealthAnalyzer(
        preprocessor=preprocessor,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Optional: Make directories for storing all latest and trending reports
    # os.makedirs('reports/latest', exist_ok=True)
    # os.makedirs('reports/trending', exist_ok=True)

    # Generate report example
    generated_report = analyzer.run_analysis(
        report_type='Trending',  # Latest | Trending
        health_measure='Hypertension',  # Overall | BMI | Hypertension | Stress | Wellness
        category='Monthly',  # (Age range, Gender type, BMI) or (Weekly, Monthly, Quarterly, Yearly)
        with_summary=False  # Set to True if have an API key
    )

    print('Generated Report:')
    print(generated_report['data'])

    # Print summary if available
    if 'summary' in generated_report:
        print('\nSummary:')
        print(generated_report['summary'])

    # Optional: Generate and save all reports
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
    """